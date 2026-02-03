# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target movement system with acceleration-based velocity control."""

from __future__ import annotations

import torch
import math
from typing import Optional

from .target_movement_cfg import TargetMovementCfg


class TargetMovement:
    """Modular target movement system for multi-agent drone environments.

    Features:
    - Two flight path modes: linear (straight) and circular (orbit)
    - Acceleration-based velocity control with damping
    - Configurable geofencing with curriculum scaling
    - Altitude constraints with bounce behavior
    - Random velocity/path updates at configurable intervals

    Usage:
        cfg = TargetMovementCfg()
        movement = TargetMovement(cfg, num_envs=4096, device=device)

        # In _apply_action:
        velocity = movement.step(
            current_position=target.data.root_pos_w,
            env_origins=env_origins,
            curriculum_progress=progress_move,
            dt=step_dt,
        )
        target.write_root_velocity_to_sim(velocity)

        # In _reset_idx:
        movement.reset(env_ids=reset_env_ids)
    """

    def __init__(
        self,
        cfg: TargetMovementCfg,
        num_envs: int,
        device: torch.device,
    ):
        """Initialize target movement system.

        Args:
            cfg: Configuration for target movement.
            num_envs: Number of parallel environments.
            device: PyTorch device (cuda or cpu).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # =====================================================================
        # Internal State Buffers
        # =====================================================================

        # Current velocity [N, 6] - linear (3) + angular (3)
        self.velocity = torch.zeros(num_envs, 6, device=device)

        # Desired velocity for tracking controller
        self.desired_velocity = torch.zeros(num_envs, 6, device=device)

        # Acceleration buffer
        self.acceleration = torch.zeros(num_envs, 6, device=device)

        # Motion mode: 0 = linear, 1 = circular
        self.motion_mode = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Circular motion parameters
        self.circular_radius = torch.zeros(num_envs, device=device)
        self.circular_angular_speed = torch.zeros(num_envs, device=device)
        self.circular_phase = torch.zeros(num_envs, device=device)
        self.circular_center = torch.zeros(num_envs, 3, device=device)
        self.circular_height = torch.zeros(num_envs, device=device)

        # Timer for velocity/path updates
        self.update_timer = torch.zeros(num_envs, device=device)
        self.next_update_time = torch.zeros(num_envs, device=device)

        # Simulation time tracker
        self.sim_time = torch.zeros(num_envs, device=device)

        # Initialize states
        self._initialize_all()

    def _initialize_all(self):
        """Initialize all internal states for all environments."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.reset(all_ids)

    def reset(self, env_ids: torch.Tensor):
        """Reset movement state for specified environments.

        Args:
            env_ids: Tensor of environment indices to reset.
        """
        n = len(env_ids)
        if n == 0:
            return

        # Reset buffers
        self.desired_velocity[env_ids] = 0.0
        self.acceleration[env_ids] = 0.0

        # Randomly assign motion mode
        mode_probs = torch.rand(n, device=self.device)
        self.motion_mode[env_ids] = (mode_probs > self.cfg.linear_weight).long()

        # Initialize circular parameters for circular mode environments
        circular_mask = self.motion_mode[env_ids] == 1
        circular_ids = env_ids[circular_mask]
        n_circular = len(circular_ids)

        if n_circular > 0:
            self.circular_radius[circular_ids] = (
                torch.rand(n_circular, device=self.device)
                * (self.cfg.circular_radius_max - self.cfg.circular_radius_min)
                + self.cfg.circular_radius_min
            )

            # Angular speed = linear_speed / radius (clamped to max angular speed)
            base_speed = torch.rand(n_circular, device=self.device) * self.cfg.max_speed * 0.5
            self.circular_angular_speed[circular_ids] = torch.clamp(
                base_speed / self.circular_radius[circular_ids],
                max=self.cfg.max_angular_speed,
            )

            self.circular_phase[circular_ids] = torch.rand(n_circular, device=self.device) * 2 * math.pi

            self.circular_height[circular_ids] = (
                torch.rand(n_circular, device=self.device)
                * (self.cfg.circular_height_max - self.cfg.circular_height_min)
                + self.cfg.circular_height_min
            )

            # Initialize circular mode with tangential velocity
            phase = self.circular_phase[circular_ids]
            radius = self.circular_radius[circular_ids]
            angular_speed = self.circular_angular_speed[circular_ids]
            tangent_speed = radius * angular_speed  # v = r * omega

            # Tangential direction: perpendicular to radial direction
            # If position is at angle phase, tangent is at angle (phase + pi/2)
            self.velocity[circular_ids, 0] = -tangent_speed * torch.sin(phase)  # -sin for CCW rotation
            self.velocity[circular_ids, 1] = tangent_speed * torch.cos(phase)   # cos for CCW rotation
            self.velocity[circular_ids, 2] = 0.0
            self.velocity[circular_ids, 3:] = 0.0

        # Initialize linear mode velocities
        linear_mask = self.motion_mode[env_ids] == 0
        linear_ids = env_ids[linear_mask]
        n_linear = len(linear_ids)

        if n_linear > 0:
            # Random initial velocity (same as desired velocity for immediate motion)
            # Generate random direction and random speed
            random_direction = torch.rand(n_linear, 3, device=self.device) * 2 - 1
            direction_norm = torch.norm(random_direction, dim=1, keepdim=True)
            random_direction = random_direction / (direction_norm + 1e-6)  # Normalize to unit vector
            random_speed = torch.rand(n_linear, 1, device=self.device) * self.cfg.max_speed
            random_velocity = random_direction * random_speed

            self.velocity[linear_ids, :3] = random_velocity
            self.velocity[linear_ids, 3:] = 0.0
            self.desired_velocity[linear_ids, :3] = random_velocity

        # Reset timers (use start interval values for reset, curriculum_progress=0)
        self.update_timer[env_ids] = 0.0
        self.next_update_time[env_ids] = (
            torch.rand(n, device=self.device)
            * (self.cfg.update_interval_max_start - self.cfg.update_interval_min_start)
            + self.cfg.update_interval_min_start
        )

        # Reset simulation time
        self.sim_time[env_ids] = 0.0

    def step(
        self,
        current_position: torch.Tensor,
        env_origins: torch.Tensor,
        curriculum_progress: float = 1.0,
        dt: float = 0.02,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute target velocity for the next simulation step.

        Args:
            current_position: Current target positions [N, 3] in world frame.
            env_origins: Environment origin positions [N, 3] for geofencing.
            curriculum_progress: Curriculum progress (0-1) for geofence scaling.
            dt: Simulation timestep (seconds).
            env_ids: Optional subset of environments to update.

        Returns:
            velocity: Target velocity [N, 6] to write to simulation.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)

        # Update simulation time
        self.sim_time[env_ids] += dt

        # Store env_origins for circular mode center reference
        self.circular_center[env_ids] = env_origins[env_ids] if env_ids.numel() < self.num_envs else env_origins

        # =====================================================================
        # Check for velocity/mode updates
        # =====================================================================
        self._check_and_apply_updates(env_ids, dt, curriculum_progress)

        # =====================================================================
        # Compute desired velocity based on mode
        # =====================================================================
        linear_mask = self.motion_mode[env_ids] == 0
        circular_mask = ~linear_mask

        # Linear mode: desired velocity already set, just track it
        # Circular mode: compute desired velocity from orbit
        if circular_mask.any():
            circular_ids = env_ids[circular_mask]
            circular_positions = current_position[circular_mask] if env_ids.numel() < self.num_envs else current_position[circular_ids]
            circular_origins = env_origins[circular_mask] if env_ids.numel() < self.num_envs else env_origins[circular_ids]
            self._compute_circular_desired_velocity(circular_ids, circular_positions, circular_origins, dt)

        # =====================================================================
        # Acceleration-based velocity control
        # =====================================================================
        vel_error = self.desired_velocity[env_ids] - self.velocity[env_ids]
        self.acceleration[env_ids] = vel_error * self.cfg.acceleration_scale

        # Clamp acceleration
        acc_magnitude = torch.norm(self.acceleration[env_ids, :3], dim=1, keepdim=True)
        acc_scale = torch.clamp(self.cfg.max_acceleration / (acc_magnitude + 1e-6), max=1.0)
        self.acceleration[env_ids, :3] = self.acceleration[env_ids, :3] * acc_scale

        # Update velocity
        self.velocity[env_ids] += self.acceleration[env_ids] * dt
        self.velocity[env_ids] *= self.cfg.velocity_damping

        # Clamp velocity magnitude
        speed = torch.norm(self.velocity[env_ids, :3], dim=1, keepdim=True)
        speed_scale = torch.clamp(self.cfg.max_speed / (speed + 1e-6), max=1.0)
        self.velocity[env_ids, :3] = self.velocity[env_ids, :3] * speed_scale

        # =====================================================================
        # Apply geofencing and altitude constraints
        # =====================================================================
        positions_to_check = current_position[env_ids] if env_ids.numel() < self.num_envs else current_position
        origins_to_check = env_origins[env_ids] if env_ids.numel() < self.num_envs else env_origins

        self._apply_geofencing(env_ids, positions_to_check, origins_to_check, curriculum_progress)
        self._apply_altitude_constraint(env_ids, positions_to_check, origins_to_check)

        # =====================================================================
        # Scale by curriculum-based speed limit
        # =====================================================================
        speed_scale = self.cfg.speed_scale_start + curriculum_progress * (
            self.cfg.speed_scale_end - self.cfg.speed_scale_start
        )
        output_velocity = self.velocity.clone()
        output_velocity[:, :3] *= speed_scale  # Scale linear velocity only

        return output_velocity

    def _check_and_apply_updates(self, env_ids: torch.Tensor, dt: float, curriculum_progress: float = 1.0):
        """Check and apply velocity/mode updates based on timer or probability."""
        n = len(env_ids)

        if self.cfg.use_timer_based_updates:
            # Timer-based updates
            self.update_timer[env_ids] += dt
            update_mask = self.update_timer[env_ids] >= self.next_update_time[env_ids]
        else:
            # Probability-based updates
            update_mask = torch.rand(n, device=self.device) < self.cfg.direction_change_prob

        if update_mask.any():
            update_ids = env_ids[update_mask]
            self._apply_velocity_update(update_ids, curriculum_progress)

        # Check for mode switching
        if self.cfg.allow_mode_switching:
            switch_mask = torch.rand(n, device=self.device) < self.cfg.mode_switch_prob
            if switch_mask.any():
                switch_ids = env_ids[switch_mask]
                self._switch_mode(switch_ids, curriculum_progress)

    def _apply_velocity_update(self, env_ids: torch.Tensor, curriculum_progress: float = 1.0):
        """Apply a new random velocity update for specified environments."""
        n = len(env_ids)

        # Linear mode: new random desired velocity
        linear_mask = self.motion_mode[env_ids] == 0
        linear_ids = env_ids[linear_mask]
        n_linear = len(linear_ids)

        if n_linear > 0:
            self.desired_velocity[linear_ids, :3] = (
                torch.rand(n_linear, 3, device=self.device) * 2 - 1
            ) * self.cfg.max_speed
            self.desired_velocity[linear_ids, 5] = (
                torch.rand(n_linear, device=self.device) * 2 - 1
            ) * self.cfg.max_angular_speed * 0.1  # Slow yaw rotation

        # Circular mode: new random radius and angular speed
        circular_mask = ~linear_mask
        circular_ids = env_ids[circular_mask]
        n_circular = len(circular_ids)

        if n_circular > 0:
            self.circular_radius[circular_ids] = (
                torch.rand(n_circular, device=self.device)
                * (self.cfg.circular_radius_max - self.cfg.circular_radius_min)
                + self.cfg.circular_radius_min
            )

            base_speed = torch.rand(n_circular, device=self.device) * self.cfg.max_speed * 0.5
            self.circular_angular_speed[circular_ids] = torch.clamp(
                base_speed / self.circular_radius[circular_ids],
                max=self.cfg.max_angular_speed,
            )

            self.circular_height[circular_ids] = (
                torch.rand(n_circular, device=self.device)
                * (self.cfg.circular_height_max - self.cfg.circular_height_min)
                + self.cfg.circular_height_min
            )

        # Reset update timer with curriculum-scaled intervals
        self.update_timer[env_ids] = 0.0
        interval_min = self.cfg.update_interval_min_start + curriculum_progress * (
            self.cfg.update_interval_min_end - self.cfg.update_interval_min_start
        )
        interval_max = self.cfg.update_interval_max_start + curriculum_progress * (
            self.cfg.update_interval_max_end - self.cfg.update_interval_max_start
        )
        self.next_update_time[env_ids] = (
            torch.rand(n, device=self.device) * (interval_max - interval_min) + interval_min
        )

    def _switch_mode(self, env_ids: torch.Tensor, curriculum_progress: float = 1.0):
        """Switch motion mode for specified environments."""
        self.motion_mode[env_ids] = 1 - self.motion_mode[env_ids]

        # Reinitialize parameters for new mode
        self._apply_velocity_update(env_ids, curriculum_progress)

    def _compute_circular_desired_velocity(
        self,
        env_ids: torch.Tensor,
        current_position: torch.Tensor,
        env_origins: torch.Tensor,
        dt: float,
    ):
        """Compute desired velocity for circular orbit mode."""
        n = len(env_ids)

        # Update phase
        self.circular_phase[env_ids] += self.circular_angular_speed[env_ids] * dt

        # Compute desired position on orbit
        radius = self.circular_radius[env_ids]
        phase = self.circular_phase[env_ids]

        center_xy = env_origins[:, :2]
        desired_x = center_xy[:, 0] + radius * torch.cos(phase)
        desired_y = center_xy[:, 1] + radius * torch.sin(phase)
        desired_z = env_origins[:, 2] + self.circular_height[env_ids]

        desired_pos = torch.stack([desired_x, desired_y, desired_z], dim=1)
        pos_error = desired_pos - current_position

        # Proportional control to compute desired velocity
        kp = 1.0
        self.desired_velocity[env_ids, :3] = kp * pos_error

        # Clamp desired velocity
        speed = torch.norm(self.desired_velocity[env_ids, :3], dim=1, keepdim=True)
        speed_clamped = torch.clamp(speed, max=self.cfg.max_speed)
        self.desired_velocity[env_ids, :3] = (
            self.desired_velocity[env_ids, :3] / (speed + 1e-6) * speed_clamped
        )

        # No rotation in circular mode
        self.desired_velocity[env_ids, 3:] = 0.0

    def _apply_geofencing(
        self,
        env_ids: torch.Tensor,
        current_position: torch.Tensor,
        env_origins: torch.Tensor,
        curriculum_progress: float,
    ):
        """Apply geofencing constraints with curriculum-scaled area."""
        # Compute current geofence size based on curriculum
        geofence_size = self.cfg.geofence_min_size + curriculum_progress * (
            self.cfg.geofence_max_size - self.cfg.geofence_min_size
        )

        # Position relative to environment origin
        relative_pos = current_position - env_origins

        # Check X bounds
        x_too_low = relative_pos[:, 0] < -geofence_size
        x_too_high = relative_pos[:, 0] > geofence_size

        # Get the indices in env_ids that violate bounds
        x_low_indices = env_ids[x_too_low]
        x_high_indices = env_ids[x_too_high]

        if len(x_low_indices) > 0:
            self.velocity[x_low_indices, 0] = (
                torch.abs(self.velocity[x_low_indices, 0]) * self.cfg.geofence_bounce_factor
            )
        if len(x_high_indices) > 0:
            self.velocity[x_high_indices, 0] = (
                -torch.abs(self.velocity[x_high_indices, 0]) * self.cfg.geofence_bounce_factor
            )

        # Check Y bounds
        y_too_low = relative_pos[:, 1] < -geofence_size
        y_too_high = relative_pos[:, 1] > geofence_size

        y_low_indices = env_ids[y_too_low]
        y_high_indices = env_ids[y_too_high]

        if len(y_low_indices) > 0:
            self.velocity[y_low_indices, 1] = (
                torch.abs(self.velocity[y_low_indices, 1]) * self.cfg.geofence_bounce_factor
            )
        if len(y_high_indices) > 0:
            self.velocity[y_high_indices, 1] = (
                -torch.abs(self.velocity[y_high_indices, 1]) * self.cfg.geofence_bounce_factor
            )

    def _apply_altitude_constraint(
        self,
        env_ids: torch.Tensor,
        current_position: torch.Tensor,
        env_origins: torch.Tensor,
    ):
        """Apply minimum and maximum altitude constraints."""
        # Altitude relative to terrain (env origin Z)
        altitude = current_position[:, 2] - env_origins[:, 2]

        # Check minimum altitude
        too_low = altitude < self.cfg.min_altitude
        low_indices = env_ids[too_low]

        # Bounce upward if too low
        if len(low_indices) > 0:
            self.velocity[low_indices, 2] = torch.clamp(
                self.velocity[low_indices, 2],
                min=self.cfg.altitude_bounce_velocity,
            )

        # Check maximum altitude
        too_high = altitude > self.cfg.max_altitude
        high_indices = env_ids[too_high]

        # Bounce downward if too high
        if len(high_indices) > 0:
            self.velocity[high_indices, 2] = torch.clamp(
                self.velocity[high_indices, 2],
                max=-self.cfg.altitude_bounce_velocity,
            )

    def get_motion_mode(self) -> torch.Tensor:
        """Get current motion mode for all environments."""
        return self.motion_mode

    def get_velocity(self) -> torch.Tensor:
        """Get current velocity buffer."""
        return self.velocity

    def set_motion_mode(self, env_ids: torch.Tensor, mode: int, curriculum_progress: float = 1.0):
        """Force set motion mode for specified environments.

        Args:
            env_ids: Environment indices.
            mode: 0 for linear, 1 for circular.
            curriculum_progress: Curriculum progress for interval scaling.
        """
        self.motion_mode[env_ids] = mode
        self._apply_velocity_update(env_ids, curriculum_progress)
