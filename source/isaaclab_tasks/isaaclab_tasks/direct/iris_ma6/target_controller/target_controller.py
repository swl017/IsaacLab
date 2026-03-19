# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-based target controller using DroneController.

This module provides target movement using the same cascaded control
architecture as agents, producing realistic physics-based motion
instead of direct velocity writes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..controller import DroneController
from ..controller.controller_cfg import DroneControllerCfg
from ..controller.velocity_controller_cfg import VelocityControllerCfg
from ..controller.attitude_controller_cfg import AttitudeControllerCfg
from ..controller.rate_controller_cfg import RateControllerCfg
from ..controller.motor_dynamics_cfg import MotorDynamicsCfg
from ..controller.gimbal_controller_cfg import GimbalControllerCfg
from ..controller.zoom_controller_cfg import ZoomControllerCfg
from ..controller.aerodynamics_cfg import AerodynamicsCfg

from .target_controller_cfg import TargetControllerCfg, BEHAVIOR_PROFILES
from .behavior_fsm import BehaviorFSM, FSMState, VelocityMode
from .velocity_generators import (
    LinearModeGenerator,
    CircularModeGenerator,
    ApproachModeGenerator,
    EvadeModeGenerator,
)


class TargetController:
    """Physics-based target controller using DroneController.

    This class wraps the DroneController to provide target movement
    with realistic physics-based dynamics. It manages:
    - Velocity generation based on FSM state and mode
    - Behavior profile assignment and management
    - Integration with DroneController for force/torque output
    - Geofencing and altitude constraints

    Usage:
        cfg = TargetControllerCfg()
        controller = TargetController(
            cfg=cfg,
            mass=1.5,
            gravity=9.81,
            num_envs=4096,
            num_targets=10,
            device=device,
        )

        # In environment step:
        F_body, tau_body = controller.step(
            current_position=target_pos,
            current_velocity=target_vel,
            current_quat=target_quat,
            current_angular_vel=target_omega,
            facility_position=facility_pos,
            interceptor_positions=defender_pos,
            interceptor_roles=defender_roles,
            curriculum_progress=progress,
            dt=sim_dt,
        )

        # Apply forces
        target.set_external_force_and_torque(forces=F_body, torques=tau_body, ...)
    """

    def __init__(
        self,
        cfg: TargetControllerCfg,
        mass: float,
        gravity: float,
        num_envs: int,
        num_targets: int,
        device: torch.device | str = "cpu",
    ):
        """Initialize the target controller.

        Args:
            cfg: Target controller configuration.
            mass: Target mass [kg].
            gravity: Gravity magnitude [m/s^2].
            num_envs: Number of parallel environments.
            num_targets: Maximum number of targets per environment.
            device: Torch device (cuda or cpu).
        """
        self.cfg = cfg
        self.mass = mass
        self.gravity = gravity
        self.num_envs = num_envs
        self.num_targets = num_targets
        self.device = torch.device(device)
        self.total_targets = num_envs * num_targets

        # Create DroneController with simplified gains
        drone_cfg = self._create_drone_controller_cfg()
        self._drone_controller = DroneController(
            cfg=drone_cfg,
            mass=mass,
            gravity=gravity,
            num_envs=self.total_targets,  # One controller per target
            device=self.device,
        )

        # Create behavior FSM
        self._fsm = BehaviorFSM(
            cfg=cfg,
            num_envs=num_envs,
            num_targets=num_targets,
            device=self.device,
        )

        # Create velocity generators
        self._velocity_generators: Dict[str, object] = {
            "linear": LinearModeGenerator(cfg, num_envs, num_targets, self.device),
            "circular": CircularModeGenerator(cfg, num_envs, num_targets, self.device),
            "approach": ApproachModeGenerator(cfg, num_envs, num_targets, self.device),
            "evade": EvadeModeGenerator(cfg, num_envs, num_targets, self.device),
        }

        # Environment origins (for geofencing and circular mode)
        self._env_origins = torch.zeros(num_envs, 3, device=self.device)

        # Behavior profile parameters (flattened)
        self._speed_multiplier = torch.ones(self.total_targets, device=self.device)
        self._evasion_agility = torch.zeros(self.total_targets, device=self.device)

    def _create_drone_controller_cfg(self) -> DroneControllerCfg:
        """Create DroneControllerCfg with simplified gains for targets.

        Returns:
            DroneControllerCfg configured for target movement.
        """
        cfg = self.cfg

        return DroneControllerCfg(
            velocity=VelocityControllerCfg(
                Kp_vel=cfg.velocity_kp,
                Ki_vel=cfg.velocity_ki,
                integral_limit=cfg.velocity_integral_limit,
                max_tilt=cfg.max_tilt,
                max_lin_vel=cfg.max_lin_vel,
            ),
            attitude=AttitudeControllerCfg(
                Kp_att=cfg.attitude_kp,
            ),
            rate=RateControllerCfg(
                Kp_rate=cfg.rate_kp,
                Ki_rate=cfg.rate_ki,
                Kd_rate=cfg.rate_kd,
            ),
            motor=MotorDynamicsCfg(
                tau_motor=cfg.motor_tau,
            ),
            gimbal=GimbalControllerCfg(),  # Unused
            zoom=ZoomControllerCfg(),  # Unused
            aerodynamics=AerodynamicsCfg(
                fidelity_level=cfg.aerodynamics_fidelity_level,
            ),
            control_dt=cfg.control_dt,
        )

    def step(
        self,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        current_quat: torch.Tensor,
        current_angular_vel: torch.Tensor,
        facility_position: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        curriculum_progress: float,
        dt: float,
        env_origins: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step target controller and return body forces/torques.

        Args:
            current_position: [N_env, N_target, 3] positions in world frame.
            current_velocity: [N_env, N_target, 3] velocities in world frame.
            current_quat: [N_env, N_target, 4] orientations (wxyz).
            current_angular_vel: [N_env, N_target, 3] angular velocities in body frame.
            facility_position: [N_env, 3] facility positions.
            interceptor_positions: [N_env, N_defender, 3] defender positions.
            interceptor_roles: [N_env, N_defender] defender roles (0=OBSERVE, 1=INTERCEPT).
            curriculum_progress: Curriculum progress (0-1).
            dt: Simulation timestep [s].
            env_origins: [N_env, 3] environment origins for geofencing (optional).

        Returns:
            F_body: [N_env, N_target, 3] body frame forces [N].
            tau_body: [N_env, N_target, 3] body frame torques [Nm].
        """
        # Update environment origins if provided
        if env_origins is not None:
            self._env_origins = env_origins

        # Flatten tensors for processing
        pos_flat = current_position.view(-1, 3)
        vel_flat = current_velocity.view(-1, 3)
        quat_flat = current_quat.view(-1, 4)
        omega_flat = current_angular_vel.view(-1, 3)

        # Expand facility position to per-target
        facility_expanded = (
            facility_position.unsqueeze(1)
            .expand(-1, self.num_targets, -1)
            .reshape(-1, 3)
        )

        # Update FSM states (check transitions)
        self._update_fsm(
            pos_flat,
            interceptor_positions,
            interceptor_roles,
            facility_position,
            curriculum_progress,
        )

        # Generate velocity commands based on FSM state and mode
        v_cmd = self._compute_velocity_command(
            pos_flat=pos_flat,
            facility_expanded=facility_expanded,
            interceptor_positions=interceptor_positions,
            interceptor_roles=interceptor_roles,
            curriculum_progress=curriculum_progress,
            dt=dt,
        )

        # Apply geofencing and altitude constraints
        v_cmd = self._apply_constraints(v_cmd, pos_flat, curriculum_progress)

        # Zero out velocity for dead targets
        alive_flat = self._fsm.alive.view(-1)
        v_cmd[~alive_flat] = 0.0

        # Store velocity command for debugging
        self._debug_v_cmd = v_cmd.clone()

        # Pass through DroneController
        # Target has no gimbal — pass zeros for joint positions
        gimbal_jp_zeros = torch.zeros(self.total_targets, 3, device=self.device)

        F_body_flat, tau_body_flat, _, _, _ = self._drone_controller.step_policy(
            v_cmd=v_cmd,
            yaw_rate_cmd=torch.zeros(self.total_targets, device=self.device),
            gimbal_yaw_rate_cmd=torch.zeros(self.total_targets, device=self.device),
            gimbal_pitch_rate_cmd=torch.zeros(self.total_targets, device=self.device),
            zoom_rate_cmd=torch.zeros(self.total_targets, device=self.device),
            q_body=quat_flat,
            v_body=vel_flat,
            omega_body=omega_flat,
            sim_dt=dt,
            gimbal_joint_positions=gimbal_jp_zeros,
        )

        # Zero out forces for dead targets
        F_body_flat[~alive_flat] = 0.0
        tau_body_flat[~alive_flat] = 0.0

        # Reshape back to [N_env, N_target, 3]
        F_body = F_body_flat.view(self.num_envs, self.num_targets, 3)
        tau_body = tau_body_flat.view(self.num_envs, self.num_targets, 3)

        return F_body, tau_body

    def _update_fsm(
        self,
        pos_flat: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        facility_position: torch.Tensor,
        curriculum_progress: float,
    ):
        """Update FSM states based on current conditions.

        Args:
            pos_flat: [N_total, 3] flattened target positions.
            interceptor_positions: [N_env, N_defender, 3] defender positions.
            interceptor_roles: [N_env, N_defender] defender roles.
            facility_position: [N_env, 3] facility positions.
            curriculum_progress: Curriculum progress (0-1).
        """
        # Reshape position for FSM
        current_position = pos_flat.view(self.num_envs, self.num_targets, 3)

        # Get evasion timer from evade generator
        evasion_timer = self._velocity_generators["evade"].evasion_timer

        # Update FSM
        transition_to_evade, transition_to_approach, _ = self._fsm.update(
            current_position=current_position,
            interceptor_positions=interceptor_positions,
            interceptor_roles=interceptor_roles,
            facility_position=facility_position,
            facility_radius=10.0,  # Default facility radius
            evasion_agility=self._evasion_agility,
            evasion_timer=evasion_timer,
        )

        # Start evasion for transitioning targets
        if len(transition_to_evade) > 0:
            self._velocity_generators["evade"].start_evasion(transition_to_evade)

    def _compute_velocity_command(
        self,
        pos_flat: torch.Tensor,
        facility_expanded: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        curriculum_progress: float,
        dt: float,
    ) -> torch.Tensor:
        """Compute velocity commands based on FSM state and mode.

        Args:
            pos_flat: [N_total, 3] flattened target positions.
            facility_expanded: [N_total, 3] facility positions per target.
            interceptor_positions: [N_env, N_defender, 3] defender positions.
            interceptor_roles: [N_env, N_defender] defender roles.
            curriculum_progress: Curriculum progress (0-1).
            dt: Timestep [s].

        Returns:
            v_cmd: [N_total, 3] velocity commands.
        """
        v_cmd = torch.zeros(self.total_targets, 3, device=self.device)

        # Get alive indices
        alive_indices = self._fsm.get_alive_indices()

        if len(alive_indices) == 0:
            return v_cmd

        # Get FSM state and velocity mode for alive targets
        fsm_flat = self._fsm.fsm_state.view(-1)
        mode_flat = self._fsm.velocity_mode.view(-1)

        # Process by FSM state
        # -----------------------------------------------------------------
        # APPROACH state: use velocity mode (linear, circular, or approach)
        # -----------------------------------------------------------------
        approach_mask = fsm_flat[alive_indices] == FSMState.APPROACH
        approach_indices = alive_indices[approach_mask]

        if len(approach_indices) > 0:
            approach_mode = mode_flat[approach_indices]

            # Linear mode
            linear_mask = approach_mode == VelocityMode.LINEAR
            if linear_mask.any():
                linear_indices = approach_indices[linear_mask]
                v_cmd[linear_indices] = self._velocity_generators["linear"].compute(
                    indices=linear_indices,
                    current_position=pos_flat[linear_indices],
                    curriculum_progress=curriculum_progress,
                    dt=dt,
                )

            # Circular mode
            circular_mask = approach_mode == VelocityMode.CIRCULAR
            if circular_mask.any():
                circular_indices = approach_indices[circular_mask]
                v_cmd[circular_indices] = self._velocity_generators["circular"].compute(
                    indices=circular_indices,
                    current_position=pos_flat[circular_indices],
                    curriculum_progress=curriculum_progress,
                    dt=dt,
                )

            # Approach mode (attacker behavior)
            approach_mode_mask = approach_mode == VelocityMode.APPROACH
            if approach_mode_mask.any():
                approach_mode_indices = approach_indices[approach_mode_mask]
                v_cmd[approach_mode_indices] = self._velocity_generators[
                    "approach"
                ].compute(
                    indices=approach_mode_indices,
                    current_position=pos_flat[approach_mode_indices],
                    curriculum_progress=curriculum_progress,
                    dt=dt,
                    facility_position=facility_expanded[approach_mode_indices],
                )

        # -----------------------------------------------------------------
        # EVADE state: use evade mode
        # -----------------------------------------------------------------
        evade_mask = fsm_flat[alive_indices] == FSMState.EVADE
        evade_indices = alive_indices[evade_mask]

        if len(evade_indices) > 0:
            v_cmd[evade_indices] = self._velocity_generators["evade"].compute(
                indices=evade_indices,
                current_position=pos_flat[evade_indices],
                curriculum_progress=curriculum_progress,
                dt=dt,
                interceptor_positions=interceptor_positions,
                interceptor_roles=interceptor_roles,
            )

        return v_cmd

    def _apply_constraints(
        self,
        v_cmd: torch.Tensor,
        pos_flat: torch.Tensor,
        curriculum_progress: float,
    ) -> torch.Tensor:
        """Apply geofencing and altitude constraints to velocity commands.

        Args:
            v_cmd: [N_total, 3] velocity commands.
            pos_flat: [N_total, 3] current positions.
            curriculum_progress: Curriculum progress (0-1).

        Returns:
            Modified velocity commands.
        """
        # Expand env_origins to per-target
        origins_expanded = (
            self._env_origins.unsqueeze(1)
            .expand(-1, self.num_targets, -1)
            .reshape(-1, 3)
        )

        # Apply geofencing
        v_cmd = self._apply_geofence_constraint(
            v_cmd, pos_flat, origins_expanded, curriculum_progress
        )

        # Apply altitude constraint
        v_cmd = self._apply_altitude_constraint(v_cmd, pos_flat, origins_expanded)

        return v_cmd

    def _apply_geofence_constraint(
        self,
        v_cmd: torch.Tensor,
        pos_flat: torch.Tensor,
        origins_expanded: torch.Tensor,
        curriculum_progress: float,
    ) -> torch.Tensor:
        """Apply geofencing velocity constraints.

        Args:
            v_cmd: [N_total, 3] velocity commands.
            pos_flat: [N_total, 3] current positions.
            origins_expanded: [N_total, 3] environment origins.
            curriculum_progress: Curriculum progress (0-1).

        Returns:
            Modified velocity commands.
        """
        # Compute geofence size based on curriculum
        geofence_size = self.cfg.geofence_min_size + curriculum_progress * (
            self.cfg.geofence_max_size - self.cfg.geofence_min_size
        )

        # Position relative to environment origin
        relative_pos = pos_flat - origins_expanded

        # Apply bounce at X boundaries
        x_too_low = relative_pos[:, 0] < -geofence_size
        x_too_high = relative_pos[:, 0] > geofence_size
        v_cmd[x_too_low, 0] = (
            torch.abs(v_cmd[x_too_low, 0]) * self.cfg.geofence_bounce_factor
        )
        v_cmd[x_too_high, 0] = (
            -torch.abs(v_cmd[x_too_high, 0]) * self.cfg.geofence_bounce_factor
        )

        # Apply bounce at Y boundaries
        y_too_low = relative_pos[:, 1] < -geofence_size
        y_too_high = relative_pos[:, 1] > geofence_size
        v_cmd[y_too_low, 1] = (
            torch.abs(v_cmd[y_too_low, 1]) * self.cfg.geofence_bounce_factor
        )
        v_cmd[y_too_high, 1] = (
            -torch.abs(v_cmd[y_too_high, 1]) * self.cfg.geofence_bounce_factor
        )

        return v_cmd

    def _apply_altitude_constraint(
        self,
        v_cmd: torch.Tensor,
        pos_flat: torch.Tensor,
        origins_expanded: torch.Tensor,
    ) -> torch.Tensor:
        """Apply altitude constraints.

        Uses the same bounce pattern as geofencing: velocity magnitude is preserved
        and direction is forced to point back within bounds.

        Args:
            v_cmd: [N_total, 3] velocity commands.
            pos_flat: [N_total, 3] current positions.
            origins_expanded: [N_total, 3] environment origins.

        Returns:
            Modified velocity commands.
        """
        altitude = pos_flat[:, 2] - origins_expanded[:, 2]

        # Minimum altitude - force upward velocity (positive Z)
        too_low = altitude < self.cfg.min_altitude
        v_cmd[too_low, 2] = (
            torch.abs(v_cmd[too_low, 2]) * self.cfg.geofence_bounce_factor
        )

        # Maximum altitude - force downward velocity (negative Z)
        too_high = altitude > self.cfg.max_altitude
        v_cmd[too_high, 2] = (
            -torch.abs(v_cmd[too_high, 2]) * self.cfg.geofence_bounce_factor
        )

        return v_cmd

    def reset(self, env_ids: torch.Tensor):
        """Reset target controller state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if len(env_ids) == 0:
            return

        # Reset FSM
        self._fsm.reset(env_ids)

        # Reset velocity generators
        for generator in self._velocity_generators.values():
            generator.reset(env_ids)

        # Reset DroneController for all targets in these environments
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            self._drone_controller.reset(flat_indices)

        # Assign behavior profiles
        self._assign_behavior_profiles(env_ids)

    def _assign_behavior_profiles(self, env_ids: torch.Tensor):
        """Assign behavior profiles to targets in specified environments.

        Args:
            env_ids: Environment indices to assign profiles.
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Profile weights
        weights = torch.tensor(
            [
                self.cfg.behavior_kamikaze_weight,
                self.cfg.behavior_standard_weight,
                self.cfg.behavior_evasive_weight,
                self.cfg.behavior_stealth_weight,
            ],
            device=self.device,
        )
        weights = weights / weights.sum()
        cumsum = torch.cumsum(weights, dim=0)

        profile_names = ["standard"]
        # profile_names = ["kamikaze", "standard", "evasive", "stealth"]

        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            n = len(flat_indices)

            # Sample profile
            probs = torch.rand(n, device=self.device)
            profile_idx = torch.zeros(n, dtype=torch.long, device=self.device)
            for i in range(len(cumsum) - 1):
                profile_idx[probs > cumsum[i]] = i + 1

            # Apply profile parameters
            for i, name in enumerate(profile_names):
                mask = profile_idx == i
                if not mask.any():
                    continue

                profile = BEHAVIOR_PROFILES[name]
                indices = flat_indices[mask]

                self._speed_multiplier[indices] = profile.speed_multiplier
                self._evasion_agility[indices] = profile.evasion_agility

                # Set path type weights for approach mode
                self._velocity_generators["approach"].set_speed_multiplier(
                    indices, torch.full((len(indices),), profile.speed_multiplier, device=self.device)
                )
                self._velocity_generators["evade"].set_evasion_agility(
                    indices, torch.full((len(indices),), profile.evasion_agility, device=self.device)
                )

    def set_circular_center(self, env_ids: torch.Tensor, center: torch.Tensor):
        """Set circular orbit center for specified environments.

        Args:
            env_ids: Environment indices.
            center: [N_env, 3] center positions.
        """
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            self._velocity_generators["circular"].set_center(flat_indices, center)

    def set_approach_waypoints(
        self,
        env_ids: torch.Tensor,
        current_position: torch.Tensor,
        facility_position: torch.Tensor,
    ):
        """Generate and set approach waypoints for specified environments.

        Args:
            env_ids: Environment indices.
            current_position: [N_env, N_target, 3] current target positions.
            facility_position: [N_env, 3] facility positions.
        """
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            target_pos = current_position[:, target_idx, :]

            self._velocity_generators["approach"].set_waypoints(
                flat_indices, target_pos, facility_position
            )

    # =========================================================================
    # Properties for external access
    # =========================================================================

    @property
    def fsm(self) -> BehaviorFSM:
        """Get behavior FSM."""
        return self._fsm

    @property
    def alive(self) -> torch.Tensor:
        """Get alive status [N_env, N_target]."""
        return self._fsm.alive

    @property
    def fsm_state(self) -> torch.Tensor:
        """Get FSM state [N_env, N_target]."""
        return self._fsm.fsm_state

    def check_intercept(
        self,
        current_position: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        capture_distance: float = 5.0,
    ) -> torch.Tensor:
        """Check for intercept events.

        Args:
            current_position: [N_env, N_target, 3] target positions.
            interceptor_positions: [N_env, N_defender, 3] defender positions.
            interceptor_roles: [N_env, N_defender] defender roles.
            capture_distance: Distance threshold for capture [m].

        Returns:
            intercepted_indices: Flat indices of intercepted targets.
        """
        return self._fsm.check_intercept(
            current_position=current_position,
            interceptor_positions=interceptor_positions,
            interceptor_roles=interceptor_roles,
            capture_distance=capture_distance,
        )
