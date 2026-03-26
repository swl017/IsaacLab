# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core generator for initial states randomization in iris_ma6.

This module implements cylinder-based agent placement with curriculum-driven
randomization to prevent catastrophic forgetting during RL training.

Key Design:
    - Curriculum sampling: uniform(min, min + progress * (max - min))
    - Rejection sampling for agent clearance with fallback to even placement
    - Designated observer has gimbal pointing at target
    - Other agents follow curriculum-based gimbal randomization
"""

from __future__ import annotations

import math
import torch
from typing import Optional

from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse

from .initial_states_cfg import InitialStatesCfg, InitialStatesResult


class InitialStatesGenerator:
    """Generator for initial states with cylinder-based agent placement.

    Usage:
        ```python
        cfg = InitialStatesCfg()
        generator = InitialStatesGenerator(cfg, num_envs=256, num_agents=3, device)

        # Generate at different curriculum stages
        result = generator.generate(curriculum_progress=0.0)  # Easy
        result = generator.generate(curriculum_progress=1.0)  # Hard
        ```
    """

    def __init__(
        self,
        cfg: InitialStatesCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the generator.

        Args:
            cfg: Initial states configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device for tensor operations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Pre-allocate index tensor for all environments
        self._ALL_INDICES = torch.arange(num_envs, dtype=torch.long, device=device)

        # Counter for rotating designated observer mode
        self._rotating_counter = 0

    def generate(
        self,
        env_ids: Optional[torch.Tensor] = None,
        curriculum_progress: float = 0.5,
    ) -> InitialStatesResult:
        """Generate initial states for specified environments.

        The generation follows these steps:
        1. Sample curriculum-controlled parameters
        2. Generate cylinder centers
        3. Place agents in cylinder with clearance
        4. Generate target position
        5. Generate target velocity
        6. Select designated observer
        7. Generate agent orientations
        8. Generate agent velocities
        9. Generate gimbal states
        10. Generate zoom levels

        Args:
            env_ids: Environment indices to generate for (None = all).
            curriculum_progress: Progress value [0, 1] controlling difficulty.
                - 0.0: Easy (close targets, low velocities, all pointing)
                - 1.0: Hard (far targets, high velocities, random gimbal)

        Returns:
            InitialStatesResult with all state tensors.
        """
        if env_ids is None:
            env_ids = self._ALL_INDICES
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, curriculum_progress))

        # ======================================================================
        # Step 1: Sample curriculum-controlled parameters
        # ======================================================================
        params = self._sample_curriculum_parameters(num_envs, progress)

        # ======================================================================
        # Step 2: Generate cylinder centers
        # ======================================================================
        cylinder_centers = self._generate_cylinder_centers(num_envs)

        # ======================================================================
        # Step 3: Place agents in cylinder with clearance
        # ======================================================================
        agent_positions = self._generate_agent_positions_in_cylinder(
            num_envs, cylinder_centers, params["diameters"], progress
        )

        # ======================================================================
        # Step 4: Generate target position
        # ======================================================================
        target_positions = self._generate_target_positions(
            num_envs, cylinder_centers, params["target_distances"]
        )

        # ======================================================================
        # Step 5: Generate target velocity
        # ======================================================================
        target_velocities = self._generate_target_velocities(
            num_envs, params["target_velocity_scales"]
        )

        # ======================================================================
        # Step 6: Select designated observer
        # ======================================================================
        designated_observer_idx = self._select_designated_observers(num_envs)

        # ======================================================================
        # Step 7: Generate agent orientations
        # ======================================================================
        agent_orientations = self._generate_agent_orientations(
            agent_positions, target_positions, designated_observer_idx, progress
        )

        # ======================================================================
        # Step 8: Generate agent velocities
        # ======================================================================
        agent_linear_velocities, agent_angular_velocities = self._generate_agent_velocities(
            num_envs, params["agent_velocity_scales"], progress
        )

        # ======================================================================
        # Step 9: Generate gimbal states
        # ======================================================================
        gimbal_joint_positions = self._generate_gimbal_states(
            agent_positions,
            agent_orientations,
            target_positions,
            designated_observer_idx,
            progress,
        )

        # ======================================================================
        # Step 10: Generate zoom levels
        # ======================================================================
        zoom_levels = self._generate_zoom_levels(num_envs, curriculum_progress)

        # ======================================================================
        # Compute distances for logging
        # ======================================================================
        distances_to_target = self._compute_distances_to_target(
            agent_positions, target_positions
        )

        # ======================================================================
        # Assemble result
        # ======================================================================
        # Target orientation: identity quaternion (no rotation)
        target_orientations = torch.zeros(num_envs, 4, device=self.device)
        target_orientations[:, 0] = 1.0  # wxyz format, identity = [1,0,0,0]

        return InitialStatesResult(
            agent_positions=agent_positions,
            agent_orientations=agent_orientations,
            agent_linear_velocities=agent_linear_velocities,
            agent_angular_velocities=agent_angular_velocities,
            target_positions=target_positions,
            target_orientations=target_orientations,
            target_velocities=target_velocities,
            gimbal_joint_positions=gimbal_joint_positions,
            zoom_levels=zoom_levels,
            designated_observer_idx=designated_observer_idx,
            cylinder_centers=cylinder_centers,
            distances_to_target=distances_to_target,
        )

    # ==========================================================================
    # Curriculum Parameter Sampling
    # ==========================================================================

    def _sample_curriculum_parameters(
        self, num_envs: int, progress: float
    ) -> dict[str, torch.Tensor]:
        """Sample curriculum-controlled parameters.

        Uses uniform(min, min + progress * (max - min)) to prevent forgetting.

        Args:
            num_envs: Number of environments.
            progress: Curriculum progress [0, 1].

        Returns:
            Dictionary with sampled parameter tensors.
        """
        cfg = self.cfg

        # Diameter: uniform(min, min + p * (max - min))
        diameter_range = progress * (cfg.cylinder_diameter_max - cfg.cylinder_diameter_min)
        diameters = (
            torch.rand(num_envs, device=self.device) * diameter_range
            + cfg.cylinder_diameter_min
        )

        # Target distance: uniform(min, min + p * (max - min))
        distance_range = progress * (cfg.target_distance_max - cfg.target_distance_min)
        target_distances = (
            torch.rand(num_envs, device=self.device) * distance_range
            + cfg.target_distance_min
        )

        # Agent velocity scale: uniform(0, p * scale_max)
        agent_velocity_scales = (
            torch.rand(num_envs, device=self.device) * progress * cfg.agent_velocity_scale_max
        )

        # Target velocity scale: uniform(0, p * scale_max)
        target_velocity_scales = (
            torch.rand(num_envs, device=self.device) * progress * cfg.target_velocity_scale_max
        )

        return {
            "diameters": diameters,
            "target_distances": target_distances,
            "agent_velocity_scales": agent_velocity_scales,
            "target_velocity_scales": target_velocity_scales,
        }

    # ==========================================================================
    # Cylinder and Agent Placement
    # ==========================================================================

    def _generate_cylinder_centers(self, num_envs: int) -> torch.Tensor:
        """Generate random cylinder center positions.

        Centers are positioned with random XY offset and random height.

        Args:
            num_envs: Number of environments.

        Returns:
            centers: [num_envs, 3] cylinder center positions.
        """
        cfg = self.cfg

        # Random XY offset (relative to env origin, applied later)
        x = torch.rand(num_envs, device=self.device) * 10.0 - 5.0  # [-5, 5]
        y = torch.rand(num_envs, device=self.device) * 10.0 - 5.0  # [-5, 5]

        # Random height
        z = (
            torch.rand(num_envs, device=self.device)
            * (cfg.cylinder_height_max - cfg.cylinder_height_min)
            + cfg.cylinder_height_min
        )

        return torch.stack([x, y, z], dim=1)

    def _generate_agent_positions_in_cylinder(
        self,
        num_envs: int,
        centers: torch.Tensor,
        diameters: torch.Tensor,
        progress: float = 1.0,
    ) -> torch.Tensor:
        """Place agents randomly within cylinder using rejection sampling.

        Uses rejection sampling to ensure minimum clearance between agents.
        Falls back to evenly distributed placement if sampling fails.

        Args:
            num_envs: Number of environments.
            centers: [num_envs, 3] cylinder center positions.
            diameters: [num_envs] cylinder diameters per environment.
            progress: Curriculum progress [0, 1] for height range scaling.

        Returns:
            positions: [num_envs, num_agents, 3] agent positions.
        """
        cfg = self.cfg
        positions = torch.zeros(num_envs, self.num_agents, 3, device=self.device)

        # Curriculum-controlled vertical spread
        height_range = cfg.cylinder_height_range_min + progress * (
            cfg.cylinder_height_range_max - cfg.cylinder_height_range_min
        )

        # NOTE: Python for-loop is acceptable here — rejection sampling with
        # sequential clearance checks cannot be easily vectorized.
        for env_idx in range(num_envs):
            center = centers[env_idx]
            radius = diameters[env_idx] / 2.0

            placed = 0
            attempts = 0
            max_attempts = cfg.max_placement_retries * self.num_agents

            while placed < self.num_agents and attempts < max_attempts:
                # Sample random position in cylinder (uniform in disk)
                r = torch.sqrt(torch.rand(1, device=self.device)) * radius
                theta = torch.rand(1, device=self.device) * 2 * math.pi
                z_offset = torch.rand(1, device=self.device) * height_range

                x = center[0] + r * torch.cos(theta)
                y = center[1] + r * torch.sin(theta)
                z = center[2] + z_offset

                candidate = torch.tensor(
                    [x.item(), y.item(), z.item()], device=self.device
                )

                # Check clearance with already placed agents
                if placed == 0:
                    positions[env_idx, placed] = candidate
                    placed += 1
                else:
                    dists = (positions[env_idx, :placed] - candidate).norm(dim=-1)
                    if dists.min() >= cfg.agent_clearance:
                        positions[env_idx, placed] = candidate
                        placed += 1

                attempts += 1

            # Fallback to even placement if rejection sampling failed
            if placed < self.num_agents:
                positions[env_idx] = self._fallback_even_placement(
                    center, radius.item(), self.num_agents, height_range
                )

        return positions

    def _fallback_even_placement(
        self,
        center: torch.Tensor,
        radius: float,
        num_agents: int,
        height_range: float,
    ) -> torch.Tensor:
        """Fallback placement: evenly distribute agents around cylinder.

        Args:
            center: [3] cylinder center.
            radius: Cylinder radius.
            num_agents: Number of agents to place.
            height_range: Height range for distribution.

        Returns:
            positions: [num_agents, 3] agent positions.
        """
        positions = torch.zeros(num_agents, 3, device=self.device)

        for i in range(num_agents):
            # Even angular distribution
            theta = 2 * math.pi * i / num_agents
            # Use 70% of radius for safety
            r = 0.7 * radius

            positions[i, 0] = center[0] + r * math.cos(theta)
            positions[i, 1] = center[1] + r * math.sin(theta)
            # Spread height evenly
            z_offset = height_range * i / max(num_agents - 1, 1)
            positions[i, 2] = center[2] + z_offset

        return positions

    # ==========================================================================
    # Target Generation
    # ==========================================================================

    def _generate_target_positions(
        self,
        num_envs: int,
        cylinder_centers: torch.Tensor,
        target_distances: torch.Tensor,
    ) -> torch.Tensor:
        """Generate target positions at curriculum-controlled distance.

        Args:
            num_envs: Number of environments.
            cylinder_centers: [num_envs, 3] cylinder centers.
            target_distances: [num_envs] distance per environment.

        Returns:
            positions: [num_envs, 3] target positions.
        """
        cfg = self.cfg

        # Random bearing from cylinder center
        bearings = torch.rand(num_envs, device=self.device) * 2 * math.pi

        # Compute XY position
        target_x = cylinder_centers[:, 0] + target_distances * torch.cos(bearings)
        target_y = cylinder_centers[:, 1] + target_distances * torch.sin(bearings)

        # Height offset from cylinder center
        height_offset = (
            torch.rand(num_envs, device=self.device)
            * (cfg.target_height_offset_max - cfg.target_height_offset_min)
            + cfg.target_height_offset_min
        )
        target_z = cylinder_centers[:, 2] + height_offset

        # Ensure above ground
        target_z = torch.clamp(target_z, min=2.0)

        return torch.stack([target_x, target_y, target_z], dim=1)

    def _generate_target_velocities(
        self,
        num_envs: int,
        velocity_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Generate random target velocities.

        Args:
            num_envs: Number of environments.
            velocity_scales: [num_envs] velocity scale per environment.

        Returns:
            velocities: [num_envs, 6] linear (3) + angular (3) velocities.
        """
        cfg = self.cfg
        velocities = torch.zeros(num_envs, 6, device=self.device)

        # Random direction (unit vector)
        direction = torch.randn(num_envs, 3, device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Magnitude = scale * max_velocity
        magnitudes = velocity_scales * cfg.target_max_velocity

        # Linear velocity (first 3 components)
        velocities[:, :3] = direction * magnitudes.unsqueeze(-1)

        # No angular velocity for target
        return velocities

    # ==========================================================================
    # Designated Observer Selection
    # ==========================================================================

    def _select_designated_observers(self, num_envs: int) -> torch.Tensor:
        """Select designated observer index for each environment.

        Args:
            num_envs: Number of environments.

        Returns:
            indices: [num_envs] designated observer indices (0 to num_agents-1).
        """
        cfg = self.cfg

        if cfg.designated_observer_mode == "fixed":
            return torch.zeros(num_envs, dtype=torch.long, device=self.device)

        elif cfg.designated_observer_mode == "random":
            return torch.randint(
                0, self.num_agents, (num_envs,), dtype=torch.long, device=self.device
            )

        elif cfg.designated_observer_mode == "rotating":
            idx = self._rotating_counter % self.num_agents
            self._rotating_counter += 1
            return torch.full(
                (num_envs,), idx, dtype=torch.long, device=self.device
            )

        else:
            raise ValueError(f"Unknown observer mode: {cfg.designated_observer_mode}")

    # ==========================================================================
    # Agent Orientations
    # ==========================================================================

    def _generate_agent_orientations(
        self,
        agent_positions: torch.Tensor,
        target_positions: torch.Tensor,
        designated_observer_idx: torch.Tensor,
        progress: float = 1.0,
    ) -> torch.Tensor:
        """Generate agent body orientations.

        Designated observer faces target; others based on config mode.

        Args:
            agent_positions: [num_envs, num_agents, 3]
            target_positions: [num_envs, 3]
            designated_observer_idx: [num_envs]
            progress: Curriculum progress [0, 1] for "curriculum" orientation mode.

        Returns:
            orientations: [num_envs, num_agents, 4] quaternions (wxyz).
        """
        cfg = self.cfg
        num_envs = agent_positions.shape[0]

        # Compute yaw toward target for all agents: [num_envs, num_agents]
        to_target = target_positions.unsqueeze(1) - agent_positions  # [E, A, 3]
        yaw_to_target = torch.atan2(to_target[:, :, 1], to_target[:, :, 0])  # [E, A]

        # Add noise to target-facing yaw
        noise = torch.randn(num_envs, self.num_agents, device=self.device) * cfg.orientation_noise_std
        yaw_facing = yaw_to_target + noise

        # Random yaw for all agents
        yaw_random = (torch.rand(num_envs, self.num_agents, device=self.device) * 2 - 1) * math.pi

        # Build observer mask: [num_envs, num_agents] bool
        agent_indices = torch.arange(self.num_agents, device=self.device).unsqueeze(0)  # [1, A]
        is_observer = agent_indices == designated_observer_idx.unsqueeze(1)  # [E, A]

        # Start with random yaw, then apply rules
        yaw = yaw_random.clone()

        if cfg.other_agents_orientation_mode == "face_target":
            # All non-observers face target
            yaw = yaw_facing
        elif cfg.other_agents_orientation_mode == "curriculum":
            # With probability (1-progress), face target; otherwise random
            face_mask = torch.rand(num_envs, self.num_agents, device=self.device) < (1.0 - progress)
            yaw = torch.where(face_mask, yaw_facing, yaw_random)
        # else: "random" — yaw stays random

        # Observer always faces target (if configured)
        if cfg.designated_observer_faces_target:
            yaw = torch.where(is_observer, yaw_facing, yaw)

        # Convert to quaternions (roll=0, pitch=0, yaw)
        zeros = torch.zeros(num_envs * self.num_agents, device=self.device)
        orientations = quat_from_euler_xyz(
            zeros,
            zeros,
            yaw.reshape(-1),
        )  # [E*A, 4]

        return orientations.reshape(num_envs, self.num_agents, 4)

    # ==========================================================================
    # Agent Velocities
    # ==========================================================================

    def _generate_agent_velocities(
        self,
        num_envs: int,
        velocity_scales: torch.Tensor,
        progress: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate agent linear and angular velocities.

        Args:
            num_envs: Number of environments.
            velocity_scales: [num_envs] velocity scale per environment.
            progress: Curriculum progress [0, 1].

        Returns:
            linear_vel: [num_envs, num_agents, 3]
            angular_vel: [num_envs, num_agents, 3]
        """
        cfg = self.cfg

        # Linear velocity — random direction and per-agent random magnitude
        direction = torch.randn(num_envs, self.num_agents, 3, device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Per-agent random magnitude in [0, scale * max_vel]
        magnitudes = torch.rand(num_envs, self.num_agents, device=self.device) * (
            velocity_scales.unsqueeze(-1) * cfg.agent_max_velocity
        )
        linear_vel = direction * magnitudes.unsqueeze(-1)

        # Angular velocity (only yaw rate, scaled by progress)
        angular_vel = torch.zeros(num_envs, self.num_agents, 3, device=self.device)
        yaw_rate_scale = progress * cfg.max_yaw_rate
        angular_vel[:, :, 2] = (
            torch.rand(num_envs, self.num_agents, device=self.device) * 2 - 1
        ) * yaw_rate_scale

        return linear_vel, angular_vel

    # ==========================================================================
    # Gimbal States
    # ==========================================================================

    def _generate_gimbal_states(
        self,
        agent_positions: torch.Tensor,
        agent_orientations: torch.Tensor,
        target_positions: torch.Tensor,
        designated_observer_idx: torch.Tensor,
        progress: float,
    ) -> torch.Tensor:
        """Generate gimbal joint positions.

        Designated observer gimbal points at target.
        Other agents follow curriculum-based randomization.

        Args:
            agent_positions: [num_envs, num_agents, 3]
            agent_orientations: [num_envs, num_agents, 4] quaternions (wxyz)
            target_positions: [num_envs, 3]
            designated_observer_idx: [num_envs]
            progress: Curriculum progress [0, 1].

        Returns:
            gimbal_joint_positions: [num_envs, num_agents, 3] = [yaw, roll, pitch]
                Joint order matches robot articulation.
        """
        cfg = self.cfg
        num_envs = agent_positions.shape[0]

        # Compute pointing angles for all agents (vectorized)
        # Flatten to [E*A, ...] for _compute_gimbal_angles_to_target
        flat_pos = agent_positions.reshape(-1, 3)  # [E*A, 3]
        flat_quat = agent_orientations.reshape(-1, 4)  # [E*A, 4]
        flat_target = target_positions.unsqueeze(1).expand(
            -1, self.num_agents, -1
        ).reshape(-1, 3)  # [E*A, 3]

        yaw_pointing, pitch_pointing = self._compute_gimbal_angles_to_target(
            flat_pos, flat_quat, flat_target
        )
        yaw_pointing = yaw_pointing.reshape(num_envs, self.num_agents)  # [E, A]
        pitch_pointing = pitch_pointing.reshape(num_envs, self.num_agents)  # [E, A]

        # Random gimbal angles
        yaw_random = (
            torch.rand(num_envs, self.num_agents, device=self.device)
            * (cfg.gimbal_yaw_max - cfg.gimbal_yaw_min)
            + cfg.gimbal_yaw_min
        )
        pitch_random = (
            torch.rand(num_envs, self.num_agents, device=self.device)
            * (cfg.gimbal_pitch_max - cfg.gimbal_pitch_min)
            + cfg.gimbal_pitch_min
        )

        # Observer mask: [E, A]
        agent_indices = torch.arange(self.num_agents, device=self.device).unsqueeze(0)
        is_observer = agent_indices == designated_observer_idx.unsqueeze(1)

        # Apply curriculum mode for non-observers
        if cfg.gimbal_curriculum_mode == "always_pointing":
            yaw = yaw_pointing.clone()
            pitch = pitch_pointing.clone()

        elif cfg.gimbal_curriculum_mode == "gradual":
            # With probability (1-progress), point at target; otherwise random
            point_mask = torch.rand(num_envs, self.num_agents, device=self.device) < (1 - progress)
            yaw = torch.where(point_mask, yaw_pointing, yaw_random)
            pitch = torch.where(point_mask, pitch_pointing, pitch_random)

        elif cfg.gimbal_curriculum_mode == "threshold":
            if progress < cfg.gimbal_randomization_threshold:
                yaw = yaw_pointing.clone()
                pitch = pitch_pointing.clone()
            else:
                yaw = yaw_random
                pitch = pitch_random

        else:
            raise ValueError(f"Unknown gimbal mode: {cfg.gimbal_curriculum_mode}")

        # Designated observer always points at target
        yaw = torch.where(is_observer, yaw_pointing, yaw)
        pitch = torch.where(is_observer, pitch_pointing, pitch)

        # Clamp to limits
        yaw = yaw.clamp(cfg.gimbal_yaw_min, cfg.gimbal_yaw_max)
        pitch = pitch.clamp(cfg.gimbal_pitch_min, cfg.gimbal_pitch_max)

        # Assemble: [yaw, roll=0, pitch]
        gimbal_positions = torch.zeros(num_envs, self.num_agents, 3, device=self.device)
        gimbal_positions[:, :, 0] = yaw
        gimbal_positions[:, :, 2] = pitch

        return gimbal_positions

    def _compute_gimbal_angles_to_target(
        self,
        agent_pos: torch.Tensor,
        agent_quat: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gimbal yaw/pitch to point camera at target.

        Uses the same computation as GimbalController._world_to_body_angles.

        Args:
            agent_pos: [N, 3] agent world position.
            agent_quat: [N, 4] agent orientation (wxyz).
            target_pos: [N, 3] target world position.

        Returns:
            yaw: [N] gimbal yaw angle (radians).
            pitch: [N] gimbal pitch angle (radians).
        """
        # Direction to target in world frame
        dir_world = target_pos - agent_pos
        dir_world = dir_world / (dir_world.norm(dim=-1, keepdim=True) + 1e-8)

        # Transform to body frame
        dir_body = quat_rotate_inverse(agent_quat, dir_world)

        # Extract gimbal angles (same as GimbalController._world_to_body_angles)
        # Yaw: rotation around body Z axis
        yaw = torch.atan2(dir_body[:, 1], dir_body[:, 0])

        # Pitch: rotation around body Y axis (after yaw)
        xy_dist = torch.sqrt(dir_body[:, 0] ** 2 + dir_body[:, 1] ** 2)
        pitch = -torch.atan2(dir_body[:, 2], xy_dist)

        return yaw, pitch

    # ==========================================================================
    # Zoom Levels
    # ==========================================================================

    def _generate_zoom_levels(self, num_envs: int, curriculum_progress: float = 0.5) -> torch.Tensor:
        """Generate random zoom levels.

        The maximum zoom level scales with curriculum progress from
        zoom_initial_max_start to zoom_initial_max_end.

        Args:
            num_envs: Number of environments.
            curriculum_progress: Curriculum progress in [0, 1].

        Returns:
            zoom_levels: [num_envs, num_agents]
        """
        cfg = self.cfg

        zoom_max = cfg.zoom_initial_max_start + curriculum_progress * (
            cfg.zoom_initial_max_end - cfg.zoom_initial_max_start
        )

        zoom_levels = (
            torch.rand(num_envs, self.num_agents, device=self.device)
            * (zoom_max - cfg.zoom_initial_min)
            + cfg.zoom_initial_min
        )

        return zoom_levels

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def _compute_distances_to_target(
        self,
        agent_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance from each agent to target.

        Args:
            agent_positions: [num_envs, num_agents, 3]
            target_positions: [num_envs, 3]

        Returns:
            distances: [num_envs, num_agents]
        """
        # Expand target for broadcasting
        target_exp = target_positions.unsqueeze(1)  # [num_envs, 1, 3]
        delta = agent_positions - target_exp
        return delta.norm(dim=-1)
