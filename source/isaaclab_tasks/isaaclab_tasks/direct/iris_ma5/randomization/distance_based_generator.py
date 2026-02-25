# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distance-based formation generator for multi-agent environments.

The primary difficulty factor is distance to target, controlled via curriculum scaling.
Target is placed at approximately mean agent height for simplified gimbal constraints.
"""

from __future__ import annotations

import math
import torch
from typing import List, Optional

from isaaclab.utils.math import quat_from_euler_xyz

from .formation_cfg import DistanceBasedFormationCfg, FormationResult


class DistanceBasedFormationGenerator:
    """
    Generates agent formations and target positions with distance as primary difficulty.

    Key Design Principles:
        1. Distance to target is the PRIMARY curriculum factor
        2. Target is placed at approximately mean agent height (simplified)
        3. Formation geometry (inter-agent distances) affects triangulation baseline
        4. All agents guaranteed to be within [distance_min, distance_max] of target

    Usage:
        ```python
        cfg = DistanceBasedFormationCfg(distance_min=10.0, distance_max=80.0)
        generator = DistanceBasedFormationGenerator(cfg, num_envs=100, num_agents=2, device)

        # At scale_factor=0: agents close to target
        result = generator.generate(scale_factor=0.0)

        # At scale_factor=1: agents far from target
        result = generator.generate(scale_factor=1.0)
        ```
    """

    def __init__(
        self,
        cfg: DistanceBasedFormationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        max_lin_vel: float = 10.0,
        max_yaw_rate: float = math.radians(90.0),
    ):
        """Initialize the generator.

        Args:
            cfg: Formation configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device for tensor operations.
            max_lin_vel: Maximum linear velocity for initial velocity sampling.
            max_yaw_rate: Maximum yaw rate for initial velocity sampling.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.max_lin_vel = max_lin_vel
        self.max_yaw_rate = max_yaw_rate

        self._ALL_INDICES = torch.arange(num_envs, dtype=torch.long, device=device)

    def generate(
        self,
        env_ids: Optional[torch.Tensor] = None,
        scale_factor: float = 0.5,
        formation_type: Optional[str] = None,
    ) -> FormationResult:
        """
        Generate formation and target positions.

        The algorithm:
            1. Compute target distance based on curriculum: d = d_min + scale * (d_max - d_min)
            2. Sample formation center position
            3. Generate local agent formation (planar/grid/line)
            4. Sample target position at distance d from formation center
            5. Set target height to mean agent height + small offset
            6. Validate and adjust agent positions to satisfy distance constraints
            7. Generate agent orientations (facing toward target)
            8. Generate initial velocities

        Args:
            env_ids: Environment indices to generate for (None = all environments).
            scale_factor: Curriculum scale (0.0 to 1.0).
                - At 0.0: distance = distance_min (easy)
                - At 1.0: distance = distance_max (hard)
            formation_type: Override formation type (None = random selection from cfg).

        Returns:
            FormationResult containing agent states, target position, and metadata.
        """
        if env_ids is None:
            env_ids = self._ALL_INDICES
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        # Clamp scale_factor to [0, 1]
        scale_factor = max(0.0, min(1.0, scale_factor))

        # ========================================================================
        # Step 1: Compute target distance based on curriculum
        # ========================================================================
        base_distance = self.cfg.distance_min + scale_factor * (
            self.cfg.distance_max - self.cfg.distance_min
        )

        # Add random variation
        variation = 1.0 + (torch.rand(num_envs, device=self.device) - 0.5) * self.cfg.distance_variation
        target_distances = base_distance * variation  # [num_envs]

        # ========================================================================
        # Step 2: Determine formation types
        # ========================================================================
        if formation_type is None:
            type_indices = torch.randint(
                0, len(self.cfg.formation_types), (num_envs,), device=self.device
            )
            formation_types = [self.cfg.formation_types[idx.item()] for idx in type_indices]
        else:
            formation_types = [formation_type] * num_envs

        # ========================================================================
        # Step 3: Generate formation centers
        # ========================================================================
        formation_centers = self._generate_formation_centers(num_envs)

        # ========================================================================
        # Step 4: Generate local agent positions (relative to formation center)
        # ========================================================================
        local_positions = torch.zeros(num_envs, self.num_agents, 3, device=self.device)

        for ft in set(formation_types):
            type_mask = torch.tensor(
                [t == ft for t in formation_types], dtype=torch.bool, device=self.device
            )
            type_indices = torch.where(type_mask)[0]

            if len(type_indices) == 0:
                continue

            if ft == "planar":
                local_pos = self._generate_planar_formation(len(type_indices), scale_factor)
            elif ft == "grid":
                local_pos = self._generate_grid_formation(len(type_indices), scale_factor)
            elif ft == "line":
                local_pos = self._generate_line_formation(len(type_indices), scale_factor)
            else:
                raise ValueError(f"Unknown formation type: {ft}")

            local_positions[type_indices] = local_pos

        # ========================================================================
        # Step 5: Ensure minimum agent separation
        # ========================================================================
        local_positions = self._ensure_minimum_separation(local_positions)

        # ========================================================================
        # Step 6: Apply formation rotation and translation
        # ========================================================================
        formation_yaws = self._generate_formation_yaws(num_envs)
        global_positions = self._apply_formation_transform(
            local_positions, formation_centers, formation_yaws
        )

        # ========================================================================
        # Step 7: Compute target position
        # ========================================================================
        # Random bearing from formation center
        bearings = torch.rand(num_envs, device=self.device) * 2 * math.pi

        target_x = formation_centers[:, 0] + target_distances * torch.cos(bearings)
        target_y = formation_centers[:, 1] + target_distances * torch.sin(bearings)

        # Target height = mean agent height + small offset
        mean_agent_height = global_positions[:, :, 2].mean(dim=1)  # [num_envs]
        height_offset = torch.rand(num_envs, device=self.device) * (
            self.cfg.target_height_offset_max - self.cfg.target_height_offset_min
        ) + self.cfg.target_height_offset_min
        target_z = mean_agent_height + height_offset

        target_positions = torch.stack([target_x, target_y, target_z], dim=1)  # [num_envs, 3]

        # ========================================================================
        # Step 8: Validate and adjust agent distances
        # ========================================================================
        if self.cfg.validate_distances:
            global_positions = self._validate_and_adjust_distances(
                global_positions, target_positions, self.cfg.distance_min, self.cfg.distance_max
            )

        # Ensure ground clearance
        global_positions = self._ensure_ground_clearance(global_positions)

        # ========================================================================
        # Step 9: Compute actual distances for logging
        # ========================================================================
        distances_to_target = self._compute_distances_to_target(global_positions, target_positions)

        # ========================================================================
        # Step 10: Generate agent orientations
        # ========================================================================
        orientations = self._generate_agent_orientations(
            global_positions, target_positions, formation_yaws
        )

        # ========================================================================
        # Step 11: Generate initial velocities
        # ========================================================================
        velocities = self._generate_velocities(num_envs, scale_factor)

        # ========================================================================
        # Step 12: Assemble output
        # ========================================================================
        agent_root_states = torch.zeros(num_envs, self.num_agents, 13, device=self.device)
        agent_root_states[:, :, 0:3] = global_positions
        agent_root_states[:, :, 3:7] = orientations
        agent_root_states[:, :, 7:10] = velocities[:, :, 0:3]  # Linear velocity
        agent_root_states[:, :, 10:13] = velocities[:, :, 3:6]  # Angular velocity

        return FormationResult(
            agent_root_states=agent_root_states,
            target_position=target_positions,
            formation_center=formation_centers,
            distances_to_target=distances_to_target,
            formation_types=formation_types,
        )

    # ==========================================================================
    # Formation Generation Methods
    # ==========================================================================

    def _generate_formation_centers(self, num_envs: int) -> torch.Tensor:
        """Generate random formation center positions."""
        x = torch.rand(num_envs, device=self.device) * 10.0 - 5.0  # [-5, 5]
        y = torch.rand(num_envs, device=self.device) * 20.0 - 10.0  # [-10, 10]
        z = torch.rand(num_envs, device=self.device) * (
            self.cfg.formation_height_max - self.cfg.formation_height_min
        ) + self.cfg.formation_height_min

        return torch.stack([x, y, z], dim=1)

    def _generate_formation_yaws(self, num_envs: int) -> torch.Tensor:
        """Generate random yaw rotations for formations."""
        return torch.rand(num_envs, device=self.device) * math.pi - math.pi / 2  # [-π/2, π/2]

    def _generate_planar_formation(self, num_envs: int, scale_factor: float) -> torch.Tensor:
        """
        Generate planar formation - agents in horizontal line, same height.

        This is the simplest formation for early curriculum.
        All agents at Z=0 in local frame.
        """
        positions = torch.zeros(num_envs, self.num_agents, 3, device=self.device)

        for env_idx in range(num_envs):
            # Random direction in XY plane
            angle = torch.rand(1, device=self.device).item() * 2 * math.pi
            direction = torch.tensor([math.cos(angle), math.sin(angle), 0.0], device=self.device)

            # Generate spacing
            distances = self._generate_agent_spacing(scale_factor)

            # Place agents along direction
            for i in range(self.num_agents):
                positions[env_idx, i] = direction * distances[i]

        return positions

    def _generate_grid_formation(self, num_envs: int, scale_factor: float) -> torch.Tensor:
        """
        Generate 2D grid formation in XY plane with optional height variation.
        """
        positions = torch.zeros(num_envs, self.num_agents, 3, device=self.device)

        rows = int(math.sqrt(self.num_agents))
        cols = math.ceil(self.num_agents / rows)

        for env_idx in range(num_envs):
            # Random spacing
            spacing = self._sample_agent_separation(scale_factor)

            agent_idx = 0
            for row in range(rows):
                for col in range(cols):
                    if agent_idx >= self.num_agents:
                        break

                    x = (col - (cols - 1) / 2) * spacing
                    y = (row - (rows - 1) / 2) * spacing

                    positions[env_idx, agent_idx, 0] = x
                    positions[env_idx, agent_idx, 1] = y
                    agent_idx += 1

            # Add height variation (curriculum-controlled)
            z_var_range = self.cfg.z_variation_min + scale_factor * (
                self.cfg.z_variation_max - self.cfg.z_variation_min
            )
            z_var = torch.rand(self.num_agents, device=self.device) * z_var_range
            positions[env_idx, :, 2] = z_var

        return positions

    def _generate_line_formation(self, num_envs: int, scale_factor: float) -> torch.Tensor:
        """
        Generate 3D line formation with curriculum-controlled Z spread.
        """
        positions = torch.zeros(num_envs, self.num_agents, 3, device=self.device)

        for env_idx in range(num_envs):
            # Generate line direction with controlled Z component
            xy_dir = torch.randn(2, device=self.device)
            xy_dir = xy_dir / (xy_dir.norm() + 1e-6)

            max_z = self.cfg.line_max_z_component * scale_factor
            z_component = (torch.rand(1, device=self.device).item() * 2 - 1) * max_z

            direction = torch.tensor(
                [xy_dir[0].item(), xy_dir[1].item(), z_component], device=self.device
            )
            direction = direction / (direction.norm() + 1e-6)

            # Generate spacing
            distances = self._generate_agent_spacing(scale_factor)

            # Place agents
            for i in range(self.num_agents):
                positions[env_idx, i] = direction * distances[i]

            # Add extra Z variation
            z_var_range = self.cfg.z_variation_min + scale_factor * (
                self.cfg.z_variation_max - self.cfg.z_variation_min
            )
            z_var = torch.rand(self.num_agents, device=self.device) * z_var_range
            positions[env_idx, :, 2] += z_var

        return positions

    def _generate_agent_spacing(self, scale_factor: float) -> torch.Tensor:
        """Generate cumulative distances along formation line."""
        distances = torch.zeros(self.num_agents, device=self.device)

        for i in range(1, self.num_agents):
            step = self._sample_agent_separation(scale_factor)
            distances[i] = distances[i - 1] + step

        # Center around origin
        distances = distances - distances.mean()
        return distances

    def _sample_agent_separation(self, scale_factor: float) -> float:
        """Sample random agent separation."""
        base_sep = torch.rand(1, device=self.device).item() * (
            self.cfg.max_agent_separation - self.cfg.min_agent_separation
        ) + self.cfg.min_agent_separation

        # Apply curriculum scaling with minimum floor
        return base_sep * max(scale_factor * self.cfg.formation_spread_scale, 0.3)

    # ==========================================================================
    # Transform and Validation Methods
    # ==========================================================================

    def _apply_formation_transform(
        self,
        local_positions: torch.Tensor,
        centers: torch.Tensor,
        yaws: torch.Tensor,
    ) -> torch.Tensor:
        """Transform local positions to global coordinates."""
        num_envs = local_positions.shape[0]

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)

        global_positions = torch.zeros_like(local_positions)

        for env_idx in range(num_envs):
            for agent_idx in range(self.num_agents):
                x_local = local_positions[env_idx, agent_idx, 0]
                y_local = local_positions[env_idx, agent_idx, 1]
                z_local = local_positions[env_idx, agent_idx, 2]

                # Rotate around Z axis
                x_rotated = cos_yaw[env_idx] * x_local - sin_yaw[env_idx] * y_local
                y_rotated = sin_yaw[env_idx] * x_local + cos_yaw[env_idx] * y_local

                # Translate
                global_positions[env_idx, agent_idx, 0] = x_rotated + centers[env_idx, 0]
                global_positions[env_idx, agent_idx, 1] = y_rotated + centers[env_idx, 1]
                global_positions[env_idx, agent_idx, 2] = z_local + centers[env_idx, 2]

        return global_positions

    def _ensure_minimum_separation(self, positions: torch.Tensor) -> torch.Tensor:
        """Ensure agents maintain minimum separation distance."""
        num_envs = positions.shape[0]
        min_dist = self.cfg.min_agent_separation

        for env_idx in range(num_envs):
            for _ in range(self.cfg.max_collision_iterations):
                collision_found = False

                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        dist = (positions[env_idx, i] - positions[env_idx, j]).norm()

                        if dist < min_dist:
                            collision_found = True
                            direction = positions[env_idx, i] - positions[env_idx, j]
                            direction = direction / (direction.norm() + 1e-6)

                            push_amount = (min_dist - dist) / 2
                            positions[env_idx, i] += direction * push_amount
                            positions[env_idx, j] -= direction * push_amount

                if not collision_found:
                    break

        return positions

    def _validate_and_adjust_distances(
        self,
        agent_positions: torch.Tensor,
        target_positions: torch.Tensor,
        distance_min: float,
        distance_max: float,
    ) -> torch.Tensor:
        """
        Adjust agent positions to satisfy distance constraints.

        Ensures all agents are within [distance_min, distance_max] of target.
        """
        # Expand target for broadcasting: [num_envs, 1, 3]
        target_exp = target_positions.unsqueeze(1)

        # Vector from target to each agent
        delta = agent_positions - target_exp  # [num_envs, num_agents, 3]
        distances = delta.norm(dim=-1)  # [num_envs, num_agents]

        # Clamp distances
        clamped_distances = distances.clamp(distance_min, distance_max)

        # Compute scale factor for adjustment
        scale = clamped_distances / (distances + 1e-6)  # [num_envs, num_agents]

        # Apply adjustment
        adjusted_positions = target_exp + delta * scale.unsqueeze(-1)

        return adjusted_positions

    def _ensure_ground_clearance(self, positions: torch.Tensor) -> torch.Tensor:
        """Ensure all agents are above minimum ground clearance."""
        positions[:, :, 2] = torch.clamp(
            positions[:, :, 2], min=self.cfg.ground_clearance_min
        )
        return positions

    def _compute_distances_to_target(
        self, agent_positions: torch.Tensor, target_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance from each agent to target."""
        target_exp = target_positions.unsqueeze(1)  # [num_envs, 1, 3]
        delta = agent_positions - target_exp
        return delta.norm(dim=-1)  # [num_envs, num_agents]

    # ==========================================================================
    # Orientation and Velocity Generation
    # ==========================================================================

    def _generate_agent_orientations(
        self,
        positions: torch.Tensor,
        target_positions: torch.Tensor,
        formation_yaws: torch.Tensor,
    ) -> torch.Tensor:
        """Generate orientations for agents."""
        num_envs = positions.shape[0]
        orientations = torch.zeros(num_envs, self.num_agents, 4, device=self.device)

        for env_idx in range(num_envs):
            for agent_idx in range(self.num_agents):
                if self.cfg.agent_faces_target:
                    # Compute direction to target
                    to_target = target_positions[env_idx] - positions[env_idx, agent_idx]
                    yaw = torch.atan2(to_target[1], to_target[0]).item()

                    # Add noise
                    yaw += torch.randn(1, device=self.device).item() * self.cfg.agent_yaw_noise_std
                else:
                    # Random yaw
                    yaw_min, yaw_max = self.cfg.agent_random_yaw_range
                    yaw = torch.rand(1, device=self.device).item() * (yaw_max - yaw_min) + yaw_min

                # Convert to quaternion (roll=0, pitch=0, yaw)
                quat = quat_from_euler_xyz(
                    torch.tensor([0.0], device=self.device),
                    torch.tensor([0.0], device=self.device),
                    torch.tensor([yaw], device=self.device),
                )
                orientations[env_idx, agent_idx] = quat[0]

        return orientations

    def _generate_velocities(self, num_envs: int, scale_factor: float) -> torch.Tensor:
        """Generate initial velocities for agents."""
        velocities = torch.zeros(num_envs, self.num_agents, 6, device=self.device)

        # Linear velocity
        lin_scale = self.cfg.initial_velocity_scale * self.max_lin_vel * scale_factor
        velocities[:, :, 0:3] = (
            torch.rand(num_envs, self.num_agents, 3, device=self.device) * 2 - 1
        ) * lin_scale

        # Angular velocity (only yaw rate)
        yaw_scale = self.cfg.initial_yaw_rate_scale * self.max_yaw_rate * scale_factor
        velocities[:, :, 5] = (
            torch.rand(num_envs, self.num_agents, device=self.device) * 2 - 1
        ) * yaw_scale

        return velocities
