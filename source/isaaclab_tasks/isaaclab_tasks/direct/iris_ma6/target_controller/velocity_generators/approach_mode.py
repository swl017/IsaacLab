# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Approach mode velocity generator for goal-directed movement toward facility.

Implements iris_ma6 attacker approach behavior with path variants.
"""

from __future__ import annotations

from enum import IntEnum

import torch

from .base_generator import BaseVelocityGenerator


class PathType(IntEnum):
    """Approach path type enumeration."""

    DIRECT = 0  # Straight line to facility
    OFFSET = 1  # Via intermediate waypoints
    LOW_ALTITUDE = 2  # Ground-hugging approach


class ApproachModeGenerator(BaseVelocityGenerator):
    """Approach mode velocity generator.

    Generates goal-directed velocity commands toward a facility position.
    Supports multiple path variants for attacker behavior diversity.

    Path Variants:
    - DIRECT: Straight-line approach to facility
    - OFFSET: Approach via intermediate waypoints
    - LOW_ALTITUDE: Ground-hugging approach with altitude constraint

    Reference: iris_ma6_env_spec.md Section 4.2.1
    """

    def _initialize_buffers(self):
        """Initialize approach mode state buffers."""
        # Path type for each target (0=direct, 1=offset, 2=low_altitude)
        self.approach_path_type = torch.zeros(
            self.total_targets, dtype=torch.long, device=self.device
        )

        # Waypoints for offset path [N_env * N_target, max_waypoints, 3]
        self.approach_waypoints = torch.zeros(
            self.total_targets, self.cfg.max_waypoints, 3, device=self.device
        )

        # Number of valid waypoints for each target
        self.num_waypoints = torch.zeros(
            self.total_targets, dtype=torch.long, device=self.device
        )

        # Current waypoint index
        self.current_waypoint_idx = torch.zeros(
            self.total_targets, dtype=torch.long, device=self.device
        )

        # Speed multiplier from behavior profile
        self.speed_multiplier = torch.ones(self.total_targets, device=self.device)

    def compute(
        self,
        indices: torch.Tensor,
        current_position: torch.Tensor,
        curriculum_progress: float,
        dt: float,
        facility_position: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute velocity commands for approach mode.

        Args:
            indices: Flat indices for targets to compute.
            current_position: [N, 3] current target positions.
            curriculum_progress: Curriculum progress (0-1).
            dt: Timestep [s].
            facility_position: [N, 3] facility positions for each target.

        Returns:
            v_cmd: [N, 3] velocity commands in world frame.
        """
        n = len(indices)
        if n == 0:
            return torch.zeros(0, 3, device=self.device)

        if facility_position is None:
            raise ValueError("facility_position must be provided for approach mode")

        v_cmd = torch.zeros(n, 3, device=self.device)
        path_types = self.approach_path_type[indices]

        # -----------------------------------------------------------------
        # Direct approach: straight to facility
        # -----------------------------------------------------------------
        direct_mask = path_types == PathType.DIRECT
        if direct_mask.any():
            direct_local_indices = torch.where(direct_mask)[0]
            direct_global_indices = indices[direct_mask]

            direction = (
                facility_position[direct_local_indices]
                - current_position[direct_local_indices]
            )
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

            speed = self._get_approach_speed(
                direct_global_indices, curriculum_progress
            )
            v_cmd[direct_local_indices] = direction * speed.unsqueeze(1)

        # -----------------------------------------------------------------
        # Offset approach: via waypoints
        # -----------------------------------------------------------------
        offset_mask = path_types == PathType.OFFSET
        if offset_mask.any():
            offset_local_indices = torch.where(offset_mask)[0]
            offset_global_indices = indices[offset_mask]

            v_cmd[offset_local_indices] = self._compute_offset_velocity(
                local_indices=offset_local_indices,
                global_indices=offset_global_indices,
                current_position=current_position[offset_local_indices],
                facility_position=facility_position[offset_local_indices],
                curriculum_progress=curriculum_progress,
            )

        # -----------------------------------------------------------------
        # Low-altitude approach
        # -----------------------------------------------------------------
        low_alt_mask = path_types == PathType.LOW_ALTITUDE
        if low_alt_mask.any():
            low_alt_local_indices = torch.where(low_alt_mask)[0]
            low_alt_global_indices = indices[low_alt_mask]

            v_cmd[low_alt_local_indices] = self._compute_low_altitude_velocity(
                local_indices=low_alt_local_indices,
                global_indices=low_alt_global_indices,
                current_position=current_position[low_alt_local_indices],
                facility_position=facility_position[low_alt_local_indices],
                curriculum_progress=curriculum_progress,
            )

        return v_cmd

    def _compute_offset_velocity(
        self,
        local_indices: torch.Tensor,
        global_indices: torch.Tensor,
        current_position: torch.Tensor,
        facility_position: torch.Tensor,
        curriculum_progress: float,
    ) -> torch.Tensor:
        """Compute velocity for offset path mode.

        Args:
            local_indices: Indices into the current batch.
            global_indices: Flat indices into global buffers.
            current_position: [N, 3] current positions.
            facility_position: [N, 3] facility positions.
            curriculum_progress: Curriculum progress (0-1).

        Returns:
            v_cmd: [N, 3] velocity commands.
        """
        n = len(local_indices)
        v_cmd = torch.zeros(n, 3, device=self.device)

        # Get current waypoint index and total waypoints
        waypoint_idx = self.current_waypoint_idx[global_indices]
        num_waypoints = self.num_waypoints[global_indices]

        # Determine if we should target a waypoint or the facility
        at_final_waypoint = waypoint_idx >= num_waypoints

        # Targets still navigating waypoints
        waypoint_mask = ~at_final_waypoint
        if waypoint_mask.any():
            wp_local = torch.where(waypoint_mask)[0]
            wp_global = global_indices[waypoint_mask]
            wp_idx = waypoint_idx[waypoint_mask]

            # Gather current waypoint positions
            # Shape: [num_waypoint_targets, 3]
            target_pos = self.approach_waypoints[wp_global, wp_idx]

            # Check if waypoint reached
            dist_to_waypoint = torch.norm(
                current_position[wp_local] - target_pos, dim=1
            )
            reached = dist_to_waypoint < self.cfg.waypoint_reach_threshold

            # Advance to next waypoint for those that reached
            if reached.any():
                reached_global = wp_global[reached]
                self.current_waypoint_idx[reached_global] += 1

            # Compute direction to waypoint
            direction = target_pos - current_position[wp_local]
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

            speed = self._get_approach_speed(wp_global, curriculum_progress) * 0.9
            v_cmd[wp_local] = direction * speed.unsqueeze(1)

        # Targets heading directly to facility (past all waypoints)
        facility_mask = at_final_waypoint
        if facility_mask.any():
            fac_local = torch.where(facility_mask)[0]
            fac_global = global_indices[facility_mask]

            direction = facility_position[fac_local] - current_position[fac_local]
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

            speed = self._get_approach_speed(fac_global, curriculum_progress) * 0.9
            v_cmd[fac_local] = direction * speed.unsqueeze(1)

        return v_cmd

    def _compute_low_altitude_velocity(
        self,
        local_indices: torch.Tensor,
        global_indices: torch.Tensor,
        current_position: torch.Tensor,
        facility_position: torch.Tensor,
        curriculum_progress: float,
    ) -> torch.Tensor:
        """Compute velocity for low-altitude approach mode.

        Args:
            local_indices: Indices into the current batch.
            global_indices: Flat indices into global buffers.
            current_position: [N, 3] current positions.
            facility_position: [N, 3] facility positions.
            curriculum_progress: Curriculum progress (0-1).

        Returns:
            v_cmd: [N, 3] velocity commands.
        """
        n = len(local_indices)

        # Horizontal direction to facility
        direction = facility_position - current_position
        direction[:, 2] = 0  # Zero out vertical component
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        # Horizontal speed (reduced for stealth)
        speed = self._get_approach_speed(global_indices, curriculum_progress) * 0.7

        v_cmd = direction * speed.unsqueeze(1)

        # Altitude regulation toward low altitude target
        target_alt = self.cfg.low_altitude_target
        alt_error = target_alt - current_position[:, 2]
        v_cmd[:, 2] = torch.clamp(
            alt_error * self.cfg.low_altitude_gain, -2.0, 2.0
        )

        return v_cmd

    def _get_approach_speed(
        self, indices: torch.Tensor, curriculum_progress: float
    ) -> torch.Tensor:
        """Get approach speed with speed multiplier.

        Args:
            indices: Flat indices for targets.
            curriculum_progress: Curriculum progress (0-1).

        Returns:
            [N] tensor of approach speeds [m/s].
        """
        base_speed = self.cfg.approach_speed_base
        curriculum_scale = 0.5 + 0.5 * curriculum_progress  # 0.5 to 1.0
        return base_speed * curriculum_scale * self.speed_multiplier[indices]

    def reset(self, env_ids: torch.Tensor):
        """Reset approach mode state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Reset all targets in specified environments
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            self._reset_approach_params(flat_indices)

    def reset_targets(self, flat_indices: torch.Tensor):
        """Reset approach mode state for specific flat indices.

        Args:
            flat_indices: Flat indices into [num_envs * num_targets] buffer.
        """
        self._reset_approach_params(flat_indices)

    def _reset_approach_params(self, flat_indices: torch.Tensor):
        """Reset approach parameters for specified indices.

        Args:
            flat_indices: Flat indices to reset.
        """
        n = len(flat_indices)
        if n == 0:
            return

        # Sample path type based on configured weights
        weights = torch.tensor(
            [
                self.cfg.approach_path_direct_weight,
                self.cfg.approach_path_offset_weight,
                self.cfg.approach_path_low_alt_weight,
            ],
            device=self.device,
        )
        weights = weights / weights.sum()  # Normalize

        # Sample path types
        path_probs = torch.rand(n, device=self.device)
        cumsum = torch.cumsum(weights, dim=0)

        self.approach_path_type[flat_indices] = PathType.DIRECT
        self.approach_path_type[flat_indices[path_probs > cumsum[0]]] = PathType.OFFSET
        self.approach_path_type[flat_indices[path_probs > cumsum[1]]] = (
            PathType.LOW_ALTITUDE
        )

        # Reset waypoint index
        self.current_waypoint_idx[flat_indices] = 0

        # Initialize waypoints (will be properly set by controller with facility position)
        self.approach_waypoints[flat_indices] = 0.0
        self.num_waypoints[flat_indices] = 0

        # Reset speed multiplier (will be set by behavior profile)
        self.speed_multiplier[flat_indices] = 1.0

    def set_waypoints(
        self,
        indices: torch.Tensor,
        current_position: torch.Tensor,
        facility_position: torch.Tensor,
    ):
        """Generate and set waypoints for offset approach.

        Args:
            indices: Flat indices for targets.
            current_position: [N, 3] current target positions.
            facility_position: [N, 3] facility positions.
        """
        n = len(indices)
        if n == 0:
            return

        # Only generate waypoints for OFFSET path type
        offset_mask = self.approach_path_type[indices] == PathType.OFFSET
        if not offset_mask.any():
            return

        offset_indices = indices[offset_mask]
        offset_current = current_position[offset_mask]
        offset_facility = facility_position[offset_mask]
        n_offset = len(offset_indices)

        # Random number of waypoints (1 to max_waypoints)
        num_wp = torch.randint(
            1, self.cfg.max_waypoints + 1, (n_offset,), device=self.device
        )
        self.num_waypoints[offset_indices] = num_wp

        # Generate waypoints along the path with lateral offset
        direct_vec = offset_facility - offset_current
        path_length = torch.norm(direct_vec, dim=1, keepdim=True)
        direct_unit = direct_vec / (path_length + 1e-6)

        # Perpendicular vector (in XY plane)
        perp_vec = torch.stack(
            [-direct_unit[:, 1], direct_unit[:, 0], torch.zeros(n_offset, device=self.device)],
            dim=1,
        )

        # Generate waypoints for each target
        for wp_idx in range(self.cfg.max_waypoints):
            # Fraction along path (evenly spaced)
            t = (wp_idx + 1) / (self.cfg.max_waypoints + 1)

            # Base position along direct path
            base_pos = offset_current + t * direct_vec

            # Random lateral offset
            lateral_offset = (
                (torch.rand(n_offset, device=self.device) * 2 - 1)
                * self.cfg.offset_waypoint_distance
            )

            # Waypoint position
            waypoint = base_pos + lateral_offset.unsqueeze(1) * perp_vec

            # Store waypoint
            self.approach_waypoints[offset_indices, wp_idx] = waypoint

    def set_speed_multiplier(self, indices: torch.Tensor, multiplier: torch.Tensor):
        """Set speed multiplier from behavior profile.

        Args:
            indices: Flat indices for targets.
            multiplier: [N] speed multiplier values.
        """
        self.speed_multiplier[indices] = multiplier

    def set_path_type(self, indices: torch.Tensor, path_type: int):
        """Force set path type for specified targets.

        Args:
            indices: Flat indices for targets.
            path_type: PathType value (0=DIRECT, 1=OFFSET, 2=LOW_ALTITUDE).
        """
        self.approach_path_type[indices] = path_type
