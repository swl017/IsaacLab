# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evade mode velocity generator for reactive evasion maneuvers.

Implements iris_ma6 attacker evasion behavior.
"""

from __future__ import annotations

import torch

from .base_generator import BaseVelocityGenerator


class EvadeModeGenerator(BaseVelocityGenerator):
    """Evade mode velocity generator.

    Generates reactive evasion velocity commands when interceptors are nearby.
    The evasion direction is perpendicular to the threat vector with a
    component away from the interceptor.

    Features:
    - Perpendicular evasion direction computation
    - Evasion timer management
    - Curriculum-scaled agility

    Reference: iris_ma6_env_spec.md Section 4.2.2
    """

    def _initialize_buffers(self):
        """Initialize evade mode state buffers."""
        # Evasion direction (normalized) [N_env * N_target, 3]
        self.evasion_direction = torch.zeros(
            self.total_targets, 3, device=self.device
        )

        # Remaining evasion time [s]
        self.evasion_timer = torch.zeros(self.total_targets, device=self.device)

        # Evasion agility (0-1) from behavior profile
        self.evasion_agility = torch.zeros(self.total_targets, device=self.device)

        # Random sign for perpendicular direction (left or right)
        self.evasion_sign = torch.zeros(self.total_targets, device=self.device)

    def compute(
        self,
        indices: torch.Tensor,
        current_position: torch.Tensor,
        curriculum_progress: float,
        dt: float,
        interceptor_positions: torch.Tensor = None,
        interceptor_roles: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute velocity commands for evasion mode.

        Args:
            indices: Flat indices for targets to compute.
            current_position: [N, 3] current target positions.
            curriculum_progress: Curriculum progress (0-1).
            dt: Timestep [s].
            interceptor_positions: [N_env, num_defenders, 3] defender positions.
            interceptor_roles: [N_env, num_defenders] defender roles (0=OBSERVE, 1=INTERCEPT).

        Returns:
            v_cmd: [N, 3] velocity commands in world frame.
        """
        n = len(indices)
        if n == 0:
            return torch.zeros(0, 3, device=self.device)

        if interceptor_positions is None or interceptor_roles is None:
            # No interceptor information - return zero velocity
            return torch.zeros(n, 3, device=self.device)

        # Convert flat indices to env_ids for interceptor lookup
        env_ids, target_ids = self._flat_to_env_target(indices)

        # Compute evasion velocity
        v_cmd = self._compute_evasion_velocity(
            indices=indices,
            env_ids=env_ids,
            current_position=current_position,
            interceptor_positions=interceptor_positions,
            interceptor_roles=interceptor_roles,
            curriculum_progress=curriculum_progress,
        )

        # Update evasion timer
        self.evasion_timer[indices] -= dt

        return v_cmd

    def _compute_evasion_velocity(
        self,
        indices: torch.Tensor,
        env_ids: torch.Tensor,
        current_position: torch.Tensor,
        interceptor_positions: torch.Tensor,
        interceptor_roles: torch.Tensor,
        curriculum_progress: float,
    ) -> torch.Tensor:
        """Compute evasion velocity away from nearest interceptor.

        Args:
            indices: Flat indices for targets.
            env_ids: Environment indices for each target.
            current_position: [N, 3] current positions.
            interceptor_positions: [N_env, num_defenders, 3] defender positions.
            interceptor_roles: [N_env, num_defenders] defender roles.
            curriculum_progress: Curriculum progress (0-1).

        Returns:
            v_cmd: [N, 3] velocity commands.
        """
        n = len(indices)

        # Get interceptor info for each target's environment
        # Shape: [N, num_defenders, 3] and [N, num_defenders]
        target_interceptor_pos = interceptor_positions[env_ids]
        target_interceptor_roles = interceptor_roles[env_ids]

        # Compute distance to each interceptor
        # current_position: [N, 3] -> [N, 1, 3]
        pos_expanded = current_position.unsqueeze(1)
        dist_to_interceptors = torch.norm(
            pos_expanded - target_interceptor_pos, dim=-1
        )  # [N, num_defenders]

        # Mask out non-INTERCEPT defenders with large distance
        intercept_mask = target_interceptor_roles == 1  # 1 = INTERCEPT
        dist_to_interceptors[~intercept_mask] = 1e6

        # Find nearest interceptor
        nearest_dist, nearest_idx = dist_to_interceptors.min(dim=1)

        # Gather nearest interceptor position
        batch_indices = torch.arange(n, device=self.device)
        nearest_pos = target_interceptor_pos[batch_indices, nearest_idx]  # [N, 3]

        # Threat vector: from interceptor toward target (escape direction)
        threat_vec = current_position - nearest_pos
        threat_vec_norm = threat_vec / (torch.norm(threat_vec, dim=1, keepdim=True) + 1e-6)

        # Perpendicular direction for evasion (use stored sign)
        sign = self.evasion_sign[indices]
        perp_vec = torch.stack(
            [
                -threat_vec_norm[:, 1] * sign,
                threat_vec_norm[:, 0] * sign,
                torch.zeros(n, device=self.device),
            ],
            dim=1,
        )

        # Blend evasion direction: perpendicular + escape
        escape_weight = self.cfg.evasion_escape_weight
        evasion_direction = (1 - escape_weight) * perp_vec + escape_weight * threat_vec_norm
        evasion_direction = evasion_direction / (
            torch.norm(evasion_direction, dim=1, keepdim=True) + 1e-6
        )

        # Store evasion direction for consistency
        self.evasion_direction[indices] = evasion_direction

        # Apply evasion agility scaling with curriculum
        agility_scale = self._get_agility_scale(curriculum_progress)
        agility = self.evasion_agility[indices] * agility_scale
        max_speed = self._get_max_speed(curriculum_progress)

        v_cmd = evasion_direction * max_speed * agility.unsqueeze(1)

        return v_cmd

    def _get_agility_scale(self, curriculum_progress: float) -> float:
        """Get curriculum-scaled evasion agility multiplier.

        Evasion starts disabled and ramps up after evasion_start_progress.

        Args:
            curriculum_progress: Progress value (0-1).

        Returns:
            Agility scale (0-1).
        """
        evasion_start = self.cfg.evasion_start_progress

        if curriculum_progress < evasion_start:
            return 0.0

        scaled_progress = (curriculum_progress - evasion_start) / (1.0 - evasion_start)
        return scaled_progress

    def reset(self, env_ids: torch.Tensor):
        """Reset evade mode state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Reset all targets in specified environments
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            self._reset_evasion_params(flat_indices)

    def reset_targets(self, flat_indices: torch.Tensor):
        """Reset evade mode state for specific flat indices.

        Args:
            flat_indices: Flat indices into [num_envs * num_targets] buffer.
        """
        self._reset_evasion_params(flat_indices)

    def _reset_evasion_params(self, flat_indices: torch.Tensor):
        """Reset evasion parameters for specified indices.

        Args:
            flat_indices: Flat indices to reset.
        """
        n = len(flat_indices)
        if n == 0:
            return

        # Reset evasion direction
        self.evasion_direction[flat_indices] = 0.0

        # Reset evasion timer
        self.evasion_timer[flat_indices] = 0.0

        # Random evasion sign (left or right)
        self.evasion_sign[flat_indices] = (
            (torch.rand(n, device=self.device) > 0.5).float() * 2 - 1
        )

        # Evasion agility will be set by behavior profile
        # Default to middle value
        self.evasion_agility[flat_indices] = 0.5

    def start_evasion(self, flat_indices: torch.Tensor):
        """Start evasion for specified targets.

        Initializes evasion timer and randomizes evasion sign.

        Args:
            flat_indices: Flat indices for targets to start evading.
        """
        n = len(flat_indices)
        if n == 0:
            return

        # Sample evasion duration
        duration = (
            torch.rand(n, device=self.device)
            * (self.cfg.evade_duration_max - self.cfg.evade_duration_min)
            + self.cfg.evade_duration_min
        )
        self.evasion_timer[flat_indices] = duration

        # Randomize evasion sign (new direction for this evasion)
        self.evasion_sign[flat_indices] = (
            (torch.rand(n, device=self.device) > 0.5).float() * 2 - 1
        )

    def is_evasion_complete(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Check if evasion is complete for specified targets.

        Args:
            flat_indices: Flat indices for targets to check.

        Returns:
            [N] boolean tensor indicating evasion completion.
        """
        return self.evasion_timer[flat_indices] <= 0

    def set_evasion_agility(self, indices: torch.Tensor, agility: torch.Tensor):
        """Set evasion agility from behavior profile.

        Args:
            indices: Flat indices for targets.
            agility: [N] evasion agility values (0-1).
        """
        self.evasion_agility[indices] = agility
