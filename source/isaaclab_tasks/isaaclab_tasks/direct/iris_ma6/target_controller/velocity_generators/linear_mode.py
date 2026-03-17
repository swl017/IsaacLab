# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linear mode velocity generator with random direction changes.

Ported from iris_ma5 TargetMovement linear mode.
"""

from __future__ import annotations

import torch

from .base_generator import BaseVelocityGenerator


class LinearModeGenerator(BaseVelocityGenerator):
    """Linear mode velocity generator.

    Generates random direction flight with periodic direction changes.
    This mode is compatible with iris_ma5 TargetMovement linear behavior.

    Features:
    - Random velocity direction sampling
    - Timer-based direction updates
    - Curriculum-scaled speed and update intervals
    """

    def _initialize_buffers(self):
        """Initialize linear mode state buffers."""
        # Desired velocity for each target [N_env * N_target, 3]
        self.desired_velocity = torch.zeros(
            self.total_targets, 3, device=self.device
        )

        # Timer tracking time since last direction change
        self.update_timer = torch.zeros(self.total_targets, device=self.device)

        # Time until next direction change
        self.next_update_time = torch.zeros(self.total_targets, device=self.device)

    def compute(
        self,
        indices: torch.Tensor,
        current_position: torch.Tensor,
        curriculum_progress: float,
        dt: float,
        **kwargs,
    ) -> torch.Tensor:
        """Compute velocity commands for linear mode.

        Args:
            indices: Flat indices for targets to compute.
            current_position: [N, 3] current positions (unused in linear mode).
            curriculum_progress: Curriculum progress (0-1).
            dt: Timestep [s].

        Returns:
            v_cmd: [N, 3] velocity commands in world frame.
        """
        n = len(indices)
        if n == 0:
            return torch.zeros(0, 3, device=self.device)

        # Update timers
        self.update_timer[indices] += dt

        # Check for direction updates
        update_mask = self.update_timer[indices] >= self.next_update_time[indices]

        if update_mask.any():
            update_indices = indices[update_mask]
            self._apply_direction_update(update_indices, curriculum_progress)

        return self.desired_velocity[indices].clone()

    def _apply_direction_update(
        self,
        indices: torch.Tensor,
        curriculum_progress: float,
    ):
        """Apply a new random direction update for specified targets.

        Args:
            indices: Flat indices of targets to update.
            curriculum_progress: Curriculum progress (0-1).
        """
        n = len(indices)
        if n == 0:
            return

        # Generate random direction (uniformly on sphere)
        direction = torch.randn(n, 3, device=self.device)
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        # Random speed within curriculum-scaled limits
        max_speed = self._get_max_speed(curriculum_progress)
        speed = torch.rand(n, 1, device=self.device) * max_speed

        # Set new desired velocity
        self.desired_velocity[indices] = direction * speed

        # Reset timer and sample new update interval
        self.update_timer[indices] = 0.0
        self.next_update_time[indices] = self._sample_update_interval(
            n, curriculum_progress
        )

    def reset(self, env_ids: torch.Tensor):
        """Reset linear mode state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Reset all targets in specified environments
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx

            # Generate initial random velocity
            n = len(flat_indices)
            direction = torch.randn(n, 3, device=self.device)
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

            # Use start curriculum values for initial speed
            max_speed = self.cfg.max_speed_start
            speed = torch.rand(n, 1, device=self.device) * max_speed

            self.desired_velocity[flat_indices] = direction * speed

            # Reset timers with initial interval sampling (progress=0)
            self.update_timer[flat_indices] = 0.0
            self.next_update_time[flat_indices] = self._sample_update_interval(
                n, curriculum_progress=0.0
            )

    def reset_targets(self, flat_indices: torch.Tensor):
        """Reset linear mode state for specific flat indices.

        Args:
            flat_indices: Flat indices into [num_envs * num_targets] buffer.
        """
        n = len(flat_indices)
        if n == 0:
            return

        # Generate initial random velocity
        direction = torch.randn(n, 3, device=self.device)
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        max_speed = self.cfg.max_speed_start
        speed = torch.rand(n, 1, device=self.device) * max_speed

        self.desired_velocity[flat_indices] = direction * speed
        self.update_timer[flat_indices] = 0.0
        self.next_update_time[flat_indices] = self._sample_update_interval(
            n, curriculum_progress=0.0
        )
