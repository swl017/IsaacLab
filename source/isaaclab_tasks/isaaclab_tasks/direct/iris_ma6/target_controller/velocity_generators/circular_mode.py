# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circular mode velocity generator for orbital flight paths.

Ported from iris_ma5 TargetMovement circular mode.
"""

from __future__ import annotations

import math

import torch

from .base_generator import BaseVelocityGenerator


class CircularModeGenerator(BaseVelocityGenerator):
    """Circular mode velocity generator.

    Generates orbital flight around a center point.
    This mode is compatible with iris_ma5 TargetMovement circular behavior.

    Features:
    - Configurable orbit radius and height
    - Proportional control for orbit tracking
    - Curriculum-scaled speed limits
    """

    def _initialize_buffers(self):
        """Initialize circular mode state buffers."""
        # Orbit center position [N_env * N_target, 3]
        self.circular_center = torch.zeros(
            self.total_targets, 3, device=self.device
        )

        # Orbit radius for each target
        self.circular_radius = torch.zeros(self.total_targets, device=self.device)

        # Angular speed (rad/s)
        self.circular_angular_speed = torch.zeros(
            self.total_targets, device=self.device
        )

        # Current phase angle
        self.circular_phase = torch.zeros(self.total_targets, device=self.device)

        # Orbit height above center
        self.circular_height = torch.zeros(self.total_targets, device=self.device)

    def compute(
        self,
        indices: torch.Tensor,
        current_position: torch.Tensor,
        curriculum_progress: float,
        dt: float,
        **kwargs,
    ) -> torch.Tensor:
        """Compute velocity commands for circular orbit mode.

        Args:
            indices: Flat indices for targets to compute.
            current_position: [N, 3] current target positions.
            curriculum_progress: Curriculum progress (0-1).
            dt: Timestep [s].

        Returns:
            v_cmd: [N, 3] velocity commands in world frame.
        """
        n = len(indices)
        if n == 0:
            return torch.zeros(0, 3, device=self.device)

        # Update phase angle
        self.circular_phase[indices] += self.circular_angular_speed[indices] * dt

        # Wrap phase to [0, 2*pi)
        self.circular_phase[indices] = torch.fmod(
            self.circular_phase[indices], 2 * math.pi
        )

        # Get orbit parameters
        phase = self.circular_phase[indices]
        radius = self.circular_radius[indices]
        center = self.circular_center[indices]
        height = self.circular_height[indices]

        # Compute desired position on orbit
        desired_pos = torch.stack(
            [
                center[:, 0] + radius * torch.cos(phase),
                center[:, 1] + radius * torch.sin(phase),
                center[:, 2] + height,
            ],
            dim=1,
        )

        # Proportional control to track orbit
        pos_error = desired_pos - current_position
        kp = self.cfg.circular_position_gain

        velocity_cmd = kp * pos_error

        # Clamp to curriculum-scaled max speed
        speed = torch.norm(velocity_cmd, dim=1, keepdim=True)
        max_speed = self._get_max_speed(curriculum_progress)
        velocity_cmd = velocity_cmd * torch.clamp(max_speed / (speed + 1e-6), max=1.0)

        return velocity_cmd

    def reset(self, env_ids: torch.Tensor):
        """Reset circular mode state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Reset all targets in specified environments
        for target_idx in range(self.num_targets):
            flat_indices = env_ids * self.num_targets + target_idx
            self._reset_circular_params(flat_indices)

    def reset_targets(self, flat_indices: torch.Tensor):
        """Reset circular mode state for specific flat indices.

        Args:
            flat_indices: Flat indices into [num_envs * num_targets] buffer.
        """
        self._reset_circular_params(flat_indices)

    def _reset_circular_params(self, flat_indices: torch.Tensor):
        """Reset circular orbit parameters for specified indices.

        Args:
            flat_indices: Flat indices to reset.
        """
        n = len(flat_indices)
        if n == 0:
            return

        # Random radius
        self.circular_radius[flat_indices] = (
            torch.rand(n, device=self.device)
            * (self.cfg.circular_radius_max - self.cfg.circular_radius_min)
            + self.cfg.circular_radius_min
        )

        # Random height
        self.circular_height[flat_indices] = (
            torch.rand(n, device=self.device)
            * (self.cfg.circular_height_max - self.cfg.circular_height_min)
            + self.cfg.circular_height_min
        )

        # Random phase
        self.circular_phase[flat_indices] = (
            torch.rand(n, device=self.device) * 2 * math.pi
        )

        # Angular speed = linear_speed / radius (clamped)
        base_speed = (
            torch.rand(n, device=self.device) * self.cfg.max_speed_start * 0.5
        )
        self.circular_angular_speed[flat_indices] = torch.clamp(
            base_speed / self.circular_radius[flat_indices],
            max=self.cfg.max_angular_speed,
        )

        # Center will be set by the controller based on env_origins
        # Initialize to zero for now
        self.circular_center[flat_indices] = 0.0

    def set_center(self, indices: torch.Tensor, center: torch.Tensor):
        """Set orbit center for specified targets.

        Args:
            indices: Flat indices for targets.
            center: [N, 3] center positions.
        """
        self.circular_center[indices] = center
