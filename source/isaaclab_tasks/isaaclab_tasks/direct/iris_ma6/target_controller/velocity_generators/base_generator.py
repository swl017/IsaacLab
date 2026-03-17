# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for velocity generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..target_controller_cfg import TargetControllerCfg


class BaseVelocityGenerator(ABC):
    """Abstract base class for velocity generation modes.

    All velocity generators must implement:
    - compute(): Generate velocity commands for specified targets
    - reset(): Reset internal state for specified environments/targets

    Subclasses can optionally override:
    - _initialize_buffers(): Create mode-specific state buffers
    """

    def __init__(
        self,
        cfg: TargetControllerCfg,
        num_envs: int,
        num_targets: int,
        device: torch.device,
    ):
        """Initialize the velocity generator.

        Args:
            cfg: Target controller configuration.
            num_envs: Number of parallel environments.
            num_targets: Maximum number of targets per environment.
            device: Torch device (cuda or cpu).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_targets = num_targets
        self.device = device

        # Total number of target instances
        self.total_targets = num_envs * num_targets

        # Initialize mode-specific buffers
        self._initialize_buffers()

    @abstractmethod
    def _initialize_buffers(self):
        """Initialize mode-specific state buffers.

        Subclasses should create any tensors needed for their velocity
        computation logic (e.g., timers, desired velocities, phases).
        """
        pass

    @abstractmethod
    def compute(
        self,
        indices: torch.Tensor,
        current_position: torch.Tensor,
        curriculum_progress: float,
        dt: float,
        **kwargs,
    ) -> torch.Tensor:
        """Compute velocity commands for specified targets.

        Args:
            indices: Flat indices into the [num_envs * num_targets] buffer.
            current_position: [N, 3] current positions for the targets.
            curriculum_progress: Curriculum progress (0-1).
            dt: Timestep [s].
            **kwargs: Additional mode-specific arguments.

        Returns:
            v_cmd: [N, 3] velocity commands in world frame.
        """
        pass

    @abstractmethod
    def reset(self, env_ids: torch.Tensor):
        """Reset internal state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        pass

    def reset_targets(self, flat_indices: torch.Tensor):
        """Reset internal state for specified flat target indices.

        This is useful when resetting specific targets without resetting
        all targets in an environment.

        Args:
            flat_indices: Flat indices into [num_envs * num_targets] buffer.
        """
        # Default implementation: subclasses can override for specific behavior
        pass

    # =========================================================================
    # Curriculum Helper Methods
    # =========================================================================

    def _get_max_speed(self, curriculum_progress: float) -> float:
        """Get curriculum-scaled maximum speed.

        Args:
            curriculum_progress: Progress value (0-1).

        Returns:
            Maximum speed [m/s].
        """
        return self.cfg.max_speed_start + curriculum_progress * (
            self.cfg.max_speed_end - self.cfg.max_speed_start
        )

    def _sample_update_interval(
        self,
        n: int,
        curriculum_progress: float,
    ) -> torch.Tensor:
        """Sample direction change intervals based on curriculum.

        Args:
            n: Number of samples.
            curriculum_progress: Progress value (0-1).

        Returns:
            [n] tensor of update intervals [s].
        """
        interval_min = self.cfg.update_interval_min_start + curriculum_progress * (
            self.cfg.update_interval_min_end - self.cfg.update_interval_min_start
        )
        interval_max = self.cfg.update_interval_max_start + curriculum_progress * (
            self.cfg.update_interval_max_end - self.cfg.update_interval_max_start
        )

        return (
            torch.rand(n, device=self.device) * (interval_max - interval_min)
            + interval_min
        )

    def _get_geofence_size(self, curriculum_progress: float) -> float:
        """Get curriculum-scaled geofence size.

        Args:
            curriculum_progress: Progress value (0-1).

        Returns:
            Geofence half-width [m].
        """
        return self.cfg.geofence_min_size + curriculum_progress * (
            self.cfg.geofence_max_size - self.cfg.geofence_min_size
        )

    # =========================================================================
    # Index Conversion Helpers
    # =========================================================================

    def _flat_to_env_target(
        self, flat_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert flat indices to (env_id, target_id) pairs.

        Args:
            flat_indices: Flat indices into [num_envs * num_targets] buffer.

        Returns:
            env_ids: Environment indices.
            target_ids: Target indices within environments.
        """
        env_ids = flat_indices // self.num_targets
        target_ids = flat_indices % self.num_targets
        return env_ids, target_ids

    def _env_target_to_flat(
        self, env_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Convert (env_id, target_id) pairs to flat indices.

        Args:
            env_ids: Environment indices.
            target_ids: Target indices within environments.

        Returns:
            Flat indices into [num_envs * num_targets] buffer.
        """
        return env_ids * self.num_targets + target_ids
