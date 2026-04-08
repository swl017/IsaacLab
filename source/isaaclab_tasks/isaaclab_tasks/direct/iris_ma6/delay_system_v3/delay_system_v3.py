# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified delay system V3.

This module provides a unified delay system that handles all four paths:
- Clean + Ego (for reward computation, self-view)
- Clean + Other (for reward computation, other-view)
- Noisy + Ego (for observations, self-view)
- Noisy + Other (for observations, other-view)

Instead of maintaining 4 separate delay systems (as in V2), we use:
- One storage with raw/noisy data
- One pipeline per field
- Perspective specified at query time
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, Optional, Literal

from .delay_cfg_v3 import UnifiedDelayCfgV3, DelayPipelineCfgV3, PerspectiveCfg
from .delay_pipeline_v3 import DelayPipelineV3
from .field_storage import FieldStorage


class UnifiedDelaySystem:
    """Unified delay system handling all delay paths.

    This system:
    1. Stores raw (ground truth) and noisy data with timestamps
    2. Maintains separate pipelines for ego and other perspectives
    3. Returns delayed data with correct timestamps at query time

    **Key simplification over V2**: Instead of 4 DelaySystemV2 instances,
    we have 1 UnifiedDelaySystem with perspective specified at query time.
    """

    def __init__(
        self,
        cfg: UnifiedDelayCfgV3,
        num_envs: int,
        device: torch.device,
    ):
        """Initialize unified delay system.

        Args:
            cfg: Configuration for the delay system.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self._cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._dt = cfg.dt

        # Field storage (raw + noisy with timestamps)
        self._storage = FieldStorage(num_envs, device)

        # Pipelines: separate for ego and other perspectives, per field
        self._ego_pipelines: Dict[str, DelayPipelineV3] = {}
        self._other_pipelines: Dict[str, DelayPipelineV3] = {}

        # Track registered fields and their shapes
        self._field_shapes: Dict[str, Tuple[int, ...]] = {}

        # Current simulation time
        self._t_current = torch.zeros(num_envs, device=device)

        # Curriculum mode
        self._mode: Literal["none", "fixed", "random"] = "none"
        self._progress: float = 1.0

    @property
    def cfg(self) -> UnifiedDelayCfgV3:
        """System configuration."""
        return self._cfg

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    @property
    def t_current(self) -> torch.Tensor:
        """Current simulation time."""
        return self._t_current

    @property
    def field_names(self) -> list[str]:
        """List of registered field names."""
        return list(self._field_shapes.keys())

    def register_field(
        self,
        field_name: str,
        data_shape: Tuple[int, ...],
        custom_cfg: Optional[DelayPipelineCfgV3] = None,
    ):
        """Register a field for delay processing.

        Args:
            field_name: Unique field identifier (e.g., "agent_0.bboxes_2d").
            data_shape: Shape of data per environment (excluding batch dim).
            custom_cfg: Optional custom pipeline config (overrides defaults).
        """
        if field_name in self._field_shapes:
            return  # Already registered

        self._field_shapes[field_name] = data_shape

        # Extract field suffix for override lookup
        # field_name is like "agent_0.bboxes_2d" -> suffix is "bboxes_2d"
        field_suffix = field_name.split(".")[-1] if "." in field_name else field_name

        # Determine pipeline configs using perspective-specific overrides
        # Priority: custom_cfg > perspective.field_overrides > perspective.pipeline
        if custom_cfg is not None:
            ego_cfg = custom_cfg
            other_cfg = custom_cfg
        else:
            # Check perspective-specific field overrides, then fall back to default
            ego_cfg = self._cfg.ego.field_overrides.get(
                field_suffix, self._cfg.ego.pipeline
            )
            other_cfg = self._cfg.other.field_overrides.get(
                field_suffix, self._cfg.other.pipeline
            )

        # Create pipelines for this field
        self._ego_pipelines[field_name] = DelayPipelineV3(
            cfg=ego_cfg,
            num_envs=self._num_envs,
            device=self._device,
            dt=self._dt,
            data_shape=data_shape,
        )
        self._other_pipelines[field_name] = DelayPipelineV3(
            cfg=other_cfg,
            num_envs=self._num_envs,
            device=self._device,
            dt=self._dt,
            data_shape=data_shape,
        )

        # Apply current mode
        self._ego_pipelines[field_name].set_mode(self._mode, self._progress)
        self._other_pipelines[field_name].set_mode(self._mode, self._progress)

    def set_time(self, t: torch.Tensor):
        """Set current simulation time.

        Args:
            t: Current time tensor of shape (num_envs,).
        """
        self._t_current = t.to(self._device)
        self._storage.set_time(t)

    def store(
        self,
        field_name: str,
        data: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
        noisy_data: Optional[torch.Tensor] = None,
    ):
        """Store field data with optional noise.

        Args:
            field_name: Field identifier.
            data: Data tensor of shape (num_envs, ...).
            timestamp: Optional capture timestamp. If None, uses current time.
            noise_std: Standard deviation for observation noise.
                Ignored when noisy_data is provided.
            noisy_data: Pre-noised data tensor (same shape as data). When provided,
                stored directly as the noisy version instead of generating Gaussian
                noise. Used by the detector replicator to pass calibrated noise.
        """
        # Auto-register if not already registered
        if field_name not in self._field_shapes:
            data_shape = tuple(data.shape[1:])
            self.register_field(field_name, data_shape)

        # Store in field storage
        self._storage.store(field_name, data, timestamp, noise_std, noisy_data=noisy_data)

    def get_delayed(
        self,
        field_name: str,
        perspective: Literal["ego", "other"],
        use_noise: bool = False,
        allow_dropout: bool = True,
        burst_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get delayed data with specified perspective and noise mode.

        Args:
            field_name: Field identifier.
            perspective: "ego" or "other" (determines delay parameters).
            use_noise: If True, use noisy data. If False, use raw.
            allow_dropout: Whether to allow dropout (typically False for rewards).
            burst_dropout_mask: Optional external dropout mask of shape (num_envs,),
                dtype bool. When provided, replaces per-pipeline i.i.d. dropout
                for this field (used by burst dropout model).

        Returns:
            Tuple of (delayed_data, delayed_timestamp).

        Raises:
            KeyError: If field not found.
        """
        if field_name not in self._field_shapes:
            raise KeyError(f"Field '{field_name}' not registered")

        # Get data from storage
        if use_noise:
            data, timestamp = self._storage.get_noisy(field_name)
        else:
            data, timestamp = self._storage.get_raw(field_name)

        # Select appropriate pipeline
        if perspective == "ego":
            pipeline = self._ego_pipelines[field_name]
            cfg = self._cfg.ego
        else:
            pipeline = self._other_pipelines[field_name]
            cfg = self._cfg.other

        # Determine if dropout should be applied
        apply_dropout = allow_dropout and (
            (use_noise and cfg.dropout_for_observations)
            or (not use_noise and cfg.dropout_for_rewards)
        )

        # Process through pipeline (burst mask only passed when dropout is active)
        effective_burst_mask = burst_dropout_mask if apply_dropout else None
        delayed_data, delayed_timestamp = pipeline.process(
            data, timestamp, self._t_current,
            allow_dropout=apply_dropout,
            burst_dropout_mask=effective_burst_mask,
        )

        return delayed_data, delayed_timestamp

    def get_aoi(
        self,
        field_name: str,
        perspective: Literal["ego", "other"],
        use_noise: bool = False,
    ) -> torch.Tensor:
        """Get Age-of-Information for a field.

        AoI = current_time - data_capture_time

        Args:
            field_name: Field identifier.
            perspective: "ego" or "other".
            use_noise: Whether to use noisy path.

        Returns:
            AoI tensor of shape (num_envs,).
        """
        _, timestamp = self.get_delayed(field_name, perspective, use_noise)
        return self._t_current - timestamp

    def set_delay_mode(
        self, mode: Literal["none", "fixed", "random"], progress: float = 1.0
    ):
        """Set delay mode for curriculum learning.

        Args:
            mode: Delay mode ('none', 'fixed', 'random').
            progress: Curriculum progress [0, 1].
        """
        self._mode = mode
        self._progress = max(0.0, min(1.0, progress))

        # Update all pipelines
        for pipeline in self._ego_pipelines.values():
            pipeline.set_mode(mode, progress)
        for pipeline in self._other_pipelines.values():
            pipeline.set_mode(mode, progress)

    def set_dropout_rate(self, rate: float):
        """Set dropout rate for curriculum control.

        Args:
            rate: Dropout probability [0, 1].
        """
        for pipeline in self._ego_pipelines.values():
            pipeline.set_dropout_rate(rate)
        for pipeline in self._other_pipelines.values():
            pipeline.set_dropout_rate(rate)

    def set_field_delay_mode(
        self,
        field_prefix: str,
        mode: Literal["none", "fixed", "random"],
        progress: float = 1.0,
    ):
        """Set delay mode for pipelines whose key starts with field_prefix.

        Args:
            field_prefix: Prefix to match (e.g., 'drone_0' for all drone_0 fields).
            mode: Delay mode ('none', 'fixed', 'random').
            progress: Curriculum progress [0, 1].
        """
        progress = max(0.0, min(1.0, progress))
        dot_prefix = field_prefix + "."
        for key, pipeline in self._ego_pipelines.items():
            if key.startswith(dot_prefix):
                pipeline.set_mode(mode, progress)
        for key, pipeline in self._other_pipelines.items():
            if key.startswith(dot_prefix):
                pipeline.set_mode(mode, progress)

    def set_field_dropout_rate(self, field_prefix: str, rate: float):
        """Set dropout rate for pipelines whose key starts with field_prefix.

        Args:
            field_prefix: Prefix to match (e.g., 'drone_0').
            rate: Dropout probability [0, 1].
        """
        dot_prefix = field_prefix + "."
        for key, pipeline in self._ego_pipelines.items():
            if key.startswith(dot_prefix):
                pipeline.set_dropout_rate(rate)
        for key, pipeline in self._other_pipelines.items():
            if key.startswith(dot_prefix):
                pipeline.set_dropout_rate(rate)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset system for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        # Reset storage
        self._storage.reset(env_ids)

        # Reset all pipelines
        for pipeline in self._ego_pipelines.values():
            pipeline.reset(env_ids)
        for pipeline in self._other_pipelines.values():
            pipeline.reset(env_ids)

        # Reset time for specified environments
        if env_ids is None:
            self._t_current.zero_()
        else:
            self._t_current[env_ids] = 0.0

    def step(self, reset_env_ids: Optional[torch.Tensor] = None):
        """Called each simulation step to update internal state.

        Args:
            reset_env_ids: Environments that were reset this step.
        """
        # Maybe resample parameters
        for pipeline in self._ego_pipelines.values():
            pipeline.maybe_resample(reset_env_ids)
        for pipeline in self._other_pipelines.values():
            pipeline.maybe_resample(reset_env_ids)


class DelayedFieldAccessor:
    """Convenience accessor for delayed fields.

    Provides a cleaner API for common access patterns.
    """

    def __init__(
        self,
        system: UnifiedDelaySystem,
        field_name: str,
        perspective: Literal["ego", "other"],
    ):
        """Initialize accessor.

        Args:
            system: Unified delay system.
            field_name: Field to access.
            perspective: View perspective.
        """
        self._system = system
        self._field_name = field_name
        self._perspective = perspective

    def get_for_rewards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get clean (no-noise) delayed data for reward computation."""
        return self._system.get_delayed(
            self._field_name, self._perspective, use_noise=False, allow_dropout=False
        )

    def get_for_observations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get noisy delayed data for observations."""
        return self._system.get_delayed(
            self._field_name, self._perspective, use_noise=True, allow_dropout=True
        )

    def get_aoi(self, use_noise: bool = True) -> torch.Tensor:
        """Get Age-of-Information."""
        return self._system.get_aoi(self._field_name, self._perspective, use_noise)
