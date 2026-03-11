# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delay pipeline V3 with timestamp guarantees.

This module implements the core delay pipeline with the fundamental invariant:
**Timestamp always travels with its associated data through all stages.**

Pipeline stages:
1. Staleness (FPS limiting) - holds data and timestamp together
2. Latency (communication delay) - parallel buffers for data and timestamp
3. Dropout (missed detections) - holds both data and timestamp together

Key design principle: Data and timestamp are ALWAYS processed together,
ensuring AoI (Age-of-Information) = t_current - timestamp is always correct.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional, Literal

from isaaclab.utils.buffers import CircularBuffer

from .delay_cfg_v3 import DelayPipelineCfgV3, LatencyCfg, StalenessCfg, DropoutCfg
from .sampling_strategies import LatencySampler, StalenessSampler, DropoutSampler, SamplingCfg


class DelayPipelineV3:
    """Single-stage delay pipeline with configurable sampling and timestamp guarantees.

    This pipeline processes data through:
    1. Staleness (FPS limiting) - simulates limited sensor update rates
    2. Latency (communication delay) - simulates network/processing delays
    3. Dropout (missed detections) - simulates communication failures

    **CRITICAL INVARIANT**: The returned timestamp is ALWAYS the capture time
    of the returned data, regardless of which stages are enabled.
    """

    def __init__(
        self,
        cfg: DelayPipelineCfgV3,
        num_envs: int,
        device: torch.device,
        dt: float,
        data_shape: Tuple[int, ...] = (3,),
    ):
        """Initialize the delay pipeline.

        Args:
            cfg: Pipeline configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
            dt: Simulation time step.
            data_shape: Shape of data per environment (excluding batch dim).
        """
        self._cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._dt = dt
        self._data_shape = data_shape

        # Curriculum mode
        self._mode: Literal["none", "fixed", "random"] = "none"
        self._progress: float = 1.0

        # =================================================================
        # Latency Stage
        # =================================================================
        # Compute max buffer size needed
        max_latency = cfg.latency.distribution.mean + 3 * cfg.latency.distribution.std
        max_steps = max(int(max_latency / dt) + 5, cfg.latency.min_steps + 5)

        # Latency sampler
        self._latency_sampler = LatencySampler(
            distribution_cfg=cfg.latency.distribution,
            sampling_cfg=cfg.latency.sampling,
            num_envs=num_envs,
            device=device,
            dt=dt,
            min_steps=cfg.latency.min_steps,
        )

        # Parallel circular buffers for data and timestamp
        # Using CircularBuffer from Isaac Lab
        self._data_buffer = CircularBuffer(max_steps, num_envs, device)
        self._timestamp_buffer = CircularBuffer(max_steps, num_envs, device)
        self._buffer_initialized = False

        # =================================================================
        # Staleness Stage
        # =================================================================
        self._staleness_sampler = StalenessSampler(
            distribution_cfg=cfg.staleness.fps_distribution,
            sampling_cfg=cfg.staleness.sampling,
            num_envs=num_envs,
            device=device,
        )

        # Held data and timestamp during staleness
        self._staleness_held_data = torch.zeros(
            num_envs, *data_shape, device=device
        )
        self._staleness_held_timestamp = torch.zeros(num_envs, device=device)
        self._last_detection_time = torch.zeros(num_envs, device=device)
        self._staleness_initialized = False

        # =================================================================
        # Dropout Stage
        # =================================================================
        self._dropout_sampler = DropoutSampler(
            probability=cfg.dropout.probability,
            rate_distribution=cfg.dropout.rate_distribution,
            sampling_cfg=cfg.dropout.sampling,
            num_envs=num_envs,
            device=device,
        )

        # Held data and timestamp during dropout
        self._dropout_held_data = torch.zeros(
            num_envs, *data_shape, device=device
        )
        self._dropout_held_timestamp = torch.zeros(num_envs, device=device)
        self._dropout_initialized = False

        # =================================================================
        # First-order lag (optional smoothing)
        # =================================================================
        self._lag_enabled = cfg.first_order_lag.enabled and cfg.first_order_lag.tau > 0
        self._lag_tau = cfg.first_order_lag.tau
        self._lag_output = torch.zeros(num_envs, *data_shape, device=device)
        self._lag_initialized = False

        # Initialize samplers
        self._latency_sampler.initialize()
        self._staleness_sampler.initialize()
        self._dropout_sampler.initialize()

    @property
    def cfg(self) -> DelayPipelineCfgV3:
        """Pipeline configuration."""
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
    def mode(self) -> str:
        """Current delay mode."""
        return self._mode

    @property
    def progress(self) -> float:
        """Current curriculum progress."""
        return self._progress

    @property
    def current_latency_steps(self) -> torch.Tensor:
        """Current latency in steps for each environment."""
        return self._latency_sampler.step_values

    @property
    def current_detection_period(self) -> torch.Tensor:
        """Current detection period for each environment."""
        return self._staleness_sampler.period_values

    @property
    def current_dropout_rates(self) -> torch.Tensor:
        """Current dropout rates for each environment."""
        return self._dropout_sampler.rates

    def set_mode(self, mode: Literal["none", "fixed", "random"], progress: float = 1.0):
        """Set delay mode for curriculum learning.

        Args:
            mode: Delay mode:
                - 'none': No delay applied (pass-through)
                - 'fixed': Fixed deterministic delay
                - 'random': Random delay with staleness
            progress: Curriculum progress [0, 1].
        """
        if mode not in ("none", "fixed", "random"):
            raise ValueError(f"Unknown delay mode: {mode}")

        self._mode = mode
        self._progress = max(0.0, min(1.0, progress))

        # Update sampler scales based on mode and progress
        if mode == "none":
            self._latency_sampler.set_scale(0.0)
        else:
            self._latency_sampler.set_scale(self._progress)

        # Staleness only in random mode
        # (In fixed mode, we still apply latency but no staleness)

        # Resample with new settings
        self._latency_sampler.initialize()

    def set_dropout_rate(self, rate: float):
        """Set dropout probability for curriculum control.

        Args:
            rate: Dropout probability [0, 1].
        """
        self._dropout_sampler.set_rate(rate)

    def process(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
        allow_dropout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process data through the delay pipeline.

        **CRITICAL INVARIANT**: The returned timestamp is ALWAYS the capture
        time of the returned data.

        Args:
            data: Input data tensor of shape (num_envs, ...).
            timestamp: Capture timestamp of shape (num_envs,).
            t_current: Current simulation time of shape (num_envs,).
            allow_dropout: Whether to apply dropout this step.

        Returns:
            Tuple of (delayed_data, delayed_timestamp).
        """
        # Ensure tensors are on correct device
        data = data.to(self._device)
        timestamp = timestamp.to(self._device)
        t_current = t_current.to(self._device)

        # In "none" mode, just pass through (minimal delay)
        if self._mode == "none":
            # Still need first-order lag if enabled
            if self._lag_enabled:
                data = self._apply_first_order_lag(data)
            return data.clone(), timestamp.clone()

        # Stage 1: Staleness (only in random mode)
        if self._cfg.staleness.enabled and self._mode == "random":
            data, timestamp = self._apply_staleness(data, timestamp, t_current)

        # Stage 2: Latency
        if self._cfg.latency.enabled:
            data, timestamp = self._apply_latency(data, timestamp)

        # Stage 3: Dropout (if allowed)
        if self._cfg.dropout.enabled and allow_dropout:
            data, timestamp = self._apply_dropout(data, timestamp)

        # Stage 4: First-order lag (optional)
        if self._lag_enabled:
            data = self._apply_first_order_lag(data)

        return data, timestamp

    def _apply_staleness(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply staleness (FPS limiting).

        Simulates limited sensor update rates. Data is only updated when
        the time since last detection exceeds the detection period.

        **KEY**: Both data AND timestamp are held together during staleness.
        """
        if not self._staleness_initialized:
            # First call - initialize held values
            self._staleness_held_data = data.clone()
            self._staleness_held_timestamp = timestamp.clone()
            self._last_detection_time = t_current.clone()
            self._staleness_initialized = True
            return data.clone(), timestamp.clone()

        # Compute time since last detection
        time_since_last = t_current - self._last_detection_time
        detection_period = self._staleness_sampler.period_values

        # Determine which environments get a new detection
        should_update = time_since_last >= detection_period

        # Update BOTH data and timestamp together
        # Shape handling for data (may have multiple dims)
        should_update_data = should_update
        for _ in range(len(data.shape) - 1):
            should_update_data = should_update_data.unsqueeze(-1)

        self._staleness_held_data = torch.where(
            should_update_data, data, self._staleness_held_data
        )
        self._staleness_held_timestamp = torch.where(
            should_update, timestamp, self._staleness_held_timestamp
        )

        # Update last detection time for environments that got new detection
        self._last_detection_time = torch.where(
            should_update, t_current, self._last_detection_time
        )

        return self._staleness_held_data.clone(), self._staleness_held_timestamp.clone()

    def _apply_latency(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply latency using parallel circular buffers.

        Data and timestamp are put through SEPARATE buffers with the
        SAME delay, ensuring they stay synchronized.

        **KEY**: Parallel buffers maintain the data-timestamp correspondence.
        """
        # Flatten data for buffer storage
        flat_data = data.reshape(self._num_envs, -1)
        flat_ts = timestamp.unsqueeze(-1)  # (num_envs, 1)

        if not self._buffer_initialized:
            # Initialize buffers with first data
            # CircularBuffer will fill all slots with this data
            self._data_buffer.append(flat_data)
            self._timestamp_buffer.append(flat_ts)
            self._buffer_initialized = True

        # Append new data to both buffers
        self._data_buffer.append(flat_data)
        self._timestamp_buffer.append(flat_ts)

        # Retrieve with the same time lag
        time_lags = self._latency_sampler.step_values

        # CircularBuffer uses __getitem__ with time_lags tensor
        delayed_flat_data = self._data_buffer[time_lags]
        delayed_flat_ts = self._timestamp_buffer[time_lags]

        # Reshape back
        delayed_data = delayed_flat_data.reshape(self._num_envs, *self._data_shape)
        delayed_timestamp = delayed_flat_ts.squeeze(-1)

        return delayed_data, delayed_timestamp

    def _apply_dropout(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout (missed detections).

        When dropout occurs, the previous data AND timestamp are retained.

        **KEY**: Both data and timestamp are held together during dropout.
        """
        if not self._dropout_initialized:
            # First call - initialize held values with incoming data
            # This ensures we never return zeros on first step
            self._dropout_held_data = data.clone()
            self._dropout_held_timestamp = timestamp.clone()
            self._dropout_initialized = True
            return data.clone(), timestamp.clone()

        # Sample dropout mask for this step
        drop_mask = self._dropout_sampler.sample_mask()

        # Shape handling
        drop_mask_data = drop_mask
        for _ in range(len(data.shape) - 1):
            drop_mask_data = drop_mask_data.unsqueeze(-1)

        # On drop: keep previous held value
        # On no drop: update held value with new data
        self._dropout_held_data = torch.where(
            drop_mask_data, self._dropout_held_data, data
        )
        self._dropout_held_timestamp = torch.where(
            drop_mask, self._dropout_held_timestamp, timestamp
        )

        return self._dropout_held_data.clone(), self._dropout_held_timestamp.clone()

    def _apply_first_order_lag(self, data: torch.Tensor) -> torch.Tensor:
        """Apply first-order lag filter for smoothing.

        This smooths sudden changes in data. Note that this does NOT
        affect the timestamp - the timestamp still represents when
        the underlying data was captured.
        """
        if not self._lag_initialized:
            self._lag_output = data.clone()
            self._lag_initialized = True
            return data.clone()

        # First-order lag: y = y_prev + dt/tau * (x - y_prev)
        alpha = self._dt / (self._lag_tau + self._dt)
        self._lag_output = self._lag_output + alpha * (data - self._lag_output)

        return self._lag_output.clone()

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset pipeline state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            # Reset all
            env_ids = torch.arange(self._num_envs, device=self._device)

            # Reset buffers
            self._data_buffer.reset()
            self._timestamp_buffer.reset()
            self._buffer_initialized = False

            # Reset staleness
            self._staleness_held_data.zero_()
            self._staleness_held_timestamp.zero_()
            self._last_detection_time.zero_()
            self._staleness_initialized = False

            # Reset dropout
            self._dropout_held_data.zero_()
            self._dropout_held_timestamp.zero_()
            self._dropout_initialized = False

            # Reset lag
            self._lag_output.zero_()
            self._lag_initialized = False

        else:
            # Reset specific environments
            self._data_buffer.reset(env_ids)
            self._timestamp_buffer.reset(env_ids)

            # Reset staleness for specific envs
            self._staleness_held_data[env_ids] = 0
            self._staleness_held_timestamp[env_ids] = 0
            self._last_detection_time[env_ids] = 0

            # Reset dropout for specific envs
            self._dropout_held_data[env_ids] = 0
            self._dropout_held_timestamp[env_ids] = 0

            # Reset lag for specific envs
            self._lag_output[env_ids] = 0

        # Resample parameters for reset environments
        self._latency_sampler.reset(env_ids)
        self._staleness_sampler.reset(env_ids)
        self._dropout_sampler.reset(env_ids)

    def maybe_resample(self, reset_env_ids: Optional[torch.Tensor] = None):
        """Potentially resample parameters based on configured frequencies.

        Args:
            reset_env_ids: Environments that were just reset.
        """
        self._latency_sampler.maybe_resample(reset_env_ids)
        self._staleness_sampler.maybe_resample(reset_env_ids)
        self._dropout_sampler.maybe_resample_rate(reset_env_ids)
