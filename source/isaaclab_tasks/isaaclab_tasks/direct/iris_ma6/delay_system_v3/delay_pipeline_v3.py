# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delay pipeline V3 with timestamp guarantees and advance/query split.

This module implements the core delay pipeline with the fundamental invariant:
**Timestamp always travels with its associated data through all stages.**

Pipeline stages (in advance order):
1. Staleness (FPS limiting) - holds data and timestamp together
2. Latency (communication delay) - parallel buffers for data and timestamp
3. First-order lag (smoothing) - filters channel signal before dropout
4. Dropout (missed detections) - holds both data and timestamp together

Calling contract:
- advance(): WRITE — runs all stateful stages once. Idempotent within a sim step.
- query(): READ — returns cached output. Safe to call multiple times.
- process(): Convenience wrapper (advance + query). Backward compatible.

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
        # Guard against multiple appends per sim step (bug fix: process() may
        # be called multiple times at the same t_current for rewards/obs/teleop)
        self._last_append_time = torch.full((num_envs,), -1.0, device=device)

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

        # =================================================================
        # Advance/query cache (ensures idempotent multi-call per step)
        # =================================================================
        self._last_advance_time = torch.full((num_envs,), -1.0, device=device)
        self._cached_data_with_dropout = torch.zeros(num_envs, *data_shape, device=device)
        self._cached_ts_with_dropout = torch.zeros(num_envs, device=device)
        self._cached_data_no_dropout = torch.zeros(num_envs, *data_shape, device=device)
        self._cached_ts_no_dropout = torch.zeros(num_envs, device=device)

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

    def advance(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
        burst_dropout_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance all stateful pipeline stages. WRITE — call once per sim step.

        Idempotent: if called again at the same t_current, skips all state mutation
        and keeps the previously cached results.

        Pipeline order: staleness → latency → FOL → dropout.
        FOL is placed before dropout so the sensor filter operates on the channel
        signal before packet-loss masking. This avoids needing separate FOL states
        for the with/without-dropout query paths.

        Args:
            data: Input data tensor of shape (num_envs, ...).
            timestamp: Capture timestamp of shape (num_envs,).
            t_current: Current simulation time of shape (num_envs,).
            burst_dropout_mask: Optional external dropout mask of shape (num_envs,),
                dtype bool. When provided, replaces the internal i.i.d. sampler for
                this step (used by burst dropout from multi-agent wrapper).
        """
        data = data.to(self._device)
        timestamp = timestamp.to(self._device)
        t_current = t_current.to(self._device)

        # Idempotency guard — skip if already advanced at this time
        time_advanced = (t_current - self._last_advance_time).abs() > 1e-6
        if not time_advanced.any():
            return
        self._last_advance_time = t_current.clone()

        # "none" mode: only FOL
        if self._mode == "none":
            if self._lag_enabled:
                data = self._apply_first_order_lag(data)
            self._cached_data_no_dropout = data.clone()
            self._cached_ts_no_dropout = timestamp.clone()
            self._cached_data_with_dropout = data.clone()
            self._cached_ts_with_dropout = timestamp.clone()
            return

        # Stage 1: Staleness (only in random mode)
        if self._cfg.staleness.enabled and self._mode == "random":
            data, timestamp = self._apply_staleness(data, timestamp, t_current)

        # Stage 2: Latency (buffer append guarded internally)
        if self._cfg.latency.enabled:
            data, timestamp = self._apply_latency(data, timestamp, t_current)

        # Stage 3: FOL (before dropout — smooths channel signal)
        if self._lag_enabled:
            data = self._apply_first_order_lag(data)

        # Cache pre-dropout result (used by reward queries with allow_dropout=False)
        self._cached_data_no_dropout = data.clone()
        self._cached_ts_no_dropout = timestamp.clone()

        # Stage 4: Dropout (mask sampled once per step)
        if self._cfg.dropout.enabled:
            data, timestamp = self._apply_dropout(data, timestamp, external_mask=burst_dropout_mask)

        self._cached_data_with_dropout = data.clone()
        self._cached_ts_with_dropout = timestamp.clone()

    def query(self, allow_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached pipeline output. READ — safe to call multiple times.

        Args:
            allow_dropout: If True, return post-dropout data.
                          If False, return pre-dropout data (for rewards).

        Returns:
            Tuple of (delayed_data, delayed_timestamp).
        """
        if allow_dropout:
            return self._cached_data_with_dropout.clone(), self._cached_ts_with_dropout.clone()
        else:
            return self._cached_data_no_dropout.clone(), self._cached_ts_no_dropout.clone()

    def process(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
        allow_dropout: bool = True,
        burst_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process data through the delay pipeline (advance + query).

        Backward-compatible convenience wrapper. advance() is idempotent,
        so multiple process() calls at the same t_current only advance once.

        **CRITICAL INVARIANT**: The returned timestamp is ALWAYS the capture
        time of the returned data.

        Args:
            data: Input data tensor of shape (num_envs, ...).
            timestamp: Capture timestamp of shape (num_envs,).
            t_current: Current simulation time of shape (num_envs,).
            allow_dropout: Whether to return post-dropout data.
            burst_dropout_mask: Optional external dropout mask from burst model.

        Returns:
            Tuple of (delayed_data, delayed_timestamp).
        """
        self.advance(data, timestamp, t_current, burst_dropout_mask=burst_dropout_mask)
        return self.query(allow_dropout=allow_dropout)

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
        t_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply latency using parallel circular buffers.

        Data and timestamp are put through SEPARATE buffers with the
        SAME delay, ensuring they stay synchronized.

        **KEY**: Parallel buffers maintain the data-timestamp correspondence.

        **GUARD**: Only appends to buffer once per sim step. process() may be
        called multiple times at the same t_current (rewards, observations,
        teleop visualization). Without this guard the buffer advances N times
        per real step, dividing effective delay by N.
        """
        # Flatten data for buffer storage
        flat_data = data.reshape(self._num_envs, -1)
        flat_ts = timestamp.unsqueeze(-1)  # (num_envs, 1)

        if not self._buffer_initialized:
            # Initialize buffers with first data
            # CircularBuffer will fill all slots with this data
            self._data_buffer.append(flat_data)
            self._timestamp_buffer.append(flat_ts)
            self._last_append_time = t_current.clone()
            self._buffer_initialized = True

        # Only append if simulation time has advanced since last append
        time_advanced = (t_current - self._last_append_time).abs() > 1e-6
        if time_advanced.any():
            self._data_buffer.append(flat_data)
            self._timestamp_buffer.append(flat_ts)
            self._last_append_time = t_current.clone()

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
        external_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout (missed detections).

        When dropout occurs, the previous data AND timestamp are retained.

        **KEY**: Both data and timestamp are held together during dropout.

        Args:
            data: Input data tensor.
            timestamp: Capture timestamp tensor.
            external_mask: Optional external dropout mask of shape (num_envs,),
                dtype bool. When provided, replaces the internal i.i.d. sampler
                (used by burst dropout). True = dropped.
        """
        if not self._dropout_initialized:
            # First call - initialize held values with incoming data
            # This ensures we never return zeros on first step
            self._dropout_held_data = data.clone()
            self._dropout_held_timestamp = timestamp.clone()
            self._dropout_initialized = True
            return data.clone(), timestamp.clone()

        # Use external mask (burst dropout) or internal i.i.d. sampler
        if external_mask is not None:
            drop_mask = external_mask
        else:
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
            self._last_append_time.fill_(-1.0)

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

            # Reset advance/query cache
            self._last_advance_time.fill_(-1.0)
            self._cached_data_with_dropout.zero_()
            self._cached_ts_with_dropout.zero_()
            self._cached_data_no_dropout.zero_()
            self._cached_ts_no_dropout.zero_()

        else:
            # Reset specific environments
            self._data_buffer.reset(env_ids)
            self._timestamp_buffer.reset(env_ids)
            self._last_append_time[env_ids] = -1.0

            # Reset staleness for specific envs
            self._staleness_held_data[env_ids] = 0
            self._staleness_held_timestamp[env_ids] = 0
            self._last_detection_time[env_ids] = 0

            # Reset dropout for specific envs
            self._dropout_held_data[env_ids] = 0
            self._dropout_held_timestamp[env_ids] = 0

            # Reset lag for specific envs
            self._lag_output[env_ids] = 0

            # Reset advance/query cache for specific envs
            self._last_advance_time[env_ids] = -1.0
            self._cached_data_with_dropout[env_ids] = 0
            self._cached_ts_with_dropout[env_ids] = 0
            self._cached_data_no_dropout[env_ids] = 0
            self._cached_ts_no_dropout[env_ids] = 0

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
