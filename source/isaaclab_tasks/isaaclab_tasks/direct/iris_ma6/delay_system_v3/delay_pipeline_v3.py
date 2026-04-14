# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delay pipeline V3 with timestamp guarantees and advance/query split.

This module implements the core delay pipeline with two fundamental invariants:
1. **Timestamp always travels with its associated data through all stages.**
2. **Raw and noisy payloads share a single delay realization.** Staleness masks,
   latency draws, dropout masks, and first-order lag filters are sampled once
   per sim step and applied to both payloads in lockstep. The payloads may
   differ in content but are always delayed by the same amount.

Pipeline stages (in advance order):
1. Staleness (FPS limiting) - single mask held across both payloads
2. Latency (communication delay) - parallel raw/noisy buffers, one timestamp buffer
3. First-order lag (smoothing) - independent filter state per payload
4. Dropout (missed detections) - single mask applied to both payloads

Calling contract:
- advance(): WRITE — runs all stateful stages once. Idempotent within a sim step.
- query(): READ — returns cached output. Safe to call multiple times.
- process(): Convenience wrapper (advance + query). Backward compatible.

When a field has no separate noisy payload, ``advance`` accepts
``noisy_data=None`` and both cache slots are filled with the raw payload —
the observation query then returns the same delayed clean data as the
reward query. Fields with a noisy payload (e.g. state fields with curriculum
Gaussian noise, or the ``bboxes_2d`` detection field carrying raycaster GT
as raw and detector-replicator output as noisy) receive distinct cache
slots populated under one shared delay realization.
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
        max_latency = cfg.latency.distribution.mean + 4 * cfg.latency.distribution.std
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

        # Parallel circular buffers for raw/noisy payloads + a single timestamp
        # buffer. Both data buffers are indexed by the same `time_lags` tensor
        # so raw and noisy stay synchronized in delay.
        self._raw_data_buffer = CircularBuffer(max_steps, num_envs, device)
        self._noisy_data_buffer = CircularBuffer(max_steps, num_envs, device)
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

        # Held data (raw + noisy) and timestamp during staleness. Update mask
        # is shared so raw and noisy stay in lockstep.
        self._staleness_held_raw = torch.zeros(num_envs, *data_shape, device=device)
        self._staleness_held_noisy = torch.zeros(num_envs, *data_shape, device=device)
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

        # Held data (raw + noisy) and timestamp during dropout. One drop mask
        # is sampled per step and applied to both payloads.
        self._dropout_held_raw = torch.zeros(num_envs, *data_shape, device=device)
        self._dropout_held_noisy = torch.zeros(num_envs, *data_shape, device=device)
        self._dropout_held_timestamp = torch.zeros(num_envs, device=device)
        self._dropout_initialized = False

        # =================================================================
        # First-order lag (optional smoothing) — independent state per payload
        # so each signal smooths to itself, but the filter coefficient is shared.
        # =================================================================
        self._lag_enabled = cfg.first_order_lag.enabled and cfg.first_order_lag.tau > 0
        self._lag_tau = cfg.first_order_lag.tau
        self._lag_output_raw = torch.zeros(num_envs, *data_shape, device=device)
        self._lag_output_noisy = torch.zeros(num_envs, *data_shape, device=device)
        self._lag_initialized_raw = False
        self._lag_initialized_noisy = False

        # =================================================================
        # Advance/query cache — four data slots (raw/noisy × with/no dropout)
        # plus two timestamp slots (with/no dropout). Timestamps are shared
        # across raw and noisy since both payloads follow the same delay.
        # =================================================================
        self._last_advance_time = torch.full((num_envs,), -1.0, device=device)
        self._cached_raw_no_dropout = torch.zeros(num_envs, *data_shape, device=device)
        self._cached_noisy_no_dropout = torch.zeros(num_envs, *data_shape, device=device)
        self._cached_raw_with_dropout = torch.zeros(num_envs, *data_shape, device=device)
        self._cached_noisy_with_dropout = torch.zeros(num_envs, *data_shape, device=device)
        self._cached_ts_no_dropout = torch.zeros(num_envs, device=device)
        self._cached_ts_with_dropout = torch.zeros(num_envs, device=device)

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
        raw_data: torch.Tensor,
        noisy_data: Optional[torch.Tensor],
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
        burst_dropout_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance all stateful pipeline stages. WRITE — call once per sim step.

        Raw and noisy payloads are processed together under a single delay
        realization (shared staleness mask, shared latency draw, shared dropout
        mask). This is the invariant that makes reward / observation queries
        consistent: when both are taken at the same ``t_current`` they refer to
        the same physical detection event.

        Idempotent: if called again at the same t_current, skips all state mutation
        and keeps the previously cached results.

        Pipeline order: staleness → latency → FOL → dropout.
        FOL is placed before dropout so the sensor filter operates on the channel
        signal before packet-loss masking. This avoids needing separate FOL states
        for the with/without-dropout query paths.

        Args:
            raw_data: Clean input tensor of shape (num_envs, ...).
            noisy_data: Optional noisy input with the same shape. When ``None``
                the field has no separate noisy payload; both caches are filled
                with the raw payload.
            timestamp: Capture timestamp of shape (num_envs,). Shared by both
                raw and noisy — they describe the same event.
            t_current: Current simulation time of shape (num_envs,).
            burst_dropout_mask: Optional external dropout mask of shape (num_envs,),
                dtype bool. When provided, replaces the internal i.i.d. sampler for
                this step (used by burst dropout from multi-agent wrapper).
        """
        raw_data = raw_data.to(self._device)
        timestamp = timestamp.to(self._device)
        t_current = t_current.to(self._device)
        if noisy_data is None:
            # Field has no separate noisy payload — mirror raw into the noisy
            # cache slots so query(use_noise=True) is still well-defined.
            noisy_data = raw_data
        else:
            noisy_data = noisy_data.to(self._device)
            if noisy_data.shape != raw_data.shape:
                raise ValueError(
                    f"DelayPipelineV3.advance: raw/noisy shape mismatch "
                    f"{raw_data.shape} vs {noisy_data.shape}"
                )

        # Idempotency guard — skip if already advanced at this time.
        # A subsequent call with a different `noisy_data` at the same t_current
        # is intentionally ignored; this is the single-advance-per-step contract.
        time_advanced = (t_current - self._last_advance_time).abs() > 1e-6
        if not time_advanced.any():
            return
        self._last_advance_time = t_current.clone()

        # "none" mode: only FOL (applied independently to each payload)
        if self._mode == "none":
            if self._lag_enabled:
                raw_out = self._apply_first_order_lag(raw_data, payload="raw")
                noisy_out = self._apply_first_order_lag(noisy_data, payload="noisy")
            else:
                raw_out = raw_data
                noisy_out = noisy_data
            self._cached_raw_no_dropout = raw_out.clone()
            self._cached_noisy_no_dropout = noisy_out.clone()
            self._cached_ts_no_dropout = timestamp.clone()
            self._cached_raw_with_dropout = raw_out.clone()
            self._cached_noisy_with_dropout = noisy_out.clone()
            self._cached_ts_with_dropout = timestamp.clone()
            return

        # Stage 1: Staleness (only in random mode). Single update mask drives
        # both payloads' held buffers.
        if self._cfg.staleness.enabled and self._mode == "random":
            raw_data, noisy_data, timestamp = self._apply_staleness_pair(
                raw_data, noisy_data, timestamp, t_current
            )

        # Stage 2: Latency (parallel raw/noisy buffers, shared time_lags)
        if self._cfg.latency.enabled:
            raw_data, noisy_data, timestamp = self._apply_latency_pair(
                raw_data, noisy_data, timestamp, t_current
            )

        # Stage 3: FOL (independent filter state per payload, shared alpha)
        if self._lag_enabled:
            raw_data = self._apply_first_order_lag(raw_data, payload="raw")
            noisy_data = self._apply_first_order_lag(noisy_data, payload="noisy")

        # Cache pre-dropout (reward queries use allow_dropout=False)
        self._cached_raw_no_dropout = raw_data.clone()
        self._cached_noisy_no_dropout = noisy_data.clone()
        self._cached_ts_no_dropout = timestamp.clone()

        # Stage 4: Dropout — single mask sampled once per step, applied to both.
        if self._cfg.dropout.enabled:
            raw_data, noisy_data, timestamp = self._apply_dropout_pair(
                raw_data, noisy_data, timestamp, external_mask=burst_dropout_mask,
            )

        self._cached_raw_with_dropout = raw_data.clone()
        self._cached_noisy_with_dropout = noisy_data.clone()
        self._cached_ts_with_dropout = timestamp.clone()

    def query(
        self,
        use_noise: bool = False,
        allow_dropout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached pipeline output. READ — safe to call multiple times.

        Args:
            use_noise: If True, return the noisy-payload cache (observations).
                       If False, return the raw-payload cache (rewards).
            allow_dropout: If True, return post-dropout data.
                           If False, return pre-dropout data (for rewards).

        Returns:
            Tuple of (delayed_data, delayed_timestamp). Timestamp is shared
            across raw/noisy because both payloads were delayed together.
        """
        if allow_dropout:
            ts = self._cached_ts_with_dropout
            data = (
                self._cached_noisy_with_dropout if use_noise
                else self._cached_raw_with_dropout
            )
        else:
            ts = self._cached_ts_no_dropout
            data = (
                self._cached_noisy_no_dropout if use_noise
                else self._cached_raw_no_dropout
            )
        return data.clone(), ts.clone()

    def process(
        self,
        raw_data: torch.Tensor,
        noisy_data: Optional[torch.Tensor],
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
        use_noise: bool = False,
        allow_dropout: bool = True,
        burst_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process raw/noisy payload pair through the delay pipeline.

        ``advance`` is idempotent, so multiple ``process()`` calls at the same
        ``t_current`` only advance once. The first call populates both caches;
        subsequent calls (including observation call after reward call) read
        from the cache. This is the structural fix for the shared-pipeline
        idempotency bug: both payloads are cached on the first advance, so
        the second call's ``use_noise`` flag selects the right cache slot.

        **CRITICAL INVARIANT**: The returned timestamp is ALWAYS the capture
        time of the returned data, and raw/noisy queries at the same step
        share a single timestamp.

        Args:
            raw_data: Clean input tensor of shape (num_envs, ...).
            noisy_data: Optional noisy input with the same shape. Pass ``None``
                for clean-only fields.
            timestamp: Capture timestamp of shape (num_envs,).
            t_current: Current simulation time of shape (num_envs,).
            use_noise: Which cache slot to return on the query side.
            allow_dropout: Whether to return post-dropout data.
            burst_dropout_mask: Optional external dropout mask from burst model.

        Returns:
            Tuple of (delayed_data, delayed_timestamp).
        """
        self.advance(
            raw_data, noisy_data, timestamp, t_current,
            burst_dropout_mask=burst_dropout_mask,
        )
        return self.query(use_noise=use_noise, allow_dropout=allow_dropout)

    def process_single(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
        allow_dropout: bool = True,
        burst_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper for clean-only fields (no noisy payload).

        Equivalent to ``process(data, noisy_data=None, timestamp, t_current,
        use_noise=False, allow_dropout=allow_dropout, ...)`` — both caches are
        populated with the raw payload and the raw cache is returned. Useful
        for test suites, teleop visualization, and any consumer that does not
        carry a separate noisy stream.
        """
        return self.process(
            data, None, timestamp, t_current,
            use_noise=False, allow_dropout=allow_dropout,
            burst_dropout_mask=burst_dropout_mask,
        )

    def _apply_staleness_pair(
        self,
        raw: torch.Tensor,
        noisy: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply staleness (FPS limiting) to a raw/noisy payload pair.

        Simulates limited sensor update rates. A single update mask is computed
        from the staleness sampler and applied to BOTH payloads in lockstep —
        raw and noisy are always held-or-refreshed together, so their
        timestamps stay synchronized.

        **KEY**: All three (raw, noisy, timestamp) share a single update mask.
        """
        if not self._staleness_initialized:
            # First call - initialize held values with incoming pair
            self._staleness_held_raw = raw.clone()
            self._staleness_held_noisy = noisy.clone()
            self._staleness_held_timestamp = timestamp.clone()
            self._last_detection_time = t_current.clone()
            self._staleness_initialized = True
            return raw.clone(), noisy.clone(), timestamp.clone()

        # Compute time since last detection (shared across payloads)
        time_since_last = t_current - self._last_detection_time
        detection_period = self._staleness_sampler.period_values

        # Shared update mask — raw and noisy refresh on the same step
        should_update = time_since_last >= detection_period

        # Shape handling for data (may have multiple dims)
        should_update_data = should_update
        for _ in range(len(raw.shape) - 1):
            should_update_data = should_update_data.unsqueeze(-1)

        self._staleness_held_raw = torch.where(
            should_update_data, raw, self._staleness_held_raw
        )
        self._staleness_held_noisy = torch.where(
            should_update_data, noisy, self._staleness_held_noisy
        )
        self._staleness_held_timestamp = torch.where(
            should_update, timestamp, self._staleness_held_timestamp
        )

        # Update last detection time for environments that got new detection
        self._last_detection_time = torch.where(
            should_update, t_current, self._last_detection_time
        )

        return (
            self._staleness_held_raw.clone(),
            self._staleness_held_noisy.clone(),
            self._staleness_held_timestamp.clone(),
        )

    def _apply_latency_pair(
        self,
        raw: torch.Tensor,
        noisy: torch.Tensor,
        timestamp: torch.Tensor,
        t_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply latency using parallel circular buffers.

        Raw and noisy payloads flow through SEPARATE buffers but share a
        single timestamp buffer and a single latency draw, ensuring both
        payloads exit the stage with the same delay and the same timestamp.

        **KEY**: One latency sample drives BOTH buffer reads. Raw and noisy
        cannot diverge in delay.

        **GUARD**: Only appends to the buffers once per sim step. process() may
        be called multiple times at the same t_current (rewards, observations,
        teleop visualization). Without this guard the buffer advances N times
        per real step, dividing effective delay by N.
        """
        # Flatten data for buffer storage
        flat_raw = raw.reshape(self._num_envs, -1)
        flat_noisy = noisy.reshape(self._num_envs, -1)
        flat_ts = timestamp.unsqueeze(-1)  # (num_envs, 1)

        if not self._buffer_initialized:
            # Initialize buffers with first payload
            # CircularBuffer will fill all slots with this data
            self._raw_data_buffer.append(flat_raw)
            self._noisy_data_buffer.append(flat_noisy)
            self._timestamp_buffer.append(flat_ts)
            self._last_append_time = t_current.clone()
            self._buffer_initialized = True

        # Only append if simulation time has advanced since last append
        time_advanced = (t_current - self._last_append_time).abs() > 1e-6
        if time_advanced.any():
            self._raw_data_buffer.append(flat_raw)
            self._noisy_data_buffer.append(flat_noisy)
            self._timestamp_buffer.append(flat_ts)
            self._last_append_time = t_current.clone()

        # Retrieve with a single time lag — shared across raw/noisy.
        time_lags = self._latency_sampler.step_values

        delayed_flat_raw = self._raw_data_buffer[time_lags]
        delayed_flat_noisy = self._noisy_data_buffer[time_lags]
        delayed_flat_ts = self._timestamp_buffer[time_lags]

        # Reshape back
        delayed_raw = delayed_flat_raw.reshape(self._num_envs, *self._data_shape)
        delayed_noisy = delayed_flat_noisy.reshape(self._num_envs, *self._data_shape)
        delayed_timestamp = delayed_flat_ts.squeeze(-1)

        return delayed_raw, delayed_noisy, delayed_timestamp

    def _apply_dropout_pair(
        self,
        raw: torch.Tensor,
        noisy: torch.Tensor,
        timestamp: torch.Tensor,
        external_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply dropout (missed detections) to a raw/noisy payload pair.

        One drop mask is sampled (or supplied via ``external_mask``) and applied
        to BOTH payloads plus the timestamp. When dropout occurs, the previous
        held raw, noisy, and timestamp are all retained together.

        **KEY**: One drop mask drives all three held buffers.

        Args:
            raw: Clean input tensor.
            noisy: Noisy input tensor (same shape as raw).
            timestamp: Capture timestamp tensor.
            external_mask: Optional external dropout mask of shape (num_envs,),
                dtype bool. When provided, replaces the internal i.i.d. sampler
                (used by burst dropout). True = dropped.
        """
        if not self._dropout_initialized:
            # First call - initialize held values with incoming pair
            # This ensures we never return zeros on first step
            self._dropout_held_raw = raw.clone()
            self._dropout_held_noisy = noisy.clone()
            self._dropout_held_timestamp = timestamp.clone()
            self._dropout_initialized = True
            return raw.clone(), noisy.clone(), timestamp.clone()

        # Use external mask (burst dropout) or internal i.i.d. sampler
        if external_mask is not None:
            drop_mask = external_mask
        else:
            drop_mask = self._dropout_sampler.sample_mask()

        # Shape handling
        drop_mask_data = drop_mask
        for _ in range(len(raw.shape) - 1):
            drop_mask_data = drop_mask_data.unsqueeze(-1)

        # On drop: keep previous held value. On no drop: refresh with new data.
        self._dropout_held_raw = torch.where(
            drop_mask_data, self._dropout_held_raw, raw
        )
        self._dropout_held_noisy = torch.where(
            drop_mask_data, self._dropout_held_noisy, noisy
        )
        self._dropout_held_timestamp = torch.where(
            drop_mask, self._dropout_held_timestamp, timestamp
        )

        return (
            self._dropout_held_raw.clone(),
            self._dropout_held_noisy.clone(),
            self._dropout_held_timestamp.clone(),
        )

    def _apply_first_order_lag(
        self,
        data: torch.Tensor,
        payload: Literal["raw", "noisy"],
    ) -> torch.Tensor:
        """Apply first-order lag filter for smoothing.

        Each payload has independent filter state (``_lag_output_raw`` /
        ``_lag_output_noisy``) so each signal smooths to itself. The filter
        coefficient ``alpha = dt / (tau + dt)`` is shared. Initialization is
        tracked per payload so raw and noisy each seed their filter from
        their own first input.

        Does NOT affect the timestamp — the timestamp still represents when
        the underlying data was captured.
        """
        init_attr = f"_lag_initialized_{payload}"
        state_attr = f"_lag_output_{payload}"

        if not getattr(self, init_attr):
            setattr(self, state_attr, data.clone())
            setattr(self, init_attr, True)
            return data.clone()

        # First-order lag: y = y_prev + dt/tau * (x - y_prev)
        alpha = self._dt / (self._lag_tau + self._dt)
        prev = getattr(self, state_attr)
        new = prev + alpha * (data - prev)
        setattr(self, state_attr, new)
        return new.clone()

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset pipeline state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            # Reset all
            env_ids = torch.arange(self._num_envs, device=self._device)

            # Reset latency buffers
            self._raw_data_buffer.reset()
            self._noisy_data_buffer.reset()
            self._timestamp_buffer.reset()
            self._buffer_initialized = False
            self._last_append_time.fill_(-1.0)

            # Reset staleness (both payloads + shared timestamp/state)
            self._staleness_held_raw.zero_()
            self._staleness_held_noisy.zero_()
            self._staleness_held_timestamp.zero_()
            self._last_detection_time.zero_()
            self._staleness_initialized = False

            # Reset dropout (both payloads + shared timestamp)
            self._dropout_held_raw.zero_()
            self._dropout_held_noisy.zero_()
            self._dropout_held_timestamp.zero_()
            self._dropout_initialized = False

            # Reset lag (independent init per payload)
            self._lag_output_raw.zero_()
            self._lag_output_noisy.zero_()
            self._lag_initialized_raw = False
            self._lag_initialized_noisy = False

            # Reset advance/query cache (4 data slots + 2 timestamp slots)
            self._last_advance_time.fill_(-1.0)
            self._cached_raw_no_dropout.zero_()
            self._cached_noisy_no_dropout.zero_()
            self._cached_raw_with_dropout.zero_()
            self._cached_noisy_with_dropout.zero_()
            self._cached_ts_no_dropout.zero_()
            self._cached_ts_with_dropout.zero_()

        else:
            # Reset specific environments
            self._raw_data_buffer.reset(env_ids)
            self._noisy_data_buffer.reset(env_ids)
            self._timestamp_buffer.reset(env_ids)
            self._last_append_time[env_ids] = -1.0

            # Reset staleness for specific envs
            self._staleness_held_raw[env_ids] = 0
            self._staleness_held_noisy[env_ids] = 0
            self._staleness_held_timestamp[env_ids] = 0
            self._last_detection_time[env_ids] = 0

            # Reset dropout for specific envs
            self._dropout_held_raw[env_ids] = 0
            self._dropout_held_noisy[env_ids] = 0
            self._dropout_held_timestamp[env_ids] = 0

            # Reset lag for specific envs (cannot partially mark initialized;
            # zero the state rows and let the next advance rebuild from input)
            self._lag_output_raw[env_ids] = 0
            self._lag_output_noisy[env_ids] = 0

            # Reset advance/query cache for specific envs
            self._last_advance_time[env_ids] = -1.0
            self._cached_raw_no_dropout[env_ids] = 0
            self._cached_noisy_no_dropout[env_ids] = 0
            self._cached_raw_with_dropout[env_ids] = 0
            self._cached_noisy_with_dropout[env_ids] = 0
            self._cached_ts_no_dropout[env_ids] = 0
            self._cached_ts_with_dropout[env_ids] = 0

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
