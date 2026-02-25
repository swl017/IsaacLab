# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-field delay pipeline for processing sensor data through multiple delay stages."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from .delay_cfg import DistributionCfg, FieldDelayCfg


class DelayPipeline:
    """Processes a single field through a multi-stage delay pipeline.

    The pipeline applies the following stages in order:
    1. First-order lag filtering (dynamics delay)
    2. Staleness (sample-and-hold at lower rate)
    3. Random latency (transport delay via history buffer)
    4. Dropout (randomly drop samples, keep stale data)

    Each stage can be independently enabled/disabled via configuration.

    Example usage:
        cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.05,
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="constant", value=30.0),
            dropout_enabled=True,
            dropout_prob=0.1,
        )

        pipeline = DelayPipeline(
            cfg=cfg,
            num_envs=4096,
            field_dim=3,
            dt=0.01,
            device="cuda:0",
        )

        # Process data each step
        delayed_data = pipeline.process(raw_data, t_current, allow_dropout=True)
    """

    def __init__(
        self,
        cfg: FieldDelayCfg,
        num_envs: int,
        field_dim: int,
        dt: float,
        device: torch.device | str,
    ):
        """Initialize the delay pipeline.

        Args:
            cfg: Configuration for the delay pipeline.
            num_envs: Number of parallel environments.
            field_dim: Dimension of the field data (e.g., 3 for position).
            dt: Simulation timestep in seconds.
            device: Device to allocate tensors on.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.field_dim = field_dim
        self.dt = dt
        self.device = torch.device(device) if isinstance(device, str) else device

        # === First-order lag state ===
        self.filtered_state: torch.Tensor | None = None

        # === Staleness state ===
        self.held_data: torch.Tensor | None = None
        self.last_sample_time = torch.zeros(num_envs, device=self.device)
        self.next_sample_period: torch.Tensor | None = None
        if cfg.staleness_enabled:
            self.next_sample_period = self._sample_period()

        # === Latency state (using Isaac Lab's DelayBuffer) ===
        self._latency_buffer: DelayBuffer | None = None
        self._latency_delays: torch.Tensor | None = None
        if cfg.latency_enabled:
            # Estimate max delay steps needed
            max_latency = self._get_max_latency()
            max_delay_steps = max(1, int(max_latency / dt) + 1)
            self._latency_buffer = DelayBuffer(
                history_length=max_delay_steps,
                batch_size=num_envs,
                device=self.device,
            )
            # Sample initial latencies
            self._latency_delays = self._sample_latency()
            # Initialize buffer with zeros
            self._latency_buffer.compute(torch.zeros(num_envs, field_dim, device=self.device))

        # === Dropout state ===
        # Track previous data for dropout (keep stale data when dropout occurs)
        self.dropout_held_data: torch.Tensor | None = None

        # === Curriculum delay mode state ===
        self._delay_mode: str = "none"  # "none", "fixed", "random"
        self._delay_progress: float = 0.0  # Curriculum progress [0, 1]

        # Store original config latency parameters for curriculum scaling
        self._cfg_latency_mean: float = self._get_config_latency_mean()
        self._cfg_latency_std: float = self._get_config_latency_std()
        self._cfg_dropout_prob: float = cfg.dropout_prob

    def process(
        self,
        data: torch.Tensor,
        t_current: torch.Tensor,
        allow_dropout: bool = True,
    ) -> torch.Tensor:
        """Process data through the delay pipeline.

        Args:
            data: Input data tensor of shape (num_envs, field_dim).
            t_current: Current simulation time per environment of shape (num_envs,).
            allow_dropout: Whether to allow dropout. Set to False for clean path.

        Returns:
            Delayed data tensor of shape (num_envs, field_dim).
        """
        result = data

        # Stage 1: First-order lag (dynamics delay)
        if self.cfg.first_order_lag_enabled:
            result = self._apply_first_order_lag(result)

        # Stage 2: Staleness (sample-and-hold at lower rate)
        # Only apply in 'random' mode (Phase 4+). In 'fixed' mode (Phase 3),
        # data must be fresh every step for deterministic temporal alignment.
        if self.cfg.staleness_enabled and self._delay_mode == "random":
            result = self._apply_staleness(result, t_current)

        # Stage 3: Random latency (transport delay via history buffer)
        if self.cfg.latency_enabled:
            result = self._apply_latency(result)

        # Stage 4: Dropout (only if allowed)
        if self.cfg.dropout_enabled and allow_dropout:
            result = self._apply_dropout(result)
        else:
            # Update dropout held data even if dropout is disabled
            if self.dropout_held_data is None:
                self.dropout_held_data = result.clone()
            else:
                self.dropout_held_data = result.clone()

        return result

    def _apply_first_order_lag(self, data: torch.Tensor) -> torch.Tensor:
        """Apply exponential first-order lag filtering.

        Implements: x_new = x_old + alpha * (x_meas - x_old)
        where alpha = dt / (dt + tau)

        This filter is invariant to simulation rate (same tau produces same response).

        Args:
            data: Input data of shape (num_envs, field_dim).

        Returns:
            Filtered data of shape (num_envs, field_dim).
        """
        if self.filtered_state is None:
            # Initialize filter state with first data
            self.filtered_state = data.clone()
            return data

        tau = self.cfg.time_constant
        if tau <= 0:
            # No filtering if tau is zero
            return data

        alpha = self.dt / (self.dt + tau)
        self.filtered_state = self.filtered_state + alpha * (data - self.filtered_state)
        return self.filtered_state.clone()

    def _apply_staleness(self, data: torch.Tensor, t_current: torch.Tensor) -> torch.Tensor:
        """Apply sample-and-hold staleness.

        Data is only updated when the sample period has elapsed. Between updates,
        the held data remains constant (stale).

        Args:
            data: Input data of shape (num_envs, field_dim).
            t_current: Current simulation time of shape (num_envs,).

        Returns:
            Held data of shape (num_envs, field_dim).
        """
        if self.held_data is None:
            # Initialize with first data
            self.held_data = data.clone()
            self.last_sample_time = t_current.clone()
            return data

        # Check if sample period has elapsed for each environment
        time_since_sample = t_current - self.last_sample_time
        should_sample = time_since_sample >= self.next_sample_period

        if should_sample.any():
            # Update held data where sampling occurs
            self.held_data[should_sample] = data[should_sample]
            self.last_sample_time[should_sample] = t_current[should_sample]

            # Sample new periods for environments that just sampled
            new_periods = self._sample_period()
            self.next_sample_period[should_sample] = new_periods[should_sample]

        return self.held_data.clone()

    def _apply_latency(self, data: torch.Tensor) -> torch.Tensor:
        """Apply transport delay using history buffer.

        Args:
            data: Input data of shape (num_envs, field_dim).

        Returns:
            Delayed data of shape (num_envs, field_dim).
        """
        if self._latency_buffer is None:
            return data

        # Update delays periodically (convert latency seconds to steps)
        delay_steps = (self._latency_delays / self.dt).to(torch.int)
        delay_steps = torch.clamp(delay_steps, min=0, max=self._latency_buffer.history_length)
        self._latency_buffer.set_time_lag(delay_steps)

        # Compute delayed output
        return self._latency_buffer.compute(data)

    def _apply_dropout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply dropout (randomly keep stale data).

        Args:
            data: Input data of shape (num_envs, field_dim).

        Returns:
            Data with dropout applied, shape (num_envs, field_dim).
        """
        if self.dropout_held_data is None:
            self.dropout_held_data = data.clone()
            return data

        # Sample dropout mask
        dropout_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.dropout_prob

        # Where dropout occurs, keep previous held data
        result = data.clone()
        result[dropout_mask] = self.dropout_held_data[dropout_mask]

        # Update held data with non-dropped values
        self.dropout_held_data[~dropout_mask] = data[~dropout_mask]

        return result

    def _sample_period(self) -> torch.Tensor:
        """Sample period from sample_rate distribution.

        Returns:
            Period tensor of shape (num_envs,) in seconds.
        """
        sample_rate = self._sample_from_distribution(self.cfg.sample_rate)
        # Clamp to avoid division by zero and ensure period >= dt
        sample_rate = torch.clamp(sample_rate, min=1.0)
        period = 1.0 / sample_rate
        return torch.clamp(period, min=self.dt)

    def _sample_latency(self) -> torch.Tensor:
        """Sample latency from latency distribution.

        Returns:
            Latency tensor of shape (num_envs,) in seconds.
        """
        latency = self._sample_from_distribution(self.cfg.latency)
        return torch.clamp(latency, min=0.0)

    def _get_max_latency(self) -> float:
        """Get maximum possible latency for buffer sizing.

        Returns:
            Maximum latency in seconds.
        """
        dist = self.cfg.latency
        if dist.type == "constant":
            return dist.value or 0.0
        elif dist.type == "uniform":
            return (dist.mean or 0.0) + (dist.half_range or 0.0)
        elif dist.type == "normal":
            # Use mean + 3*std as max
            return (dist.mean or 0.0) + 3 * (dist.std or 0.0)
        return 0.0

    def _sample_from_distribution(self, dist: DistributionCfg) -> torch.Tensor:
        """Sample values from a distribution configuration.

        Args:
            dist: Distribution configuration.

        Returns:
            Sampled values tensor of shape (num_envs,).
        """
        if dist.type == "constant":
            return torch.full((self.num_envs,), dist.value, device=self.device)
        elif dist.type == "uniform":
            # Sample from [mean - half_range, mean + half_range]
            return dist.mean + (torch.rand(self.num_envs, device=self.device) * 2 - 1) * dist.half_range
        elif dist.type == "normal":
            return torch.randn(self.num_envs, device=self.device) * dist.std + dist.mean
        else:
            raise ValueError(f"Unknown distribution type: {dist.type}")

    def reset(self, env_ids: torch.Tensor | None = None, initial_data: torch.Tensor | None = None):
        """Reset pipeline state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
            initial_data: Initial data for filter state. If None, resets to zeros.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset first-order lag state
        if self.filtered_state is not None:
            if initial_data is not None:
                self.filtered_state[env_ids] = initial_data
            else:
                self.filtered_state[env_ids] = 0.0

        # Reset staleness state
        if self.held_data is not None:
            if initial_data is not None:
                self.held_data[env_ids] = initial_data
            else:
                self.held_data[env_ids] = 0.0

        self.last_sample_time[env_ids] = 0.0
        if self.next_sample_period is not None:
            new_periods = self._sample_period()
            self.next_sample_period[env_ids] = new_periods[env_ids]

        # Reset latency buffer with curriculum-aware delay mode
        if self._latency_buffer is not None:
            self._latency_buffer.reset(env_ids.tolist())

            # Handle latency based on curriculum delay mode
            if self._delay_mode == "none":
                # No delay
                self._latency_delays[env_ids] = 0.0
            elif self._delay_mode == "fixed":
                # Fixed: all envs get same delay (mean * progress)
                effective_mean = self._get_effective_latency_mean()
                self._latency_delays[env_ids] = effective_mean
            else:  # random
                # Random: sample from N(mean * progress, std * progress)
                effective_mean = self._get_effective_latency_mean()
                effective_std = self._get_effective_latency_std()
                if effective_std > 0:
                    # Sample and ensure non-negative
                    sampled = torch.randn(len(env_ids), device=self.device) * effective_std + effective_mean
                    self._latency_delays[env_ids] = torch.clamp(sampled, min=0.0)
                else:
                    self._latency_delays[env_ids] = effective_mean

        # Reset dropout state
        if self.dropout_held_data is not None:
            if initial_data is not None:
                self.dropout_held_data[env_ids] = initial_data
            else:
                self.dropout_held_data[env_ids] = 0.0

    def update_time_constant(self, time_constant: float):
        """Update the first-order lag time constant.

        Useful for curriculum learning.

        Args:
            time_constant: New time constant in seconds.
        """
        self.cfg.time_constant = time_constant

    def update_dropout_prob(self, dropout_prob: float):
        """Update the dropout probability.

        Useful for curriculum learning.

        Args:
            dropout_prob: New dropout probability in [0, 1].
        """
        if not 0 <= dropout_prob <= 1:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
        self.cfg.dropout_prob = dropout_prob

    # ==========================================================================
    # Curriculum Delay Mode Control
    # ==========================================================================

    def _get_config_latency_mean(self) -> float:
        """Extract latency mean from config distribution."""
        if not self.cfg.latency_enabled:
            return 0.0
        dist = self.cfg.latency
        if dist.type == "constant":
            return dist.value or 0.0
        elif dist.type in ("uniform", "normal"):
            return dist.mean or 0.0
        return 0.0

    def _get_config_latency_std(self) -> float:
        """Extract latency std from config distribution."""
        if not self.cfg.latency_enabled:
            return 0.0
        dist = self.cfg.latency
        if dist.type == "constant":
            return 0.0
        elif dist.type == "uniform":
            # For uniform [mean - half_range, mean + half_range], std = half_range / sqrt(3)
            return (dist.half_range or 0.0) / 1.732
        elif dist.type == "normal":
            return dist.std or 0.0
        return 0.0

    def set_delay_mode(self, mode: str, progress: float = 1.0):
        """Set delay mode for curriculum learning.

        Args:
            mode: Delay mode - 'none', 'fixed', or 'random'
                - 'none': No latency delay applied
                - 'fixed': Fixed deterministic delay (all envs get same delay)
                - 'random': Random delay sampled per environment
            progress: Curriculum progress [0, 1] for ramping delay magnitude/variance
                - In 'fixed' mode: delay = mean * progress
                - In 'random' mode: delay ~ N(mean, std * progress)
        """
        if mode not in ("none", "fixed", "random"):
            raise ValueError(f"Unknown delay mode: {mode}. Expected 'none', 'fixed', or 'random'")
        self._delay_mode = mode
        self._delay_progress = max(0.0, min(1.0, progress))

    def set_dropout_rate(self, dropout_rate: float):
        """Set dropout rate for curriculum control.

        Args:
            dropout_rate: Dropout probability [0, 1].
        """
        self.cfg.dropout_prob = max(0.0, min(1.0, dropout_rate))

    def _get_effective_latency_mean(self) -> float:
        """Get effective latency mean based on mode and progress."""
        if self._delay_mode == "none":
            return 0.0
        # Ramp from 0 to config mean based on progress
        return self._cfg_latency_mean * self._delay_progress

    def _get_effective_latency_std(self) -> float:
        """Get effective latency std based on mode and progress."""
        if self._delay_mode in ("none", "fixed"):
            return 0.0  # No variance in fixed mode
        # Ramp from 0 to config std based on progress
        return self._cfg_latency_std * self._delay_progress
