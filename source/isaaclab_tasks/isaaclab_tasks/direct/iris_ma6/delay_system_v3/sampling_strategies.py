# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampling strategies for delay parameters.

This module provides samplers that control WHEN parameters are resampled:
- Per-step: Every simulation step
- Per-episode: At episode start
- Per-env-reset: Only for reset environments
"""

from __future__ import annotations

import torch
from typing import Literal

from .delay_cfg_v3 import DistributionCfg, SamplingCfg


class DistributionSampler:
    """Samples values from a configured distribution."""

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        num_envs: int,
        device: torch.device,
    ):
        """Initialize the distribution sampler.

        Args:
            distribution_cfg: Configuration specifying the distribution type and parameters.
            num_envs: Number of parallel environments.
            device: Torch device for tensor operations.
        """
        self._cfg = distribution_cfg
        self._num_envs = num_envs
        self._device = device

    def sample(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Sample values from the distribution.

        Args:
            env_ids: Optional tensor of environment indices to sample for.
                     If None, samples for all environments.

        Returns:
            Tensor of sampled values with shape (num_envs,) or (len(env_ids),).
        """
        if env_ids is None:
            n = self._num_envs
        else:
            n = len(env_ids)

        if self._cfg.type == "constant":
            values = torch.full((n,), self._cfg.value, device=self._device)
        elif self._cfg.type == "normal":
            values = torch.normal(
                mean=self._cfg.mean,
                std=self._cfg.std,
                size=(n,),
                device=self._device,
            )
        elif self._cfg.type == "uniform":
            low = self._cfg.mean - self._cfg.half_range
            high = self._cfg.mean + self._cfg.half_range
            values = torch.rand(n, device=self._device) * (high - low) + low
        else:
            raise ValueError(f"Unknown distribution type: {self._cfg.type}")

        # Apply clipping if configured
        if self._cfg.min_value is not None:
            values = torch.clamp(values, min=self._cfg.min_value)
        if self._cfg.max_value is not None:
            values = torch.clamp(values, max=self._cfg.max_value)

        return values

    def sample_scaled(
        self, scale: float, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample values scaled by a curriculum progress factor.

        Args:
            scale: Scale factor [0, 1] to multiply mean/value by.
            env_ids: Optional environment indices.

        Returns:
            Scaled sampled values.
        """
        if env_ids is None:
            n = self._num_envs
        else:
            n = len(env_ids)

        if self._cfg.type == "constant":
            values = torch.full((n,), self._cfg.value * scale, device=self._device)
        elif self._cfg.type == "normal":
            values = torch.normal(
                mean=self._cfg.mean * scale,
                std=self._cfg.std * scale,
                size=(n,),
                device=self._device,
            )
        elif self._cfg.type == "uniform":
            scaled_mean = self._cfg.mean * scale
            scaled_range = self._cfg.half_range * scale
            low = scaled_mean - scaled_range
            high = scaled_mean + scaled_range
            values = torch.rand(n, device=self._device) * (high - low) + low
        else:
            raise ValueError(f"Unknown distribution type: {self._cfg.type}")

        # Apply clipping
        if self._cfg.min_value is not None:
            values = torch.clamp(values, min=self._cfg.min_value)
        if self._cfg.max_value is not None:
            values = torch.clamp(values, max=self._cfg.max_value)

        return values


class ParameterSampler:
    """Manages sampling of a delay parameter with configurable frequency.

    Handles when to resample based on the configured frequency:
    - per_step: Resample every step
    - per_episode: Resample at episode boundaries
    - per_env_reset: Resample only for reset environments
    """

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        device: torch.device,
    ):
        """Initialize the parameter sampler.

        Args:
            distribution_cfg: Distribution configuration.
            sampling_cfg: Sampling frequency configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self._distribution = DistributionSampler(distribution_cfg, num_envs, device)
        self._frequency = sampling_cfg.frequency
        self._num_envs = num_envs
        self._device = device

        # Current values for each environment
        self._values = torch.zeros(num_envs, device=device)

        # Track initialization
        self._initialized = False

        # Curriculum scale factor
        self._scale = 1.0

    @property
    def values(self) -> torch.Tensor:
        """Current parameter values for all environments."""
        return self._values

    @property
    def frequency(self) -> Literal["per_step", "per_episode", "per_env_reset"]:
        """Sampling frequency."""
        return self._frequency

    def set_scale(self, scale: float):
        """Set curriculum scale factor.

        Args:
            scale: Scale factor [0, 1] for curriculum learning.
        """
        self._scale = max(0.0, min(1.0, scale))

    def initialize(self):
        """Initialize values for all environments."""
        self._values = self._distribution.sample_scaled(self._scale)
        self._initialized = True

    def maybe_resample(self, reset_env_ids: torch.Tensor | None = None):
        """Potentially resample values based on frequency.

        Args:
            reset_env_ids: Tensor of environment IDs that were reset.
                          Used for per_episode and per_env_reset modes.
        """
        if not self._initialized:
            self.initialize()
            return

        if self._frequency == "per_step":
            # Resample all environments every step
            self._values = self._distribution.sample_scaled(self._scale)

        elif self._frequency == "per_episode":
            # Resample all environments at episode boundary
            # This is called when ANY environment resets
            if reset_env_ids is not None and len(reset_env_ids) > 0:
                # Resample ALL envs (episode = global)
                self._values = self._distribution.sample_scaled(self._scale)

        elif self._frequency == "per_env_reset":
            # Resample only the reset environments
            if reset_env_ids is not None and len(reset_env_ids) > 0:
                new_values = self._distribution.sample_scaled(
                    self._scale, env_ids=reset_env_ids
                )
                self._values[reset_env_ids] = new_values

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset sampler for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self.initialize()
        else:
            self.maybe_resample(reset_env_ids=env_ids)


class LatencySampler(ParameterSampler):
    """Specialized sampler for latency with step conversion.

    Converts latency in seconds to delay steps, enforcing minimum steps.
    """

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        device: torch.device,
        dt: float,
        min_steps: int = 0,
    ):
        """Initialize latency sampler.

        Args:
            distribution_cfg: Distribution for latency (in seconds).
            sampling_cfg: Sampling frequency configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
            dt: Simulation time step.
            min_steps: Minimum delay steps. Default 0: time_lag=0 returns
                the most-recently appended data (verified safe with CircularBuffer).
                The previous default of 2 caused a 2×dt latency cliff at curriculum start.
        """
        super().__init__(distribution_cfg, sampling_cfg, num_envs, device)
        self._dt = dt
        self._min_steps = min_steps

        # Step values (integer)
        self._step_values = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def step_values(self) -> torch.Tensor:
        """Current delay in steps (integer) for all environments."""
        return self._step_values

    def _update_steps(self):
        """Convert latency seconds to steps with minimum enforcement."""
        # Convert to steps: round to nearest integer
        steps = torch.round(self._values / self._dt).long()
        # Enforce minimum
        self._step_values = torch.clamp(steps, min=self._min_steps)

    def initialize(self):
        """Initialize values for all environments."""
        super().initialize()
        self._update_steps()

    def maybe_resample(self, reset_env_ids: torch.Tensor | None = None):
        """Resample and update step values."""
        super().maybe_resample(reset_env_ids)
        self._update_steps()


class StalenessSampler(ParameterSampler):
    """Specialized sampler for staleness (FPS to period conversion).

    Converts FPS to detection period in seconds.
    """

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        device: torch.device,
    ):
        """Initialize staleness sampler.

        Args:
            distribution_cfg: Distribution for FPS values.
            sampling_cfg: Sampling frequency configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        super().__init__(distribution_cfg, sampling_cfg, num_envs, device)

        # Detection period in seconds
        self._period = torch.zeros(num_envs, device=device)

    @property
    def fps_values(self) -> torch.Tensor:
        """Current FPS values for all environments."""
        return self._values

    @property
    def period_values(self) -> torch.Tensor:
        """Current detection period (1/FPS) for all environments."""
        return self._period

    def _update_period(self):
        """Convert FPS to period."""
        # Avoid division by zero
        safe_fps = torch.clamp(self._values, min=1.0)
        self._period = 1.0 / safe_fps

    def initialize(self):
        """Initialize values for all environments."""
        super().initialize()
        self._update_period()

    def maybe_resample(self, reset_env_ids: torch.Tensor | None = None):
        """Resample and update period values."""
        super().maybe_resample(reset_env_ids)
        self._update_period()


class DropoutSampler:
    """Sampler for dropout events.

    Handles both per-step dropout mask generation and
    optional per-episode dropout rate variation.
    """

    def __init__(
        self,
        probability: float,
        rate_distribution: DistributionCfg | None,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        device: torch.device,
    ):
        """Initialize dropout sampler.

        Args:
            probability: Base dropout probability.
            rate_distribution: Optional distribution for per-episode rate variation.
            sampling_cfg: Sampling frequency for dropout rate.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self._base_probability = probability
        self._num_envs = num_envs
        self._device = device
        self._frequency = sampling_cfg.frequency

        # Per-environment dropout rate (may vary if rate_distribution is set)
        self._rates = torch.full((num_envs,), probability, device=device)

        # Optional rate sampler for per-episode variation
        self._rate_sampler: ParameterSampler | None = None
        if rate_distribution is not None:
            self._rate_sampler = ParameterSampler(
                rate_distribution, sampling_cfg, num_envs, device
            )

        # Current dropout mask
        self._mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def rates(self) -> torch.Tensor:
        """Current dropout rates for all environments."""
        return self._rates

    @property
    def mask(self) -> torch.Tensor:
        """Current dropout mask (True = dropped)."""
        return self._mask

    def set_rate(self, rate: float):
        """Set dropout rate for curriculum control.

        Args:
            rate: Dropout probability [0, 1].
        """
        self._base_probability = max(0.0, min(1.0, rate))
        if self._rate_sampler is None:
            self._rates.fill_(self._base_probability)

    def initialize(self):
        """Initialize dropout sampler."""
        if self._rate_sampler is not None:
            self._rate_sampler.initialize()
            self._rates = self._rate_sampler.values
        else:
            self._rates.fill_(self._base_probability)

    def sample_mask(self) -> torch.Tensor:
        """Sample a new dropout mask for this step.

        Returns:
            Boolean tensor of shape (num_envs,) where True = dropped.
        """
        self._mask = torch.rand(self._num_envs, device=self._device) < self._rates
        return self._mask

    def maybe_resample_rate(self, reset_env_ids: torch.Tensor | None = None):
        """Potentially resample dropout rates.

        Args:
            reset_env_ids: Environment IDs that were reset.
        """
        if self._rate_sampler is not None:
            self._rate_sampler.maybe_resample(reset_env_ids)
            self._rates = self._rate_sampler.values

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset dropout state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            self.initialize()
        else:
            self.maybe_resample_rate(reset_env_ids=env_ids)


# =============================================================================
# Per-Agent Samplers (extended versions with num_agents support)
# =============================================================================


class PerAgentDistributionSampler:
    """Samples values from a configured distribution with per-agent support.

    When num_agents > 1, samples independent values for each agent.
    """

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the per-agent distribution sampler.

        Args:
            distribution_cfg: Configuration specifying the distribution type and parameters.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device for tensor operations.
        """
        self._cfg = distribution_cfg
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device

    def sample(
        self, env_ids: torch.Tensor | None = None, agent_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample values from the distribution.

        Args:
            env_ids: Optional tensor of environment indices to sample for.
            agent_ids: Optional tensor of agent indices to sample for.

        Returns:
            Tensor of shape (num_envs, num_agents) or subset based on ids.
        """
        n_envs = len(env_ids) if env_ids is not None else self._num_envs
        n_agents = len(agent_ids) if agent_ids is not None else self._num_agents
        shape = (n_envs, n_agents)

        if self._cfg.type == "constant":
            values = torch.full(shape, self._cfg.value, device=self._device)
        elif self._cfg.type == "normal":
            values = torch.normal(
                mean=self._cfg.mean,
                std=self._cfg.std,
                size=shape,
                device=self._device,
            )
        elif self._cfg.type == "uniform":
            low = self._cfg.mean - self._cfg.half_range
            high = self._cfg.mean + self._cfg.half_range
            values = torch.rand(*shape, device=self._device) * (high - low) + low
        else:
            raise ValueError(f"Unknown distribution type: {self._cfg.type}")

        # Apply clipping if configured
        if self._cfg.min_value is not None:
            values = torch.clamp(values, min=self._cfg.min_value)
        if self._cfg.max_value is not None:
            values = torch.clamp(values, max=self._cfg.max_value)

        return values

    def sample_scaled(
        self,
        scale: float,
        env_ids: torch.Tensor | None = None,
        agent_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample values scaled by a curriculum progress factor.

        Args:
            scale: Scale factor [0, 1] to multiply mean/value by.
            env_ids: Optional environment indices.
            agent_ids: Optional agent indices.

        Returns:
            Scaled sampled values of shape (num_envs, num_agents) or subset.
        """
        n_envs = len(env_ids) if env_ids is not None else self._num_envs
        n_agents = len(agent_ids) if agent_ids is not None else self._num_agents
        shape = (n_envs, n_agents)

        if self._cfg.type == "constant":
            values = torch.full(shape, self._cfg.value * scale, device=self._device)
        elif self._cfg.type == "normal":
            values = torch.normal(
                mean=self._cfg.mean * scale,
                std=self._cfg.std * scale,
                size=shape,
                device=self._device,
            )
        elif self._cfg.type == "uniform":
            scaled_mean = self._cfg.mean * scale
            scaled_range = self._cfg.half_range * scale
            low = scaled_mean - scaled_range
            high = scaled_mean + scaled_range
            values = torch.rand(*shape, device=self._device) * (high - low) + low
        else:
            raise ValueError(f"Unknown distribution type: {self._cfg.type}")

        # Apply clipping
        if self._cfg.min_value is not None:
            values = torch.clamp(values, min=self._cfg.min_value)
        if self._cfg.max_value is not None:
            values = torch.clamp(values, max=self._cfg.max_value)

        return values


class PerAgentParameterSampler:
    """Manages sampling of a delay parameter with per-agent support.

    Each agent gets independently sampled parameters.
    """

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the per-agent parameter sampler.

        Args:
            distribution_cfg: Distribution configuration.
            sampling_cfg: Sampling frequency configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device.
        """
        self._distribution = PerAgentDistributionSampler(
            distribution_cfg, num_envs, num_agents, device
        )
        self._frequency = sampling_cfg.frequency
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device

        # Current values for each environment and agent: (num_envs, num_agents)
        self._values = torch.zeros(num_envs, num_agents, device=device)

        # Track initialization
        self._initialized = False

        # Curriculum scale factor
        self._scale = 1.0

    @property
    def values(self) -> torch.Tensor:
        """Current parameter values: (num_envs, num_agents)."""
        return self._values

    @property
    def frequency(self) -> Literal["per_step", "per_episode", "per_env_reset"]:
        """Sampling frequency."""
        return self._frequency

    def set_scale(self, scale: float):
        """Set curriculum scale factor.

        Args:
            scale: Scale factor [0, 1] for curriculum learning.
        """
        self._scale = max(0.0, min(1.0, scale))

    def initialize(self):
        """Initialize values for all environments and agents."""
        self._values = self._distribution.sample_scaled(self._scale)
        self._initialized = True

    def maybe_resample(self, reset_env_ids: torch.Tensor | None = None):
        """Potentially resample values based on frequency.

        Args:
            reset_env_ids: Tensor of environment IDs that were reset.
        """
        if not self._initialized:
            self.initialize()
            return

        if self._frequency == "per_step":
            # Resample all environments and agents every step
            self._values = self._distribution.sample_scaled(self._scale)

        elif self._frequency == "per_episode":
            # Resample all environments at episode boundary
            if reset_env_ids is not None and len(reset_env_ids) > 0:
                self._values = self._distribution.sample_scaled(self._scale)

        elif self._frequency == "per_env_reset":
            # Resample only the reset environments (all agents in those envs)
            if reset_env_ids is not None and len(reset_env_ids) > 0:
                new_values = self._distribution.sample_scaled(
                    self._scale, env_ids=reset_env_ids
                )
                self._values[reset_env_ids] = new_values

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset sampler for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self.initialize()
        else:
            self.maybe_resample(reset_env_ids=env_ids)


class PerAgentLatencySampler(PerAgentParameterSampler):
    """Specialized per-agent sampler for latency with step conversion."""

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        dt: float,
        min_steps: int = 2,
    ):
        """Initialize per-agent latency sampler.

        Args:
            distribution_cfg: Distribution for latency (in seconds).
            sampling_cfg: Sampling frequency configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device.
            dt: Simulation time step.
            min_steps: Minimum delay steps (for CircularBuffer warmup).
        """
        super().__init__(distribution_cfg, sampling_cfg, num_envs, num_agents, device)
        self._dt = dt
        self._min_steps = min_steps

        # Step values (integer): (num_envs, num_agents)
        self._step_values = torch.zeros(
            num_envs, num_agents, dtype=torch.long, device=device
        )

    @property
    def step_values(self) -> torch.Tensor:
        """Current delay in steps: (num_envs, num_agents)."""
        return self._step_values

    def _update_steps(self):
        """Convert latency seconds to steps with minimum enforcement."""
        steps = torch.round(self._values / self._dt).long()
        self._step_values = torch.clamp(steps, min=self._min_steps)

    def initialize(self):
        """Initialize values for all environments and agents."""
        super().initialize()
        self._update_steps()

    def maybe_resample(self, reset_env_ids: torch.Tensor | None = None):
        """Resample and update step values."""
        super().maybe_resample(reset_env_ids)
        self._update_steps()


class PerAgentStalenessSampler(PerAgentParameterSampler):
    """Specialized per-agent sampler for staleness (FPS to period)."""

    def __init__(
        self,
        distribution_cfg: DistributionCfg,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize per-agent staleness sampler.

        Args:
            distribution_cfg: Distribution for FPS values.
            sampling_cfg: Sampling frequency configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device.
        """
        super().__init__(distribution_cfg, sampling_cfg, num_envs, num_agents, device)

        # Detection period in seconds: (num_envs, num_agents)
        self._period = torch.zeros(num_envs, num_agents, device=device)

    @property
    def fps_values(self) -> torch.Tensor:
        """Current FPS values: (num_envs, num_agents)."""
        return self._values

    @property
    def period_values(self) -> torch.Tensor:
        """Current detection period: (num_envs, num_agents)."""
        return self._period

    def _update_period(self):
        """Convert FPS to period."""
        safe_fps = torch.clamp(self._values, min=1.0)
        self._period = 1.0 / safe_fps

    def initialize(self):
        """Initialize values for all environments and agents."""
        super().initialize()
        self._update_period()

    def maybe_resample(self, reset_env_ids: torch.Tensor | None = None):
        """Resample and update period values."""
        super().maybe_resample(reset_env_ids)
        self._update_period()


class PerAgentDropoutSampler:
    """Per-agent sampler for dropout events.

    Each agent gets independent dropout decisions.
    """

    def __init__(
        self,
        probability: float,
        rate_distribution: DistributionCfg | None,
        sampling_cfg: SamplingCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize per-agent dropout sampler.

        Args:
            probability: Base dropout probability.
            rate_distribution: Optional distribution for per-episode rate variation.
            sampling_cfg: Sampling frequency for dropout rate.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device.
        """
        self._base_probability = probability
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device
        self._frequency = sampling_cfg.frequency

        # Per-environment per-agent dropout rate: (num_envs, num_agents)
        self._rates = torch.full(
            (num_envs, num_agents), probability, device=device
        )

        # Optional rate sampler for per-episode variation
        self._rate_sampler: PerAgentParameterSampler | None = None
        if rate_distribution is not None:
            self._rate_sampler = PerAgentParameterSampler(
                rate_distribution, sampling_cfg, num_envs, num_agents, device
            )

        # Current dropout mask: (num_envs, num_agents)
        self._mask = torch.zeros(
            num_envs, num_agents, dtype=torch.bool, device=device
        )

    @property
    def rates(self) -> torch.Tensor:
        """Current dropout rates: (num_envs, num_agents)."""
        return self._rates

    @property
    def mask(self) -> torch.Tensor:
        """Current dropout mask (True = dropped): (num_envs, num_agents)."""
        return self._mask

    def set_rate(self, rate: float):
        """Set dropout rate for curriculum control.

        Args:
            rate: Dropout probability [0, 1].
        """
        self._base_probability = max(0.0, min(1.0, rate))
        if self._rate_sampler is None:
            self._rates.fill_(self._base_probability)

    def initialize(self):
        """Initialize dropout sampler."""
        if self._rate_sampler is not None:
            self._rate_sampler.initialize()
            self._rates = self._rate_sampler.values
        else:
            self._rates.fill_(self._base_probability)

    def sample_mask(self) -> torch.Tensor:
        """Sample a new dropout mask for this step.

        Returns:
            Boolean tensor of shape (num_envs, num_agents) where True = dropped.
        """
        self._mask = (
            torch.rand(self._num_envs, self._num_agents, device=self._device)
            < self._rates
        )
        return self._mask

    def maybe_resample_rate(self, reset_env_ids: torch.Tensor | None = None):
        """Potentially resample dropout rates.

        Args:
            reset_env_ids: Environment IDs that were reset.
        """
        if self._rate_sampler is not None:
            self._rate_sampler.maybe_resample(reset_env_ids)
            self._rates = self._rate_sampler.values

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset dropout state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            self.initialize()
        else:
            self.maybe_resample_rate(reset_env_ids=env_ids)
