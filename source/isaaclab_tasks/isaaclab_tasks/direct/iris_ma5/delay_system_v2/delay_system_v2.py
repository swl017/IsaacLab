# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enhanced delay system with central data bus architecture."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.buffers import DelayBuffer

from .data_bus import DataBus
from .delay_pipeline import DelayPipeline
from .derived_fields import DerivedFieldComputer, DerivedFieldDef

if TYPE_CHECKING:
    from .delay_cfg import DelayCfg, DelaySystemCfgV2, FieldDelayCfg


class DelaySystemV2:
    """Enhanced delay system with central data bus architecture.

    This class implements the architecture described in REQUIREMENTS.md:
    1. Central data bus storing clean (rewards) and noisy (observations) data
    2. Per-field delay pipelines with first-order lag, staleness, latency, dropout
    3. Derived field computation from delayed raw fields

    The system supports two modes:
    - Legacy mode (use_enhanced_mode=False): Uses original step-based DelayBuffer
    - Enhanced mode (use_enhanced_mode=True): Uses full data bus architecture

    Data Flow (Enhanced Mode):
        Raw Data → DataBus (clean/noisy) → DelayPipeline → DerivedFieldComputer → Output

    Example usage:
        cfg = DelaySystemCfgV2(
            dt=0.01,
            use_enhanced_mode=True,
            field_configs={
                "imu_accel": FieldDelayCfg(
                    first_order_lag_enabled=True,
                    time_constant=0.005,
                    noise=NoiseCfg(enabled=True, std=0.1),
                ),
                "gps_position": FieldDelayCfg(
                    staleness_enabled=True,
                    sample_rate=DistributionCfg(type="constant", value=10.0),
                ),
            },
        )

        system = DelaySystemV2(
            cfg=cfg,
            num_envs=4096,
            device="cuda:0",
            field_dims={"imu_accel": 3, "gps_position": 3},
        )

        # Store sensor data
        system.store("imu_accel", raw_accel)
        system.store("gps_position", raw_gps)

        # Get delayed data for observations
        obs_accel = system.get_delayed_noisy("imu_accel")

        # Get delayed data for rewards (clean, no dropout)
        reward_pos = system.get_delayed_clean("gps_position")

        system.step()
    """

    def __init__(
        self,
        cfg: DelaySystemCfgV2,
        num_envs: int,
        device: str,
        field_dims: dict[str, int] | None = None,
        derived_fields: list[DerivedFieldDef] | None = None,
        # Legacy API compatibility
        action_dim: int | None = None,
        observation_dim: int | None = None,
    ):
        """Initialize the delay system.

        Args:
            cfg: Delay system configuration.
            num_envs: Number of parallel environments.
            device: Device to allocate tensors on.
            field_dims: Mapping of field names to dimensions. Required for enhanced mode.
            derived_fields: List of derived field definitions. Optional.
            action_dim: Action dimension (legacy API compatibility).
            observation_dim: Observation dimension (legacy API compatibility).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device

        # Legacy mode buffers
        self._action_buffer: DelayBuffer | None = None
        self._observation_buffer: DelayBuffer | None = None
        self._action_dim = action_dim
        self._observation_dim = observation_dim

        # Enhanced mode components
        self.data_bus: DataBus | None = None
        self._clean_pipelines: dict[str, DelayPipeline] = {}
        self._noisy_pipelines: dict[str, DelayPipeline] = {}
        self._derived_computer: DerivedFieldComputer | None = None

        if cfg.use_enhanced_mode:
            self._init_enhanced_mode(field_dims, derived_fields)
        else:
            self._init_legacy_mode(action_dim, observation_dim)

    def _init_legacy_mode(self, action_dim: int | None, observation_dim: int | None):
        """Initialize legacy step-based delay buffers.

        Args:
            action_dim: Action dimension.
            observation_dim: Observation dimension.
        """
        # Initialize action delay buffer
        if self.cfg.action_delay.enabled:
            history_length = max(self.cfg.action_delay.max_delay, 1)
            self._action_buffer = DelayBuffer(
                history_length=history_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            if action_dim is not None:
                # Initialize with zeros
                self._action_buffer.compute(
                    torch.zeros(self.num_envs, action_dim, device=self.device)
                )

        # Initialize observation delay buffer
        if self.cfg.observation_delay.enabled:
            history_length = max(self.cfg.observation_delay.max_delay, 1)
            self._observation_buffer = DelayBuffer(
                history_length=history_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            if observation_dim is not None:
                # Initialize with zeros
                self._observation_buffer.compute(
                    torch.zeros(self.num_envs, observation_dim, device=self.device)
                )

    def _init_enhanced_mode(
        self,
        field_dims: dict[str, int] | None,
        derived_fields: list[DerivedFieldDef] | None,
    ):
        """Initialize enhanced mode with data bus architecture.

        Args:
            field_dims: Mapping of field names to dimensions.
            derived_fields: List of derived field definitions.
        """
        # Central data bus
        self.data_bus = DataBus(self.num_envs, self.device)

        # Derived field computer
        self._derived_computer = DerivedFieldComputer(derived_fields)

        # Initialize pipelines for each configured field
        if field_dims:
            for field_name, dim in field_dims.items():
                field_cfg = self.cfg.field_configs.get(
                    field_name, self.cfg.default_field_config
                )

                # Create clean path pipeline (no dropout)
                self._clean_pipelines[field_name] = DelayPipeline(
                    cfg=field_cfg,
                    num_envs=self.num_envs,
                    field_dim=dim,
                    dt=self.cfg.dt,
                    device=self.device,
                )

                # Create noisy path pipeline (with dropout)
                self._noisy_pipelines[field_name] = DelayPipeline(
                    cfg=field_cfg,
                    num_envs=self.num_envs,
                    field_dim=dim,
                    dt=self.cfg.dt,
                    device=self.device,
                )

    # ==========================================================================
    # Core Enhanced API
    # ==========================================================================

    def store(self, field_name: str, data: torch.Tensor):
        """Store raw sensor data to the data bus.

        In enhanced mode, data is stored to both clean and noisy buffers.
        Noise is injected into the noisy buffer based on field configuration.

        Args:
            field_name: Name of the field.
            data: Raw sensor data tensor of shape (num_envs, ...).

        Raises:
            RuntimeError: If not in enhanced mode.
        """
        if not self.cfg.use_enhanced_mode:
            raise RuntimeError("store() is only available in enhanced mode")

        field_cfg = self.cfg.field_configs.get(field_name, self.cfg.default_field_config)
        noise_std = field_cfg.noise.std if field_cfg.noise.enabled else 0.0

        self.data_bus.store(field_name, data, noise_std)

    def get_delayed_clean(self, field_name: str) -> torch.Tensor:
        """Get delayed clean data for reward computation.

        Clean data has delays applied but no dropout (for unbiased rewards).

        Args:
            field_name: Name of the field.

        Returns:
            Delayed clean data tensor.

        Raises:
            RuntimeError: If not in enhanced mode.
            KeyError: If field not found.
        """
        if not self.cfg.use_enhanced_mode:
            raise RuntimeError("get_delayed_clean() is only available in enhanced mode")

        raw = self.data_bus.get_clean(field_name)

        pipeline = self._clean_pipelines.get(field_name)
        if pipeline:
            return pipeline.process(raw, self.data_bus.t_current, allow_dropout=False)

        return raw

    def get_delayed_noisy(self, field_name: str) -> torch.Tensor:
        """Get delayed noisy data for observation computation.

        Noisy data has noise injected and all delays applied (including dropout).

        Args:
            field_name: Name of the field.

        Returns:
            Delayed noisy data tensor.

        Raises:
            RuntimeError: If not in enhanced mode.
            KeyError: If field not found.
        """
        if not self.cfg.use_enhanced_mode:
            raise RuntimeError("get_delayed_noisy() is only available in enhanced mode")

        noisy = self.data_bus.get_noisy(field_name)

        pipeline = self._noisy_pipelines.get(field_name)
        if pipeline:
            return pipeline.process(noisy, self.data_bus.t_current, allow_dropout=True)

        return noisy

    def compute_derived_fields(self, use_clean: bool = False) -> dict[str, torch.Tensor]:
        """Compute derived fields from delayed raw fields.

        Derived fields are computed FROM delayed raw fields, ensuring consistency.

        Args:
            use_clean: If True, use clean (reward) path. If False, use noisy path.

        Returns:
            Dictionary containing all fields (raw + derived).

        Raises:
            RuntimeError: If not in enhanced mode.
        """
        if not self.cfg.use_enhanced_mode:
            raise RuntimeError("compute_derived_fields() is only available in enhanced mode")

        # Gather delayed raw fields
        delayed_fields = {}
        for field_name in self.data_bus.get_field_names():
            if use_clean:
                delayed_fields[field_name] = self.get_delayed_clean(field_name)
            else:
                delayed_fields[field_name] = self.get_delayed_noisy(field_name)

        # Compute derived fields
        if self._derived_computer:
            return self._derived_computer.compute_all(delayed_fields)

        return delayed_fields

    def step(self, dt: float | None = None):
        """Advance simulation time.

        Args:
            dt: Time step in seconds. If None, uses cfg.dt.
        """
        if self.cfg.use_enhanced_mode and self.data_bus:
            self.data_bus.step(dt or self.cfg.dt)

    # ==========================================================================
    # Legacy Backward-Compatible API
    # ==========================================================================

    def compute_delayed_action(self, action: torch.Tensor) -> torch.Tensor:
        """Compute delayed action (legacy API).

        Args:
            action: Current action tensor of shape (num_envs, action_dim).

        Returns:
            Delayed action tensor.
        """
        if self.cfg.use_enhanced_mode:
            # Use enhanced mode via data bus
            self.store("action", action)
            return self.get_delayed_noisy("action")
        else:
            # Legacy mode
            if self._action_buffer is None:
                return action
            return self._action_buffer.compute(action)

    def compute_delayed_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute delayed observation (legacy API).

        Args:
            observation: Current observation tensor of shape (num_envs, obs_dim).

        Returns:
            Delayed observation tensor.
        """
        if self.cfg.use_enhanced_mode:
            # Use enhanced mode via data bus
            self.store("observation", observation)
            return self.get_delayed_noisy("observation")
        else:
            # Legacy mode
            if self._observation_buffer is None:
                return observation
            return self._observation_buffer.compute(observation)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset delay system for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            batch_ids = None
        else:
            batch_ids = env_ids.tolist()

        if self.cfg.use_enhanced_mode:
            # Reset data bus
            if self.data_bus:
                self.data_bus.reset(env_ids)

            # Reset pipelines
            for pipeline in self._clean_pipelines.values():
                pipeline.reset(env_ids)
            for pipeline in self._noisy_pipelines.values():
                pipeline.reset(env_ids)
        else:
            # Legacy mode
            if self._action_buffer is not None:
                self._action_buffer.reset(batch_ids)
                self._randomize_delays(
                    self._action_buffer, self.cfg.action_delay, batch_ids
                )

            if self._observation_buffer is not None:
                self._observation_buffer.reset(batch_ids)
                self._randomize_delays(
                    self._observation_buffer, self.cfg.observation_delay, batch_ids
                )

    def _randomize_delays(
        self,
        buffer: DelayBuffer,
        delay_cfg: DelayCfg,
        batch_ids: list[int] | None = None,
    ):
        """Randomize delays for legacy mode buffer.

        Args:
            buffer: The delay buffer to configure.
            delay_cfg: The delay configuration.
            batch_ids: Batch indices to randomize. If None, randomizes all.
        """
        if delay_cfg.randomize_on_reset and delay_cfg.min_delay < delay_cfg.max_delay:
            if batch_ids is None:
                num_to_sample = self.num_envs
            else:
                num_to_sample = len(batch_ids)

            random_delays = torch.randint(
                low=delay_cfg.min_delay,
                high=delay_cfg.max_delay + 1,
                size=(num_to_sample,),
                device=self.device,
                dtype=torch.int,
            )
            buffer.set_time_lag(random_delays, batch_ids)
        else:
            buffer.set_time_lag(delay_cfg.max_delay, batch_ids)

    def get_delay_info(self) -> dict:
        """Get current delay information for logging (legacy API).

        Returns:
            Dictionary containing delay statistics.
        """
        info = {}

        if self.cfg.use_enhanced_mode:
            # Enhanced mode info
            info["mode"] = "enhanced"
            info["num_fields"] = len(self._clean_pipelines)
        else:
            # Legacy mode info
            info["mode"] = "legacy"
            if self._action_buffer is not None:
                info["action_delay_min"] = self._action_buffer.min_time_lag
                info["action_delay_max"] = self._action_buffer.max_time_lag
                info["action_delay_mean"] = (
                    self._action_buffer.time_lags.float().mean().item()
                )
            if self._observation_buffer is not None:
                info["observation_delay_min"] = self._observation_buffer.min_time_lag
                info["observation_delay_max"] = self._observation_buffer.max_time_lag
                info["observation_delay_mean"] = (
                    self._observation_buffer.time_lags.float().mean().item()
                )

        return info

    # ==========================================================================
    # Legacy Properties
    # ==========================================================================

    @property
    def action_delay_enabled(self) -> bool:
        """Whether action delay is enabled (legacy API)."""
        return self._action_buffer is not None or self.cfg.use_enhanced_mode

    @property
    def observation_delay_enabled(self) -> bool:
        """Whether observation delay is enabled (legacy API)."""
        return self._observation_buffer is not None or self.cfg.use_enhanced_mode

    # ==========================================================================
    # Curriculum Learning Hooks
    # ==========================================================================

    def set_field_time_constant(self, field_name: str, time_constant: float):
        """Update the first-order lag time constant for a field.

        Args:
            field_name: Name of the field.
            time_constant: New time constant in seconds.
        """
        if field_name in self._clean_pipelines:
            self._clean_pipelines[field_name].update_time_constant(time_constant)
        if field_name in self._noisy_pipelines:
            self._noisy_pipelines[field_name].update_time_constant(time_constant)

    def set_field_dropout_rate(self, field_name: str, dropout_prob: float):
        """Update the dropout probability for a field.

        Args:
            field_name: Name of the field.
            dropout_prob: New dropout probability in [0, 1].
        """
        # Only update noisy pipeline (clean path doesn't have dropout)
        if field_name in self._noisy_pipelines:
            self._noisy_pipelines[field_name].update_dropout_prob(dropout_prob)

    def set_all_time_constants(self, time_constant: float):
        """Update time constant for all fields.

        Args:
            time_constant: New time constant in seconds.
        """
        for pipeline in self._clean_pipelines.values():
            pipeline.update_time_constant(time_constant)
        for pipeline in self._noisy_pipelines.values():
            pipeline.update_time_constant(time_constant)

    def set_all_dropout_rates(self, dropout_prob: float):
        """Update dropout probability for all fields.

        Args:
            dropout_prob: New dropout probability in [0, 1].
        """
        for pipeline in self._noisy_pipelines.values():
            pipeline.update_dropout_prob(dropout_prob)

    def set_all_delay_modes(self, mode: str, progress: float = 1.0):
        """Set delay mode for all fields (curriculum learning).

        Args:
            mode: Delay mode - 'none', 'fixed', or 'random'
                - 'none': No latency delay applied
                - 'fixed': Fixed deterministic delay (all envs get same delay)
                - 'random': Random delay sampled per environment
            progress: Curriculum progress [0, 1] for ramping delay magnitude/variance
        """
        for pipeline in self._clean_pipelines.values():
            pipeline.set_delay_mode(mode, progress)
        for pipeline in self._noisy_pipelines.values():
            pipeline.set_delay_mode(mode, progress)

    def set_field_delay_mode(self, field_name: str, mode: str, progress: float = 1.0):
        """Set delay mode for a specific field.

        Args:
            field_name: Name of the field.
            mode: Delay mode - 'none', 'fixed', or 'random'
            progress: Curriculum progress [0, 1] for ramping
        """
        if field_name in self._clean_pipelines:
            self._clean_pipelines[field_name].set_delay_mode(mode, progress)
        if field_name in self._noisy_pipelines:
            self._noisy_pipelines[field_name].set_delay_mode(mode, progress)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def register_field(self, field_name: str, field_dim: int, field_cfg: FieldDelayCfg | None = None):
        """Dynamically register a new field (enhanced mode only).

        Args:
            field_name: Name of the new field.
            field_dim: Dimension of the field.
            field_cfg: Configuration for the field. Uses default if None.

        Raises:
            RuntimeError: If not in enhanced mode.
        """
        if not self.cfg.use_enhanced_mode:
            raise RuntimeError("register_field() is only available in enhanced mode")

        if field_cfg is None:
            field_cfg = self.cfg.default_field_config

        # Create pipelines
        self._clean_pipelines[field_name] = DelayPipeline(
            cfg=field_cfg,
            num_envs=self.num_envs,
            field_dim=field_dim,
            dt=self.cfg.dt,
            device=self.device,
        )
        self._noisy_pipelines[field_name] = DelayPipeline(
            cfg=field_cfg,
            num_envs=self.num_envs,
            field_dim=field_dim,
            dt=self.cfg.dt,
            device=self.device,
        )

    def register_derived_field(self, field_def: DerivedFieldDef):
        """Register a derived field definition.

        Args:
            field_def: The derived field definition.

        Raises:
            RuntimeError: If not in enhanced mode.
        """
        if not self.cfg.use_enhanced_mode:
            raise RuntimeError("register_derived_field() is only available in enhanced mode")

        if self._derived_computer:
            self._derived_computer.register(field_def)
