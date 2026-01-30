# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for the delay system."""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from isaaclab.utils import configclass


@configclass
class DelayCfg:
    """Configuration for a single delay buffer.

    This class defines the delay parameters for either actions or observations.
    The delay is specified in terms of simulation time steps (not decimated steps).

    Example usage:
        # Fixed 10-step delay
        delay_cfg = DelayCfg(enabled=True, min_delay=10, max_delay=10)

        # Randomized delay between 5 and 15 steps
        delay_cfg = DelayCfg(enabled=True, min_delay=5, max_delay=15, randomize_on_reset=True)
    """

    enabled: bool = False
    """Whether to enable delay for this buffer. Default is False."""

    min_delay: int = 0
    """Minimum delay in simulation steps. Default is 0 (no delay).

    This value must be non-negative and less than or equal to max_delay.
    """

    max_delay: int = 0
    """Maximum delay in simulation steps. Default is 0 (no delay).

    If max_delay > min_delay, delays are randomized per-environment within [min_delay, max_delay].
    This value determines the history buffer size.
    """

    randomize_on_reset: bool = True
    """Whether to randomize delays for each environment on reset. Default is True.

    If True, each environment gets a random delay uniformly sampled from [min_delay, max_delay].
    If False, all environments use max_delay.
    """

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_delay < 0:
            raise ValueError(f"min_delay must be non-negative, got {self.min_delay}")
        if self.max_delay < self.min_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= min_delay ({self.min_delay})"
            )


@configclass
class DelaySystemCfg:
    """Configuration for the complete delay system.

    This class holds configurations for both action and observation delays.
    Action and observation delays can be configured independently.

    Example usage:
        # Action delay only
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        )

        # Both action and observation delays
        cfg = DelaySystemCfg(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
            observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
        )
    """

    action_delay: DelayCfg = DelayCfg()
    """Configuration for action delay.

    Actions are delayed before being applied to the robot.
    This simulates actuator latency and control loop delays.
    """

    observation_delay: DelayCfg = DelayCfg()
    """Configuration for observation delay.

    Observations are delayed before being returned to the policy.
    This simulates sensor latency and communication delays.
    """


# ==============================================================================
# Enhanced Configuration Classes (V2)
# ==============================================================================


@configclass
class DistributionCfg:
    """Configuration for stochastic distributions.

    Supports three distribution types:
    - constant: Returns a fixed value
    - uniform: Samples uniformly from [mean - half_range, mean + half_range]
    - normal: Samples from a Gaussian with given mean and std

    Example usage:
        # Constant value
        dist = DistributionCfg(type="constant", value=100.0)

        # Uniform distribution (samples from [10.0, 50.0] with mean 30.0)
        dist = DistributionCfg(type="uniform", mean=30.0, half_range=20.0)

        # Normal distribution
        dist = DistributionCfg(type="normal", mean=30.0, std=5.0)
    """

    type: Literal["constant", "uniform", "normal"] = "constant"
    """Distribution type: 'constant', 'uniform', or 'normal'."""

    # For constant distribution
    value: float | None = None
    """Value for constant distribution."""

    # For uniform and normal distributions
    mean: float | None = None
    """Mean for uniform and normal distributions."""

    # For uniform distribution
    half_range: float | None = None
    """Half-range for uniform distribution. Samples from [mean - half_range, mean + half_range]."""

    # For normal distribution
    std: float | None = None
    """Standard deviation for normal distribution."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.type == "constant":
            if self.value is None:
                raise ValueError("Constant distribution requires 'value' to be set")
        elif self.type == "uniform":
            if self.mean is None or self.half_range is None:
                raise ValueError("Uniform distribution requires 'mean' and 'half_range' to be set")
            if self.half_range < 0:
                raise ValueError(f"half_range must be non-negative, got {self.half_range}")
        elif self.type == "normal":
            if self.mean is None or self.std is None:
                raise ValueError("Normal distribution requires 'mean' and 'std' to be set")
            if self.std < 0:
                raise ValueError(f"std must be non-negative, got {self.std}")


@configclass
class NoiseCfg:
    """Configuration for noise injection.

    Configures Gaussian noise to be added to sensor data.

    Example usage:
        # Enable noise with std=0.1
        noise = NoiseCfg(enabled=True, std=0.1)

        # Disable noise
        noise = NoiseCfg(enabled=False)
    """

    enabled: bool = False
    """Whether to enable noise injection. Default is False."""

    std: float = 0.0
    """Standard deviation of Gaussian noise. Default is 0.0."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.std < 0:
            raise ValueError(f"std must be non-negative, got {self.std}")


@configclass
class FieldDelayCfg:
    """Configuration for a single field's delay pipeline.

    This class configures the full delay pipeline for a sensor field, including:
    - First-order lag (dynamics delay/filtering)
    - Staleness (FPS throttling / sample-and-hold)
    - Random latency (transport delay)
    - Dropout (packet loss)
    - Noise injection

    The pipeline processes data in this order:
    1. First-order lag filtering
    2. Staleness (sample-and-hold at lower rate)
    3. Random latency (transport delay via history buffer)
    4. Dropout (randomly drop samples)

    Example usage:
        # IMU with fast dynamics and small noise
        imu_cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.005,
            noise=NoiseCfg(enabled=True, std=0.1),
        )

        # GPS with slower update rate and staleness
        gps_cfg = FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.1,
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="constant", value=10.0),
            noise=NoiseCfg(enabled=True, std=0.5),
        )

        # Camera with dropout and latency
        camera_cfg = FieldDelayCfg(
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="normal", mean=30.0, std=5.0),
            dropout_enabled=True,
            dropout_prob=0.1,
            latency_enabled=True,
            latency=DistributionCfg(type="normal", mean=0.05, std=0.01),
        )
    """

    # First-order lag (dynamics delay)
    first_order_lag_enabled: bool = False
    """Enable first-order lag filtering. Default is False."""

    time_constant: float = 0.0
    """Time constant (tau) for first-order lag in seconds. Default is 0.0.

    The filter implements: x_new = x_old + alpha * (x_meas - x_old)
    where alpha = dt / (dt + tau)
    """

    # Staleness (FPS throttling / sample-and-hold)
    staleness_enabled: bool = False
    """Enable staleness (sample-and-hold at lower rate). Default is False."""

    sample_rate: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(type="constant", value=100.0)
    )
    """Sample rate in Hz. Default is 100 Hz (constant).

    Data is held constant between samples, simulating sensors with lower update rates.
    """

    # Random latency (transport delay)
    latency_enabled: bool = False
    """Enable random transport latency. Default is False."""

    latency: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(type="constant", value=0.0)
    )
    """Transport latency in seconds. Default is 0.0 (constant).

    Simulates communication delay, processing delay, etc.
    """

    # Dropout
    dropout_enabled: bool = False
    """Enable dropout (packet loss). Default is False."""

    dropout_prob: float = 0.0
    """Probability of dropping a sample [0, 1]. Default is 0.0.

    When dropout occurs, the previous held data is kept.
    """

    # Noise
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    """Noise configuration. Default is disabled."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.time_constant < 0:
            raise ValueError(f"time_constant must be non-negative, got {self.time_constant}")
        if not 0 <= self.dropout_prob <= 1:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")


@configclass
class DelaySystemCfgV2:
    """Enhanced delay system configuration with data bus architecture.

    This configuration extends the original DelaySystemCfg with support for:
    - Per-field delay configurations
    - First-order lag, staleness, latency, and dropout
    - Noise injection
    - Clean/noisy data separation via central data bus

    The system can operate in two modes:
    - Legacy mode (use_enhanced_mode=False): Uses original step-based delays
    - Enhanced mode (use_enhanced_mode=True): Uses full delay pipeline with data bus

    Example usage:
        # Legacy mode (backward compatible)
        cfg = DelaySystemCfgV2(
            action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        )

        # Enhanced mode with per-field configuration
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
    """

    dt: float = 0.01
    """Simulation timestep in seconds. Default is 0.01 (100 Hz)."""

    # Legacy compatibility
    action_delay: DelayCfg = field(default_factory=DelayCfg)
    """Legacy action delay configuration (step-based)."""

    observation_delay: DelayCfg = field(default_factory=DelayCfg)
    """Legacy observation delay configuration (step-based)."""

    use_enhanced_mode: bool = False
    """Whether to use enhanced mode with data bus architecture. Default is False.

    If False, uses legacy step-based delays (backward compatible).
    If True, uses enhanced per-field delay pipelines.
    """

    # Per-field configuration
    field_configs: dict[str, FieldDelayCfg] = field(default_factory=dict)
    """Per-field delay configurations. Keys are field names, values are FieldDelayCfg."""

    # Default config for unconfigured fields
    default_field_config: FieldDelayCfg = field(default_factory=FieldDelayCfg)
    """Default configuration for fields not in field_configs."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
