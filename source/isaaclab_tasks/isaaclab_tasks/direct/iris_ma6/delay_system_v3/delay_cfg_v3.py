# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for delay system V3.

This module provides a flexible configuration system for the delay pipeline
with support for:
- Per-step, per-episode, per-env-reset sampling frequencies
- Curriculum-aware delay modes (none, fixed, random)
- Separate ego/other perspective configurations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DistributionCfg:
    """Configuration for a probability distribution.

    Supports constant values, normal distributions, and uniform distributions.
    """

    type: Literal["constant", "normal", "uniform"] = "constant"
    """Type of distribution: 'constant', 'normal', or 'uniform'."""

    value: float = 0.0
    """Value for constant distribution."""

    mean: float = 0.0
    """Mean for normal/uniform distributions."""

    std: float = 0.0
    """Standard deviation for normal distribution."""

    half_range: float = 0.0
    """Half-range for uniform distribution: samples from [mean - half_range, mean + half_range]."""

    min_value: float | None = None
    """Optional minimum value (clipping)."""

    max_value: float | None = None
    """Optional maximum value (clipping)."""


@dataclass
class SamplingCfg:
    """Configuration for when a parameter is sampled.

    Controls the frequency at which delay parameters are resampled.
    """

    frequency: Literal["per_step", "per_episode", "per_env_reset"] = "per_episode"
    """When to sample the parameter:
    - 'per_step': Resample every simulation step (creates jitter)
    - 'per_episode': Resample at the start of each episode
    - 'per_env_reset': Resample only for environments that are reset
    """


@dataclass
class LatencyCfg:
    """Configuration for communication/processing latency.

    Latency represents the time delay between when data is captured
    and when it becomes available to the agent.
    """

    enabled: bool = True
    """Whether latency is applied."""

    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    """When to resample latency values."""

    distribution: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="normal", mean=0.1, std=0.08, min_value=0.0
        )
    )
    """Distribution for latency values (in seconds)."""

    min_steps: int = 2
    """Minimum delay in simulation steps.

    This handles CircularBuffer warmup - the buffer needs at least 2 steps
    to return meaningful delayed data.
    """


@dataclass
class StalenessCfg:
    """Configuration for detection staleness (FPS limiting).

    Staleness simulates limited sensor update rates (e.g., camera FPS).
    When enabled, detections are only updated at the configured FPS,
    causing data to become "stale" between updates.
    """

    enabled: bool = True
    """Whether staleness is applied."""

    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    """When to resample FPS values."""

    fps_distribution: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="uniform", mean=25.0, half_range=5.0, min_value=5.0, max_value=60.0
        )
    )
    """Distribution for detection FPS. Higher FPS = less staleness."""


@dataclass
class DropoutCfg:
    """Configuration for detection dropout.

    Dropout simulates missed detections or communication failures.
    When a dropout occurs, the previous data is retained.
    """

    enabled: bool = True
    """Whether dropout is applied."""

    sampling: SamplingCfg = field(
        default_factory=lambda: SamplingCfg(frequency="per_step")
    )
    """When to sample dropout events. Default is per-step for random drops."""

    probability: float = 0.05
    """Base dropout probability [0, 1]."""

    rate_distribution: DistributionCfg | None = None
    """Optional: Distribution for per-episode dropout rate variation.

    If set, the dropout rate is sampled from this distribution per episode,
    allowing some episodes to have higher/lower dropout rates.
    """


@dataclass
class NoiseCfg:
    """Configuration for observation noise.

    Noise is added to observations to simulate sensor inaccuracies.
    """

    enabled: bool = True
    """Whether noise is applied."""

    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    """When to resample noise standard deviation."""

    position_std: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="constant", value=0.1, min_value=0.0
        )
    )
    """Standard deviation for position noise (meters)."""

    velocity_std: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="constant", value=0.05, min_value=0.0
        )
    )
    """Standard deviation for velocity noise (m/s)."""

    orientation_std: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="constant", value=0.01, min_value=0.0
        )
    )
    """Standard deviation for orientation noise (radians)."""


@dataclass
class FirstOrderLagCfg:
    """Configuration for first-order lag filter.

    Smooths sudden changes in observations, simulating
    sensor filtering or physical response delays.
    """

    enabled: bool = False
    """Whether first-order lag is applied."""

    tau: float = 0.0
    """Time constant in seconds. Larger = more smoothing. 0 = disabled."""


@dataclass
class DelayPipelineCfgV3:
    """Configuration for a single delay pipeline.

    A pipeline processes data through the following stages:
    1. Staleness (FPS limiting)
    2. Latency (communication delay)
    3. Dropout (missed detections)
    4. First-order lag (optional smoothing)
    """

    latency: LatencyCfg = field(default_factory=LatencyCfg)
    """Latency configuration."""

    staleness: StalenessCfg = field(default_factory=StalenessCfg)
    """Staleness (FPS limiting) configuration."""

    dropout: DropoutCfg = field(default_factory=DropoutCfg)
    """Dropout configuration."""

    first_order_lag: FirstOrderLagCfg = field(default_factory=FirstOrderLagCfg)
    """First-order lag filter configuration."""


@dataclass
class PerspectiveCfg:
    """Configuration for a perspective (ego or other agent).

    Ego perspective: Agent's view of itself (typically lower latency)
    Other perspective: Agent's view of other agents (typically higher latency)
    """

    pipeline: DelayPipelineCfgV3 = field(default_factory=DelayPipelineCfgV3)
    """Delay pipeline configuration for this perspective."""

    dropout_for_observations: bool = True
    """Whether dropout applies to observations (noisy path)."""

    dropout_for_rewards: bool = False
    """Whether dropout applies to reward computation (clean path)."""


@dataclass
class UnifiedDelayCfgV3:
    """Configuration for the unified delay system.

    The unified system handles both ego and other perspectives,
    as well as clean (reward) and noisy (observation) paths.
    """

    dt: float = 0.04
    """Simulation time step in seconds."""

    ego: PerspectiveCfg = field(
        default_factory=lambda: PerspectiveCfg(
            pipeline=DelayPipelineCfgV3(
                latency=LatencyCfg(
                    distribution=DistributionCfg(
                        type="normal", mean=0.05, std=0.02, min_value=0.0
                    )
                )
            )
        )
    )
    """Configuration for ego perspective (agent viewing itself)."""

    other: PerspectiveCfg = field(
        default_factory=lambda: PerspectiveCfg(
            pipeline=DelayPipelineCfgV3(
                latency=LatencyCfg(
                    distribution=DistributionCfg(
                        type="normal", mean=0.1, std=0.08, min_value=0.0
                    )
                )
            )
        )
    )
    """Configuration for other perspective (agent viewing others)."""

    field_overrides: dict[str, DelayPipelineCfgV3] = field(default_factory=dict)
    """Per-field pipeline overrides. Key: field name, Value: custom pipeline config."""


@dataclass
class MultiAgentDelayCfgV3:
    """Configuration for multi-agent delay system.

    Wraps UnifiedDelayCfgV3 with multi-agent specific settings.
    """

    delay_cfg: UnifiedDelayCfgV3 = field(default_factory=UnifiedDelayCfgV3)
    """Underlying delay configuration."""

    noise: NoiseCfg = field(default_factory=NoiseCfg)
    """Observation noise configuration."""

    # Field definitions for multi-agent state tracking
    position_field: str = "body_position_w"
    """Field name for body position."""

    velocity_field: str = "body_velocity_w"
    """Field name for body velocity."""

    orientation_field: str = "body_orientation_w"
    """Field name for body orientation (quaternion)."""

    angular_velocity_field: str = "body_angular_velocity_w"
    """Field name for body angular velocity."""

    gimbal_orientation_field: str = "gimbal_orientation_w"
    """Field name for gimbal orientation."""

    bbox_field: str = "bbox"
    """Field name for bounding box detections."""


# =============================================================================
# Preset Configurations
# =============================================================================


def create_no_delay_cfg() -> MultiAgentDelayCfgV3:
    """Create configuration with no delay (clean observations)."""
    cfg = MultiAgentDelayCfgV3()
    cfg.delay_cfg.ego.pipeline.latency.enabled = False
    cfg.delay_cfg.ego.pipeline.staleness.enabled = False
    cfg.delay_cfg.ego.pipeline.dropout.enabled = False
    cfg.delay_cfg.other.pipeline.latency.enabled = False
    cfg.delay_cfg.other.pipeline.staleness.enabled = False
    cfg.delay_cfg.other.pipeline.dropout.enabled = False
    cfg.noise.enabled = False
    return cfg


def create_fixed_delay_cfg(
    ego_latency: float = 0.05, other_latency: float = 0.1
) -> MultiAgentDelayCfgV3:
    """Create configuration with fixed (deterministic) delay."""
    cfg = MultiAgentDelayCfgV3()

    # Fixed latency = constant distribution, no variance
    cfg.delay_cfg.ego.pipeline.latency.distribution = DistributionCfg(
        type="constant", value=ego_latency
    )
    cfg.delay_cfg.other.pipeline.latency.distribution = DistributionCfg(
        type="constant", value=other_latency
    )

    # Disable staleness in fixed mode
    cfg.delay_cfg.ego.pipeline.staleness.enabled = False
    cfg.delay_cfg.other.pipeline.staleness.enabled = False

    return cfg


def create_random_delay_cfg(
    ego_latency_mean: float = 0.05,
    ego_latency_std: float = 0.02,
    other_latency_mean: float = 0.1,
    other_latency_std: float = 0.08,
    staleness_fps_mean: float = 25.0,
    staleness_fps_range: float = 5.0,
    dropout_prob: float = 0.05,
) -> MultiAgentDelayCfgV3:
    """Create configuration with full random delay, staleness, and dropout."""
    cfg = MultiAgentDelayCfgV3()

    # Random latency
    cfg.delay_cfg.ego.pipeline.latency.distribution = DistributionCfg(
        type="normal", mean=ego_latency_mean, std=ego_latency_std, min_value=0.0
    )
    cfg.delay_cfg.other.pipeline.latency.distribution = DistributionCfg(
        type="normal", mean=other_latency_mean, std=other_latency_std, min_value=0.0
    )

    # Staleness (FPS limiting)
    cfg.delay_cfg.ego.pipeline.staleness.enabled = True
    cfg.delay_cfg.ego.pipeline.staleness.fps_distribution = DistributionCfg(
        type="uniform",
        mean=staleness_fps_mean,
        half_range=staleness_fps_range,
        min_value=5.0,
        max_value=60.0,
    )
    cfg.delay_cfg.other.pipeline.staleness.enabled = True
    cfg.delay_cfg.other.pipeline.staleness.fps_distribution = (
        cfg.delay_cfg.ego.pipeline.staleness.fps_distribution
    )

    # Dropout
    cfg.delay_cfg.ego.pipeline.dropout.probability = dropout_prob
    cfg.delay_cfg.other.pipeline.dropout.probability = dropout_prob

    return cfg
