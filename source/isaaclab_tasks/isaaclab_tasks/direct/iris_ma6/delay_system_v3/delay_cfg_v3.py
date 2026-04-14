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

    min_steps: int = 0
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
    """Standard deviation for orientation noise (radians). Applied as axis-angle perturbation."""

    angular_velocity_std: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="constant", value=0.02, min_value=0.0
        )
    )
    """Standard deviation for angular velocity noise (rad/s)."""

    acceleration_std: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="constant", value=0.1, min_value=0.0
        )
    )
    """Standard deviation for linear acceleration noise (m/s²)."""

    bbox_std: DistributionCfg = field(
        default_factory=lambda: DistributionCfg(
            type="constant", value=7.0, min_value=0.0
        )
    )
    """Standard deviation for bounding box noise (pixels).

    Applied to bbox center (x, y) and dimensions (w, h).
    Default 7.0 pixels matches iris_ma5 implementation.
    """


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
class PerAgentDelayCfg:
    """Per-agent delay configuration with curriculum scaling.

    When per_agent_randomization is enabled, each agent samples its own
    delay parameters independently. Parameters scale with curriculum progress [0, 1].
    """

    # Communication delay (motion states from other agents)
    max_comm_delay: float = 0.2
    """Maximum communication delay in seconds (at progress=1.0)."""

    comm_delay_distribution: Literal["uniform", "normal"] = "uniform"
    """Distribution type for comm delay sampling."""

    # Detection delay (bbox observations)
    max_detection_delay: float = 0.15
    """Maximum detection delay in seconds (at progress=1.0)."""

    detection_delay_distribution: Literal["uniform", "normal"] = "uniform"
    """Distribution type for detection delay sampling."""

    # Staleness (FPS limiting)
    min_fps: float = 10.0
    """Minimum FPS for staleness (at progress=1.0). Lower FPS = more staleness."""

    max_fps: float = 60.0
    """Maximum FPS (at progress=0.0 or when staleness disabled)."""

    # Dropout
    max_dropout_rate: float = 0.1
    """Maximum dropout rate (at progress=1.0)."""

    # Per-agent heterogeneity (multiplicative scales on base parameters)
    latency_scale_range: tuple[float, float] = (0.5, 2.0)
    """Per-agent latency multiplier range. E.g., (0.5, 2.0) means agent latency
    ranges from 50% to 200% of configured base latency."""

    noise_scale_range: tuple[float, float] = (0.5, 2.0)
    """Per-agent noise multiplier range on configured noise stds."""

    dropout_offset_range: tuple[float, float] = (0.0, 0.05)
    """Per-agent additive dropout offset range. Added to curriculum dropout rate."""


@dataclass
class RewardStateCfg:
    """Configuration for reward state computation.

    Controls whether delay and/or noise are applied to states used for rewards.
    This enables 4 different modes for reward computation:

    | use_delay | use_noise | Result                           |
    |-----------|-----------|----------------------------------|
    | False     | False     | Pure GT (privileged)             |
    | True      | False     | Delayed, clean (default)         |
    | False     | True      | GT with noise                    |
    | True      | True      | Full noisy delayed               |
    """

    use_delay: bool = True
    """If True, apply delay pipeline. If False, use ground-truth timing."""

    use_noise: bool = False
    """If True, add observation noise to reward states."""


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

    field_overrides: dict[str, DelayPipelineCfgV3] = field(default_factory=dict)
    """Per-field pipeline overrides for this perspective.

    Key: field name suffix (e.g., 'bboxes_2d')
    Value: custom pipeline config for that field type.

    This allows different delay behavior for motion fields vs detection fields
    within the same perspective.
    """


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

    burst_dropout: "BurstDropoutCfg | None" = None
    """Optional burst dropout configuration (Gilbert-Elliott model).

    When set and enabled, replaces per-pipeline i.i.d. dropout with
    correlated burst dropout on directional communication channels.
    Import: from .burst_dropout import BurstDropoutCfg
    """

    # Per-agent randomization settings
    per_agent_randomization: bool = True
    """If True, each agent samples its own delay parameters independently.
    If False, all agents share the same delay parameters."""

    per_agent_cfg: PerAgentDelayCfg = field(default_factory=PerAgentDelayCfg)
    """Per-agent delay configuration with max values scaled by curriculum."""

    # Reward state configuration
    reward_state_cfg: RewardStateCfg = field(default_factory=RewardStateCfg)
    """Configuration for reward state computation (delay/noise toggles)."""

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


@dataclass
class DelaySystemKeyParams:
    """Key parameters for delay system configuration.

    This dataclass exposes the most important tunable parameters for the delay system,
    making them easy to configure in the main environment config without navigating
    the nested configuration structure.

    These parameters are the most critical for sim-to-real transfer and should be
    tuned based on real-world communication/sensor characteristics.

    Ego Latency Architecture (matches v2):
    - Ego MOTION fields (pos, vel, orientation): First-order lag only (fast proprioceptive)
    - Ego DETECTION fields (bboxes): Latency + staleness + dropout (NN inference time)
    """

    # === Ego Motion Latency (proprioceptive sensing) ===
    ego_motion_latency_enabled: bool = False
    """Whether to apply latency pipeline to ego motion fields (pos/vel/orientation).

    When False (default), ego motion uses first-order lag only - nearly instant.
    This matches real-world proprioceptive sensing (IMU, GPS) which is very fast.
    """

    ego_motion_latency_mean: float = 0.005
    """Mean transport latency for ego motion (IMU/GPS) in seconds.

    Only used when ego_motion_latency_enabled=True.
    Default 5ms represents typical IMU/GPS processing time.
    """

    ego_motion_latency_std: float = 0.002
    """Standard deviation of ego motion transport latency in seconds."""

    ego_motion_fol_tau: float = 0.005
    """First-order lag time constant for ego motion fields in seconds.

    Applied regardless of ego_motion_latency_enabled to smooth sensor noise.
    """

    # === Ego Detection Latency (NN inference) ===
    ego_detection_latency_mean: float = 0.05
    """Mean latency for ego detection (running neural network on own camera) in seconds.

    Detection requires running a NN which takes time even on ego camera.
    Default 50ms represents typical GPU inference + post-processing time.
    """

    ego_detection_latency_std: float = 0.015
    """Standard deviation of ego detection latency in seconds."""

    # === Other Agent Latency (communication) ===
    other_latency_mean: float = 0.1
    """Mean latency for observing other agents in seconds.
    Higher than ego due to network communication delays."""

    other_latency_std: float = 0.08
    """Standard deviation of other-agent latency in seconds."""

    # === Staleness (FPS) Parameters ===
    staleness_fps_mean: float = 25.0
    """Mean detection FPS. Lower FPS = more staleness (older detections)."""

    staleness_fps_range: float = 5.0
    """Half-range for FPS sampling: samples from [mean - range, mean + range]."""

    # === Dropout Parameters ===
    dropout_prob: float = 0.05
    """Probability of detection dropout (missed detections) per step."""

    # === Burst Dropout Parameters ===
    burst_dropout_enabled: bool = True
    """Whether burst dropout (Gilbert-Elliott model) is active.

    When enabled, replaces i.i.d. per-pipeline dropout with correlated
    burst dropout on directional communication channels."""

    burst_p_onset: float = 0.01
    """Good->Bad transition probability per step.
    Controls burst frequency. Mean good-run = 1/p_onset steps."""

    burst_p_recovery: float = 0.1
    """Bad->Good transition probability per step.
    Controls burst duration. Mean burst length = 1/p_recovery steps."""

    burst_good_dropout_prob: float = 0.01
    """Dropout probability in Good state (low baseline loss)."""

    burst_bad_dropout_prob: float = 0.9
    """Dropout probability in Bad state (near-total loss during burst)."""

    # === Noise Parameters ===
    noise_enabled: bool = True
    """Whether observation noise is applied."""

    noise_position_std: float = 0.1
    """Standard deviation for position noise in meters."""

    noise_velocity_std: float = 0.05
    """Standard deviation for velocity noise in m/s."""

    noise_orientation_std: float = 0.01
    """Standard deviation for orientation noise in radians."""

    noise_angular_velocity_std: float = 0.02
    """Standard deviation for angular velocity noise in rad/s."""

    noise_acceleration_std: float = 0.1
    """Standard deviation for linear acceleration noise in m/s²."""

    noise_bbox_std: float = 7.0
    """Standard deviation for bounding box noise in pixels.
    Applied to bbox center (x, y) and dimensions (w, h)."""

    # === Reward State Configuration ===
    reward_use_delay: bool = True
    """If True, reward computation uses delayed states. If False, uses GT."""

    reward_use_noise: bool = False
    """If True, reward computation uses noisy states."""


def create_delay_cfg_from_params(params: DelaySystemKeyParams) -> MultiAgentDelayCfgV3:
    """Create delay system configuration from key parameters.

    This function creates separate pipelines for motion vs detection fields:
    - Ego MOTION fields: First-order lag only (fast proprioceptive sensing)
    - Ego DETECTION fields: Latency + staleness + dropout (NN inference)
    - Other agent fields: Full delay pipeline (communication delay)

    Args:
        params: Key parameters dataclass with tunable values.

    Returns:
        Fully configured MultiAgentDelayCfgV3.
    """
    cfg = MultiAgentDelayCfgV3()

    # === Ego Motion Pipeline (first-order lag only) ===
    # Motion fields (pos, vel, orientation) use fast proprioceptive sensing
    ego_motion_pipeline = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=params.ego_motion_latency_enabled,
            distribution=DistributionCfg(
                type="normal",
                mean=params.ego_motion_latency_mean,
                std=params.ego_motion_latency_std,
                min_value=0.0,
            ),
        ),
        staleness=StalenessCfg(enabled=False),  # No staleness for motion
        dropout=DropoutCfg(enabled=False),  # No dropout for motion
        first_order_lag=FirstOrderLagCfg(
            enabled=True, tau=params.ego_motion_fol_tau
        ),
    )

    # === Ego Detection Pipeline (latency + staleness + dropout) ===
    # Detection fields (bboxes) require NN processing time
    fps_dist = DistributionCfg(
        type="uniform",
        mean=params.staleness_fps_mean,
        half_range=params.staleness_fps_range,
        min_value=5.0,
        max_value=60.0,
    )
    ego_detection_pipeline = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(
                type="normal",
                mean=params.ego_detection_latency_mean,
                std=params.ego_detection_latency_std,
                min_value=0.0,
            ),
        ),
        staleness=StalenessCfg(enabled=True, fps_distribution=fps_dist),
        dropout=DropoutCfg(enabled=True, probability=params.dropout_prob),
        first_order_lag=FirstOrderLagCfg(enabled=False),
    )

    # === Other Agent Pipeline (full delay for communication) ===
    other_pipeline = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(
                type="normal",
                mean=params.other_latency_mean,
                std=params.other_latency_std,
                min_value=0.0,
            ),
        ),
        staleness=StalenessCfg(enabled=True, fps_distribution=fps_dist),
        dropout=DropoutCfg(enabled=True, probability=params.dropout_prob),
        first_order_lag=FirstOrderLagCfg(enabled=True, tau=params.ego_motion_fol_tau),
    )

    # Build perspective configs
    # Ego: motion pipeline as default, detection (bbox) field uses override.
    # The bbox field carries both raycaster (raw) and replicator (noisy)
    # payloads; one pipeline applies one delay realization to both.
    cfg.delay_cfg.ego = PerspectiveCfg(
        pipeline=ego_motion_pipeline,
        field_overrides={
            "bboxes_2d": ego_detection_pipeline,
        },
    )

    # Other: full delay pipeline for all fields
    cfg.delay_cfg.other = PerspectiveCfg(
        pipeline=other_pipeline,
    )

    # Noise configuration
    cfg.noise.enabled = params.noise_enabled
    cfg.noise.position_std = DistributionCfg(
        type="constant", value=params.noise_position_std, min_value=0.0
    )
    cfg.noise.velocity_std = DistributionCfg(
        type="constant", value=params.noise_velocity_std, min_value=0.0
    )
    cfg.noise.orientation_std = DistributionCfg(
        type="constant", value=params.noise_orientation_std, min_value=0.0
    )
    cfg.noise.angular_velocity_std = DistributionCfg(
        type="constant", value=params.noise_angular_velocity_std, min_value=0.0
    )
    cfg.noise.acceleration_std = DistributionCfg(
        type="constant", value=params.noise_acceleration_std, min_value=0.0
    )
    cfg.noise.bbox_std = DistributionCfg(
        type="constant", value=params.noise_bbox_std, min_value=0.0
    )

    # Reward state configuration
    cfg.reward_state_cfg.use_delay = params.reward_use_delay
    cfg.reward_state_cfg.use_noise = params.reward_use_noise

    # Burst dropout configuration
    if params.burst_dropout_enabled:
        from .burst_dropout import BurstDropoutCfg

        cfg.burst_dropout = BurstDropoutCfg(
            enabled=True,
            p_onset=params.burst_p_onset,
            p_recovery=params.burst_p_recovery,
            good_dropout_prob=params.burst_good_dropout_prob,
            bad_dropout_prob=params.burst_bad_dropout_prob,
        )

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
