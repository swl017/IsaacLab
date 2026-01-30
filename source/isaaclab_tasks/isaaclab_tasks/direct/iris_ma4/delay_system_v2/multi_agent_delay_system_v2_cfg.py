# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for multi-agent delay system V2."""

from __future__ import annotations

from dataclasses import field
from isaaclab.utils import configclass


@configclass
class MultiAgentDelaySystemV2Cfg:
    """Configuration for multi-agent delay system using V2 field-based architecture.

    This configuration wraps the quadcopter delay system V2 (DelaySystemV2) to provide
    a multi-agent delay system with:
    - Dual pipelines: Clean (rewards) + Noisy (observations)
    - Per-field delay configurations (first-order lag, staleness, latency, dropout)
    - Perspective-aware delays: Ego (fast) vs Inter-agent (slow)
    - Noise injection before delays

    Example usage:
        cfg = MultiAgentDelaySystemV2Cfg(
            dt=0.01,
            motion_time_constant=0.1,
            detection_fps_mean=30.0,
            position_noise_std=0.05,
            enable_noise=True,
        )

        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=["agent_0", "agent_1"],
            num_envs=512,
            num_joints_per_agent={"agent_0": 3, "agent_1": 3},
            num_targets_per_agent={"agent_0": 1, "agent_1": 1},
            device="cuda:0",
        )
    """

    # =========================================================================
    # Simulation
    # =========================================================================
    dt: float = 0.01
    """Simulation timestep in seconds. Default is 0.01 (100 Hz)."""

    # =========================================================================
    # Motion Field Configuration (First-Order Lag)
    # =========================================================================
    motion_time_constant: float = 0.1
    """Time constant for motion fields (position, velocity, acceleration) in seconds.

    First-order lag implements: x_new = x_old + alpha * (x_meas - x_old)
    where alpha = dt / (dt + tau)
    """

    orientation_time_constant: float = 0.1
    """Time constant for orientation fields (quaternions) in seconds.

    Quaternions are filtered with first-order lag and then normalized.
    """

    joint_time_constant: float = 0.05
    """Time constant for joint fields (gimbal positions, velocities) in seconds."""

    zoom_time_constant: float = 0.1
    """Time constant for camera zoom level in seconds."""

    # =========================================================================
    # Detection Field Configuration (Staleness + Latency + Dropout)
    # =========================================================================
    detection_fps_mean: float = 30.0
    """Mean detection FPS for staleness (sample-and-hold rate) in Hz."""

    detection_fps_std: float = 5.0
    """Standard deviation of detection FPS."""

    detection_latency_mean: float = 0.03
    """Mean detection latency (transport delay) in seconds."""

    detection_latency_std: float = 0.005
    """Standard deviation of detection latency in seconds."""

    detection_dropout_rate: float = 0.05
    """Probability of detection dropout [0, 1]."""

    # =========================================================================
    # Ego Communication (Fast Local Processing)
    # =========================================================================
    ego_comm_time_constant: float = 0.005
    """Time constant for ego communication delay (local processing) in seconds.

    Ego sees its own state with minimal delay.
    """

    # =========================================================================
    # Inter-Agent Communication (Slow Network)
    # =========================================================================
    inter_agent_comm_fps_mean: float = 30.0
    """Mean FPS for inter-agent communication staleness in Hz."""

    inter_agent_comm_fps_std: float = 5.0
    """Standard deviation of inter-agent communication FPS."""

    inter_agent_comm_latency_mean: float = 0.1
    """Mean inter-agent communication latency in seconds."""

    inter_agent_comm_latency_std: float = 0.02
    """Standard deviation of inter-agent communication latency in seconds."""

    inter_agent_comm_dropout_rate: float = 0.05
    """Probability of inter-agent communication dropout [0, 1]."""

    # =========================================================================
    # Noise Configuration (Injected Before Delays)
    # =========================================================================
    enable_noise: bool = True
    """Whether to enable noise injection for observations."""

    position_noise_std: float = 0.0
    """Standard deviation of position noise in meters."""

    orientation_noise_std: float = 0.0
    """Standard deviation of orientation noise (added to quaternion components)."""

    linear_velocity_noise_std: float = 0.0
    """Standard deviation of linear velocity noise in m/s."""

    angular_velocity_noise_std: float = 0.0
    """Standard deviation of angular velocity noise in rad/s."""

    linear_acceleration_noise_std: float = 0.0
    """Standard deviation of linear acceleration noise in m/s^2."""

    gimbal_noise_std: float = 0.0
    """Standard deviation of gimbal joint angle noise in radians."""

    bbox_noise_std: float = 0.0
    """Standard deviation of bounding box noise in pixels."""

    zoom_noise_std: float = 0.0
    """Standard deviation of zoom level noise."""

    noise_seed: int = 0
    """Random seed for noise generation (for reproducibility)."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        if self.motion_time_constant < 0:
            raise ValueError(f"motion_time_constant must be non-negative, got {self.motion_time_constant}")

        if not 0 <= self.detection_dropout_rate <= 1:
            raise ValueError(f"detection_dropout_rate must be in [0, 1], got {self.detection_dropout_rate}")

        if not 0 <= self.inter_agent_comm_dropout_rate <= 1:
            raise ValueError(f"inter_agent_comm_dropout_rate must be in [0, 1], got {self.inter_agent_comm_dropout_rate}")

        # Validate noise STDs are non-negative
        noise_params = [
            ("position_noise_std", self.position_noise_std),
            ("orientation_noise_std", self.orientation_noise_std),
            ("linear_velocity_noise_std", self.linear_velocity_noise_std),
            ("angular_velocity_noise_std", self.angular_velocity_noise_std),
            ("linear_acceleration_noise_std", self.linear_acceleration_noise_std),
            ("gimbal_noise_std", self.gimbal_noise_std),
            ("bbox_noise_std", self.bbox_noise_std),
            ("zoom_noise_std", self.zoom_noise_std),
        ]
        for name, value in noise_params:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
