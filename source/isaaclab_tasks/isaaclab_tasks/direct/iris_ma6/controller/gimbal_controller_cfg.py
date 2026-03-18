# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for gimbal controller."""

from __future__ import annotations

import math

from isaaclab.utils import configclass


@configclass
class GimbalControllerCfg:
    """Configuration for gimbal controller with world-frame stabilization.

    Frame Convention:
    - Gimbal Base: FLU (Forward-Left-Up), attached to drone body
    - Joint Order: Yaw (outer) -> Roll (middle) -> Pitch (inner)
    - Yaw: CCW positive when viewed from above
    - Pitch: Nose down positive (looking down)
    - Roll: Left side up positive

    Note: Physical servo dynamics (time constant, bandwidth) are handled by
    the implicit actuator (stiffness/damping) in the asset config, not here.
    """

    max_gimbal_rate: float = 2 * math.pi
    """Maximum gimbal angular rate [rad/s]. Default 2*pi (360 deg/s)."""

    yaw_limits: tuple[float, float] = (-math.radians(200), math.radians(200))
    """Yaw joint limits [min, max] [rad]. Default +-180 deg."""

    pitch_limits: tuple[float, float] = (-math.radians(45), math.radians(45))
    """Pitch joint limits [min, max] [rad]. Default -45 to 45 deg.
    Limited to stay away from 90 deg gimbal lock singularity."""

    roll_limits: tuple[float, float] = (-math.pi / 4, math.pi / 4)
    """Roll joint limits [min, max] [rad]. Default +-45 deg for better stabilization."""

    auto_stabilize_roll: bool = True
    """Whether to automatically compute roll to keep horizon level."""

    feedback_blend: float = 0.1
    """Blend factor for closed-loop joint position feedback.
    Corrects internal state drift: state += beta * (actual - state).
    Range [0.05, 0.2]. Higher values correct faster but may oscillate."""

    initial_yaw: float = 0.0 #-math.radians(90)
    """Initial gimbal yaw angle [rad]. Default -90 deg."""
