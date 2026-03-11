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
    """Configuration for gimbal controller with first-order dynamics.

    Frame Convention:
    - Gimbal Base: FLU (Forward-Left-Up), attached to drone body
    - Joint Order: Yaw (outer) -> Roll (middle) -> Pitch (inner)
    - Yaw: CCW positive when viewed from above
    - Pitch: Nose down positive (looking down)
    - Roll: Left side up positive
    """

    tau_gimbal: float = 0.05
    """Gimbal motor time constant [s]. Default 50ms for servo motors."""

    max_gimbal_rate: float = 2 * math.pi
    """Maximum gimbal angular rate [rad/s]. Default 2*pi (360 deg/s)."""

    yaw_limits: tuple[float, float] = (-math.pi, math.pi)
    """Yaw joint limits [min, max] [rad]. Default +-180 deg."""

    pitch_limits: tuple[float, float] = (0.0, math.pi / 2)
    """Pitch joint limits [min, max] [rad]. Default 0 to 90 deg (looking horizontal to down)."""

    roll_limits: tuple[float, float] = (-math.pi / 6, math.pi / 6)
    """Roll joint limits [min, max] [rad]. Default +-30 deg."""

    auto_stabilize_roll: bool = True
    """Whether to automatically compute roll to keep horizon level."""
