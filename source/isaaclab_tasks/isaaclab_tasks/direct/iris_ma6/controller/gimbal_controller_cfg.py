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

    Controllers read actual joint state from simulation each step rather than
    maintaining parallel integrated state.
    """

    max_gimbal_rate: float = 1.0 * math.pi
    """Maximum gimbal angular rate [rad/s]. Default 2*pi (360 deg/s)."""

    yaw_limits: tuple[float, float] = (-math.radians(160), math.radians(160))
    """Yaw joint limits [min, max] [rad]. Default +-180 deg."""

    pitch_limits: tuple[float, float] = (-math.radians(45), math.radians(45))
    """Pitch joint limits [min, max] [rad]. Default -45 to 45 deg.
    Limited to stay away from 90 deg gimbal lock singularity."""

    roll_limits: tuple[float, float] = (-math.radians(45), math.radians(45))
    """Roll joint limits [min, max] [rad]. Default +-45 deg for better stabilization."""

    auto_stabilize_roll: bool = True
    """Whether to automatically compute roll to keep horizon level."""

    pointing_gain: float = 32.5
    """Proportional gain converting pointing error (rad) to angular velocity command (rad/s).
    Converts body-frame attitude error to a rate command fed through J^{-1}.
    Discrete-time stability: K*dt < 2 (K < 200 at dt=0.01s). Diverges above K≈200.
    With implicit actuator (PD loop): optimal K≈30 (score 5.8 deg).
    With direct state setting: optimal K≈112 (score 5.8 deg)."""

    initial_yaw: float = 0.0  # was -math.radians(90) when visual forward = body +Y
    """Initial gimbal yaw angle [rad]."""

    mode: str = "jacobian"
    """Gimbal control mode.

    "analytical": Direct atan2 IK — exact, zero-lag, no gain tuning needed.
    "jacobian":   J^{-1} velocity tracking — uses pointing_gain, has tracking dynamics.
    """
