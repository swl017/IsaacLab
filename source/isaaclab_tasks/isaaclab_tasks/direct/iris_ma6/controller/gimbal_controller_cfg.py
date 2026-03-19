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

    Note: When using direct joint state setting (write_joint_state_to_sim),
    set feedback_blend=0.0 since internal state always matches actual state.
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

    feedback_blend: float = 0.0
    """Blend factor for closed-loop joint position feedback.
    Corrects internal state drift: state += beta * (actual - state).
    Set to 0.0 when using direct joint state setting (no actuator lag).
    Use 0.05-0.2 when using implicit actuator PD loop."""

    pointing_gain: float = 10.0
    """Proportional gain converting pointing error (rad) to angular velocity command (rad/s).
    Converts body-frame attitude error to a rate command fed through J^{-1}.
    Range [5.0, 20.0]. Higher = faster convergence but may overshoot."""

    initial_yaw: float = 0.0 #-math.radians(90)
    """Initial gimbal yaw angle [rad]. Default -90 deg."""
