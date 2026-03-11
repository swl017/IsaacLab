# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for attitude controller.

This middle-loop controller converts attitude errors to rate setpoints,
following the PX4 control architecture (mc_att_control).
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass


@configclass
class AttitudeControllerCfg:
    """Configuration for quaternion-based P attitude controller.

    Control law (P-only, outputs rate setpoint):
        rate_setpoint = -Kp_att * attitude_error + yaw_rate_feedforward

    Where attitude_error = 2 * sign(q_err.w) * q_err.xyz

    The rate setpoint is then fed to the inner rate controller (PID)
    which computes the actual torque commands.

    Based on PX4 mc_att_control gains:
        MC_ROLL_P = 6.5 (rad/s per rad)
        MC_PITCH_P = 6.5
        MC_YAW_P = 2.8 (lower for yaw)
    """

    Kp_att: tuple[float, float, float] = (6.5, 6.5, 2.8)
    """Proportional gains [roll, pitch, yaw] [(rad/s)/rad].

    Converts attitude error to desired angular rate.
    Yaw gain is lower to deprioritize yaw tracking.
    """

    rate_limit: tuple[float, float, float] = (
        math.radians(360.0),
        math.radians(360.0),
        math.radians(360.0),
    )
    """Maximum rate setpoint output [roll, pitch, yaw] [rad/s].

    Prevents excessive rate commands that could destabilize the drone.
    Matches PX4 MC_*RATE_MAX defaults.
    """

    yaw_weight: float = 0.4
    """Yaw control weight for mixing with roll/pitch [0-1].

    When roll/pitch error is large, yaw tracking is reduced.
    Matches PX4 MC_YAW_WEIGHT default.
    """
