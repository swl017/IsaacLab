# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for attitude controller."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class AttitudeControllerCfg:
    """Configuration for quaternion-based PD attitude controller.

    Control law:
        tau_cmd = -Kp_att * attitude_error - Kd_att * (omega - omega_des)

    Where attitude_error = 2 * sign(q_err.w) * q_err.xyz
    """

    Kp_att: tuple[float, float, float] = (14.45, 14.45, 4.01)
    """Proportional gains [roll, pitch, yaw] [Nm/rad]. Auto-tuned 2026-03-11."""

    Kd_att: tuple[float, float, float] = (1.13, 1.13, 0.88)
    """Derivative gains [roll, pitch, yaw] [Nm/(rad/s)]. Auto-tuned 2026-03-11."""

    tau_max: tuple[float, float, float] = (5.0, 5.0, 2.0)
    """Maximum moment output [roll, pitch, yaw] [Nm]."""
