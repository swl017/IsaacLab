# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for velocity controller."""

from __future__ import annotations

from isaaclab.utils import configclass

@configclass
class VelocityControllerCfg:
    """Configuration for PI velocity controller with anti-windup.

    Control law:
        a_des = Kp_vel * (v_des - v) + Ki_vel * integral + g * z_hat
        thrust_cmd = m * norm(a_des)
        attitude_des = rotation aligning body z with a_des
    """

    Kp_vel: tuple[float, float, float] = (4.5127, 4.5127, 3.3880)
    """Proportional gains [x, y, z] [1/s]. Auto-tuned 2026-03-12, score=5.3831."""

    Ki_vel: tuple[float, float, float] = (0.6495, 0.6495, 0.3384)
    """Integral gains [x, y, z] [1/s^2]. Auto-tuned 2026-03-12, score=5.3831."""

    integral_limit: tuple[float, float, float] = (4.0, 4.0, 2.0)
    """Integral anti-windup limits [x, y, z] [m/s equivalent]."""

    max_tilt: float = 30.0
    """Maximum tilt angle [deg]. Limits roll/pitch commands."""

    max_lin_vel: float = 30.0
    """Maximum linear velocity [m/s]."""
