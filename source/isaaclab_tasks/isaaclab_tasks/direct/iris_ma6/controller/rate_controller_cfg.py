# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for angular rate controller.

This inner-loop controller converts angular rate setpoints to body torques,
following the PX4 control architecture.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from .motor_dynamics_cfg import MotorDynamicsCfg

# Compute thrust limits from motor dynamics (single source of truth)
_motor_cfg = MotorDynamicsCfg()
_THRUST_MIN = _motor_cfg.k_f * _motor_cfg.omega_min**2
_THRUST_MAX = _motor_cfg.k_f * _motor_cfg.omega_max**2


@configclass
class RateControllerCfg:
    """Configuration for PID angular rate controller.

    Control law (parallel form):
        tau_cmd = Kp * rate_error + Ki * integral(rate_error) - Kd * angular_accel

    Note: The D-term uses angular acceleration (derivative of omega), not
    derivative of error. This provides damping without amplifying setpoint changes.

    Based on PX4 mc_rate_control gains:
        MC_ROLLRATE_P = 0.15
        MC_ROLLRATE_I = 0.2
        MC_ROLLRATE_D = 0.003
    """

    Kp_rate: tuple[float, float, float] = (0.15, 0.15, 0.2)
    """Proportional gains [roll, pitch, yaw] [Nm/(rad/s)].

    Scaled from PX4 defaults which assume different inertia.
    """

    Ki_rate: tuple[float, float, float] = (0.2, 0.2, 0.15)
    """Integral gains [roll, pitch, yaw] [Nm/rad].

    Provides steady-state tracking for constant rate commands.
    """

    Kd_rate: tuple[float, float, float] = (0.003, 0.003, 0.001)
    """Derivative gains [roll, pitch, yaw] [Nm/(rad/s^2)].

    Acts on angular acceleration (measured), not rate error derivative.
    """

    integral_limit: tuple[float, float, float] = (0.3, 0.3, 0.15)
    """Integral saturation limits [roll, pitch, yaw] [Nm].

    Prevents integral windup. Based on PX4 defaults.
    """

    tau_max: tuple[float, float, float] = (20.0, 20.0, 8.0)
    """Maximum moment output [roll, pitch, yaw] [Nm].

    Conservative limit to prevent actuator saturation.
    """

    rate_limit: tuple[float, float, float] = (
        math.radians(360.0),
        math.radians(360.0),
        math.radians(360.0),
    )
    """Maximum angular rate limits [roll, pitch, yaw] [rad/s].

    Clamps rate setpoints before processing.
    """

    # Anti-windup configuration
    antiwindup_i_factor_limit: float = math.radians(400.0)
    """Rate error threshold for I-gain reduction [rad/s].

    When rate error exceeds this, I-gain is reduced quadratically.
    Matches PX4 behavior: i_factor = max(0, 1 - (error/limit)^2)
    """

    # Thrust allocation limits (used by mixer)
    thrust_min: float = _THRUST_MIN
    """Minimum thrust per rotor [N]. Auto-computed from k_f * omega_min^2."""

    thrust_max: float = _THRUST_MAX
    """Maximum thrust per rotor [N]. Auto-computed from k_f * omega_max^2."""
