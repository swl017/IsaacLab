# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for attitude controller."""

from __future__ import annotations

from isaaclab.utils import configclass

from .motor_dynamics_cfg import MotorDynamicsCfg

# Compute thrust limits from motor dynamics (single source of truth)
_motor_cfg = MotorDynamicsCfg()
_THRUST_MIN = _motor_cfg.k_f * _motor_cfg.omega_min**2
_THRUST_MAX = _motor_cfg.k_f * _motor_cfg.omega_max**2


@configclass
class AttitudeControllerCfg:
    """Configuration for quaternion-based PD attitude controller.

    Control law:
        tau_cmd = -Kp_att * attitude_error - Kd_att * (omega - omega_des)

    Where attitude_error = 2 * sign(q_err.w) * q_err.xyz
    """

    Kp_att: tuple[float, float, float] = (8.9308, 8.9308, 7.3526)
    """Proportional gains [roll, pitch, yaw] [Nm/rad]. Auto-tuned 2026-03-12, score=5.3831."""

    Kd_att: tuple[float, float, float] = (1.1283, 1.1283, 1.4365)
    """Derivative gains [roll, pitch, yaw] [Nm/(rad/s)]. Auto-tuned 2026-03-12, score=5.3831."""

    tau_max: tuple[float, float, float] = (20.0, 20.0, 8.0)
    """Maximum moment output [roll, pitch, yaw] [Nm].

    Conservative limit based on motor capabilities. Theoretical max during
    full differential thrust is much higher, but this prevents overshoot.
    """

    thrust_min: float = _THRUST_MIN
    """Minimum thrust per rotor [N]. Auto-computed from k_f * omega_min^2."""

    thrust_max: float = _THRUST_MAX
    """Maximum thrust per rotor [N]. Auto-computed from k_f * omega_max^2."""
