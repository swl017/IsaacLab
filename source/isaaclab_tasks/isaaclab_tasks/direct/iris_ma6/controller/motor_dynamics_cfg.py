# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for motor dynamics."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class MotorDynamicsCfg:
    """Configuration for first-order motor dynamics.

    Motor speed follows commanded speed with first-order lag:
        d(omega)/dt = (omega_cmd - omega) / tau_motor

    Discrete-time: omega_new = omega + alpha * (omega_cmd - omega)
    where alpha = 1 - exp(-dt / tau_motor) for exact first-order response
    """

    k_f: float = 8.54858e-06
    """Thrust coefficient [N/(rad/s)^2]. T = k_f * omega^2."""

    k_m: float = 1.0e-07
    """Torque coefficient [Nm/(rad/s)^2]. Q = k_m * omega^2."""

    tau_motor: float = 0.02
    """Motor time constant [s]. Default 20ms for fast brushless DC motors."""

    omega_min: float = 100.0
    """Minimum rotor speed (idle) [rad/s]."""

    omega_max: float = 1100.0
    """Maximum rotor speed [rad/s]."""

    arm_length: float = 0.22
    """Arm length from center to rotor [m]."""
