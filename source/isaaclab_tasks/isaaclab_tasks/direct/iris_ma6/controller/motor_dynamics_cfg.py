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

    k_f: float = 1.2e-05
    """Thrust coefficient [N/(rad/s)^2]. T = k_f * omega^2.

    Increased from 8.54858e-06 to provide higher thrust authority.
    With omega_max=5000: T_max = 300 N per motor (1200 N total, ~82:1 T/W for 1.5kg).
    This is an aggressive racing-drone-class configuration for fast attitude recovery.
    """

    k_m: float = 1.4e-07
    """Torque coefficient [Nm/(rad/s)^2]. Q = k_m * omega^2.

    Scaled proportionally with k_f. Ratio k_m/k_f ≈ 0.012 typical for 5" props.
    """

    tau_motor: float = 0.01
    """Motor time constant [s]. Reduced to 15ms for faster response."""

    omega_min: float = 50.0
    """Minimum rotor speed (idle) [rad/s]. Reduced for more downward torque headroom."""

    omega_max: float = 5000.0
    """Maximum rotor speed [rad/s]. Increased for higher thrust ceiling."""

    arm_length: float = 0.22
    """Arm length from center to rotor [m]."""
