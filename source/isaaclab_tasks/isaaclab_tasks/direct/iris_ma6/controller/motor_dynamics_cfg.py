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

    Mode selector (ticket 040):
        ``model = "default"`` — pre-040 racing-class numerics (k_f=1.2e-5, omega_max=5000, etc).
        ``model = "pegasus"`` — Pegasus IrisConfig parity (k_f=8.54858e-6, omega_max=1100, tau≈0).

    The effective numerics are read via ``get_effective_*`` accessors so both
    ``MotorDynamics`` and the shared ``MixerMatrix`` (built in ``DroneController``)
    resolve the same values without duplicating dispatch logic.
    """

    model: str = "default"
    """Motor model selector. ``"default"`` (pre-040 racing-class) or ``"pegasus"``
    (Pegasus IrisConfig parity, ticket 040). Default preserves bit-exact pre-040 behavior."""

    k_f: float = 1.2e-05
    """Thrust coefficient [N/(rad/s)^2]. T = k_f * omega^2.

    Default-mode value. With omega_max=5000: T_max = 300 N per motor (1200 N total,
    ~82:1 T/W for 1.5 kg). Aggressive racing-drone-class configuration for fast
    attitude recovery.
    """

    k_m: float = 1.4e-07
    """Torque coefficient [Nm/(rad/s)^2]. Q = k_m * omega^2.

    Default-mode value. Scaled proportionally with k_f. Ratio k_m/k_f ≈ 0.012 typical for 5" props.
    """

    tau_motor: float = 0.01
    """Motor time constant [s]. Default-mode value (10 ms symmetric)."""

    omega_min: float = 50.0
    """Minimum rotor speed (idle) [rad/s]. Kept in both modes — mixer's thrust-preserving
    clamp needs a strictly-positive floor to avoid a zero denominator in the headroom
    redistribution (pegasus-mode docstring in doc/pegasus_physics_parity_spec.md §4.4)."""

    omega_max: float = 5000.0
    """Maximum rotor speed [rad/s]. Default-mode value (racing-class ceiling)."""

    arm_length: float = 0.22
    """Arm length from center to rotor [m]. Kept in both modes (USD geometry)."""

    # ----------------------------------------------------------------------
    # Ticket 040 — Pegasus IrisConfig parity values.
    # Read only when ``model == "pegasus"`` via ``get_effective_*`` accessors.
    # Defaults match PegasusSimulator's QuadraticThrustCurve defaults (k_f, k_m,
    # omega_max) and an effectively-zero tau_motor (Pegasus's plant has no
    # first-order lag on the thrust path).
    # ----------------------------------------------------------------------

    k_f_pegasus: float = 8.54858e-6
    """Pegasus-mode thrust coefficient. From PegasusSimulator quadratic_thrust_curve.py:36.
    With omega_max_pegasus=1100: T_max = 10.34 N per rotor (41.4 N total, T/W ≈ 2.81 at 1.5 kg)."""

    k_m_pegasus: float = 1.0e-6
    """Pegasus-mode torque coefficient. From PegasusSimulator quadratic_thrust_curve.py:40.
    Yields k_m/k_f ≈ 0.117 (10× higher yaw authority per thrust than default mode)."""

    omega_max_pegasus: float = 1100.0
    """Pegasus-mode max rotor speed [rad/s]. From PegasusSimulator quadratic_thrust_curve.py:51."""

    tau_motor_pegasus: float = 1.0e-4
    """Pegasus-mode motor time constant [s]. Pegasus's QuadraticThrustCurve is instantaneous;
    we keep a tiny non-zero value so the first-order-lag code path (and its DR knob) stays active
    while contributing negligible lag at any sim dt down to 1/1000."""

    # ----------------------------------------------------------------------
    # Effective-value accessors — single dispatch point for both MotorDynamics
    # and DroneController's shared MixerMatrix.
    # ----------------------------------------------------------------------

    def get_effective_k_f(self) -> float:
        """Return ``k_f`` dispatched on ``self.model``."""
        return self.k_f_pegasus if self.model == "pegasus" else self.k_f

    def get_effective_k_m(self) -> float:
        """Return ``k_m`` dispatched on ``self.model``."""
        return self.k_m_pegasus if self.model == "pegasus" else self.k_m

    def get_effective_omega_max(self) -> float:
        """Return ``omega_max`` dispatched on ``self.model``."""
        return self.omega_max_pegasus if self.model == "pegasus" else self.omega_max

    def get_effective_tau_motor(self) -> float:
        """Return ``tau_motor`` dispatched on ``self.model``."""
        return self.tau_motor_pegasus if self.model == "pegasus" else self.tau_motor
