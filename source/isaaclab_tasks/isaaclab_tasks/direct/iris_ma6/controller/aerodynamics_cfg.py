# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for aerodynamic effects."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class AerodynamicsCfg:
    """Configuration for configurable aerodynamic effects.

    Fidelity Levels (mode="default"):
    - Level 0: Disabled (fastest training)
    - Level 1: Basic drag (F_drag = -0.5 * rho * Cd * A * |v|^2 * v_hat)
    - Level 2: Level 1 + Wind (constant + Dryden gust model)
    - Level 3: Level 2 + Rotor effects (H-force, blade flapping)

    Ticket 040 — Pegasus parity mode:
    - ``mode = "pegasus"`` forces the controller to use the linear-diagonal drag
      formula ``F = -diag(d) · v_body`` with ``drag_coefs`` (default Pegasus
      IrisConfig values), bypasses wind / gust, and skips the H-force / blade-flap
      branch regardless of the ``fidelity_level`` int. Default ``mode = "default"``
      preserves bit-exact pre-040 behavior.
    """

    mode: str = "default"
    """Aero mode selector. ``"default"`` (pre-040 — quadratic drag + Dryden + rotor effects)
    or ``"pegasus"`` (linear-diagonal drag, no wind, no rotor effects — PegasusSimulator parity)."""

    fidelity_level: int = 3
    """Aerodynamic fidelity level (0-3). Default 3. Ignored when ``mode == "pegasus"``."""

    drag_model: str = "quadratic"
    """Drag formula selector. ``"quadratic"`` (default) — ``F = -½ ρ C_d A · |v| · v``.
    ``"linear_diag"`` — ``F = -diag(d_x, d_y, d_z) · v_body``. Force-set to ``"linear_diag"``
    by the controller when ``mode == "pegasus"`` regardless of this field's value."""

    drag_coefs: tuple[float, float, float] = (0.50, 0.30, 0.00)
    """Linear-diagonal drag coefficients (X, Y, Z body) [Ns/m]. Used when
    ``drag_model == "linear_diag"``. Defaults match PegasusSimulator IrisConfig
    (iris.py:30): forward 0.5, lateral 0.3, vertical 0.0 (anisotropic)."""

    # Level 1: Basic drag parameters (quadratic model — default mode)
    C_d: float = 0.03
    """Drag coefficient [-]. Used when ``drag_model == "quadratic"``."""

    A: float = 0.1
    """Frontal area [m^2]. Used when ``drag_model == "quadratic"``."""

    rho: float = 1.225
    """Air density [kg/m^3] at sea level. Used when ``drag_model == "quadratic"``."""

    # Level 2: Wind parameters
    v_wind_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Mean wind velocity in world frame [m/s]. Forced to zero when ``mode == "pegasus"``."""

    sigma_gust: float = 0.0
    """Gust standard deviation [m/s]. Bypassed when ``mode == "pegasus"``."""

    gust_bandwidth: float = 0.5
    """Gust bandwidth [Hz]. Lower = slower varying gusts. Bypassed when ``mode == "pegasus"``."""

    # Level 3: Rotor effects
    k_H: float = 0.001
    """H-force coefficient [Ns/m]. Bypassed when ``mode == "pegasus"``."""

    k_flap: float = 0.01
    """Blade flapping coefficient [Nm/(m/s)]. Bypassed when ``mode == "pegasus"``."""
