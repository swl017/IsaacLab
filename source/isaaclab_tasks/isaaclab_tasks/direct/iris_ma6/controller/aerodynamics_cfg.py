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

    Fidelity Levels:
    - Level 0: Disabled (fastest training)
    - Level 1: Basic drag (F_drag = -0.5 * rho * Cd * A * |v|^2 * v_hat)
    - Level 2: Level 1 + Wind (constant + Dryden gust model)
    - Level 3: Level 2 + Rotor effects (H-force, blade flapping)
    """

    fidelity_level: int = 3
    """Aerodynamic fidelity level (0-3). Default 0 (disabled)."""

    # Level 1: Basic drag parameters
    C_d: float = 0.03
    """Drag coefficient [-]."""

    A: float = 0.1
    """Frontal area [m^2]."""

    rho: float = 1.225
    """Air density [kg/m^3] at sea level."""

    # Level 2: Wind parameters
    v_wind_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Mean wind velocity in world frame [m/s]."""

    sigma_gust: float = 0.0
    """Gust standard deviation [m/s]."""

    gust_bandwidth: float = 0.5
    """Gust bandwidth [Hz]. Lower = slower varying gusts."""

    # Level 3: Rotor effects
    k_H: float = 0.001
    """H-force coefficient [Ns/m]."""

    k_flap: float = 0.01
    """Blade flapping coefficient [Nm/(m/s)]."""
