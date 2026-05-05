# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for controller gain randomization."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class GainRandomizationCfg:
    """Configuration for per-env controller gain randomization.

    At each episode reset, gains are sampled as:
        effective_gain = nominal_gain * uniform(effective_low, effective_high)

    where:
        effective_low = 1.0 - progress * (1.0 - scale_range[0])
        effective_high = 1.0 + progress * (scale_range[1] - 1.0)

    This ramps from no randomization (progress=0) to full range (progress=1).
    Curriculum-gated to the dynamics phase (160k-200k steps).
    """

    enabled: bool = True
    """Enable controller gain randomization."""

    scale_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale range at full curriculum progress (+-20%)."""

    randomize_velocity: bool = True
    """Randomize velocity controller gains (Kp_vel, Ki_vel)."""

    randomize_attitude: bool = True
    """Randomize attitude controller gain (Kp_att)."""

    randomize_rate: bool = True
    """Randomize rate controller gains (Kp_rate, Ki_rate, Kd_rate)."""

    randomize_motor: bool = True
    """Randomize motor time constant (tau_motor)."""

    randomize_zoom: bool = True
    """Randomize zoom dynamics (tau_zoom, max_zoom_rate)."""

    zoom_scale_range: tuple[float, float] = (0.01, 1.0)
    """Multiplicative scale range for tau_zoom at full curriculum progress.

    Wide asymmetric range applied to the *live* curriculum-gated tau_zoom
    base (which the env writes as ``max(cfg.tau_zoom * progress, 1e-4)`` at
    every reset). At full progress this gives tau_zoom in
    ``[cfg.tau_zoom * 0.01, cfg.tau_zoom * 1.0] = [1e-3, 0.1] s``,
    preserving near-instant samples in the post-curriculum distribution.
    At progress=0 randomization is skipped entirely (env-side base of 1e-4
    is used as-is — bootstrap-time instant zoom). See iris_ma6
    dynamics-curriculum review (2026-05-05).
    """

    randomize_max_lin_vel: bool = True
    """Randomize max linear velocity (action scaling)."""

    max_lin_vel_scale_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale range for max_lin_vel at full curriculum progress.

    +-20% around the agent-velocity-curriculum value (10 m/s at full
    progress_agent_velocity). At progress_dynamics=0 the range collapses
    to (1, 1).
    """
