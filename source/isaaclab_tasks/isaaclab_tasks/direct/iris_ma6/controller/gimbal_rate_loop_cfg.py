# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SIYI A8 mini gimbal rate loop.

The rate loop sits between the policy-emitted (azimuth_rate, elevation_rate)
command and the existing world-frame setpoint integration inside
`gimbal_controller_jacobian.py`. It models the SIYI hardware's user-command
path:
    - first-order lag toward the saturated rate command (τ per axis),
    - hard saturation at ±max_rate_per_axis,
    - optional deadband (default-disabled),
    - per-env command-to-first-move dead-time buffer (mas/036; default
      curriculum scale = 0 keeps it disabled until the env advances it).

Body-motion compensation does NOT flow through this loop — the existing IK
inside the gimbal controller continues to use ``q_body_NOW`` every step,
preserving instantaneous LOS stabilization. See mas/035 ticket for the
architectural invariant.

Source for the defaults:
    /home/usrg/mas/src/gimbal_controller/scripts/gimbal_rate_step_followspeed_tune/rate_model.json
        yaw   τ = 0.0995 s,  k = 73.31 deg/s per unit u
        pitch τ = 0.0954 s,  k = 73.40 deg/s per unit u
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class GimbalRateLoopCfg:
    """Configuration for `GimbalRateLoop`.

    The rate loop is per-axis (azimuth ⊥ elevation at the policy interface).
    Joint-level coupling continues to live inside `J^{-1}` downstream and is
    not affected by this configuration.
    """

    enabled: bool = True
    """Master switch. When False, the rate loop is pass-through (preserves
    pre-mas/035 behavior for regression / smoke runs)."""

    # =========================================================================
    # Per-axis time constants (rate_model.json fit)
    # =========================================================================

    tau_yaw_s: float = 0.0995
    """First-order lag time constant on the yaw / azimuth-rate channel [s]."""

    tau_pitch_s: float = 0.0954
    """First-order lag time constant on the pitch / elevation-rate channel [s]."""

    # =========================================================================
    # Saturation
    # =========================================================================

    max_rate_per_axis: float = 1.28
    """Hard saturation on the user-command rate per axis [rad/s].
    1.28 rad/s ≈ 73.3 deg/s — the measured `k_deg_s_per_u` at u=1.0 from
    rate_model.json. Independent of the asset's `velocity_limit_sim` (which
    controls the inner motor stabilization loop, kept much higher)."""

    # =========================================================================
    # mas/036: command-to-first-move dead time
    #
    # Source: gimbal_dead_time_fit.json (mas/036 fit of rate_step_summary.csv
    # `latency_s` column). Yaw mean 68 ms, pitch mean 63 ms, union mean 66 ms.
    # =========================================================================

    deadband_rad_s: float = 0.0
    """Symmetric deadband on the rate-command magnitude [rad/s]. Commands
    below this threshold are zeroed before the lag stage. Defaulted to 0;
    mas/036 may set this from the measurement if a deadband is confirmed."""

    dead_time_mean_s: float = 0.066
    """Mean dead time at the rate-loop input [s]. Sampled per env at
    episode reset; effective mean is scaled by ``dead_time_curriculum_scale``."""

    dead_time_std_s: float = 0.016
    """Standard deviation of the dead-time sampler [s]. Used at episode
    reset together with ``dead_time_mean_s`` to draw a per-env Gaussian
    sample, scaled by ``dead_time_curriculum_scale``."""

    dead_time_max_s: float = 0.120
    """Hard cap on per-env dead time [s]. Bounds the circular-buffer
    depth (``ceil(dead_time_max_s / dt)`` ring slots are pre-allocated)
    and clips Gaussian samples that would exceed it."""

    dead_time_curriculum_scale: float = 0.0
    """Curriculum-driven scaling on the sampled dead time ∈ [0, 1].
    0 = no delay (rate-loop input is identical to the command — preserves
    pre-mas/036 behavior), 1 = full measured distribution. Set by the
    env's curriculum driver each step; defaults to 0 so importing this
    cfg does not silently introduce delay."""

    # =========================================================================
    # Domain randomization on τ (replaces the retired joint-PD scale ranges)
    # =========================================================================

    tau_scale_range_yaw: tuple[float, float] = (0.8, 1.2)
    """Multiplicative DR range on `tau_yaw_s`, sampled per env per episode."""

    tau_scale_range_pitch: tuple[float, float] = (0.8, 1.2)
    """Multiplicative DR range on `tau_pitch_s`, sampled per env per episode."""
