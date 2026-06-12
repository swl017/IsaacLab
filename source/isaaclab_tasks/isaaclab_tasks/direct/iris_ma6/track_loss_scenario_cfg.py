# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ceiling-raise overrides that manufacture single-agent track-loss events (ticket 050, Slice A).

These values are overlaid onto ``initial_states`` / ``target_controller`` / ``curriculum`` /
``delay_system_params`` in :meth:`IrisMA6TestEnvCfg.__post_init__` ONLY when
``enable_track_loss_scenario=True``. Each field raises a curriculum *ceiling* (a ``*_max`` /
``*_end`` value) or activates the independent per-(env, agent) dropout earlier — the easy
*floor* (``*_min`` / ``*_start``) is never touched, so the t034 anti-forgetting sampler keeps
mass on the easy regime. Flag off = bit-exact baseline.

The values below are starting points; the gate run confirms they drive
``Coop/track_loss_event_rate`` to a non-trivial rate (target >= 0.5 qualifying mid-loss
events/episode), and the engineer tunes from there.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class TrackLossScenarioCfg:
    """Ceiling-only scenario overrides for the cooperation trigger (Slice A)."""

    # ===== Far sub-mode (recovery needs zoom-in) — initial conditions =====

    target_distance_max: float = 40.0
    """Curriculum ceiling on target spawn distance [m] (env baseline 15 = t046 closer-spawn).
    Far targets shrink below the detector-miss sigmoid for one agent at a time. Floor
    ``target_distance_min`` unchanged."""

    zoom_initial_max_end: float = 6.0
    """Curriculum ceiling on initial zoom at progress=1 (baseline 4). Floor unchanged."""

    cylinder_diameter_max: float = 40.0
    """Curriculum ceiling on agent spawn-cylinder diameter [m] (env baseline 30 = t046). Wider
    spread makes agent viewpoints diverge, so the target exits one frustum while a peer retains it.
    Floor ``cylinder_diameter_min`` unchanged. (Spawn geometry = initial conditions; in scope.)"""

    # ===== Edge sub-mode (recovery needs gimbal slew) — target behaviour =====

    target_max_speed_end: float = 8.0
    """Curriculum ceiling on target speed [m/s] at progress=1 (baseline 5). Faster targets cross a
    single agent's FOV boundary. Floor ``max_speed_start`` unchanged."""

    target_update_interval_min_end: float = 1.0
    """Curriculum ceiling (shorter = harder) on min re-heading interval [s] at progress=1
    (baseline 2.0). Choppier motion stresses single-agent tracking."""

    target_update_interval_max_end: float = 2.5
    """Curriculum ceiling on max re-heading interval [s] at progress=1 (baseline 4.0)."""

    # ===== Dropout sub-mode (independent per-(env, agent)) =====

    dropout_start_step: int = 40000
    """Bring the i.i.d. detector-dropout curriculum into the cooperative training window
    (baseline 160000). Dropout stays INDEPENDENT per directional channel, so single-agent loss
    arises naturally (P(exactly one of 2 lost) = 2p(1-p))."""

    dropout_end_step: int = 120000
    """End step of the dropout ramp (baseline 180000)."""

    dropout_prob: float = 0.15
    """Per-step dropout probability ceiling at full progress (baseline 0.05). The per-(env, agent)
    Uniform(0, global_p) sampler keeps a zero-dropout floor."""
