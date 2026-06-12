# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cooperative track re-acquisition tracker (ticket 050, Slice A)."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ReacquisitionTrackerCfg:
    """Configuration for :class:`ReacquisitionTracker`.

    The tracker instruments single-agent, peer-assisted track-loss and re-acquisition events.
    It is a measurement-only component: it does not modify observations, rewards, or dynamics.
    """

    # ===== Activation =====

    enable: bool = False
    """Whether the tracker is active. Default False keeps the env bit-exact with the baseline."""

    # ===== Effective-track gate (reachable-set within FOV) =====

    k_fov: float = 1.0
    """FOV-fraction multiplier for the reachable-set gate. A detection is an actionable track
    while ``v_max * bbox_age < k_fov * range * tan(fov_eff_half)``. k_fov=1.0 is the literal
    "the target could still be inside the frame"; k_fov<1.0 declares loss earlier (with margin).
    Dimensionless sensitivity knob — NOT a staleness time in seconds."""

    use_gt_range: bool = True
    """Whether the env should feed privileged GT range into the gate (recommended; removes the
    monocular range ambiguity). Consumed by the env when assembling inputs — the tracker itself
    uses whatever ``target_range`` it receives. Metric-only; GT range must NOT leak into obs/reward."""

    # ===== Event-counting hygiene (flicker filters) =====

    tau_min_s: float = 0.2
    """Minimum peer-assisted-deficit duration [s] to count as a qualifying event. Filters
    single-step flicker. Event-counting hygiene, not a physical claim."""

    tau_hold_s: float = 0.4
    """Minimum post-recovery hold [s] for a re-acquisition to count as a success. Filters
    one-step recovery flicker."""

    # ===== Cause tagging =====

    tag_far_edge: bool = False
    """If True and a target-pixel-in-bounds signal is supplied, split empty-detection losses into
    ``far`` (in frame but too small -> needs zoom-in) vs ``edge`` (out of frame -> needs gimbal slew).
    If False, both collapse to a single ``fov`` cause."""

    # ===== Diagnostics =====

    record_diagnostics: bool = True
    """Whether to record recovery-window diagnostics (inter-agent distance change and ego/peer
    bearing alignment) to disambiguate emergent re-acquisition behaviour. Requires the env to
    supply agent positions / bearings; silently skipped if absent."""

    collect_episode_values: bool = False
    """When True, the env stashes per-env episode values into a buffer at each reset (for eval-time
    aggregation in experiments/evaluate.py). Default False so training does not grow an unbounded
    buffer — TensorBoard logging via episode_summary() is unaffected."""
