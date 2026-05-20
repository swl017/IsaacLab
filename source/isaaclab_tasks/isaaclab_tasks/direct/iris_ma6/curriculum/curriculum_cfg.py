# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for training curriculum phases."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Configuration for curriculum learning phases.

    The curriculum defines when different aspects of training are enabled
    and their difficulty progression. Each phase has a start and end step,
    with linear interpolation between them.

    Training phases are designed to increase difficulty along (mostly) one axis
    at a time:
    - Coupling/geometry first (triangulation + collision avoidance)
    - Then target dynamics (track moving target with clean observations)
    - Then partial observability (noise → delay/staleness → dropout)

    The phases can overlap, allowing gradual transitions.

    Step:    0k   20k   40k   60k   80k  100k  120k  140k  160k  180k
             |     |     |     |     |     |     |     |     |     |
    AgentVel:[──ramp──]full───────────────────────────────────────────
    Safety:  [──ramp──]full───────────────────────────────────────────
    Tracking:[────────ramp────────]full───────────────────────────────
    FP/FN bg:     [───────── flat 0.1 ──────────]
    Target:        [──────────ramp──────────]full─────────────────────
    Coord:                  [────ramp────]full────────────────────────
    FP/FN:                                [──ramp──]full─────────────
    Noise:                                [──ramp──]full─────────────
    FixDelay:                                   [──ramp──]full───────
    RndDelay:                                         [──ramp──]full─
    Dropout:                                                [rmp]full
    Dynamics:                                                  [ramp]

    The FP/FN background phase (15k-100k at 10% of calibrated rates) is
    the one addition over the 492f baseline. It gives the policy ~85k steps
    to learn bbox-empty handling before the full ramp at 100k.

    """

    # ==========================================================================
    # Global Curriculum Settings
    # ==========================================================================

    all_end_step: int = 400000
    """Step at which all curriculum factors reach their final values."""

    # ==========================================================================
    # Phase 0/1: Geometry + Formation (clean observations, static target)
    # ==========================================================================

    tracking_start_step: int = 20000
    """Step to start increasing formation/initialization difficulty."""

    tracking_end_step: int = 60000
    """Step when formation/initialization difficulty reaches maximum."""

    # ==========================================================================
    # Phase 0: Agent Velocity Ramp (learn tilt dynamics on slow target)
    # ==========================================================================

    agent_velocity_start_step: int = 0
    """Step to start ramping agent max linear velocity."""

    agent_velocity_end_step: int = 0
    """Step when agent velocity reaches its configured maximum.

    Decoupled from target motion so the agent learns tilt compensation
    and preemptive zoom-out before the target starts moving fast.
    """

    # ==========================================================================
    # Phase 2: Noise Introduction (before delay)
    # ==========================================================================

    noise_start_step: int = 100000
    """Step to start introducing observation noise.

    Starts after target motion ramp completes so the agent masters
    fast-target tracking with clean observations first.
    """

    noise_end_step: int = 120000
    """Step when noise reaches maximum realistic values."""

    # ==========================================================================
    # FP/FN — Detection Misses and False Positives
    # Uses a two-phase schedule:
    #   15k-100k: constant background at fp_fn_background_scale (0.1 by default)
    #   100k-120k: ramp from background to full calibrated rate
    # Rationale: the policy needs training coverage of bbox-empty states
    # during early training to develop hold-last-position and search behaviors,
    # so it has a learned response when the full FP/FN ramp hits at 100k.
    # ==========================================================================

    fp_fn_background_start_step: int = 2200000
    """Step to start constant-rate background FP/FN exposure.

    Starts at 15k (just after bootstrap completes) so the policy encounters
    bbox-empty states while learning base tracking, ensuring a learned
    hold-last-position response rather than the fragile 'bbox always valid'
    assumption.
    """

    fp_fn_background_scale: float = 0.1
    """Constant fp_fn_scale during the background phase (15k-100k).

    0.1 = 10% of calibrated rates. With fp_rate=0.0347, this gives ~0.35%
    effective FP rate per slot. Miss rate is size-dependent, roughly 1-5%
    effective depending on target size. Rare enough to not dominate loss,
    frequent enough for RNN sequences (length 32) to encounter bbox-empty.
    """

    fp_fn_ramp_start_step: int = 2_200_000
    """Step to start ramping FP/FN from background (0.1) to full calibrated (1.0).

    Aligned with noise onset at 100k — observation corruption starts
    after coordination shift completes and sigma has had time to grow.
    """

    fp_fn_end_step: int = 3_000_000
    """Step when FP/FN rates reach calibrated full values."""

    fp_fn_start_step: int = 4_000_000
    """[DEPRECATED] Use fp_fn_ramp_start_step. Kept for backward compatibility."""

    # ==========================================================================
    # Phase 3: Fixed Delay (Deterministic Latency)
    # ==========================================================================

    fixed_delay_start_step: int = 120000
    """Step to start introducing fixed (deterministic) delay."""

    fixed_delay_end_step: int = 140000
    """Step when fixed delay reaches maximum value (uses config latency means)."""

    # NOTE: Fixed delay magnitude uses existing detection_latency_mean and
    # inter_agent_comm_latency_mean from MultiAgentDelaySystemV2Cfg.

    # ==========================================================================
    # Phase 4: Random Delay + Staleness
    # ==========================================================================

    random_delay_start_step: int = 140000
    """Step to transition from fixed to random delay."""

    random_delay_end_step: int = 160000
    """Step when random delay variance reaches maximum."""

    # NOTE: Random delay uses existing latency_std parameters from config.

    # ==========================================================================
    # Phase 5: Dropout (after delay phases)
    # ==========================================================================

    dropout_start_step: int = 160000
    """Step to start introducing dropout."""

    dropout_end_step: int = 180000
    """Step when dropout reaches maximum rate."""

    # ==========================================================================
    # Phase 6: Burst Dropout (after i.i.d. dropout)
    # ==========================================================================

    burst_dropout_start_step: int = 200000
    """Step to start introducing burst dropout (correlated packet loss).

    Placed after i.i.d. dropout phase so the policy first learns to handle
    isolated single-frame drops before experiencing sustained blackouts.
    """

    burst_dropout_end_step: int = 220000
    """Step when burst dropout onset probability reaches target value."""

    # Legacy alias for backward compatibility
    delay_start_step: int = 120000
    """[DEPRECATED] Use noise_start_step, fixed_delay_start_step, etc."""

    delay_end_step: int = 160000
    """[DEPRECATED] Use random_delay_end_step."""

    # ==========================================================================
    # Phase 0: Multi-Agent Coupling (triangulation geometry)
    # ==========================================================================

    coordination_start_step: int = 20000
    """Step at which the triangulation reward switches on, as a step-at-
    episode-boundary signal (per-env, computed in _reset_idx).

    The default 30k is just past the bootstrap window (pair_valid plateaus
    around 0.85 by 14k–24k in the SIYI stack) — the policy already has a
    solid tracking baseline when triangulation is added, so the new gradient
    is informative rather than noisy.

    The step-vs-ramp choice is deliberate: triangulation is a *task signal*
    (do it / don't), not a *difficulty knob* (do harder triangulation), so a
    smooth ramp creates a 40k window where the gradient is some random
    fraction of the real one and the policy can't disambiguate
    "I'm bad at triangulation" from "the reward isn't fully on yet."
    """

    coordination_end_step: int = 20000
    """[Legacy] Retained for backward compatibility but ignored under the
    step-at-episode-boundary semantics. Held equal to coordination_start_step
    so that any code reading _linear_progress(start, end) also sees a step.
    """

    # ==========================================================================
    # Phase 2: Safety (CBF collision avoidance — after basic tracking is learned)
    # ==========================================================================

    safety_start_step: int = 20000
    """Step to start enforcing safety constraints (CBF penalty)."""

    safety_end_step: int = 40000
    """Step when safety penalties reach full scale."""

    # ==========================================================================
    # Phase 1: Target Dynamics (still clean observations)
    # ==========================================================================

    moving_target_start_step: int = 40000
    """Step to start introducing target motion.

    Starts after agent velocity ramp completes (20k end),
    so the agent has learned tilt compensation before tracking fast targets.
    """

    moving_target_end_step: int = 80000
    """Step when target reaches maximum speed/maneuverability.

    40k-step ramp gives time to learn tilt-coupled tracking.
    """

    # ==========================================================================
    # Phase 3+: Robot/Camera Dynamics Randomization (last)
    # ==========================================================================

    dynamics_start_step: int = 60000
    """Step to start dynamics randomization.

    Placed during the coordination ramp (60k-100k) and BEFORE noise
    (100k-120k). LR is still high here so the policy can actually adapt
    to the new dynamics distribution. By the noise/delay phases the policy
    has already robustified against gain/τ/mass variation.
    """

    dynamics_end_step: int = 100000
    """Step when dynamics randomization reaches full range.

    Aligned with noise_start_step so dynamics randomization is at full
    distribution before observation-corruption phases begin (clean handoff).
    """

    # ==========================================================================
    # Ticket 037 Slice 7 — Per-axis decoupling of `dynamics_*`
    #
    # `dynamics_*` currently bundles 8 physically-independent randomization
    # axes (gimbal τ, drone gains, max_lin_vel scale, FOV, gimbal mech
    # offsets, mass/inertia, gimbal stiff/damp, target scale). Real hardware
    # has no such correlation; bundling masks sim-to-real failure modes.
    #
    # Each axis has its own (start_step, end_step). Default `None` means
    # "inherit from dynamics_*" — so existing experiments are bit-exact
    # unless the cfg flag `enable_axis_independence` is True AND per-axis
    # values are set.
    #
    # See doc/critic_obs_design.md §2.3 / §3 for motivation, and ticket 037
    # for the decoupling sub-scope.
    # ==========================================================================

    gimbal_rate_tau_start_step: int | None = None
    """Per-axis start step for gimbal rate-loop τ randomization.
    None → inherits ``dynamics_start_step``."""

    gimbal_rate_tau_end_step: int | None = None
    """Per-axis end step for gimbal rate-loop τ randomization."""

    drone_gains_start_step: int | None = None
    """Per-axis start step for drone controller gain randomization
    (Kp_vel/Ki_vel, Kp_att, Kp/Ki/Kd_rate, τ_motor, τ_zoom, max_zoom_rate)."""

    drone_gains_end_step: int | None = None

    max_lin_vel_scale_start_step: int | None = None
    """Per-axis start step for max_lin_vel multiplicative scale
    (±20% via gain_randomization.max_lin_vel_scale_range)."""

    max_lin_vel_scale_end_step: int | None = None

    camera_fov_start_step: int | None = None
    """Per-axis start step for camera FOV scale randomization."""

    camera_fov_end_step: int | None = None

    gimbal_mech_offsets_start_step: int | None = None
    """Per-axis start step for gimbal mechanical yaw/pitch/roll offsets."""

    gimbal_mech_offsets_end_step: int | None = None

    mass_inertia_start_step: int | None = None
    """Per-axis start step for robot mass and inertia randomization."""

    mass_inertia_end_step: int | None = None

    gimbal_stiff_damp_start_step: int | None = None
    """Per-axis start step for gimbal joint stiffness/damping randomization."""

    gimbal_stiff_damp_end_step: int | None = None

    target_scale_start_step: int | None = None
    """Per-axis start step for target object xy/z scale randomization."""

    target_scale_end_step: int | None = None

    # ==========================================================================
    # Phase 3+: Gimbal command-to-first-move dead time (mas/036)
    #
    # Independent of `dynamics_*` so dead time and rate-loop τ can ramp on
    # different cadences. Defaults match mas/034's nominal full-bring-up
    # cadence (5e6 steps): full delay only late in training.
    # ==========================================================================

    gimbal_dead_time_start_step: int = 180000
    """Step to start ramping the gimbal dead-time curriculum scale (0 → 1).

    Starts where dynamics_end_step lands so the policy first masters the
    rate-loop lag (mas/035) on its own before the input-side dead time
    (mas/036) is layered on top.
    """

    gimbal_dead_time_end_step: int = 220000
    """Step when the gimbal dead-time curriculum scale reaches 1.0
    (full measured distribution). 5e6-step ramp matches the mas/034
    nominal cadence.
    """

    # ==========================================================================
    # Phase 3+: Zoom dynamics (mas/037)
    #
    # Two independent ramps:
    #   - tau ramp: τ₁ (post-integrator first-order lag) ramps from near-instant
    #     at bootstrap (< zoom_tau_start_step) to the configured tau_zoom over a
    #     long window. Decoupled from gimbal `dynamics_*` because zoom and gimbal
    #     have different timescales (mas/037 vs mas/035).
    #   - dead-time ramp: input-side dead-time scale ramps from 0 (no delay) to
    #     1 (full measured Gaussian). Mirrors `gimbal_dead_time_*` cadence.
    # ==========================================================================

    zoom_tau_start_step: int = 20000
    """Step to start ramping the zoom-controller τ₁ (mas/037).

    Bootstrap (< zoom_tau_start_step) keeps τ₁ near-instant (the env writes
    ``max(cfg.zoom.tau_zoom * progress_zoom_tau, 1e-4)`` at every reset, so
    progress=0 → tau ≈ 1e-4 → effectively pass-through). After this step, τ₁
    ramps slowly to its configured value.
    """

    zoom_tau_end_step: int = 200000
    """Step when the zoom-controller τ₁ ramp reaches 1.0 (full configured τ).

    Long ramp (default ~180k window) lets the policy gradually adapt to the
    measured τ₁ = 0.091 s without a step change. Zoom misalignment cascades
    into bbox-size and triangulation noise, so a slower ramp than the gimbal
    `dynamics_*` cadence is preferred.
    """

    zoom_dead_time_start_step: int = 180000
    """Step to start ramping the zoom dead-time curriculum scale (mas/037).

    Mirrors ``gimbal_dead_time_start_step`` so the policy first masters the
    rate-loop lag (τ₁ via ``zoom_tau_*``) before the input-side dead-time is
    layered on top.
    """

    zoom_dead_time_end_step: int = 220000
    """Step when the zoom dead-time curriculum scale reaches 1.0 (full measured
    Gaussian τ_d ~ N(0.100, 0.018) clipped to [0, 0.150]).
    """

    # ==========================================================================
    # Task Reward Level Curriculum (FIM → GT-anchored → Composite)
    # ==========================================================================

    task_level_2_start_step: int = 40000
    """Step to begin transitioning from Level 1 (FIM) to Level 2 (GT-anchored)."""

    task_level_2_end_step: int = 80000
    """Step when Level 2 fully replaces Level 1."""

    task_level_3_start_step: int = 80000
    """Step to begin blending in E2E component (Level 3 composite)."""

    task_level_3_end_step: int = 120000
    """Step when Level 3 composite reaches full weight."""

    def get_progress(self, current_step: int, start_step: int, end_step: int) -> float:
        """Calculate curriculum progress for a given phase.

        Args:
            current_step: Current training step.
            start_step: Phase start step.
            end_step: Phase end step.

        Returns:
            Progress value between 0.0 (not started) and 1.0 (completed).
        """
        if current_step < start_step:
            return 0.0
        elif current_step >= end_step:
            return 1.0
        else:
            return (current_step - start_step) / (end_step - start_step)

    # ==========================================================================
    # Phase 3/4 Delay Curriculum Methods
    # ==========================================================================

    def get_delay_mode(self, current_step: int) -> str:
        """Get current delay mode based on curriculum progress.

        Returns:
            'none': No delay applied (pre-Phase 3)
            'fixed': Fixed deterministic delay (Phase 3)
            'random': Random delay with staleness (Phase 4+)
        """
        if current_step < self.fixed_delay_start_step:
            return "none"
        elif current_step < self.random_delay_start_step:
            return "fixed"
        else:
            return "random"

    def get_agent_velocity_progress(self, current_step: int) -> float:
        """Get progress within agent velocity ramp phase [0, 1].

        Used to ramp agent max_lin_vel from min to max, decoupled from target motion.
        """
        return self.get_progress(current_step, self.agent_velocity_start_step, self.agent_velocity_end_step)

    def get_noise_progress(self, current_step: int) -> float:
        """Get progress within noise phase [0, 1]."""
        return self.get_progress(current_step, self.noise_start_step, self.noise_end_step)

    def get_fp_fn_progress(self, current_step: int) -> float:
        """Get FP/FN curriculum scale using the two-phase schedule.

        - Before fp_fn_background_start_step: 0.0 (no FP/FN)
        - Background phase (15k-100k): constant fp_fn_background_scale (flat low rate)
        - Ramp phase (100k-120k): linear ramp from background_scale to 1.0
        - After fp_fn_end_step: 1.0 (full calibrated rate)

        This replaces the standard linear ramp to give the policy training
        coverage of bbox-empty states during early training.
        """
        if current_step < self.fp_fn_background_start_step:
            return 0.0
        if current_step < self.fp_fn_ramp_start_step:
            return self.fp_fn_background_scale
        if current_step >= self.fp_fn_end_step:
            return 1.0
        # Ramp phase: linear from background_scale to 1.0
        ramp_progress = (current_step - self.fp_fn_ramp_start_step) / (
            self.fp_fn_end_step - self.fp_fn_ramp_start_step
        )
        return self.fp_fn_background_scale + ramp_progress * (1.0 - self.fp_fn_background_scale)

    def get_fixed_delay_progress(self, current_step: int) -> float:
        """Get progress within fixed delay phase [0, 1].

        Used to ramp delay magnitude from 0 to config mean values.
        """
        return self.get_progress(current_step, self.fixed_delay_start_step, self.fixed_delay_end_step)

    def get_random_delay_progress(self, current_step: int) -> float:
        """Get progress within random delay phase [0, 1].

        Used to ramp delay variance from 0 to config std values.
        """
        return self.get_progress(current_step, self.random_delay_start_step, self.random_delay_end_step)

    def get_dropout_progress(self, current_step: int) -> float:
        """Get progress within dropout phase [0, 1]."""
        return self.get_progress(current_step, self.dropout_start_step, self.dropout_end_step)

    def get_burst_dropout_progress(self, current_step: int) -> float:
        """Get progress within burst dropout phase [0, 1].

        Used to ramp burst onset probability (p_onset) from 0 to target value.
        """
        return self.get_progress(current_step, self.burst_dropout_start_step, self.burst_dropout_end_step)

    def get_dynamics_progress(self, current_step: int) -> float:
        """Get progress within the dynamics-randomization phase [0, 1].

        Used by mas/035 to gate the gimbal rate-loop (and any future
        actuator-dynamics knobs) so the policy first learns with instant
        gimbal response, then transitions to the measured first-order lag
        between `dynamics_start_step` and `dynamics_end_step`.
        """
        return self.get_progress(current_step, self.dynamics_start_step, self.dynamics_end_step)

    # ----------------------------------------------------------------- ticket 037
    # Per-axis decoupling of `dynamics_*`. Each returns progress in [0, 1]
    # using its own (start, end) if set, else inherits `dynamics_*`. When
    # the env cfg's `enable_axis_independence=False` (default), no per-axis
    # value is meant to be set, so all 8 methods return the same value as
    # `get_dynamics_progress` and downstream behavior is bit-exact.

    def _axis_progress(self, current_step: int, axis: str) -> float:
        """Helper: progress for one of the 8 decoupled dynamics axes.

        Resolves to ``get_progress(current_step, start, end)`` where
        ``start = self.{axis}_start_step or self.dynamics_start_step``,
        same for ``end``.
        """
        start = getattr(self, f"{axis}_start_step", None)
        end = getattr(self, f"{axis}_end_step", None)
        if start is None:
            start = self.dynamics_start_step
        if end is None:
            end = self.dynamics_end_step
        return self.get_progress(current_step, start, end)

    def get_gimbal_rate_tau_progress(self, current_step: int) -> float:
        """Progress for the gimbal rate-loop τ axis."""
        return self._axis_progress(current_step, "gimbal_rate_tau")

    def get_drone_gains_progress(self, current_step: int) -> float:
        """Progress for the drone controller gain randomization axis."""
        return self._axis_progress(current_step, "drone_gains")

    def get_max_lin_vel_scale_progress(self, current_step: int) -> float:
        """Progress for the max_lin_vel multiplicative scale axis."""
        return self._axis_progress(current_step, "max_lin_vel_scale")

    def get_camera_fov_progress(self, current_step: int) -> float:
        """Progress for the camera FOV scale axis."""
        return self._axis_progress(current_step, "camera_fov")

    def get_gimbal_mech_offsets_progress(self, current_step: int) -> float:
        """Progress for the gimbal mechanical offsets axis."""
        return self._axis_progress(current_step, "gimbal_mech_offsets")

    def get_mass_inertia_progress(self, current_step: int) -> float:
        """Progress for the robot mass/inertia axis."""
        return self._axis_progress(current_step, "mass_inertia")

    def get_gimbal_stiff_damp_progress(self, current_step: int) -> float:
        """Progress for the gimbal joint stiffness/damping axis."""
        return self._axis_progress(current_step, "gimbal_stiff_damp")

    def get_target_scale_progress(self, current_step: int) -> float:
        """Progress for the target object scale axis."""
        return self._axis_progress(current_step, "target_scale")

    def get_gimbal_dead_time_progress(self, current_step: int) -> float:
        """Get progress within the gimbal dead-time phase [0, 1] (mas/036).

        Drives ``GimbalRateLoop.set_dead_time_curriculum_scale`` so the
        per-env dead-time samples ramp from 0 (no delay) at
        ``gimbal_dead_time_start_step`` to the full measured Gaussian at
        ``gimbal_dead_time_end_step``. Independent of the rate-loop τ
        ramp (``dynamics_*``) so the two effects can be staged separately.
        """
        return self.get_progress(
            current_step,
            self.gimbal_dead_time_start_step,
            self.gimbal_dead_time_end_step,
        )

    def get_zoom_tau_progress(self, current_step: int) -> float:
        """Get progress within the zoom τ₁ ramp phase [0, 1] (mas/037).

        Multiplies the configured ``cfg.zoom.tau_zoom`` at episode reset:
        ``tau_eff = max(tau_zoom * progress, 1e-4)``. Bootstrap (progress=0)
        gives near-instant zoom; full progress gives the measured τ₁ =
        0.091 s. Decoupled from the gimbal rate-loop τ ramp
        (``dynamics_*``) — zoom and gimbal have different timescales.
        """
        return self.get_progress(
            current_step,
            self.zoom_tau_start_step,
            self.zoom_tau_end_step,
        )

    def get_zoom_dead_time_progress(self, current_step: int) -> float:
        """Get progress within the zoom dead-time phase [0, 1] (mas/037).

        Drives ``ZoomController.set_dead_time_curriculum_scale`` so the
        per-env dead-time samples ramp from 0 (no delay) at
        ``zoom_dead_time_start_step`` to the full measured Gaussian at
        ``zoom_dead_time_end_step``. Independent of the τ₁ ramp
        (``zoom_tau_*``) so the two effects can be staged separately.
        """
        return self.get_progress(
            current_step,
            self.zoom_dead_time_start_step,
            self.zoom_dead_time_end_step,
        )

    def get_task_level_progress(self, current_step: int) -> tuple:
        """Get task reward level curriculum progress.

        Returns:
            Tuple of (level2_progress, level3_progress), each in [0, 1].
            - level2_progress: Blend from Level 1 (FIM) to Level 2 (GT-anchored).
            - level3_progress: Blend from pure Level 2 to Level 3 composite (GT + E2E).
        """
        l2 = self.get_progress(current_step, self.task_level_2_start_step, self.task_level_2_end_step)
        l3 = self.get_progress(current_step, self.task_level_3_start_step, self.task_level_3_end_step)
        return l2, l3
