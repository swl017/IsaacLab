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

    agent_velocity_start_step: int = 20000
    """Step to start ramping agent max linear velocity."""

    agent_velocity_end_step: int = 40000
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

    fp_fn_background_start_step: int = 4000000
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

    fp_fn_ramp_start_step: int = 4000000
    """Step to start ramping FP/FN from background (0.1) to full calibrated (1.0).

    Aligned with noise onset at 100k — observation corruption starts
    after coordination shift completes and sigma has had time to grow.
    """

    fp_fn_end_step: int = 4000000
    """Step when FP/FN rates reach calibrated full values."""

    fp_fn_start_step: int = 4000000
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

    coordination_start_step: int = 60000
    """Step to start rewarding coordination (triangulation)."""

    coordination_end_step: int = 100000
    """Step when coordination rewards reach full scale."""

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

    dynamics_start_step: int = 180000
    """Step to start dynamics randomization (mass, inertia)."""

    dynamics_end_step: int = 220000
    """Step when dynamics randomization reaches full range."""

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
