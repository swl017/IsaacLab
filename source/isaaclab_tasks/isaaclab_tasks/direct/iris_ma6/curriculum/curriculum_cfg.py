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

    Schedule: free warmup + framework-compliant staging + background FN.

    Design rules applied (Chapter 3 §A.0-A.6):
    - §A.0 Bootstrap: 0-10k free warmup (no curriculum, clean observations)
    - §A.1 Physics vs observation: velocity/safety/tracking/DR first;
      noise/delay/dropout/burst last; FP/FN background is baseline modification
    - §A.3 Phase matching: shrinks in early Phase 1; shifts strictly in
      mid-to-late Phase 2
    - §A.4 Stacking: compatible stages overlap; two shifts sequenced
      (target+velocity → coordination)
    - §A.5 Goldilocks: all ramps in [10k, 50k] = [2Ts, 5Ts] window
    - Physical feasibility: agent velocity and moving target ramps are
      COUPLED (same window) so the agent can always catch the target

    Stage classifications (see Chapter 3 §A.1):
    - Shrink (S):   Safety/CBF
    - Expand (E):   Tracking init, Dynamics DR (physics), Noise/Delay/
                    Dropout/Burst/FP/FN (observation)
    - Shift (Sh):   Coordination
    - Complex:      Target+Velocity coupled (expand + shift — chase velocity
                    grows with cap, so required normalized action stays ~0.8+)

    Step:    0k  10k 15k 20k         40k      60k      80k   90k   120k 130k 320k
             |    |   |   |           |        |        |     |     |    |    |
    [warmup] [───]
    Tracking (E):[─────ramp─────]full───────────────────────────────────
    Safety (S):  [─────ramp─────]full───────────────────────────────────
    FP/FN bg:        [─────────── flat 0.1 ───────────]
    Target+Vel:          [──────────ramp──────────]full──────────────── (coupled)
    DynDR (E):               [────────ramp────────]full────────────────
    Coord (Sh):                              [────ramp────]full────────
    FP/FN ramp:                                      [────ramp────]full
    Noise (E-obs):                                   [────ramp────]full
    Delay (E-obs):                                   [────ramp────]full
    Dropout (E-obs):                                      [────ramp────]full
    Burst (E-obs):                                            [──ramp──]full
    Post-curr:                                                        [──190k──]

    """

    # ==========================================================================
    # Global Curriculum Settings
    # ==========================================================================

    all_end_step: int = 320000
    """Step at which all curriculum factors reach their final values."""

    # ==========================================================================
    # Tracking & Formation (10-40k ramp, starts after free warmup)
    # ==========================================================================

    tracking_start_step: int = 10000
    """Step to start increasing formation/initialization difficulty."""

    tracking_end_step: int = 40000
    """Step when formation/initialization difficulty reaches maximum."""

    # ==========================================================================
    # Agent Velocity Ramp (20-60k) — COUPLED with moving target
    # ==========================================================================
    # Classification: when coupled with moving target ramp, agent velocity is
    # NOT a standalone shrink. If target speed ramps together with agent cap,
    # the required physical chase velocity grows proportionally, keeping
    # normalized action magnitude ~0.7-1.0 throughout. Net effect becomes
    # an expand+shift (complex) — upward σ pressure, not downward.
    #
    # HARD PHYSICAL CONSTRAINT: agent velocity cap must be ≥ target speed at
    # every point in the curriculum, or the policy cannot physically catch
    # the target. Coupling agent velocity and moving target ramps from the
    # same start step (20k) satisfies this at every intermediate progress level.
    # ==========================================================================

    agent_velocity_start_step: int = 20000
    """Step to start ramping agent max linear velocity.

    Coupled with moving_target_start_step so the agent always has at least
    enough velocity cap to catch the target at every curriculum progress.
    """

    agent_velocity_end_step: int = 60000
    """Step when agent velocity reaches its configured maximum.

    Coupled with moving_target_end_step so cap and target speed reach
    full range together.
    """

    # ==========================================================================
    # Safety — CBF collision avoidance (10-40k)
    # ==========================================================================

    safety_start_step: int = 10000
    """Step to start enforcing safety constraints (CBF penalty)."""

    safety_end_step: int = 40000
    """Step when safety penalties reach full scale."""

    # ==========================================================================
    # Target Dynamics (20-60k)
    # ==========================================================================

    moving_target_start_step: int = 20000
    """Step to start introducing target motion.

    Starts after velocity ramp has begun so the agent has some tilt
    compensation ability before the target starts moving.
    """

    moving_target_end_step: int = 60000
    """Step when target reaches maximum speed/maneuverability."""

    # ==========================================================================
    # Multi-Agent Coupling / Coordination (60-90k) — SHIFT stage
    # ==========================================================================
    # Classification: coordination is a SHIFT (moves the optimum laterally —
    # each agent must find a different viewpoint). §A.3 requires shifts to
    # run in mid-to-late Phase 2 with sufficient σ for exploration.
    #
    # Staggered to start at 60k (after moving target full at 60k) so:
    #   1. Sequential with moving target (§A.4 rule 3: no two shifts overlap)
    #   2. Sequential with agent velocity shrink ending at 60k (§A.4 rule 4)
    #   3. σ has grown through Phase 2 for exploration budget
    #   4. Matches 928b precedent (coord 60-100k)
    # ==========================================================================

    coordination_start_step: int = 60000
    """Step to start rewarding coordination (triangulation)."""

    coordination_end_step: int = 90000
    """Step when coordination rewards reach full scale."""

    # ==========================================================================
    # Observation Noise (80-120k) — after coordination shift completes
    # ==========================================================================

    noise_start_step: int = 80000
    """Step to start introducing observation noise.

    Delayed until coordination is 66%+ complete so the policy has a
    committed multi-agent strategy to defend. The 10k overlap with the
    coord tail is acceptable because coordination gradients dominate
    during that window.
    """

    noise_end_step: int = 120000
    """Step when noise reaches maximum realistic values."""

    # ==========================================================================
    # FP/FN — Detection Misses and False Positives
    # Uses a two-phase schedule:
    #   15k-80k: constant background at fp_fn_background_scale (0.1 by default)
    #   80k-120k: ramp from background to full calibrated rate
    # Rationale: the policy needs training coverage of bbox-empty states
    # during early training to develop hold-last-position and search behaviors.
    # The 0.1 background runs through the coord shift window (60-90k) so the
    # policy learns coordination in a mildly-corrupted environment already.
    # ==========================================================================

    fp_fn_background_start_step: int = 15000
    """Step to start constant-rate background FP/FN exposure.

    Starts at 15k (just after bootstrap completes) so the policy encounters
    bbox-empty states while learning base tracking, ensuring a learned
    hold-last-position response rather than the fragile 'bbox always valid'
    assumption.
    """

    fp_fn_background_scale: float = 0.1
    """Constant fp_fn_scale during the background phase (15k-80k).

    0.1 = 10% of calibrated rates. With fp_rate=0.0347, this gives ~0.35%
    effective FP rate per slot. Miss rate is size-dependent, roughly 1-5%
    effective depending on target size. Rare enough to not dominate loss,
    frequent enough for RNN sequences (length 32) to encounter bbox-empty.
    """

    fp_fn_ramp_start_step: int = 80000
    """Step to start ramping FP/FN from background (0.1) to full calibrated (1.0).

    Aligned with noise/delay onset — all observation corruption starts
    together after the coordination shift completes.
    """

    fp_fn_end_step: int = 120000
    """Step when FP/FN rates reach calibrated full values."""

    # Legacy alias — points to the start of the background phase
    fp_fn_start_step: int = 15000
    """[DEPRECATED] Use fp_fn_background_start_step."""

    # ==========================================================================
    # Random Delay + Staleness (80-120k, skip fixed delay)
    # ==========================================================================

    fixed_delay_start_step: int = 0
    """[UNUSED] Fixed delay phase removed — random delay handles all latency."""

    fixed_delay_end_step: int = 0
    """[UNUSED] Fixed delay phase removed."""

    random_delay_start_step: int = 80000
    """Step to start introducing random delay with staleness.

    Aligned with noise onset at 80k — all non-FP/FN observation corruption
    starts together after the coordination shift has committed.
    """

    random_delay_end_step: int = 120000
    """Step when random delay variance reaches maximum."""

    # ==========================================================================
    # Dropout (90-130k) — starts after iid-side observation corruption is underway
    # ==========================================================================

    dropout_start_step: int = 90000
    """Step to start introducing i.i.d. detection dropout."""

    dropout_end_step: int = 130000
    """Step when dropout reaches maximum rate."""

    # ==========================================================================
    # Burst Dropout (100-130k)
    # ==========================================================================

    burst_dropout_start_step: int = 100000
    """Step to start introducing burst dropout (correlated packet loss).

    Starts after iid dropout begins so the policy first handles single-frame
    drops before sustained blackouts.
    """

    burst_dropout_end_step: int = 130000
    """Step when burst dropout onset probability reaches target value."""

    # Legacy aliases
    delay_start_step: int = 80000
    """[DEPRECATED] Use random_delay_start_step."""

    delay_end_step: int = 120000
    """[DEPRECATED] Use random_delay_end_step."""

    # ==========================================================================
    # Dynamics Randomization (30-80k)
    # ==========================================================================

    dynamics_start_step: int = 30000
    """Step to start dynamics randomization (mass, inertia, gains).

    Starts after base tracking is established (bootstrap complete + first 20k
    of velocity/safety/tracking ramps). Physics DR is filtered through the
    controller, so early low-rate DR is barely perceived, but it must not
    overlap with bootstrap.
    """

    dynamics_end_step: int = 80000
    """Step when dynamics randomization reaches full range."""

    # ==========================================================================
    # Task Reward Level Curriculum (FIM → GT-anchored → Composite)
    # ==========================================================================

    task_level_2_start_step: int = 20000
    """Step to begin transitioning from Level 1 (FIM) to Level 2 (GT-anchored)."""

    task_level_2_end_step: int = 40000
    """Step when Level 2 fully replaces Level 1."""

    task_level_3_start_step: int = 40000
    """Step to begin blending in E2E component (Level 3 composite)."""

    task_level_3_end_step: int = 60000
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

        Returns 'random' from step 0 — delay magnitude is controlled by
        progress (0.0 at start → 1.0 at random_delay_end_step). The old
        none→fixed→random state machine is removed; the progress ramp
        naturally passes through negligible-delay territory.
        """
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
        """Get FP/FN curriculum scale using the three-phase schedule.

        - Before fp_fn_background_start_step: 0.0 (no FP/FN)
        - Background phase: constant fp_fn_background_scale (flat low rate)
        - Ramp phase: linear ramp from background_scale to 1.0
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
