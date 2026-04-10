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

    All-from-zero schedule: every difficulty knob ramps from step 0. No free
    warmup, no curriculum shocks, no sigma collapse window. The policy never
    sees a trivially easy or observation-clean world.

    Step:    0k        40k        60k       100k       320k
             |          |          |          |          |
    AgentVel:[────────ramp────────]full───────────────────
    Safety:  [────────ramp────────]full───────────────────
    Tracking:[────────ramp────────]full───────────────────
    Target:  [──────────────ramp──────────────]full───────
    Coord:   [──────────────ramp──────────────]full───────
    Noise:   [──────────────────────ramp──────────────]full
    Delay:   [──────────────────────ramp──────────────]full  (random from 0)
    Dropout: [──────────────────────ramp──────────────]full
    Burst:   [──────────────────────ramp──────────────]full
    FP/FN:   [──────────────────────ramp──────────────]full
    DynDR:   [──────────────────────ramp──────────────]full
    Post:                                    [───220k────]

    """

    # ==========================================================================
    # Global Curriculum Settings
    # ==========================================================================

    all_end_step: int = 320000
    """Step at which all curriculum factors reach their final values."""

    # ==========================================================================
    # Tracking & Formation (0-40k ramp)
    # ==========================================================================

    tracking_start_step: int = 0
    """Step to start increasing formation/initialization difficulty."""

    tracking_end_step: int = 40000
    """Step when formation/initialization difficulty reaches maximum."""

    # ==========================================================================
    # Agent Velocity Ramp (0-40k)
    # ==========================================================================

    agent_velocity_start_step: int = 0
    """Step to start ramping agent max linear velocity."""

    agent_velocity_end_step: int = 40000
    """Step when agent velocity reaches its configured maximum."""

    # ==========================================================================
    # Safety — CBF collision avoidance (0-40k)
    # ==========================================================================

    safety_start_step: int = 0
    """Step to start enforcing safety constraints (CBF penalty)."""

    safety_end_step: int = 40000
    """Step when safety penalties reach full scale."""

    # ==========================================================================
    # Target Dynamics (0-60k)
    # ==========================================================================

    moving_target_start_step: int = 0
    """Step to start introducing target motion."""

    moving_target_end_step: int = 60000
    """Step when target reaches maximum speed/maneuverability."""

    # ==========================================================================
    # Multi-Agent Coupling / Coordination (0-60k)
    # ==========================================================================

    coordination_start_step: int = 0
    """Step to start rewarding coordination (triangulation)."""

    coordination_end_step: int = 60000
    """Step when coordination rewards reach full scale."""

    # ==========================================================================
    # Observation Noise (0-100k)
    # ==========================================================================

    noise_start_step: int = 0
    """Step to start introducing observation noise.

    Ramps from zero — early steps have negligible noise.
    """

    noise_end_step: int = 100000
    """Step when noise reaches maximum realistic values."""

    # ==========================================================================
    # FP/FN — Detection Misses and False Positives (0-100k)
    # ==========================================================================

    fp_fn_start_step: int = 0
    """Step to start introducing false positives and miss rate.

    From step 0 so the policy never knows a world without occasional
    missed detections — prevents overfitting to 'bbox always valid'.
    """

    fp_fn_end_step: int = 100000
    """Step when FP/FN rates reach calibrated values."""

    # ==========================================================================
    # Random Delay + Staleness (0-100k, skip fixed delay)
    # ==========================================================================

    fixed_delay_start_step: int = 0
    """[UNUSED] Fixed delay phase removed — random delay ramps from zero,
    naturally passing through small-delay territory."""

    fixed_delay_end_step: int = 0
    """[UNUSED] Fixed delay phase removed."""

    random_delay_start_step: int = 0
    """Step to start introducing random delay with staleness.

    Ramps from zero — at low progress, delays are negligible.
    Replaces the old none→fixed→random state machine.
    """

    random_delay_end_step: int = 100000
    """Step when random delay variance reaches maximum."""

    # ==========================================================================
    # Dropout (0-100k)
    # ==========================================================================

    dropout_start_step: int = 0
    """Step to start introducing dropout."""

    dropout_end_step: int = 100000
    """Step when dropout reaches maximum rate."""

    # ==========================================================================
    # Burst Dropout (0-100k)
    # ==========================================================================

    burst_dropout_start_step: int = 0
    """Step to start introducing burst dropout (correlated packet loss).

    At low progress, p_onset ≈ 0 so bursts never trigger.
    Becomes meaningful around 30-40% progress.
    """

    burst_dropout_end_step: int = 100000
    """Step when burst dropout onset probability reaches target value."""

    # Legacy aliases
    delay_start_step: int = 0
    """[DEPRECATED] Random delay starts from step 0."""

    delay_end_step: int = 100000
    """[DEPRECATED] Use random_delay_end_step."""

    # ==========================================================================
    # Dynamics Randomization (0-100k)
    # ==========================================================================

    dynamics_start_step: int = 0
    """Step to start dynamics randomization (mass, inertia, gains).

    From step 0 with 100k ramp — early steps have negligible randomization.
    The policy never overfits to nominal dynamics.
    """

    dynamics_end_step: int = 100000
    """Step when dynamics randomization reaches full range."""

    # ==========================================================================
    # Task Reward Level Curriculum (FIM → GT-anchored → Composite)
    # ==========================================================================

    task_level_2_start_step: int = 0
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
        """Get progress within FP/FN phase [0, 1]."""
        return self.get_progress(current_step, self.fp_fn_start_step, self.fp_fn_end_step)

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
