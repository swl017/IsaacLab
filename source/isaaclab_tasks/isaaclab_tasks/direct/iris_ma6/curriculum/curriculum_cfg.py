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
    Target:        [──────────ramp──────────]full─────────────────────
    Coord:                  [────ramp────]full────────────────────────
    Noise:                                [──ramp──]full─────────────
    FixDelay:                                   [──ramp──]full───────
    RndDelay:                                         [──ramp──]full─
    Dropout:                                                [rmp]full
    Dynamics:                                                  [ramp]

    """

    # ==========================================================================
    # Global Curriculum Settings
    # ==========================================================================

    all_end_step: int = 180000
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

    dynamics_end_step: int = 200000
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
