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

    Training phases are designed to introduce complexity gradually:
    - Phase 1: Single-agent tracking (basic skills)
    - Phase 2: Delay and noise system (realism)
    - Phase 3: Multi-agent coordination (teamwork)

    The phases can overlap, allowing gradual transitions.
    """

    # ==========================================================================
    # Global Curriculum Settings
    # ==========================================================================

    all_end_step: int = 200000
    """Step at which all curriculum factors reach their final values."""

    # ==========================================================================
    # Phase 1: Single-Agent Tracking
    # ==========================================================================

    tracking_start_step: int = 10000
    """Step to start increasing tracking difficulty."""

    tracking_end_step: int = 80000
    """Step when tracking difficulty reaches maximum."""

    # ==========================================================================
    # Phase 2: Delay and Noise System
    # ==========================================================================

    delay_start_step: int = 60000
    """Step to start introducing communication delays."""

    delay_end_step: int = 150000
    """Step when delays reach maximum realistic values."""

    # ==========================================================================
    # Phase 3: Multi-Agent Coordination
    # ==========================================================================

    coordination_start_step: int = 120000
    """Step to start rewarding coordination (triangulation)."""

    coordination_end_step: int = 200000
    """Step when coordination rewards reach full scale."""

    # ==========================================================================
    # Safety Curriculum
    # ==========================================================================

    safety_start_step: int = 180000
    """Step to start enforcing safety constraints (collision, TTC)."""

    safety_end_step: int = 230000
    """Step when safety penalties reach full scale."""

    # ==========================================================================
    # Moving Target Curriculum
    # ==========================================================================

    moving_target_start_step: int = 10000
    """Step to start introducing target motion."""

    moving_target_end_step: int = 80000
    """Step when target reaches maximum speed/maneuverability."""

    # ==========================================================================
    # Dynamics Randomization Curriculum
    # ==========================================================================

    dynamics_start_step: int = 40000
    """Step to start dynamics randomization (mass, inertia)."""

    dynamics_end_step: int = 120000
    """Step when dynamics randomization reaches full range."""

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
