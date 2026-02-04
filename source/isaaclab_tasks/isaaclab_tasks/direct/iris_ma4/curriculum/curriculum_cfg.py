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

    tracking_end_step: int = 60000
    """Step when tracking difficulty reaches maximum."""

    # ==========================================================================
    # Phase 2: Delay and Noise System
    # ==========================================================================

    delay_start_step: int = 10000
    """Step to start introducing communication delays."""

    delay_end_step: int = 80000
    """Step when delays reach maximum realistic values."""

    # ==========================================================================
    # Phase 3: Multi-Agent Coordination
    # ==========================================================================

    coordination_start_step: int = 40000
    """Step to start rewarding coordination (triangulation).

    FIX: Moved from 80k to 40k to provide earlier coordination signal.
    This prevents agent from overfitting to single-agent tracking for too long.
    """

    coordination_end_step: int = 60000
    """Step when coordination rewards reach full scale.

    FIX: Moved from 100k to 60k to match earlier start.
    """

    # ==========================================================================
    # Safety Curriculum
    # ==========================================================================

    safety_start_step: int = 150000
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

    # ==========================================================================
    # Zoom Capability Curriculum
    # ==========================================================================

    zoom_phase1_end_step: int = 30000
    """End of phase 1: zoom locked at 1.0 (no zoom capability)."""

    zoom_phase2_end_step: int = 80000
    """End of phase 2: zoom allowed up to 2.0 (conservative zoom)."""

    zoom_phase3_end_step: int = 150000
    """End of phase 3: zoom allowed up to 4.0 (moderate zoom)."""

    zoom_phase1_max: float = 1.5
    """Maximum zoom in phase 1.

    FIX: Changed from 3.0 to 1.5 (was contradicting "no zoom" comment).
    Allowing limited zoom (1.0-1.5) from start prevents action dead zone.
    """

    zoom_phase2_max: float = 6.0
    """Maximum zoom in phase 2 (conservative)."""

    zoom_phase3_max: float = 6.0
    """Maximum zoom in phase 3 (moderate)."""

    zoom_final_max: float = 6.0
    """Final maximum zoom level (full capability)."""

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

    def get_max_zoom_level(self, current_step: int) -> float:
        """Get curriculum-scaled maximum zoom level.

        The zoom curriculum has 4 phases with piecewise linear interpolation:
        - Phase 1 (0 to zoom_phase1_end_step): Locked at zoom_phase1_max (1.0)
        - Phase 2 (phase1_end to phase2_end): Ramp from phase1_max to phase2_max
        - Phase 3 (phase2_end to phase3_end): Ramp from phase2_max to phase3_max
        - Phase 4 (phase3_end to all_end): Ramp from phase3_max to zoom_final_max

        Args:
            current_step: Current training step.

        Returns:
            Maximum allowed zoom level for this training phase.
        """
        if current_step < self.zoom_phase1_end_step:
            # Phase 1: No zoom
            return self.zoom_phase1_max
        elif current_step < self.zoom_phase2_end_step:
            # Phase 2: Linear interpolation from phase1 to phase2
            progress = (current_step - self.zoom_phase1_end_step) / (
                self.zoom_phase2_end_step - self.zoom_phase1_end_step
            )
            return self.zoom_phase1_max + progress * (self.zoom_phase2_max - self.zoom_phase1_max)
        elif current_step < self.zoom_phase3_end_step:
            # Phase 3: Linear interpolation from phase2 to phase3
            progress = (current_step - self.zoom_phase2_end_step) / (
                self.zoom_phase3_end_step - self.zoom_phase2_end_step
            )
            return self.zoom_phase2_max + progress * (self.zoom_phase3_max - self.zoom_phase2_max)
        elif current_step < self.all_end_step:
            # Phase 4: Linear interpolation from phase3 to final
            progress = (current_step - self.zoom_phase3_end_step) / (
                self.all_end_step - self.zoom_phase3_end_step
            )
            return self.zoom_phase3_max + progress * (self.zoom_final_max - self.zoom_phase3_max)
        else:
            # After all_end_step: Full zoom capability
            return self.zoom_final_max
