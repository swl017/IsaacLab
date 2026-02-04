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
    """

    # ==========================================================================
    # Global Curriculum Settings
    # ==========================================================================

    all_end_step: int = 200000
    """Step at which all curriculum factors reach their final values."""

    # ==========================================================================
    # Phase 0/1: Geometry + Formation (clean observations, static target)
    # ==========================================================================

    tracking_start_step: int = 0
    """Step to start increasing formation/initialization difficulty."""

    tracking_end_step: int = 60000
    """Step when formation/initialization difficulty reaches maximum."""

    # ==========================================================================
    # Phase 2: Noise Introduction (before delay)
    # ==========================================================================

    noise_start_step: int = 80000
    """Step to start introducing observation noise."""

    noise_end_step: int = 100000
    """Step when noise reaches maximum realistic values."""

    # ==========================================================================
    # Phase 3: Fixed Delay (Deterministic Latency)
    # ==========================================================================

    fixed_delay_start_step: int = 100000
    """Step to start introducing fixed (deterministic) delay."""

    fixed_delay_end_step: int = 130000
    """Step when fixed delay reaches maximum value (uses config latency means)."""

    # NOTE: Fixed delay magnitude uses existing detection_latency_mean and
    # inter_agent_comm_latency_mean from MultiAgentDelaySystemV2Cfg.

    # ==========================================================================
    # Phase 4: Random Delay + Staleness
    # ==========================================================================

    random_delay_start_step: int = 130000
    """Step to transition from fixed to random delay."""

    random_delay_end_step: int = 160000
    """Step when random delay variance reaches maximum."""

    # NOTE: Random delay uses existing latency_std parameters from config.

    # ==========================================================================
    # Phase 5: Dropout (after delay phases)
    # ==========================================================================

    dropout_start_step: int = 160000
    """Step to start introducing dropout."""

    dropout_end_step: int = 200000
    """Step when dropout reaches maximum rate."""

    # Legacy alias for backward compatibility
    delay_start_step: int = 80000
    """[DEPRECATED] Use noise_start_step, fixed_delay_start_step, etc."""

    delay_end_step: int = 160000
    """[DEPRECATED] Use random_delay_end_step."""

    # ==========================================================================
    # Phase 0: Multi-Agent Coupling (triangulation geometry)
    # ==========================================================================

    coordination_start_step: int = 0
    """Step to start rewarding coordination (triangulation)."""

    coordination_end_step: int = 20000
    """Step when coordination rewards reach full scale."""

    # ==========================================================================
    # Phase 0: Safety (collision/TTC) should be present from the start
    # ==========================================================================

    safety_start_step: int = 0
    """Step to start enforcing safety constraints (collision, TTC)."""

    safety_end_step: int = 20000
    """Step when safety penalties reach full scale."""

    # ==========================================================================
    # Phase 1: Target Dynamics (still clean observations)
    # ==========================================================================

    moving_target_start_step: int = 20000
    """Step to start introducing target motion."""

    moving_target_end_step: int = 80000
    """Step when target reaches maximum speed/maneuverability."""

    # ==========================================================================
    # Phase 3+: Robot/Camera Dynamics Randomization (last)
    # ==========================================================================

    dynamics_start_step: int = 160000
    """Step to start dynamics randomization (mass, inertia)."""

    dynamics_end_step: int = 200000
    """Step when dynamics randomization reaches full range."""

    # ==========================================================================
    # Zoom Capability Curriculum
    # ==========================================================================

    zoom_phase1_end_step: int = 20000
    """End of phase 1: limited zoom (geometry-first phase)."""

    zoom_phase2_end_step: int = 80000
    """End of phase 2: zoom ramp during target dynamics phase."""

    zoom_phase3_end_step: int = 160000
    """End of phase 3: zoom ramp during observability degradation phase."""

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
        - Phase 1 (0 to zoom_phase1_end_step): Locked at zoom_phase1_max
        - Phase 2 (phase1_end to phase2_end): Ramp from phase1_max to phase2_max
        - Phase 3 (phase2_end to phase3_end): Ramp from phase2_max to phase3_max
        - Phase 4 (phase3_end to all_end): Ramp from phase3_max to zoom_final_max

        Args:
            current_step: Current training step.

        Returns:
            Maximum allowed zoom level for this training phase.
        """
        if current_step < self.zoom_phase1_end_step:
            # Phase 1: Limited zoom
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
