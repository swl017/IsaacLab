# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-A Minimum Viable Task (MVT) config for iris_ma6.

Subclasses the working IrisMA6TestEnvCfg and narrows every non-observation-
corruption axis so the policy only has to learn robustness to noise/delay/dropout.
Curriculum ramps for task geometry, target motion, coordination, and dynamics are
pushed past all_end_step so they never fire during an MVT run.

Axes kept random (required for sim-to-sim transfer to IsaacSim+PX4+ROS2):
  - Cylinder vertical spread (2..5 m)
  - Target height offset (0..4 m)
  - Orientation noise (0.2 rad)

Everything else is collapsed to the easy end of the existing distribution.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from .curriculum import CurriculumCfg
from .initial_states import InitialStatesCfg
from .iris_ma_env6_test_cfg import IrisMA6TestEnvCfg


@configclass
class IrisMA6MVTCfg(IrisMA6TestEnvCfg):
    """Phase-A Minimum Viable Task configuration."""

    # ==========================================================================
    # Motion envelope — halved peak agent speed
    # ==========================================================================

    max_lin_vel: float = 5.0
    max_lin_vel_min: float = 2.0

    # ==========================================================================
    # Initial states — collapse to easy slice, keep s2s-critical randomizations
    # ==========================================================================

    initial_states: InitialStatesCfg = InitialStatesCfg(
        # formation geometry
        cylinder_diameter_min=20.0,
        cylinder_diameter_max=25.0,
        cylinder_height_min=20.0,
        cylinder_height_max=25.0,
        cylinder_height_range_min=2.0,
        cylinder_height_range_max=5.0,  # KEPT baseline for sim-to-sim
        # target placement
        target_distance_min=10.0,
        target_distance_max=15.0,
        target_height_offset_min=0.0,
        target_height_offset_max=4.0,  # KEPT baseline for sim-to-sim
        # velocities (halved)
        agent_max_velocity=5.0,
        agent_velocity_scale_max=0.5,
        target_max_velocity=0.5,
        target_velocity_scale_max=0.5,
        max_yaw_rate=math.radians(20.0),
        # orientation
        designated_observer_faces_target=True,
        other_agents_orientation_mode="face_target",
        orientation_noise_std=0.2,  # KEPT baseline for sim-to-sim
        # gimbal
        gimbal_curriculum_mode="always_pointing",
        # zoom
        zoom_min=1.0,
        zoom_max=6.0,
        zoom_initial_min=1.0,
        zoom_initial_max_start=1.0,
        zoom_initial_max_end=2.0,
        # observer role
        designated_observer_mode="fixed",
    )

    # ==========================================================================
    # Curriculum — freeze non-obs axes, keep obs-corruption axes live
    # ==========================================================================

    curriculum: CurriculumCfg = CurriculumCfg(
        all_end_step=400_000,
        # Non-obs ramps pushed past all_end_step (progress stays at 0)
        tracking_start_step=2_000_000,
        tracking_end_step=2_000_000,
        agent_velocity_start_step=2_000_000,
        agent_velocity_end_step=2_000_000,
        moving_target_start_step=2_000_000,
        moving_target_end_step=2_000_000,
        coordination_start_step=2_000_000,
        coordination_end_step=2_000_000,
        safety_start_step=2_000_000,
        safety_end_step=2_000_000,
        dynamics_start_step=2_000_000,
        dynamics_end_step=2_000_000,
        task_level_2_start_step=2_000_000,
        task_level_2_end_step=2_000_000,
        task_level_3_start_step=2_000_000,
        task_level_3_end_step=2_000_000,
        # Observation-corruption ramps stay live (this is the study)
        noise_start_step=100_000,
        noise_end_step=120_000,
        fixed_delay_start_step=120_000,
        fixed_delay_end_step=140_000,
        random_delay_start_step=140_000,
        random_delay_end_step=160_000,
        dropout_start_step=160_000,
        dropout_end_step=180_000,
        burst_dropout_start_step=200_000,
        burst_dropout_end_step=220_000,
        # FP/FN deferred (same as base)
        fp_fn_background_start_step=2_200_000,
        fp_fn_ramp_start_step=2_200_000,
        fp_fn_end_step=3_000_000,
    )

    # ==========================================================================
    # Reward level — stay on Level-1 FIM proxy (no level curriculum)
    # ==========================================================================

    task_reward_level: int = 1
    curriculum_task_levels: bool = False
