# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for initial states randomization in iris_ma6.

This module provides configuration dataclasses for cylinder-based agent placement
with curriculum-driven randomization to prevent catastrophic forgetting.

Key Design:
    - Curriculum sampling: uniform(min, min + progress * (max - min))
    - All previous difficulty levels remain in sampling distribution
    - Designated observer concept ensures at least one agent sees target
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import torch

from isaaclab.utils import configclass


@configclass
class InitialStatesCfg:
    """Configuration for initial states randomization in iris_ma6.

    Curriculum Sampling Strategy:
        To prevent catastrophic forgetting, curriculum-controlled parameters
        use the sampling strategy:

            value ~ Uniform(min, min + progress * (max - min))

        This ensures that as training progresses:
        - At progress=0: Only easy samples (min values)
        - At progress=0.5: Easy to medium samples (min to midpoint)
        - At progress=1: Full range (min to max)

    Designated Observer:
        Each environment has exactly one "designated observer" - an agent that:
        1. Has its gimbal pointing at the target
        2. (Optionally) has its body facing the target direction
        3. Ensures at least one detection per reset
    """

    # ==========================================================================
    # Cylinder Configuration (Agent Placement)
    # ==========================================================================

    cylinder_diameter_min: float = 20.0
    """Minimum cylinder diameter (meters).

    At curriculum progress=0, agents are placed within this diameter.
    """

    cylinder_diameter_max: float = 100.0
    """Maximum cylinder diameter (meters).

    At curriculum progress=1, agents can be placed within up to this diameter.
    Sampled as: uniform(min, min + progress * (max - min))
    """

    cylinder_height_min: float = 10.0
    """Minimum cylinder base height above ground (meters)."""

    cylinder_height_max: float = 50.0
    """Maximum cylinder base height above ground (meters)."""

    cylinder_height_range_min: float = 2.0
    """Vertical spread at curriculum progress=0 (meters).

    0 = planar formation (all agents at same height). Matches iris_ma5 behavior
    at early training for easier gimbal learning.
    """

    cylinder_height_range_max: float = 10.0
    """Vertical spread at curriculum progress=1 (meters).

    Agents are distributed within [center_z, center_z + height_range].
    Sampled as: min + progress * (max - min).
    """

    agent_clearance: float = 5.0
    """Minimum distance between any two agents (meters).

    Enforced via rejection sampling during placement.
    """

    max_placement_retries: int = 50
    """Maximum retries for rejection sampling per agent."""

    # ==========================================================================
    # Target Configuration
    # ==========================================================================

    target_distance_min: float = 10.0
    """Minimum distance from cylinder center to target (meters).

    At curriculum progress=0, target is placed at this distance.
    """

    target_distance_max: float = 40.0
    """Maximum distance from cylinder center to target (meters).

    At curriculum progress=1, target can be placed up to this distance.
    Sampled as: uniform(min, min + progress * (max - min))
    """

    target_height_offset_min: float = -2.0
    """Minimum target height offset from cylinder center (meters)."""

    target_height_offset_max: float = 2.0
    """Maximum target height offset from cylinder center (meters)."""

    # ==========================================================================
    # Velocity Configuration
    # ==========================================================================

    agent_max_velocity: float = 10.0
    """Maximum agent initial velocity magnitude (m/s)."""

    agent_velocity_scale_max: float = 1.0
    """Maximum velocity scale factor at progress=1.

    Actual velocity = uniform(0, progress * scale_max) * max_velocity
    """

    target_max_velocity: float = 2.0
    """Maximum target initial velocity magnitude (m/s)."""

    target_velocity_scale_max: float = 1.0
    """Maximum target velocity scale factor at progress=1."""

    max_yaw_rate: float = math.radians(45.0)
    """Maximum initial yaw rate for agents (rad/s)."""

    # ==========================================================================
    # Agent Body Orientation Configuration
    # ==========================================================================

    designated_observer_faces_target: bool = True
    """Whether the designated observer's body faces toward target."""

    other_agents_orientation_mode: Literal["random", "face_target", "curriculum"] = "face_target"
    """Orientation mode for non-designated agents.

    - "random": Random yaw orientation
    - "face_target": All agents face toward target
    - "curriculum": Gradual transition from face_target (progress=0) to random (progress=1).
      With probability (1 - progress), agent faces target; otherwise random yaw.
    """

    orientation_noise_std: float = 0.2
    """Standard deviation of orientation noise (radians).

    Applied to body yaw when facing target. ~0.2 rad ≈ 11° matches iris_ma5.
    """

    # ==========================================================================
    # Gimbal Configuration
    # ==========================================================================

    gimbal_yaw_min: float = -math.pi
    """Minimum gimbal yaw angle (radians)."""

    gimbal_yaw_max: float = math.pi
    """Maximum gimbal yaw angle (radians)."""

    gimbal_pitch_min: float = -math.radians(45.0)
    """Minimum gimbal pitch angle (radians)."""

    gimbal_pitch_max: float = math.radians(45.0)
    """Maximum gimbal pitch angle (radians)."""

    gimbal_curriculum_mode: Literal["gradual", "threshold", "always_pointing"] = "always_pointing"
    """How non-observers transition from pointing to random gimbal.

    - "gradual": With probability (1-progress), agent points at target
    - "threshold": Switch to random at gimbal_randomization_threshold
    - "always_pointing": All agents always point at target (for debugging)
    """

    gimbal_randomization_threshold: float = 0.5
    """Curriculum progress threshold for "threshold" mode."""

    # ==========================================================================
    # Zoom Configuration
    # ==========================================================================

    zoom_min: float = 1.0
    """Minimum zoom level."""

    zoom_max: float = 10.0
    """Maximum zoom level."""

    zoom_initial_min: float = 1.0
    """Minimum initial zoom level for sampling."""

    zoom_initial_max_start: float = 1.0
    """Maximum initial zoom level at curriculum progress 0."""

    zoom_initial_max_end: float = 6.0
    """Maximum initial zoom level at curriculum progress 1."""

    # ==========================================================================
    # Designated Observer Selection
    # ==========================================================================

    designated_observer_mode: Literal["random", "fixed", "rotating"] = "random"
    """How to select the designated observer per environment.

    - "random": Random agent each reset
    - "fixed": Always agent 0
    - "rotating": Cycle through agents across resets
    """


@dataclass
class InitialStatesResult:
    """Result of initial states generation.

    Contains all tensors needed by the environment for reset.
    All agent tensors have shape [num_envs, num_agents, ...].
    All target tensors have shape [num_envs, ...].
    """

    # ==========================================================================
    # Agent States
    # ==========================================================================

    agent_positions: torch.Tensor
    """Agent world positions. Shape: [num_envs, num_agents, 3]."""

    agent_orientations: torch.Tensor
    """Agent orientations as quaternions (wxyz). Shape: [num_envs, num_agents, 4]."""

    agent_linear_velocities: torch.Tensor
    """Agent linear velocities in world frame. Shape: [num_envs, num_agents, 3]."""

    agent_angular_velocities: torch.Tensor
    """Agent angular velocities in world frame. Shape: [num_envs, num_agents, 3]."""

    # ==========================================================================
    # Target States
    # ==========================================================================

    target_positions: torch.Tensor
    """Target world positions. Shape: [num_envs, 3]."""

    target_orientations: torch.Tensor
    """Target orientations as quaternions (wxyz). Shape: [num_envs, 4]."""

    target_velocities: torch.Tensor
    """Target velocities (linear + angular). Shape: [num_envs, 6]."""

    # ==========================================================================
    # Gimbal and Zoom States
    # ==========================================================================

    gimbal_joint_positions: torch.Tensor
    """Gimbal joint positions [yaw, roll, pitch]. Shape: [num_envs, num_agents, 3].

    Order matches joint order in robot articulation.
    """

    zoom_levels: torch.Tensor
    """Zoom levels per agent. Shape: [num_envs, num_agents]."""

    # ==========================================================================
    # Metadata
    # ==========================================================================

    designated_observer_idx: torch.Tensor
    """Index of designated observer per environment. Shape: [num_envs]."""

    cylinder_centers: torch.Tensor
    """Center position of each cylinder. Shape: [num_envs, 3]."""

    distances_to_target: torch.Tensor
    """Distance from each agent to target. Shape: [num_envs, num_agents]."""
