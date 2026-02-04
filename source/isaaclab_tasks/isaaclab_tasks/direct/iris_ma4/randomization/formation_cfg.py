# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration classes for distance-based formation generation.

The primary difficulty factor is distance to target, controlled via curriculum scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from isaaclab.utils import configclass


@configclass
class DistanceBasedFormationCfg:
    """Configuration for distance-based formation and target generation.

    The primary difficulty factor is distance to target, controlled via curriculum.

    Key Design:
        - Distance is the PRIMARY curriculum factor
        - Target is placed at approximately mean agent height (simplified)
        - Formation geometry (inter-agent distances) affects triangulation baseline
        - All agents guaranteed to be within [distance_min, distance_max] of target

    Curriculum Behavior:
        - At scale_factor=0: distance ≈ distance_min (easy, close range)
        - At scale_factor=1: distance ≈ distance_max (hard, far range)
    """

    # ==========================================================================
    # Distance Configuration (PRIMARY DIFFICULTY FACTOR)
    # ==========================================================================

    distance_min: float = 10.0
    """Minimum distance from formation center to target (meters).

    This is the distance used when scale_factor=0 (easiest difficulty).
    Should be set such that target is reliably detected (bbox > min_bbox_size).
    """

    distance_max: float = 50.0
    """Maximum distance from formation center to target (meters).

    This is the distance used when scale_factor=1 (hardest difficulty).
    Should be set based on camera resolution and target size for detection limits.
    """

    distance_variation: float = 0.1
    """Random variation in distance as fraction of computed distance.

    Actual distance = computed_distance * (1 ± distance_variation/2).
    Default 0.1 means ±5% variation.
    """

    # ==========================================================================
    # Formation Configuration (Agent Arrangement)
    # ==========================================================================

    formation_types: List[str] = field(default_factory=lambda: ["planar", "grid", "line"])
    """Available formation types.

    - "planar": Agents in a horizontal line, all at same height (easiest)
    - "grid": Agents in 2D grid pattern with optional height variation
    - "line": Agents along 3D line with curriculum-controlled Z spread
    """

    min_agent_separation: float = 5.0
    """Minimum distance between any two agents (meters).

    Enforced via collision resolution after formation generation.
    """

    max_agent_separation: float = 30.0
    """Maximum nominal distance between adjacent agents (meters).

    Actual separation is sampled uniformly between min and max.
    """

    formation_spread_scale: float = 1.0
    """Scale factor applied to formation spread.

    Can be used to curriculum-scale formation size independent of target distance.
    """

    # ==========================================================================
    # Height Configuration
    # ==========================================================================

    formation_height_min: float = 10.0
    """Minimum formation center height above ground (meters)."""

    formation_height_max: float = 30.0
    """Maximum formation center height above ground (meters)."""

    target_height_offset_min: float = -2.0
    """Minimum target height offset from mean agent height (meters).

    Negative means target can be below agents.
    """

    target_height_offset_max: float = 2.0
    """Maximum target height offset from mean agent height (meters).

    Positive means target can be above agents.
    """

    z_variation_min: float = 0.0
    """Minimum height variation between agents at scale_factor=0 (meters)."""

    z_variation_max: float = 3.0
    """Maximum height variation between agents at scale_factor=1 (meters)."""

    line_max_z_component: float = 0.3
    """Maximum Z component in line formation direction (slope constraint).

    At scale_factor=1, line direction can have up to this Z component.
    At scale_factor=0, line is horizontal (Z component = 0).
    """

    ground_clearance_min: float = 2.0
    """Minimum height above ground for any agent (meters)."""

    # ==========================================================================
    # Agent Orientation Configuration
    # ==========================================================================

    agent_faces_target: bool = True
    """If True, agents roughly face toward target. If False, random yaw."""

    agent_yaw_noise_std: float = 0.2
    """Standard deviation of yaw noise when facing target (radians).

    ~0.2 rad ≈ 11 degrees of noise around the target direction.
    """

    agent_random_yaw_range: Tuple[float, float] = (-math.pi / 2, math.pi / 2)
    """Yaw range when agent_faces_target=False (radians)."""

    # ==========================================================================
    # Velocity Initialization
    # ==========================================================================

    initial_velocity_scale: float = 0.2
    """Scale factor for initial velocities as fraction of max_vel.

    Velocities are sampled uniformly in [-max_vel * scale, +max_vel * scale].
    """

    initial_yaw_rate_scale: float = 0.2
    """Scale factor for initial yaw rate as fraction of max_yaw_rate."""

    # ==========================================================================
    # Safety and Validation
    # ==========================================================================

    max_collision_iterations: int = 10
    """Maximum iterations for enforcing minimum agent separation."""

    validate_distances: bool = True
    """If True, validate all agents are within distance bounds after generation."""


@dataclass
class FormationResult:
    """Result of formation generation.

    Contains all data needed by the environment for reset.
    """

    import torch

    agent_root_states: torch.Tensor
    """Agent root states tensor of shape [num_envs, num_agents, 13].

    Format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    - positions: [:, :, 0:3]
    - quaternions: [:, :, 3:7] (wxyz format)
    - linear velocities: [:, :, 7:10]
    - angular velocities: [:, :, 10:13]
    """

    target_position: torch.Tensor
    """Target position tensor of shape [num_envs, 3]."""

    formation_center: torch.Tensor
    """Formation center position tensor of shape [num_envs, 3]."""

    distances_to_target: torch.Tensor
    """Distance from each agent to target, shape [num_envs, num_agents]."""

    formation_types: List[str]
    """Formation type used for each environment."""
