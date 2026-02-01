# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for target movement system."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class TargetMovementCfg:
    """Configuration for target movement with acceleration-based velocity control.

    This module provides randomized target movement with:
    - Two flight path modes: linear (random direction) and circular (orbit)
    - Acceleration-based velocity control with damping
    - Geofencing with curriculum-based area expansion
    - Altitude constraints
    - Random velocity/path updates at configurable intervals
    """

    # ==========================================================================
    # Velocity Control
    # ==========================================================================

    max_speed: float = 5.0
    """Maximum target speed (m/s)."""

    max_angular_speed: float = 0.5
    """Maximum angular speed for yaw rotation (rad/s)."""

    max_acceleration: float = 3.0
    """Maximum acceleration magnitude (m/s^2)."""

    acceleration_scale: float = 2.0
    """Proportional gain for velocity tracking."""

    velocity_damping: float = 1.0
    """Velocity damping factor applied each step (0-1). Set to 1.0 for no damping."""

    # ==========================================================================
    # Flight Path Modes
    # ==========================================================================

    linear_weight: float = 0.5
    """Probability of linear mode vs circular mode (0-1)."""

    circular_radius_min: float = 10.0
    """Minimum circular orbit radius (m)."""

    circular_radius_max: float = 50.0
    """Maximum circular orbit radius (m)."""

    circular_height_min: float = 15.0
    """Minimum height above env origin for circular mode (m)."""

    circular_height_max: float = 30.0
    """Maximum height above env origin for circular mode (m)."""

    # ==========================================================================
    # Direction/Path Change (Curriculum-Scaled)
    # ==========================================================================

    direction_change_prob: float = 0.01
    """Probability of changing direction per step (for stochastic updates)."""

    use_timer_based_updates: bool = True
    """If True, use timer-based updates instead of probability-based."""

    # Curriculum-scaled update intervals: longer intervals early (easier), shorter late (harder)
    update_interval_min_start: float = 6.0
    """Minimum time between velocity updates at curriculum progress 0.0 (s)."""

    update_interval_min_end: float = 2.0
    """Minimum time between velocity updates at curriculum progress 1.0 (s)."""

    update_interval_max_start: float = 12.0
    """Maximum time between velocity updates at curriculum progress 0.0 (s)."""

    update_interval_max_end: float = 6.0
    """Maximum time between velocity updates at curriculum progress 1.0 (s)."""

    # ==========================================================================
    # Speed Curriculum
    # ==========================================================================

    speed_scale_start: float = 0.3
    """Speed scale factor at curriculum progress 0.0 (fraction of max_speed)."""

    speed_scale_end: float = 1.0
    """Speed scale factor at curriculum progress 1.0 (fraction of max_speed)."""

    # ==========================================================================
    # Geofencing (Curriculum-Scaled)
    # ==========================================================================

    geofence_min_size: float = 50.0
    """Minimum geofence half-width at curriculum progress 0.0 (m)."""

    geofence_max_size: float = 500.0
    """Maximum geofence half-width at curriculum progress 1.0 (m)."""

    geofence_bounce_factor: float = 0.8
    """Velocity reduction factor when bouncing off geofence (0-1)."""

    # ==========================================================================
    # Altitude Constraints
    # ==========================================================================

    min_altitude: float = 10.0
    """Minimum altitude above ground (m)."""

    altitude_bounce_velocity: float = 1.0
    """Upward velocity applied when hitting altitude floor (m/s)."""

    # ==========================================================================
    # Mode Switching
    # ==========================================================================

    mode_switch_prob: float = 0.001
    """Probability of switching between linear/circular mode per step."""

    allow_mode_switching: bool = True
    """Whether to allow runtime switching between modes."""
