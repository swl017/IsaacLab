# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for physics-based target movement using DroneController."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from isaaclab.utils import configclass


@dataclass
class BehaviorProfile:
    """Attacker behavior profile definition.

    Defines speed, evasion agility, and path type preferences for
    different attacker behavior profiles.
    """

    name: str
    """Profile name for identification."""

    speed_multiplier: float
    """Speed multiplier relative to max_speed (0-2)."""

    evasion_agility: float
    """Evasion intensity (0 = no evasion, 1 = maximum)."""

    path_type_weights: Tuple[float, float, float]
    """Weights for (direct, offset, low_altitude) path types. Must sum to 1."""


# Predefined behavior profiles following iris_ma6_env_spec.md §4.2.3
BEHAVIOR_PROFILES = {
    "kamikaze": BehaviorProfile(
        name="kamikaze",
        speed_multiplier=1.5,  # Fast: ~12 m/s at full curriculum
        evasion_agility=0.0,  # No evasion
        path_type_weights=(1.0, 0.0, 0.0),  # Direct only
    ),
    "standard": BehaviorProfile(
        name="standard",
        speed_multiplier=1.0,  # Medium: ~7 m/s
        evasion_agility=0.5,  # Medium evasion
        path_type_weights=(0.2, 0.6, 0.2),  # Mostly offset
    ),
    "evasive": BehaviorProfile(
        name="evasive",
        speed_multiplier=1.0,  # Medium: ~7 m/s
        evasion_agility=0.9,  # High evasion
        path_type_weights=(0.1, 0.7, 0.2),  # Mostly offset
    ),
    "stealth": BehaviorProfile(
        name="stealth",
        speed_multiplier=0.6,  # Slow: ~3 m/s
        evasion_agility=0.5,  # Medium evasion
        path_type_weights=(0.0, 0.3, 0.7),  # Mostly low-altitude
    ),
}


@configclass
class TargetControllerCfg:
    """Configuration for physics-based target movement.

    This module provides target movement using the same DroneController
    architecture as agents, producing realistic physics-based motion
    instead of direct velocity writes.

    Key differences from iris_ma5 TargetMovement:
    - Uses DroneController for force/torque generation
    - Supports attacker behavior modes (approach, evade)
    - Integrates with FSM state management
    - Lower controller gains for smoother target motion
    """

    # ==========================================================================
    # DroneController Override (Simplified for targets)
    # ==========================================================================

    use_simplified_controller: bool = True
    """Use simplified controller gains (less aggressive than agents)."""

    velocity_kp: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Velocity P gains for targets [x, y, z]. Lower than agents (4.5) for smoother motion."""

    velocity_ki: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    """Velocity I gains for targets [x, y, z]."""

    velocity_integral_limit: Tuple[float, float, float] = (2.0, 2.0, 1.0)
    """Velocity integral limits [x, y, z] [m/s]."""

    max_tilt: float = 30.0
    """Maximum tilt angle for targets [deg]. Lower than agents (45) for stability."""

    max_lin_vel: float = 15.0
    """Maximum linear velocity for controller [m/s]."""

    attitude_kp: Tuple[float, float, float] = (4.0, 4.0, 2.0)
    """Attitude P gains [roll, pitch, yaw]. Lower than agents (6.5)."""

    rate_kp: Tuple[float, float, float] = (0.10, 0.10, 0.15)
    """Rate P gains [roll, pitch, yaw]. Lower than agents (0.15)."""

    rate_ki: Tuple[float, float, float] = (0.10, 0.10, 0.05)
    """Rate I gains [roll, pitch, yaw]."""

    rate_kd: Tuple[float, float, float] = (0.002, 0.002, 0.0)
    """Rate D gains [roll, pitch, yaw]."""

    motor_tau: float = 0.015
    """Motor time constant [s]. Slightly slower than agents (0.01)."""

    aerodynamics_fidelity_level: int = 0
    """Aerodynamic fidelity level (0=disabled, 1=drag, 2=wind, 3=rotor effects)."""

    control_dt: float = 0.01
    """Control loop timestep [s]. Should match the sim dt at which targets are controlled."""

    # ==========================================================================
    # Speed Control (Curriculum-Scaled)
    # ==========================================================================

    max_speed_start: float = 1.0
    """Maximum speed at curriculum progress 0 [m/s]."""

    max_speed_end: float = 5.0
    """Maximum speed at curriculum progress 1 [m/s]."""

    max_acceleration: float = 5.0
    """Maximum acceleration [m/s^2]."""

    # ==========================================================================
    # Linear Mode Parameters
    # ==========================================================================

    linear_weight: float = 0.5
    """Probability of linear mode vs circular mode (for iris_ma5 compatibility)."""

    update_interval_min_start: float = 6.0
    """Minimum direction change interval at progress 0 [s]."""

    update_interval_min_end: float = 2.0
    """Minimum direction change interval at progress 1 [s]."""

    update_interval_max_start: float = 12.0
    """Maximum direction change interval at progress 0 [s]."""

    update_interval_max_end: float = 4.0
    """Maximum direction change interval at progress 1 [s]."""

    # ==========================================================================
    # Circular Mode Parameters
    # ==========================================================================

    circular_radius_min: float = 15.0
    """Minimum orbit radius [m]."""

    circular_radius_max: float = 60.0
    """Maximum orbit radius [m]."""

    circular_height_min: float = 15.0
    """Minimum orbit altitude above center [m]."""

    circular_height_max: float = 40.0
    """Maximum orbit altitude above center [m]."""

    max_angular_speed: float = 0.3
    """Maximum angular speed for circular orbit [rad/s]."""

    circular_position_gain: float = 1.0
    """Proportional gain for circular orbit tracking."""

    # ==========================================================================
    # Approach Mode Parameters (Attacker)
    # ==========================================================================

    approach_speed_base: float = 7.0
    """Base approach speed [m/s]."""

    approach_path_direct_weight: float = 0.3
    """Probability of direct approach path."""

    approach_path_offset_weight: float = 0.4
    """Probability of offset approach path."""

    approach_path_low_alt_weight: float = 0.3
    """Probability of low-altitude approach path."""

    offset_waypoint_distance: float = 100.0
    """Distance of offset waypoints from direct path [m]."""

    max_waypoints: int = 3
    """Maximum number of waypoints for offset path."""

    waypoint_reach_threshold: float = 5.0
    """Distance to consider waypoint reached [m]."""

    low_altitude_target: float = 7.5
    """Target altitude for low-altitude approach [m]."""

    low_altitude_gain: float = 0.5
    """Proportional gain for low-altitude regulation."""

    # ==========================================================================
    # Evasion Mode Parameters (Attacker)
    # ==========================================================================

    evade_trigger_distance: float = 50.0
    """Distance at which evasion triggers [m]."""

    evade_duration_min: float = 2.0
    """Minimum evasion duration [s]."""

    evade_duration_max: float = 5.0
    """Maximum evasion duration [s]."""

    evasion_agility_min: float = 0.3
    """Minimum evasion agility (speed multiplier)."""

    evasion_agility_max: float = 1.0
    """Maximum evasion agility (speed multiplier)."""

    evasion_escape_weight: float = 0.3
    """Weight of escape direction vs perpendicular direction (0-1)."""

    evasion_start_progress: float = 0.3
    """Curriculum progress at which evasion becomes enabled."""

    # ==========================================================================
    # Geofencing
    # ==========================================================================

    geofence_min_size: float = 50.0
    """Geofence half-width at progress 0 [m]."""

    geofence_max_size: float = 200.0
    """Geofence half-width at progress 1 [m]."""

    geofence_margin: float = 10.0
    """Distance from boundary to start slowing [m]."""

    geofence_bounce_factor: float = 0.8
    """Velocity reduction factor when bouncing off geofence (0-1)."""

    # ==========================================================================
    # Altitude Constraints
    # ==========================================================================

    min_altitude: float = 10.0
    """Minimum altitude above ground [m]."""

    max_altitude: float = 40.0
    """Maximum altitude above ground [m]."""

    # ==========================================================================
    # Behavior Profiles (Attacker Domain Randomization)
    # ==========================================================================

    behavior_kamikaze_weight: float = 0.2
    """Probability of kamikaze profile (fast, no evasion)."""

    behavior_standard_weight: float = 0.4
    """Probability of standard profile (medium speed, medium evasion)."""

    behavior_evasive_weight: float = 0.25
    """Probability of evasive profile (medium speed, high evasion)."""

    behavior_stealth_weight: float = 0.15
    """Probability of stealth profile (slow, low altitude)."""

    # ==========================================================================
    # Velocity Mode Selection (for iris_ma5 compatibility)
    # ==========================================================================

    use_attacker_mode: bool = False
    """If True, use approach/evade modes. If False, use linear/circular only."""

    default_velocity_mode: str = "linear"
    """Default velocity mode: 'linear', 'circular', or 'approach'."""
