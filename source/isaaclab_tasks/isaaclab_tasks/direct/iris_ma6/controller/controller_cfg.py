# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composite configuration for drone controller."""

from __future__ import annotations

from isaaclab.utils import configclass

from .aerodynamics_cfg import AerodynamicsCfg
from .attitude_controller_cfg import AttitudeControllerCfg
from .gimbal_controller_cfg import GimbalControllerCfg
from .motor_dynamics_cfg import MotorDynamicsCfg
from .velocity_controller_cfg import VelocityControllerCfg
from .zoom_controller_cfg import ZoomControllerCfg


@configclass
class DroneControllerCfg:
    """Complete drone controller configuration.

    Combines all sub-controller configurations into a single configuration
    for the DroneController orchestrator.
    """

    velocity: VelocityControllerCfg = VelocityControllerCfg()
    """Velocity controller configuration (outer loop)."""

    attitude: AttitudeControllerCfg = AttitudeControllerCfg()
    """Attitude controller configuration (inner loop)."""

    motor: MotorDynamicsCfg = MotorDynamicsCfg()
    """Motor dynamics configuration."""

    gimbal: GimbalControllerCfg = GimbalControllerCfg()
    """Gimbal controller configuration."""

    zoom: ZoomControllerCfg = ZoomControllerCfg()
    """Zoom controller configuration."""

    aerodynamics: AerodynamicsCfg = AerodynamicsCfg()
    """Aerodynamic effects configuration."""

    control_dt: float = 0.01
    """Inner loop control timestep [s] (100 Hz)."""
