# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""iris_ma6 controller module.

This module provides a realistic quadcopter control system with:
- Rotor-level physics (thrust = k_f * omega^2)
- First-order motor dynamics
- Cascaded velocity -> attitude -> motor control
- Gimbal control with world-frame LOS stabilization
- Optical zoom control
- Configurable aerodynamic effects
"""

# Configurations
from .aerodynamics_cfg import AerodynamicsCfg
from .attitude_controller_cfg import AttitudeControllerCfg
from .controller_cfg import DroneControllerCfg
from .gain_randomization_cfg import GainRandomizationCfg
from .gimbal_controller_cfg import GimbalControllerCfg
from .gimbal_rate_loop_cfg import GimbalRateLoopCfg
from .motor_dynamics_cfg import MotorDynamicsCfg
from .rate_controller_cfg import RateControllerCfg
from .velocity_controller_cfg import VelocityControllerCfg
from .zoom_controller_cfg import ZoomControllerCfg

# Core classes
from .aerodynamics import AerodynamicEffects
from .attitude_controller import AttitudeController
from .drone_controller import DroneController
from .gimbal_controller import GimbalController
from .gimbal_rate_loop import GimbalRateLoop
from .mixer import MixerMatrix
from .motor_dynamics import MotorDynamics
from .rate_controller import RateController
from .velocity_controller import VelocityController
from .zoom_controller import ZoomController

# Tuned configurations (see controller/tuning/tuning_results/__init__.py for
# the description of each constant and the plant pairing).
from .tuning import (
    PX4_MATCHED_CONTROLLER_CFG,
    PX4_MATCHED_PEGASUS_CONTROLLER_CFG,
    TUNED_CONTROLLER_CFG,
)

__all__ = [
    # Configurations
    "AerodynamicsCfg",
    "AttitudeControllerCfg",
    "DroneControllerCfg",
    "GainRandomizationCfg",
    "GimbalControllerCfg",
    "GimbalRateLoopCfg",
    "MotorDynamicsCfg",
    "RateControllerCfg",
    "VelocityControllerCfg",
    "ZoomControllerCfg",
    # Core classes
    "AerodynamicEffects",
    "AttitudeController",
    "DroneController",
    "GimbalController",
    "GimbalRateLoop",
    "MixerMatrix",
    "MotorDynamics",
    "RateController",
    "VelocityController",
    "ZoomController",
    # Tuned configurations
    "PX4_MATCHED_CONTROLLER_CFG",
    "PX4_MATCHED_PEGASUS_CONTROLLER_CFG",
    "TUNED_CONTROLLER_CFG",
]
