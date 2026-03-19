# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tuning results package.

Provides TUNED_CONTROLLER_CFG with PX4-style default gains.

The control architecture follows PX4:
- Velocity controller (PI): converts velocity error to attitude + thrust
- Attitude controller (P): converts attitude error to rate setpoint
- Rate controller (PID): converts rate error to torque
"""

from isaaclab_tasks.direct.iris_ma6.controller.controller_cfg import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg

# Default configuration using PX4-style gains
TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        # Velocity PI gains (adjusted for our drone mass/inertia)
        Kp_vel=(1.5, 1.5, 3.0),
        Ki_vel=(0.25, 0.25, 0.3),
    ),
    attitude=AttitudeControllerCfg(
        # Attitude P gains (outputs rate setpoint in rad/s per rad error)
        # Based on PX4 MC_ROLL_P=6.5, MC_PITCH_P=6.5, MC_YAW_P=2.8
        Kp_att=(6.5, 6.5, 2.8),
    ),
    rate=RateControllerCfg(
        # Rate PID gains (outputs torque in Nm)
        # Based on PX4 MC_ROLLRATE_P=0.15, MC_ROLLRATE_I=0.2, MC_ROLLRATE_D=0.003
        Kp_rate=(0.15, 0.15, 0.2),
        Ki_rate=(0.2, 0.2, 0.15),
        Kd_rate=(0.003, 0.003, 0.001),
    ),
)

__all__ = ["TUNED_CONTROLLER_CFG"]
