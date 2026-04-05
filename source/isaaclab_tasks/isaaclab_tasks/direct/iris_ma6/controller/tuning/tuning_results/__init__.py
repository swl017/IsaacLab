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

TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(2.0416, 2.0416, 1.5355),
        Ki_vel=(1.3005, 1.3005, 0.7767),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(5.2060, 5.2060, 1.9334),
    ),
    rate=RateControllerCfg(
        Kp_rate=(0.3932, 0.3932, 0.3956),
        Ki_rate=(0.1805, 0.1805, 0.0900),
        Kd_rate=(0.01451, 0.01451, 0.00000),
    ),
)


__all__ = ["TUNED_CONTROLLER_CFG"]
