# Auto-tuned DroneController configuration
# Generated: 20260311_212825
# Score: 1.1684

from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg

TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(4.6013, 4.6013, 1.9587),
        Ki_vel=(0.1931, 0.1931, 0.4321),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(14.4536, 14.4536, 4.0059),
        Kd_att=(1.1304, 1.1304, 0.8839),
    ),
)
