# Auto-tuned DroneController configuration
# Generated: 20260312_024405
# Score: 9999.0000

from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg

TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(5.8046, 5.8046, 2.9083),
        Ki_vel=(0.4888, 0.4888, 0.5700),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(14.5767, 14.5767, 3.5489),
        Kd_att=(1.2967, 1.2967, 1.0966),
    ),
)
