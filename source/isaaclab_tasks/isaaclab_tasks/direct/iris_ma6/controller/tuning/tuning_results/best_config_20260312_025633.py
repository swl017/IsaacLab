# Auto-tuned DroneController configuration
# Generated: 20260312_025633
# Score: 3.1297

from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg

TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(2.0208, 2.0208, 3.0826),
        Ki_vel=(0.3702, 0.3702, 0.5073),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(12.1648, 12.1648, 6.0166),
        Kd_att=(1.2094, 1.2094, 0.7265),
    ),
)
