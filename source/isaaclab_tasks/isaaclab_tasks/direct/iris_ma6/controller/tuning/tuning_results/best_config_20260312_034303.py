# Auto-tuned DroneController configuration (4-loop architecture)
# Generated: 20260312_034303
# Score: 9999.0000
#
# Architecture: Velocity (PI) -> Attitude (P) -> Rate (PID) -> Motor

from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg

TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(4.8080, 4.8080, 3.7229),
        Ki_vel=(0.2649, 0.2649, 0.1572),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(9.8748, 9.8748, 4.2106),
    ),
    rate=RateControllerCfg(
        Kp_rate=(0.1617, 0.1617, 0.1651),
        Ki_rate=(0.2736, 0.2736, 0.1397),
        Kd_rate=(0.00446, 0.00446, 0.00000),
    ),
)
