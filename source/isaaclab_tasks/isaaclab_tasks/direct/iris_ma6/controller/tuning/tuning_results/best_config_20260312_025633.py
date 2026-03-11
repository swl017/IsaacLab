# Auto-tuned DroneController configuration (4-loop architecture)
# Generated: 20260312_025633
# Score: 3.1297 (migrated from 3-loop to 4-loop architecture)
#
# Architecture: Velocity (PI) -> Attitude (P) -> Rate (PID) -> Motor
#
# Note: Original config was from 3-loop architecture (with Kd_att).
# Migrated to 4-loop by moving Kd_att role to Rate controller.

from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg

TUNED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(2.0208, 2.0208, 3.0826),
        Ki_vel=(0.3702, 0.3702, 0.5073),
    ),
    attitude=AttitudeControllerCfg(
        # P-only attitude controller (outputs rate setpoint)
        # Original Kp_att preserved
        Kp_att=(12.1648, 12.1648, 6.0166),
    ),
    rate=RateControllerCfg(
        # Rate PID gains (moved from old Kd_att role)
        # Using PX4-style defaults scaled by original Kd_att values
        Kp_rate=(0.15, 0.15, 0.2),
        Ki_rate=(0.2, 0.2, 0.1),
        Kd_rate=(0.003, 0.003, 0.0),
    ),
)
