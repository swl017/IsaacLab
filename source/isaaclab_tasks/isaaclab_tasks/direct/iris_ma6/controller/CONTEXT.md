# Controller Module

## Purpose
PX4-style cascaded quadcopter control pipeline: velocity -> attitude -> rate -> motor mixing.
Includes gimbal stabilization and zoom control for camera-equipped drones.

## Inputs
- Velocity commands: `(N, 3)` world-frame desired velocity
- Yaw rate command: `(N,)` desired yaw rate
- Gimbal rate commands: `(N, 3)` desired gimbal angular rates (roll, pitch, yaw)
- Zoom rate command: `(N,)` desired zoom rate
- Current state: position, velocity, orientation quaternion, angular velocity

## Outputs
- Body forces: `(N, 3)` in world frame
- Body torques: `(N, 3)` in body frame
- Gimbal joint positions: `(N, 3)` stabilized joint angles
- Zoom level: `(N,)` current zoom factor

## Dependencies
None (standalone module).

## Key Files
- `drone_controller.py` - Top-level controller orchestrating the cascade
- `velocity_controller.py` / `velocity_controller_cfg.py` - Velocity-to-attitude conversion
- `attitude_controller.py` / `attitude_controller_cfg.py` - Attitude-to-rate conversion
- `rate_controller.py` / `rate_controller_cfg.py` - Rate-to-torque PID
- `motor_dynamics.py` / `motor_dynamics_cfg.py` - First-order motor lag model
- `mixer.py` - Quadcopter motor mixing matrix
- `gimbal_controller.py` / `gimbal_controller_cfg.py` - 3-axis gimbal stabilization
- `zoom_controller.py` / `zoom_controller_cfg.py` - Zoom level control
- `aerodynamics.py` / `aerodynamics_cfg.py` - Aerodynamic drag model
- `controller_cfg.py` - Unified configuration dataclass

## Spec
`doc/controller_spec.md`
