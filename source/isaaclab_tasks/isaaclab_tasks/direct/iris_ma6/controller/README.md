# iris_ma6 Controller Module

A realistic quadcopter control system with rotor-level physics, cascaded control architecture, and configurable aerodynamic effects for the Isaac Lab simulation framework.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Control Loop Rates: PX4 vs iris_ma6](#control-loop-rates-px4-vs-iris_ma6)
- [Components](#components)
- [Configuration](#configuration)
- [Usage](#usage)
- [Frame Conventions](#frame-conventions)
- [Test Suite](#test-suite)
- [API Reference](#api-reference)

## Overview

The iris_ma6 controller module provides a high-fidelity quadcopter control system designed for reinforcement learning research. Unlike simpler direct force/moment controllers, this module implements:

- **Rotor-level physics**: Thrust computed as `T = k_f * omega^2`
- **First-order motor dynamics**: Realistic motor response with configurable time constants
- **Cascaded control**: Velocity → Attitude → Rate → Motor → Physics pipeline (PX4-style 4-loop)
- **Gimbal control**: World-frame line-of-sight stabilization with auto-roll
- **Optical zoom control**: First-order zoom dynamics with FOV computation
- **Configurable aerodynamics**: 4 fidelity levels from disabled to full rotor effects

### Key Differences from iris_ma5

| Feature | iris_ma5 | iris_ma6 |
|---------|----------|----------|
| Physics | Direct force/moment | Rotor-level (T = k_f * omega²) |
| Motor dynamics | None | First-order lag (tau=0.02s) |
| Gimbal dynamics | None | First-order lag (tau=0.05s) |
| Zoom control | None | First-order lag (tau=0.1s) |
| Control cascade | Velocity → Force | Velocity → Attitude → Rate → Motor (4-loop) |
| Aerodynamics | None | 4 fidelity levels |

## Architecture

### Control Cascade (PX4-style 4-loop)

```
Policy (25 Hz)
    │
    ├─► VelocityController ─► AttitudeController ─► RateController ─► MotorDynamics ─► Physics
    │   (outer loop, PI)      (middle loop, P)      (inner loop, PID)  (100 Hz)
    │   25 Hz                 100 Hz                100 Hz
    │
    ├─► GimbalController ─► Joint Targets
    │
    └─► ZoomController ─► Zoom Level
```

**Key Architecture Points:**
- Velocity Controller (PI): Converts velocity error → desired attitude + thrust
- Attitude Controller (P-only): Converts attitude error → rate setpoint (NOT torque)
- Rate Controller (PID): Converts rate error → torque commands
- This matches PX4's cascaded architecture for improved stability

### Data Flow

```
Inputs:                          Outputs:
  v_cmd (velocity)          ─┐
  yaw_rate_cmd              ─┤
  gimbal_yaw_rate_cmd       ─┼─► DroneController ─┬─► F_body (force, body frame)
  gimbal_pitch_rate_cmd     ─┤                    ├─► tau_body (torque, body frame)
  zoom_rate_cmd             ─┤                    ├─► gimbal_targets (yaw, roll, pitch)
  q_body (quaternion)       ─┤                    └─► zoom_level
  v_body (velocity)         ─┤
  omega_body (angular vel)  ─┘
```

### Control Loop Rates: PX4 vs iris_ma6

The iris_ma6 controller architecture is inspired by PX4, but operates at different loop rates due to simulation constraints.

| Loop Level | PX4 Rate | iris_ma6 Rate | Ratio |
|------------|----------|---------------|-------|
| **Rate Controller** | ~400 Hz (IMU-triggered) | 100 Hz | 4x slower |
| **Attitude Controller** | ~400 Hz | 100 Hz | 4x slower |
| **Velocity Controller** | ~50 Hz | 25 Hz | 2x slower |

#### PX4 Architecture

**Rate/Attitude Controllers (~400 Hz):**
- Triggered by IMU gyroscope callbacks
- Run in `rate_ctrl` work queue (priority 0, highest)
- dt constraints: Rate=0.125-20ms, Attitude=0.2-20ms
- Configurable via `IMU_GYRO_RATEMAX` parameter

**Position/Velocity Controller (~50 Hz):**
- Runs in `nav_and_controllers` work queue (priority -13)
- dt constraints: 2-40ms
- Lower priority than inner loops

#### iris_ma6 Implementation

- `control_dt = 0.01s` → **100 Hz** for attitude and rate controllers
- Velocity controller runs at policy rate → **25 Hz** (decimation=4 at 100 Hz sim)

**Constraint:** The simulation physics runs at 100 Hz (`sim.dt = 1/100`), which limits the inner loop to 100 Hz maximum.

#### Implications

1. **Reduced Phase Margin:** Inner loops run 4x slower than PX4, which may reduce stability margins for aggressive maneuvers.

2. **Gain Adjustment:** PX4 gains were tuned for 400 Hz. Running at 100 Hz may require:
   - Scaled integral gains (slower accumulation)
   - Adjusted derivative gains (larger dt = more noise sensitivity)

3. **To Match PX4 Rates:** Increase `sim.dt` from `1/100` to `1/400` (4x more physics computation).

### Module Structure

```
controller/
├── __init__.py                    # Module exports
├── README.md                      # This file
│
├── # Configurations
├── aerodynamics_cfg.py            # Fidelity levels, drag coefficients
├── attitude_controller_cfg.py     # PD gains, moment limits
├── controller_cfg.py              # DroneControllerCfg (composite)
├── gimbal_controller_cfg.py       # Time constant, joint limits
├── motor_dynamics_cfg.py          # k_f, k_m, tau_motor, omega limits
├── velocity_controller_cfg.py     # PI gains, max tilt, anti-windup
├── zoom_controller_cfg.py         # Time constant, zoom range
│
├── # Core Classes
├── aerodynamics.py                # Drag, wind, rotor effects
├── attitude_controller.py         # Quaternion PD control
├── drone_controller.py            # Full cascade orchestrator
├── gimbal_controller.py           # World-frame LOS stabilization
├── mixer.py                       # X-config thrust allocation
├── motor_dynamics.py              # First-order motor model
├── velocity_controller.py         # PI velocity control
├── zoom_controller.py             # Optical zoom control
│
└── tests/
    ├── __init__.py
    ├── run_tests.py               # Standalone test runner
    └── README.md                  # Test documentation
```

## Components

### MixerMatrix

Thrust allocation for PX4 X-configuration quadcopter.

**Motor Layout (top view, +X forward):**
```
    M1 (CW)     M2 (CCW)
       \       /
        \     /
         -----
        /     \
       /       \
    M3 (CCW)   M4 (CW)
```

**Functions:**
- `allocate(thrust_cmd, moment_cmd)` → Individual rotor thrusts [T1, T2, T3, T4]
- `aggregate(thrusts)` → Total thrust and moments [F, τx, τy, τz]
- `thrust_to_omega(thrusts)` → Rotor angular velocities
- `omega_to_thrust(omega)` → Rotor thrusts

### MotorDynamics

First-order motor response model.

**Dynamics:**
```
d(omega)/dt = (omega_cmd - omega) / tau_motor
```

**Discrete-time (exact):**
```python
alpha = 1 - exp(-dt / tau_motor)
omega_new = omega + alpha * (omega_cmd - omega)
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| tau_motor | 0.02s | Motor time constant (20ms) |
| k_f | 8.54858e-06 | Thrust coefficient [N/(rad/s)²] |
| k_m | 1.0e-07 | Torque coefficient [Nm/(rad/s)²] |
| omega_min | 100 | Minimum rotor speed [rad/s] |
| omega_max | 1100 | Maximum rotor speed [rad/s] |

### AttitudeController

Quaternion-based PD attitude controller.

**Control Law:**
```
q_err = q_des^(-1) * q_current
attitude_error = 2 * sign(q_err.w) * q_err.xyz
tau_cmd = -Kp * attitude_error - Kd * (omega - omega_des)
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Kp_att | (8, 8, 4) | Proportional gains [roll, pitch, yaw] |
| Kd_att | (2.5, 2.5, 1.0) | Derivative gains |
| tau_max | (5, 5, 2) | Maximum moments [Nm] |

### VelocityController

PI velocity controller with velocity-to-attitude conversion.

**Control Law:**
```
vel_error = v_des - v_current
a_des = Kp * vel_error + Ki * integral + g * z_hat
thrust = m * |a_des|
q_des = rotation aligning body z with a_des
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Kp_vel | (3, 3, 2) | Proportional gains [x, y, z] |
| Ki_vel | (0.5, 0.5, 0.3) | Integral gains |
| max_tilt | 30° | Maximum tilt angle |
| integral_limit | (5, 5, 5) | Anti-windup limits |

### GimbalController

World-frame line-of-sight stabilization with auto-roll.

**Features:**
- Rate command integration for yaw/pitch
- Auto roll stabilization (horizon leveling)
- First-order joint dynamics (tau=0.05s)
- Joint limit enforcement

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| tau_gimbal | 0.05s | Gimbal time constant (50ms) |
| max_gimbal_rate | 2π rad/s | Maximum gimbal rate |
| yaw_limits | (-π, π) | Yaw joint limits |
| pitch_limits | (-π/2, π/4) | Pitch joint limits |
| roll_limits | (-π/4, π/4) | Roll joint limits |

### ZoomController

Optical zoom control with first-order dynamics.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| tau_zoom | 0.1s | Zoom time constant (100ms) |
| zoom_min | 1.0 | Minimum zoom (1x) |
| zoom_max | 30.0 | Maximum zoom (30x) |
| max_zoom_rate | 2.0 | Maximum zoom rate [1/s] |

**FOV Computation:**
```python
fov = base_fov / zoom_level  # e.g., 90° / 2x = 45°
```

### AerodynamicEffects

Configurable aerodynamic effects with 4 fidelity levels.

| Level | Features |
|-------|----------|
| 0 | Disabled (no effects) |
| 1 | Basic quadratic drag: F = -0.5 * ρ * Cd * A * |v|² * v_hat |
| 2 | Level 1 + Wind (constant + Dryden gust model) |
| 3 | Level 2 + Rotor effects (H-force, blade flapping) |

## Configuration

### DroneControllerCfg

Composite configuration combining all sub-controllers:

```python
from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg

cfg = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(3.0, 3.0, 2.0),
        Ki_vel=(0.5, 0.5, 0.3),
        max_tilt=30.0,
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(8.0, 8.0, 4.0),
        Kd_att=(2.5, 2.5, 1.0),
    ),
    motor=MotorDynamicsCfg(
        tau_motor=0.02,
        k_f=8.54858e-06,
    ),
    gimbal=GimbalControllerCfg(
        tau_gimbal=0.05,
    ),
    zoom=ZoomControllerCfg(
        tau_zoom=0.1,
        zoom_max=30.0,
    ),
    aerodynamics=AerodynamicsCfg(
        fidelity_level=0,  # Disabled by default
    ),
    control_dt=0.01,  # Inner loop at 100 Hz
)
```

## Usage

### Basic Usage

```python
import torch
from isaaclab_tasks.direct.iris_ma6.controller import DroneController, DroneControllerCfg

# Create controller
cfg = DroneControllerCfg()
num_envs = 16
mass = 1.5  # kg
gravity = 9.81  # m/s²
device = "cuda"

controller = DroneController(
    cfg=cfg,
    mass=mass,
    gravity=gravity,
    num_envs=num_envs,
    device=device,
)

# Step the controller
F_body, tau_body, gimbal_targets, zoom_level = controller.step_policy(
    v_cmd=torch.zeros((num_envs, 3), device=device),
    yaw_rate_cmd=torch.zeros(num_envs, device=device),
    gimbal_yaw_rate_cmd=torch.zeros(num_envs, device=device),
    gimbal_pitch_rate_cmd=torch.zeros(num_envs, device=device),
    zoom_rate_cmd=torch.zeros(num_envs, device=device),
    q_body=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4),
    v_body=torch.zeros((num_envs, 3), device=device),
    omega_body=torch.zeros((num_envs, 3), device=device),
    sim_dt=0.04,  # Policy at 25 Hz
)

# Apply forces to simulation (body frame for Isaac Lab)
# F_body: (N, 3) force in body frame [N]
# tau_body: (N, 3) torque in body frame [Nm]
# gimbal_targets: tuple of (yaw, roll, pitch) joint targets [rad]
# zoom_level: (N,) current zoom level
```

### Reset Environments

```python
# Reset specific environments
env_ids = torch.tensor([0, 5, 10], device=device)
controller.reset(env_ids)

# Reset all environments
controller.reset()
```

### Access Sub-components

```python
# Get motor dynamics
motor = controller.motor_dynamics
current_omega = motor.omega  # (N, 4) rotor speeds

# Get zoom controller
zoom = controller.zoom_controller
current_zoom = zoom.zoom  # (N,) zoom levels
current_fov = zoom.get_fov(base_fov=90.0)  # (N,) field of view

# Get gimbal controller
gimbal = controller.gimbal_controller
gimbal_yaw = gimbal.yaw  # (N,) yaw angles
```

## Frame Conventions

### World Frame: ENU (East-North-Up)
- +X: East
- +Y: North
- +Z: Up

### Body Frame: FLU (Forward-Left-Up)
- +X: Forward (nose)
- +Y: Left (port wing)
- +Z: Up (dorsal)

### Quaternion Convention: wxyz (scalar first)
```python
q = [w, x, y, z]  # w is the scalar component
```

### Moment Sign Convention
- +τx (roll): Right wing down, left wing up
- +τy (pitch): Nose down
- +τz (yaw): Counter-clockwise from above

### PX4 Motor Convention (X-configuration)
```
Motor 1 (M1): Front-Right, CW rotation
Motor 2 (M2): Rear-Left, CCW rotation
Motor 3 (M3): Front-Left, CCW rotation
Motor 4 (M4): Rear-Right, CW rotation
```

## Test Suite

### Running Tests

```bash
# Run all tests
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py --headless

# Run with verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py --headless --test-verbose
```

### Test Coverage

| Component | Tests | Description |
|-----------|-------|-------------|
| MixerMatrix | 8 | Allocation, aggregation, thrust-omega conversion |
| MotorDynamics | 7 | First-order response, saturation, reset |
| ZoomController | 4 | First-order response, range limits, FOV |
| GimbalController | 4 | Rate integration, joint limits, dynamics |
| AttitudeController | 4 | Quaternion error, moment generation |
| VelocityController | 4 | Hover, tilt, anti-windup |
| AerodynamicEffects | 4 | Fidelity levels, drag, gust reset |
| DroneController | 4 | Integration, inner loop, reset |
| **Total** | **39** | |

### Test Results

```
================================================================================
TEST SUMMARY
================================================================================
Total Tests: 39
Passed:      39 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

### Performance Criteria

| Component | Test | Criteria |
|-----------|------|----------|
| Motor | Step response | 63.2% at t = tau_motor (0.02s) |
| Gimbal | Step response | 63.2% at t = tau_gimbal (0.05s) |
| Zoom | Step response | 63.2% at t = tau_zoom (0.1s) |
| Zoom | Range | Clamped to [1x, 30x] |
| Attitude | Identity | Near-zero moment output |
| Velocity | Hover | Identity attitude, m*g thrust |
| Integration | Hover | Positive Z force in body frame |

## API Reference

### DroneController

```python
class DroneController:
    def __init__(
        self,
        cfg: DroneControllerCfg,
        mass: float,
        gravity: float,
        num_envs: int,
        device: str | torch.device = "cpu",
    ): ...

    def step_policy(
        self,
        v_cmd: torch.Tensor,           # (N, 3) velocity command [m/s]
        yaw_rate_cmd: torch.Tensor,    # (N,) yaw rate command [rad/s]
        gimbal_yaw_rate_cmd: torch.Tensor,   # (N,) gimbal yaw rate [-1, 1]
        gimbal_pitch_rate_cmd: torch.Tensor, # (N,) gimbal pitch rate [-1, 1]
        zoom_rate_cmd: torch.Tensor,   # (N,) zoom rate [-1, 1]
        q_body: torch.Tensor,          # (N, 4) body quaternion (wxyz)
        v_body: torch.Tensor,          # (N, 3) body velocity [m/s]
        omega_body: torch.Tensor,      # (N, 3) body angular velocity [rad/s]
        sim_dt: float,                 # Simulation timestep [s]
    ) -> tuple[
        torch.Tensor,  # F_body: (N, 3) force in body frame [N]
        torch.Tensor,  # tau_body: (N, 3) torque in body frame [Nm]
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # gimbal (yaw, roll, pitch)
        torch.Tensor,  # zoom_level: (N,)
    ]: ...

    def reset(self, env_ids: torch.Tensor | None = None): ...

    @property
    def motor_dynamics(self) -> MotorDynamics: ...
    @property
    def attitude_controller(self) -> AttitudeController: ...
    @property
    def velocity_controller(self) -> VelocityController: ...
    @property
    def gimbal_controller(self) -> GimbalController: ...
    @property
    def zoom_controller(self) -> ZoomController: ...
    @property
    def aerodynamics(self) -> AerodynamicEffects: ...
```

### Time Constants Summary

| Component | Time Constant | Frequency | Notes |
|-----------|---------------|-----------|-------|
| Drone motors | 0.02s (20ms) | - | Fast brushless DC |
| Gimbal motors | 0.05s (50ms) | - | Servo motors |
| Zoom mechanism | 0.1s (100ms) | - | Mechanical/optical |
| **Control Loops** | | | |
| Rate controller | 0.01s (10ms) | 100 Hz | PX4: ~400 Hz |
| Attitude controller | 0.01s (10ms) | 100 Hz | PX4: ~400 Hz |
| Velocity controller | 0.04s (40ms) | 25 Hz | PX4: ~50 Hz |
| Policy loop | 0.04s (40ms) | 25 Hz | RL agent decision rate |

## References

- [PX4 Mixer Documentation](https://docs.px4.io/main/en/concept/mixing.html)
- [Quaternion-based attitude control](https://www.research-collection.ethz.ch/handle/20.500.11850/154099)
- [Dryden Wind Turbulence Model](https://en.wikipedia.org/wiki/Dryden_Wind_Turbulence_Model)
