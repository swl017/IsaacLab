# Controller Design Specification

## Overview

Main components and features:
- Actuator dynamics (motor model)
- Attitude control
- Velocity control
- Aerodynamic effects
- Gimbal joint control
- Zoom control
- State estimation
- Randomization support

## Design Goal

### Limitations of Current Implementation
- Current implementation (`iris_ma5`) uses 6-DOF force/moment input as a work-around for keeping the drone horizontal
- Gravity is ignored in the IRIS_GIMBAL2 model
- No realistic motor dynamics or saturation

### Target Implementation
- Realistic thrust-based attitude control with actuator limits (saturation)
- Individual rotor dynamics with mixer matrix
- Configurable aerodynamic fidelity levels
- Dual-path state estimation for observation/reward ablation studies

---

## System Architecture

### Block Diagram
```
                              ┌─────────────────────────────────────────────────────┐
                              │                    CONTROLLER                       │
                              │                                                     │
┌────────┐   25 Hz   ┌────────┴───────┐  100 Hz  ┌──────────────┐  100 Hz  ┌───────┴───────┐
│ Policy │ ────────► │   Velocity     │ ───────► │   Attitude   │ ───────► │    Motor      │
│        │  v_cmd    │   Controller   │  att_cmd │   Controller │  omega   │   Dynamics    │
└────────┘  yaw_rate └────────────────┘          └──────────────┘   cmd    └───────┬───────┘
                              │                                                     │
                              │                                            F_body, tau_body
                              │                                                     │
                              │         ┌──────────────────────────────────────────┘
                              │         ▼
                              │  ┌─────────────────┐      ┌──────────────────────────────┐
                              │  │     Physics     │ ◄─── │    Aerodynamic Effects       │
                              │  │   Simulation    │      │  (drag, wind, rotor effects) │
                              │  └────────┬────────┘      └──────────────────────────────┘
                              │           │
                              │           ▼
                              │  ┌─────────────────┐
                              │  │ State Estimation│
                              │  │  (dual pipeline)│
                              │  └────────┬────────┘
                              │           │
                              │           ├────────► Observation Path (to Policy)
                              │           └────────► Reward Path (configurable)
                              │
                              │  ┌─────────────────────────────────────────────────────┐
                              │  │                 GIMBAL CONTROLLER                   │
                              │  │                                                     │
                              │  │  ┌──────────────┐  100 Hz  ┌─────────────┐         │
              gimbal_rate_cmd │  │  │ Stabilization│ ───────► │   Joint     │         │
              ───────────────►│  │  │   (world LOS)│  q_joint │   Dynamics  │         │
                              │  │  └──────────────┘   cmd    └─────────────┘         │
                              │  │         ▲                                          │
                              │  │         │ body orientation feedback                │
                              │  └─────────┼──────────────────────────────────────────┘
                              │            │
                              └────────────┘
```

### Timing Diagram
```
Time (ms):  0    10    20    30    40    50    60    70    80
            │     │     │     │     │     │     │     │     │
Policy      ├─────────────────────────────────────────┤     │  (25 Hz, every 40ms)
(action)    │                                         │     │
            │                                         │     │
Velocity    ├────┤────┤────┤────┤────┤────┤────┤────┤────┤  (100 Hz, every 10ms)
Controller  │    │    │    │    │    │    │    │    │    │
            │                                         │     │
Attitude    ├────┤────┤────┤────┤────┤────┤────┤────┤────┤  (100 Hz, every 10ms)
Controller  │    │    │    │    │    │    │    │    │    │
            │                                         │     │
Motor       ├────┤────┤────┤────┤────┤────┤────┤────┤────┤  (100 Hz, every 10ms)
Dynamics    │    │    │    │    │    │    │    │    │    │
            │                                         │     │
Gimbal      ├────┤────┤────┤────┤────┤────┤────┤────┤────┤  (100 Hz, every 10ms)
Controller  │    │    │    │    │    │    │    │    │    │

Legend: ├────┤ = one control loop iteration
```

---

## Actuator Dynamics (Motor Model)

### Motor Thrust and Torque
Each rotor generates thrust and reaction torque proportional to the square of angular velocity:

```
T_i = k_f * omega_i^2        (thrust, N)
Q_i = k_m * omega_i^2        (reaction torque, Nm)
```

Where:
- `T_i`: Thrust from rotor i (positive = upward in body frame)
- `Q_i`: Reaction torque from rotor i
- `omega_i`: Angular velocity of rotor i (rad/s)
- `k_f`: Thrust coefficient
- `k_m`: Torque coefficient

### First-Order Motor Dynamics
Motor speed follows commanded speed with first-order lag:

```
d(omega)/dt = (omega_cmd - omega) / tau_motor
```

Discrete-time implementation (for simulation timestep dt):
```
alpha = dt / (dt + tau_motor)
omega_new = omega + alpha * (omega_cmd - omega)
```

### Frame Conventions

#### World Frame (ENU - East-North-Up)
```
                 +Y (North)
                    ^
                    │
                    │
                    │
   -X ◄─────────────┼─────────────► +X
  (West)            │              (East)
                    │
                    │
                    ▼
                 -Y (South)

        +Z points OUT of page (Up)
```

#### Body Frame (FLU - Forward-Left-Up)
```
              +X (Forward)
                    ^
                    │
                    │
                    │
    +Y ◄────────────┼──────────────► -Y
   (Left)           │              (Right)
                    │
                    │
                    ▼
              -X (Backward)

        +Z points OUT of page (Up)
```

#### Rotation Conventions (Right-Hand Rule about each axis)
| Axis | Positive Rotation | Physical Effect |
|------|-------------------|-----------------|
| +τx (roll) | RH thumb along +X (forward) | Left wing up, right wing down |
| +τy (pitch) | RH thumb along +Y (left) | Nose down, tail up |
| +τz (yaw) | RH thumb along +Z (up) | CCW from above, nose left |

**Note:** +τz in ENU is CCW from above (nose LEFT), opposite of NED convention.

#### Quaternion Convention
- Format: wxyz (scalar first)
- Isaac Lab convention throughout

#### Gimbal Frame (FLU, attached to drone body)

**Joint Order:** Yaw (outer) → Roll (middle) → Pitch (inner)

**Rotation Definitions:**
- **Gimbal Yaw:** Rotation around Up axis (+Z in FLU)
  - Positive = CCW when viewed from above (nose left)
  - Range: [-π, +π] rad
- **Gimbal Roll:** Rotation around Forward axis (+X in yawed frame)
  - Applied after yaw rotation
  - Positive = left side up (right-hand rule)
  - Range: [-π, +π] rad (typically limited for stabilization)
- **Gimbal Pitch:** Rotation around Left axis (+Y in yawed+rolled frame)
  - Applied after yaw and roll
  - Positive = nose down (looking down)
  - Range: [-π/2, +π/2] rad

**Camera Pointing Direction:**
- Camera optical axis points along gimbal +X (forward) after all rotations
- With all angles at zero, camera points same direction as drone nose

**Stabilization Mode:**
- Roll is automatically computed to keep horizon level in camera view
- World up vector [0, 0, 1] in ENU is used as reference
- Stabilization compensates for drone body roll/pitch

---

### Mixer Matrix (X-Configuration, PX4 Convention, ENU Frame)

**Motor Positions (body frame coordinates, FLU):**
| Motor | Position | Coordinates |
|-------|----------|-------------|
| M1 | Front-Right | (+L/√2, -L/√2, 0) |
| M2 | Rear-Left | (-L/√2, +L/√2, 0) |
| M3 | Front-Left | (+L/√2, +L/√2, 0) |
| M4 | Rear-Right | (-L/√2, -L/√2, 0) |

**PX4 Rotor Layout** (top view, looking down from +Z):
```
              Front (+X)
                  ^
                  │
        M3        │        M1
        (CW)      │      (CCW)
           \      │      /
            \     │     /
             \    │    /
              \   │   /
               \  │  /
                \ │ /
   Left (+Y) ◄───\│/───► Right (-Y)
                 /│\
                / │ \
               /  │  \
              /   │   \
             /    │    \
            /     │     \
        M2        │        M4
       (CCW)      │       (CW)
                  │
                  ▼
              Rear (-X)
```

**Derivation of Mixer Matrix:**

Each rotor creates:
1. **Thrust (F):** Upward force = +Z direction in ENU = [0, 0, +T_i]
2. **Moment from thrust (τ):** τ = r × F where r is motor position

Computing r × F for each motor (with d = L/√2):

| Motor | Position r | Thrust F | τ = r × F | τx | τy |
|-------|------------|----------|-----------|-----|-----|
| M1 | (d, -d, 0) | (0, 0, T₁) | (-dT₁, -dT₁, 0) | -dT₁ | -dT₁ |
| M2 | (-d, d, 0) | (0, 0, T₂) | (dT₂, dT₂, 0) | +dT₂ | +dT₂ |
| M3 | (d, d, 0) | (0, 0, T₃) | (dT₃, -dT₃, 0) | +dT₃ | -dT₃ |
| M4 | (-d, -d, 0) | (0, 0, T₄) | (-dT₄, dT₄, 0) | -dT₄ | +dT₄ |

3. **Yaw torque from prop reaction (ENU frame):**
   - In ENU, +Z is up, so +τz = CCW from above
   - CCW prop creates CW reaction torque on body → -τz (in ENU)
   - CW prop creates CCW reaction torque on body → +τz (in ENU)
   - Magnitude: Q_i = c × T_i where c = k_m/k_f

| Motor | Spin Direction | Reaction on Body | τz contribution (ENU) |
|-------|----------------|------------------|-----------------------|
| M1 | CCW | CW (nose right) | -cT₁ |
| M2 | CCW | CW (nose right) | -cT₂ |
| M3 | CW | CCW (nose left) | +cT₃ |
| M4 | CW | CCW (nose left) | +cT₄ |

**Summing all contributions:**
```
F  = T₁ + T₂ + T₃ + T₄
τx = -dT₁ + dT₂ + dT₃ - dT₄  = d(-T₁ + T₂ + T₃ - T₄)
τy = -dT₁ + dT₂ - dT₃ + dT₄  = d(-T₁ + T₂ - T₃ + T₄)
τz = -cT₁ - cT₂ + cT₃ + cT₄  = c(-T₁ - T₂ + T₃ + T₄)
```

**Final Mixer Matrix (ENU/FLU):**
```
┌    ┐       ┌                              ┐   ┌    ┐
│ F  │       │  1       1       1       1   │   │ T1 │
│ τx │   =   │ -L/√2    L/√2    L/√2   -L/√2│ * │ T2 │
│ τy │       │ -L/√2    L/√2   -L/√2    L/√2│   │ T3 │
│ τz │       │ -c      -c       c       c   │   │ T4 │
└    ┘       └                              ┘   └    ┘
```

Where:
- `L`: Arm length (center to rotor)
- `d = L/√2`: Moment arm for X-configuration
- `c = k_m / k_f`: Torque-to-thrust ratio

**Motor Properties Summary (ENU frame):**
| Motor | Position | Spin | Yaw Contribution (ENU) |
|-------|----------|------|------------------------|
| M1 | Front-Right | CCW | -τz (CW reaction → nose right) |
| M2 | Rear-Left | CCW | -τz (CW reaction → nose right) |
| M3 | Front-Left | CW | +τz (CCW reaction → nose left) |
| M4 | Rear-Right | CW | +τz (CCW reaction → nose left) |

**Moment Generation (physical intuition, ENU/FLU):**
- **Roll right (+τx in FLU = left wing up):** Increase LEFT motors (M2, M3), Decrease RIGHT motors (M1, M4)
- **Pitch up (+τy):** Increase REAR motors (M2, M4), Decrease FRONT motors (M1, M3)
- **Yaw right (nose right = -τz in ENU):** Increase CCW motors (M1, M2), Decrease CW motors (M3, M4)

### Default Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `k_f` | 8.54858e-06 | N/(rad/s)^2 | Thrust coefficient |
| `k_m` | 1.0e-07 | Nm/(rad/s)^2 | Torque coefficient |
| `tau_motor` | 0.02 | s | Motor time constant (20ms) |
| `omega_max` | 1100 | rad/s | Maximum rotor speed |
| `omega_min` | 100 | rad/s | Minimum rotor speed (idle) |
| `L` | 0.22 | m | Arm length |

### Test Criteria
- First-order step response: 63% of final value reached in `tau_motor` seconds
- Saturation: `omega` clamped to `[omega_min, omega_max]`

---

## Attitude Control

### Quaternion Error Computation
Attitude error using quaternion multiplication (avoids gimbal lock):

```
q_err = q_des^(-1) * q_current
attitude_error = 2 * sign(q_err.w) * q_err.xyz
```

Where:
- `q_des`: Desired orientation quaternion (wxyz convention)
- `q_current`: Current orientation quaternion
- `q_err`: Error quaternion
- `attitude_error`: 3D attitude error vector (roll, pitch, yaw)

The `sign(q_err.w)` ensures shortest path rotation.

### Control Law
PD control on attitude with rate damping:

```
tau_cmd = -Kp_att * attitude_error - Kd_att * (omega - omega_des)
```

Where:
- `tau_cmd`: Commanded body moments (Nm)
- `Kp_att`: Proportional gain matrix (diagonal)
- `Kd_att`: Derivative gain matrix (diagonal)
- `omega`: Current body angular velocity (rad/s)
- `omega_des`: Desired angular velocity (typically [0, 0, yaw_rate_cmd])

### Allocation
Convert moment commands to rotor speed commands via inverse mixer:
```
[T1, T2, T3, T4] = M^(-1) * [F_cmd, tau_x, tau_y, tau_z]
omega_i_cmd = sqrt(T_i / k_f)    (clamped to [omega_min, omega_max])
```

### Default Parameters

| Parameter | Roll | Pitch | Yaw | Unit |
|-----------|------|-------|-----|------|
| `Kp_att` | 8.0 | 8.0 | 4.0 | Nm/rad |
| `Kd_att` | 2.5 | 2.5 | 1.0 | Nm/(rad/s) |
| `tau_max` | 5.0 | 5.0 | 2.0 | Nm |

### Loop Timing
- Sample rate: 100 Hz
- Reference command update rate: 100 Hz (from velocity controller)

### Test Criteria
- Step response: settling time < 0.5s, overshoot < 10%
- Disturbance rejection: recover from 15 deg perturbation within 1s
- Hover stability: attitude error < 2 deg in steady hover

---

## Velocity Control

### Velocity-to-Attitude Conversion
The velocity controller computes desired acceleration, then converts to attitude command:

```
# Velocity error and desired acceleration
v_err = v_des - v
a_des = Kp_vel * v_err + Ki_vel * integral(v_err) + g * z_hat

# Thrust magnitude
thrust_cmd = m * norm(a_des)

# Desired attitude (rotation aligning body z-axis with a_des)
z_des = a_des / norm(a_des)
attitude_des = rotation_from_z_axis(z_des, yaw_des)
```

Where:
- `v_des`: Desired velocity (m/s) from policy
- `v`: Current velocity (m/s)
- `g`: Gravity magnitude (9.81 m/s^2)
- `m`: Drone mass (kg)
- `z_hat`: World up vector [0, 0, 1]

### Anti-Windup Strategy
Conditional integration with saturation:

```python
if not is_saturated:
    integral += v_err * dt
integral = clamp(integral, -integral_limit, integral_limit)
```

Saturation detected when:
- `thrust_cmd` exceeds limits, OR
- `tilt_angle` exceeds `max_tilt`

### Default Parameters

| Parameter | X | Y | Z | Unit |
|-----------|---|---|---|------|
| `Kp_vel` | 3.0 | 3.0 | 2.0 | 1/s |
| `Ki_vel` | 0.5 | 0.5 | 0.3 | 1/s^2 |
| `integral_limit` | 2.0 | 2.0 | 1.0 | m/s |

| Parameter | Value | Unit |
|-----------|-------|------|
| `max_tilt` | 30 | deg |
| `max_lin_vel` | 10.0 | m/s |

### Loop Timing
- Sample rate: 100 Hz
- Reference command (from policy) update rate: 25 Hz

### Test Criteria
- Velocity tracking: steady-state error < 0.1 m/s
- Step response: settling time < 1.0s
- Position hold: drift < 0.5m over 10s with zero velocity command

---

## Aerodynamic Effects

Configurable fidelity levels allow tradeoff between realism and training speed.

### Level 0: Disabled (Default)
No aerodynamic effects. Fastest training, suitable for initial policy learning.

### Level 1: Basic Drag
Quadratic drag on drone body:

```
F_drag = -0.5 * rho * C_d * A * |v|^2 * v_hat
```

Where:
- `rho`: Air density (1.225 kg/m^3 at sea level)
- `C_d`: Drag coefficient
- `A`: Frontal area
- `v`: Body velocity
- `v_hat`: Velocity unit vector

**Default Parameters:**
| Parameter | Value | Unit |
|-----------|-------|------|
| `C_d` | 1.0 | - |
| `A` | 0.1 | m^2 |

### Level 2: Level 1 + Wind
Wind disturbance added to velocity:

```
v_rel = v_body - v_wind
F_drag = -0.5 * rho * C_d * A * |v_rel|^2 * v_rel_hat
```

Wind model options:
- **Constant**: Fixed wind vector
- **Dryden Gust**: Stochastic turbulence model

**Default Parameters:**
| Parameter | Value | Unit |
|-----------|-------|------|
| `v_wind_mean` | [0, 0, 0] | m/s |
| `sigma_gust` | 2.0 | m/s |
| `gust_bandwidth` | 0.5 | Hz |

### Level 3: Level 2 + Rotor Effects
Additional rotor-specific aerodynamic effects:

**Rotor Drag (H-force):**
```
F_H = -k_H * sum(omega_i) * v_xy
```

**Blade Flapping Moment:**
```
tau_flap = -k_flap * v_xy
```

**Default Parameters:**
| Parameter | Value | Unit |
|-----------|-------|------|
| `k_H` | 0.001 | Ns/m |
| `k_flap` | 0.01 | Nm/(m/s) |

### Level 4: Level 3 + Thermal Effects (Future Work)
Motor efficiency degradation with temperature:
- `tau_motor` increases with motor temperature
- Thrust coefficient `k_f` decreases with temperature
- Temperature modeled as function of power dissipation

*Marked for future implementation.*

---

## Gimbal Joint Control

### Stabilization Mode
The gimbal maintains a world-frame line-of-sight (LOS) direction despite body motion.

**LOS Tracking:**
```
# Integrate rate commands to get desired world-frame LOS
LOS_world = update_los(LOS_world_prev, gimbal_yaw_rate_cmd, gimbal_pitch_rate_cmd, dt)

# Convert to gimbal joint angles
q_LOS_world = quaternion_from_los(LOS_world)
q_gimbal_des = q_body^(-1) * q_LOS_world

# Extract joint angles
[yaw_joint, pitch_joint, roll_joint] = euler_from_quaternion(q_gimbal_des)
```

**Roll Stabilization:**
Roll is always stabilized to horizontal (independent of pitch/yaw commands):
```
roll_joint = -body_roll    (compensates body roll)
```

### First-Order Joint Dynamics
```
d(q_joint)/dt = (q_joint_cmd - q_joint) / tau_gimbal
```

### Joint Limits
| Joint | Min | Max | Unit |
|-------|-----|-----|------|
| Yaw | -180 | 180 | deg |
| Pitch | 0 | 90 | deg |
| Roll | -30 | 30 | deg |

### Default Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| `tau_gimbal` | 0.02 | s |
| `max_gimbal_rate` | 2*pi | rad/s |

### Loop Timing
- Sample rate: 100 Hz
- Reference command (from policy) update rate: 25 Hz

### Test Criteria
- **LOS Stability**: Gimbal LOS in world frame stays constant (< 1 deg drift) with zero rate command even as drone body changes orientation
- **Rate Tracking**: Gimbal follows rate commands with < 5% steady-state error
- **Disturbance Rejection**: Recovers from 10 deg body perturbation within 0.5s

---

## Zoom Control
- Mechanical, optical zoom would have its transient motor dynamics and maximum rate.

## State Estimation

### Architecture
Dual-pipeline design supporting ablation studies:

```
                    ┌─────────────────────────────────────────────────────┐
Ground Truth ─────► │              State Estimation System                │
States              │                                                     │
                    │  ┌─────────────┐    ┌─────────────┐                │
                    │  │ First-Order │    │  Staleness  │                │
                    │  │    Lag      │───►│ (sample &   │                │
                    │  └─────────────┘    │   hold)     │                │
                    │                      └──────┬──────┘                │
                    │                             │                       │
                    │           ┌─────────────────┼─────────────────┐    │
                    │           ▼                 ▼                 │    │
                    │    ┌─────────────┐   ┌─────────────┐         │    │
                    │    │   Latency   │   │   Dropout   │         │    │
                    │    │   (delay)   │   │  (missing)  │         │    │
                    │    └──────┬──────┘   └──────┬──────┘         │    │
                    │           │                 │                 │    │
                    │           ▼                 ▼                 │    │
                    │    ┌─────────────┐   ┌─────────────┐         │    │
                    │    │    Noise    │   │    Noise    │         │    │
                    │    │  Injection  │   │  Injection  │         │    │
                    │    └──────┬──────┘   └──────┬──────┘         │    │
                    │           │                 │                 │    │
                    │           ▼                 ▼                 │    │
                    │    Observation Path    Reward Path            │    │
                    │    (always degraded)   (configurable)         │    │
                    └─────────────────────────────────────────────────────┘
```

### Observation Path
States provided to the policy always include configured delays, noise, and degradation.

### Reward Path Configuration
States used for reward computation are configurable for ablation studies:

| Option | Description |
|--------|-------------|
| `use_ground_truth` | Use GT states (no delay/noise/lag) |
| `match_observation` | Use same degradation as observation path |
| `custom` | Independent configuration per component |

Ablation combinations:
- w/ vs w/o noise
- w/ vs w/o delay (latency)
- w/ vs w/o lag filtering
- w/ vs w/o staleness
- w/ vs w/o dropout

### Per-Sensor Configuration (Observation Path Defaults)

| Sensor | tau (s) | Noise std | Staleness (Hz) | Latency (s) |
|--------|---------|-----------|----------------|-------------|
| IMU accel | 0.005 | 0.1 m/s^2 | 200 | 0.001 |
| IMU gyro | 0.005 | 0.01 rad/s | 200 | 0.001 |
| Position (GPS) | 0.05 | 0.1 m | 50 | 0.05 |
| Velocity | 0.05 | 0.1 m/s | 50 | 0.05 |
| Gimbal angles | 0.01 | 0.01 rad | 100 | 0.01 |

### Future Work
- EKF-based GPS/INS fusion
- Position bias estimation
- Magnetometer heading fusion

---

## Randomization

### Randomization Levels
| Level | Description | Purpose |
|-------|-------------|---------|
| **per-episode** | Varies with curriculum progress | Progressive difficulty |
| **per-env** | Varies across parallel environments | Diversity within batch |
| **per-step** | Varies at each simulation step | Non-stationary dynamics |

### Per-Episode Randomization (Curriculum-Scaled)

Parameters scale linearly with curriculum progress `p` in [0, 1]:

| Parameter | Min (p=0) | Max (p=1) | Formula |
|-----------|-----------|-----------|---------|
| `tau_motor` | 0.01 s | 0.05 s | `tau = 0.01 + 0.04 * p` |
| `tau_estimation` | 0.01 s | 0.1 s | `tau = 0.01 + 0.09 * p` |
| `Kp_att variance` | 0% | 20% | `var = 0.2 * p` |
| `drone_mass` | 1.0x | 1.2x | `m = m_nom * (1 + 0.2 * p)` |
| `wind_std` | 0 m/s | 3 m/s | `sigma = 3 * p` (Level 2+) |

maximum zoom change rate
zoom lag

### Per-Env Randomization (Within Episode Bounds)

Applied at environment reset, varies within the per-episode bounds:

| Parameter | Distribution | Notes |
|-----------|--------------|-------|
| `tau_motor` | U[0.01, tau_ep] | Diversity across envs |
| `tau_estimation` | U[0.01, tau_ep] | Diversity across envs |
| `Kp_att` | U[0.9, 1.1] * Kp_nom | Multiplicative |
| `Kd_att` | U[0.9, 1.1] * Kd_nom | Multiplicative |
| `Kp_vel` | U[0.9, 1.1] * Kp_nom | Multiplicative |
| `drone_mass` | U[0.95, 1.05] * m_ep | Around episode mass |
| `gimbal_inertia` | U[0.9, 1.1] * I_nom | Payload variation |

maximum zoom change rate
zoom lag

### Per-Step Randomization

Applied at each simulation step:

| Parameter | Model | Conditions |
|-----------|-------|------------|
| `wind_velocity` | Dryden gust | Level 2+ enabled |
| `motor_efficiency` | Random walk | Level 4 enabled |
| `sensor_noise` | Gaussian | Always (configurable std) |

---

## Interfaces

### Action Space (from Policy, 25 Hz)

| Index | Name | Range | Scaled By | Unit |
|-------|------|-------|-----------|------|
| 0 | vx | [-1, 1] | max_lin_vel | m/s |
| 1 | vy | [-1, 1] | max_lin_vel | m/s |
| 2 | vz | [-1, 1] | max_lin_vel / 2 | m/s |
| 3 | yaw_rate | [-1, 1] | max_yaw_rate | rad/s |
| 4 | gimbal_yaw_rate | [-1, 1] | max_gimbal_rate | rad/s |
| 5 | gimbal_pitch_rate | [-1, 1] | max_gimbal_rate | rad/s |
| 6 | zoom_rate | [-1, 1] | max_zoom_rate | 1/s |

**Scaling Defaults:**
| Parameter | Value | Unit |
|-----------|-------|------|
| `max_lin_vel` | 10.0 | m/s |
| `max_yaw_rate` | pi/2 | rad/s |
| `max_gimbal_rate` | 2*pi | rad/s |
| `max_zoom_rate` | 2.0 | 1/s |

### Velocity Controller Output (to Attitude Controller, 100 Hz)

| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0 | roll_des | rad | Desired roll angle |
| 1 | pitch_des | rad | Desired pitch angle |
| 2 | yaw_rate_des | rad/s | Desired yaw rate |
| 3 | thrust_cmd | N | Total thrust command |

### Attitude Controller Output (to Motor Dynamics, 100 Hz)

| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0 | omega_1_cmd | rad/s | Rotor 1 speed command |
| 1 | omega_2_cmd | rad/s | Rotor 2 speed command |
| 2 | omega_3_cmd | rad/s | Rotor 3 speed command |
| 3 | omega_4_cmd | rad/s | Rotor 4 speed command |

### Motor Dynamics Output (to Physics, 100 Hz)

| Name | Dimension | Unit | Description |
|------|-----------|------|-------------|
| F_body | 3 | N | Force in body frame [0, 0, F_z] |
| tau_body | 3 | Nm | Moment in body frame [tau_x, tau_y, tau_z] |

---

## Verification

### Unit Tests

**Motor Dynamics:**
- First-order step response matches `tau_motor`
- Speed saturates at `[omega_min, omega_max]`
- Thrust/torque ratios match `k_f`, `k_m`

**Attitude Controller:**
- Step response: settling time < 0.5s, overshoot < 10%
- Disturbance rejection: recover from 15 deg in 1s
- Moment saturation: outputs clamped to `tau_max`

**Velocity Controller:**
- Velocity tracking: steady-state error < 0.1 m/s
- Step response: settling time < 1.0s
- Anti-windup: integral resets appropriately

**Gimbal Controller:**
- LOS stability with body motion
- Rate limiting enforced
- Joint limits respected

### Integration Tests

| Test | Criteria | Duration |
|------|----------|----------|
| Hover stability | Position drift < 0.5m | 10s |
| Velocity tracking | Follow 5 m/s step within 1s | 5s |
| Yaw rotation | 180 deg rotation, settling < 2s | 5s |
| Gimbal tracking | LOS error < 1 deg during maneuvers | 10s |
| Wind rejection | Maintain position in 5 m/s wind (Level 2) | 10s |

### Randomization Tests

- Verify parameter ranges at curriculum extremes (p=0 and p=1)
- Check per-env diversity: histogram of parameters across batch
- Validate per-step evolution: temporal correlation matches model
- Confirm aerodynamic level switching: enable/disable cleanly

### Performance Benchmarks

| Metric | Target |
|--------|--------|
| Control loop rate | 100 Hz sustained |
| Policy inference rate | 25 Hz sustained |
| Parallel environments | 4096+ |
| GPU memory per env | < 1 MB |

---

## Auto-Tuning (Pre-Training Calibration)

### Purpose

Establish rational parameter ranges for control gains **before** full RL training. This is a calibration phase that:
- Finds stable operating regions for each controller
- Determines feasible randomization bounds
- Validates that the control architecture can stabilize the drone
- Provides baseline performance metrics

**Not** intended for online adaptation during training.

### Tuning Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTO-TUNING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Nominal Tuning                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Attitude  │───►│  Velocity   │───►│   Gimbal    │                     │
│  │  Controller │    │  Controller │    │  Controller │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│        │                  │                  │                              │
│        ▼                  ▼                  ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              Nominal Gains (Kp_nom, Kd_nom, Ki_nom)         │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                │                                            │
│  Phase 2: Sensitivity Analysis │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │   Sweep gains ±50% around nominal, measure stability        │           │
│  │   Record: settling time, overshoot, steady-state error      │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                │                                            │
│  Phase 3: Bound Determination  │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │   Identify stable region boundaries                         │           │
│  │   Set randomization bounds with safety margin               │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              Output: Parameter Ranges for Training          │           │
│  └─────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Nominal Gain Tuning

#### Method: Modified Ziegler-Nichols for Cascaded Control

**Step 1: Attitude Controller (Inner Loop)**
```
1. Set Ki = 0, Kd = 0
2. Increase Kp until sustained oscillation (critical gain Ku)
3. Measure oscillation period Tu
4. Set gains:
   Kp = 0.6 * Ku
   Kd = Kp * Tu / 8
```

**Step 2: Velocity Controller (Outer Loop)**
```
1. With tuned attitude controller
2. Set Ki = 0
3. Increase Kp until sluggish but stable response
4. Add Ki for steady-state error elimination
5. Typical ratio: Ki = Kp / 5
```

**Step 3: Gimbal Controller**
```
1. Tune independently (decoupled from body dynamics)
2. Target: fast response with minimal overshoot
3. Rate limiting prevents excessive commands
```

#### Alternative: Optimization-Based Tuning

For automated tuning, use Bayesian optimization:

```python
# Objective function
def tune_objective(params):
    Kp_att, Kd_att, Kp_vel, Ki_vel = params

    # Run step response test
    metrics = run_step_response(Kp_att, Kd_att, Kp_vel, Ki_vel)

    # Multi-objective cost
    cost = (
        w1 * metrics.settling_time +
        w2 * metrics.overshoot +
        w3 * metrics.steady_state_error +
        w4 * metrics.control_effort
    )

    # Penalty for instability
    if not metrics.is_stable:
        cost += 1000

    return cost

# Search bounds
bounds = {
    'Kp_att': (1.0, 20.0),
    'Kd_att': (0.5, 5.0),
    'Kp_vel': (1.0, 10.0),
    'Ki_vel': (0.1, 2.0),
}
```

### Phase 2: Sensitivity Analysis

Sweep each parameter around nominal to find stability boundaries.

#### Sweep Protocol

| Parameter | Sweep Range | Step Size | Hold Others |
|-----------|-------------|-----------|-------------|
| `Kp_att` | 0.5x - 1.5x nominal | 0.1x | Nominal |
| `Kd_att` | 0.5x - 1.5x nominal | 0.1x | Nominal |
| `Kp_vel` | 0.5x - 1.5x nominal | 0.1x | Nominal |
| `Ki_vel` | 0.5x - 1.5x nominal | 0.1x | Nominal |
| `tau_motor` | 0.01s - 0.1s | 0.01s | Nominal |
| `drone_mass` | 0.8x - 1.5x nominal | 0.1x | Nominal |

#### Test Scenarios per Sweep Point

| Scenario | Description | Pass Criteria |
|----------|-------------|---------------|
| Hover | Hold position at origin | Drift < 0.5m / 10s |
| Step X | 5 m/s velocity step in X | Settle < 2s, OS < 20% |
| Step Z | 3 m/s velocity step in Z | Settle < 2s, OS < 20% |
| Yaw 90° | Yaw rotation command | Settle < 1.5s |
| Disturbance | 5N impulse force | Recover < 2s |

#### Metrics Recorded

```python
@dataclass
class TuningMetrics:
    # Stability
    is_stable: bool           # No divergence
    gain_margin_db: float     # dB (target > 6dB)
    phase_margin_deg: float   # degrees (target > 45°)

    # Time-domain response
    settling_time_s: float    # 2% criterion
    rise_time_s: float        # 10% to 90%
    overshoot_pct: float      # Peak overshoot
    steady_state_error: float # Final error magnitude

    # Control effort
    peak_thrust_N: float      # Maximum thrust used
    peak_moment_Nm: float     # Maximum moment used
    rms_control: float        # RMS control effort
```

### Phase 3: Bound Determination

#### Stability Region Mapping

From sensitivity sweep, identify boundaries where system becomes unstable or violates criteria:

```
          Kd_att
            ▲
            │      ┌─────────────────┐
     3.0    │      │                 │
            │      │   STABLE        │
            │      │   REGION        │
     2.0    │  ────┤                 │
            │      │    ★ nominal    │
            │      │                 │
     1.0    │      │                 │
            │      └─────────────────┘
            │
     0.0    └────────────────────────────► Kp_att
                   4.0    8.0   12.0  16.0
```

#### Safety Margin Application

Set randomization bounds with margin from stability boundary:

```
randomization_bound = nominal ± (stability_margin * distance_to_boundary)

where stability_margin = 0.7 (30% safety margin from boundary)
```

#### Output: Parameter Ranges

| Parameter | Nominal | Stable Min | Stable Max | Rand Min | Rand Max |
|-----------|---------|------------|------------|----------|----------|
| `Kp_att[roll]` | 8.0 | 4.0 | 14.0 | 5.6 | 11.2 |
| `Kp_att[pitch]` | 8.0 | 4.0 | 14.0 | 5.6 | 11.2 |
| `Kp_att[yaw]` | 4.0 | 2.0 | 8.0 | 2.8 | 6.4 |
| `Kd_att[roll]` | 2.5 | 1.0 | 4.0 | 1.4 | 3.3 |
| `Kd_att[pitch]` | 2.5 | 1.0 | 4.0 | 1.4 | 3.3 |
| `Kd_att[yaw]` | 1.0 | 0.4 | 2.0 | 0.6 | 1.6 |
| `Kp_vel[xy]` | 3.0 | 1.5 | 6.0 | 2.1 | 4.8 |
| `Kp_vel[z]` | 2.0 | 1.0 | 4.0 | 1.4 | 3.2 |
| `Ki_vel[xy]` | 0.5 | 0.1 | 1.5 | 0.2 | 1.2 |
| `Ki_vel[z]` | 0.3 | 0.1 | 1.0 | 0.15 | 0.8 |

### Implementation Notes

#### Parallelization Strategy

Run sweep tests in parallel across environments:
```python
# Each env tests different parameter combination
num_sweep_points = 11  # per parameter
num_parameters = 6
total_combinations = num_sweep_points ** num_parameters  # Full grid (expensive)

# Alternative: Latin Hypercube Sampling
num_samples = 1000  # Much cheaper
param_samples = latin_hypercube_sample(bounds, num_samples)

# Run in parallel
for i, env in enumerate(envs):
    env.set_gains(param_samples[i % num_samples])
```

#### Batch Testing Protocol

```python
def run_auto_tuning(num_envs=4096, test_duration=10.0):
    """
    Run auto-tuning across parallel environments.

    Returns:
        Dict mapping parameter combinations to TuningMetrics
    """
    results = {}

    # Phase 1: Find nominal via Bayesian optimization
    nominal = bayesian_optimize(tune_objective, bounds, n_iter=100)

    # Phase 2: Sensitivity sweep
    for param_name in ['Kp_att', 'Kd_att', 'Kp_vel', 'Ki_vel']:
        sweep_results = sweep_parameter(
            param_name,
            center=nominal[param_name],
            range_factor=0.5,
            num_points=11,
            num_envs=num_envs
        )
        results[param_name] = sweep_results

    # Phase 3: Determine bounds
    bounds = compute_stable_bounds(results, safety_margin=0.7)

    return nominal, bounds
```

#### Output Format

Results saved to YAML for use in training config:

```yaml
# auto_tuning_results.yaml
nominal_gains:
  Kp_att: [8.0, 8.0, 4.0]
  Kd_att: [2.5, 2.5, 1.0]
  Kp_vel: [3.0, 3.0, 2.0]
  Ki_vel: [0.5, 0.5, 0.3]

randomization_bounds:
  Kp_att:
    min_factor: 0.7   # multiply nominal by this
    max_factor: 1.4
  Kd_att:
    min_factor: 0.6
    max_factor: 1.3
  Kp_vel:
    min_factor: 0.7
    max_factor: 1.6
  Ki_vel:
    min_factor: 0.4
    max_factor: 2.0

stability_metrics:
  gain_margin_db: 8.5
  phase_margin_deg: 52.0
  max_stable_tau_motor: 0.08

test_date: "2026-03-11"
test_duration_s: 10.0
num_envs_tested: 4096
```

### Integration with Training

The auto-tuning output feeds directly into randomization config:

```python
# In environment config
@configclass
class ControllerRandomizationCfg:
    # Load from auto-tuning results
    auto_tuning_file: str = "auto_tuning_results.yaml"

    # Override with curriculum scaling
    curriculum_scale_factor: float = 1.0  # 0 to 1

    def get_bounds(self, param_name: str, curriculum_progress: float):
        base_bounds = self.load_bounds(param_name)
        # Expand bounds with curriculum
        expanded = lerp(base_bounds, full_range, curriculum_progress)
        return expanded
```

### Validation Criteria

Auto-tuning is considered successful when:

| Criterion | Requirement |
|-----------|-------------|
| Nominal stability | All test scenarios pass |
| Gain margin | > 6 dB |
| Phase margin | > 45° |
| Randomization coverage | > 80% of sampled points stable |
| Bound validity | No instability within determined bounds |
| Repeatability | Results consistent across 3 runs |
