# iris_ma5: Multi-Agent Drone Environment

A GPU-accelerated multi-agent reinforcement learning environment for cooperative drone target tracking with realistic sensor delays and communication imperfections.

## Overview

**iris_ma5** simulates multiple drones equipped with gimbal-stabilized cameras that must cooperatively track and triangulate a target. Each drone observes the target from its own perspective, and the agents share information (with realistic delays) to improve localization accuracy through multi-view triangulation.

For a full technical breakdown (including mathematical formulations per submodule), see `IRIS_MA4_TECHNICAL_REPORT.md`.

This environment is designed for research in:
- Multi-agent reinforcement learning (MARL)
- Cooperative perception and sensor fusion
- Communication-aware decision making
- Sim-to-real transfer with realistic sensor models

## Key Features

- **Multi-Agent Coordination**: 2+ drones with individual action/observation spaces
- **Gimbal-Stabilized Cameras**: Independent camera control for target tracking
- **Realistic Delay System**: Ego states are fast, inter-agent communication is slow with dropout
- **Triangulation Rewards**: Encourages agents to maintain good viewing geometry
- **Safety Constraints**: Collision avoidance and time-to-collision penalties
- **Curriculum Learning**: 5-phase progressive difficulty training
- **Domain Randomization**: Mass, dynamics, and trajectory randomization

## Architecture

```
                              iris_ma5 Environment
    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
    │  │   drone_0    │     │   drone_1    │     │   drone_N    │          │
    │  │              │     │              │     │     ...      │          │
    │  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │          │
    │  │ │PointMass │ │     │ │PointMass │ │     │ │PointMass │ │          │
    │  │ │Controller│ │     │ │Controller│ │     │ │Controller│ │          │
    │  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │          │
    │  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │          │
    │  │ │  Gimbal  │ │     │ │  Gimbal  │ │     │ │  Gimbal  │ │          │
    │  │ │Stabilizer│ │     │ │Stabilizer│ │     │ │Stabilizer│ │          │
    │  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │          │
    │  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │          │
    │  │ │  BBox    │ │     │ │  BBox    │ │     │ │  BBox    │ │          │
    │  │ │Raycaster │ │     │ │Raycaster │ │     │ │Raycaster │ │          │
    │  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │          │
    │  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘          │
    │         │                    │                    │                   │
    │         └────────────────────┼────────────────────┘                   │
    │                              ▼                                        │
    │                  ┌───────────────────────┐                            │
    │                  │  MultiAgentDelaySystem │                           │
    │                  │          V2            │                           │
    │                  ├───────────────────────┤                            │
    │                  │ ┌───────┐ ┌─────────┐ │                            │
    │                  │ │ Clean │ │  Noisy  │ │                            │
    │                  │ │Pipeline│ │Pipeline │ │                            │
    │                  │ │(rewards)│(observations)│                          │
    │                  │ └───────┘ └─────────┘ │                            │
    │                  └───────────┬───────────┘                            │
    │                              │                                        │
    │         ┌────────────────────┼────────────────────┐                   │
    │         ▼                    ▼                    ▼                   │
    │  ┌─────────────┐    ┌───────────────┐    ┌─────────────┐             │
    │  │Triangulation│    │SafetyManager  │    │ Curriculum  │             │
    │  │  Reward     │    │(Collision/TTC)│    │  Manager    │             │
    │  └─────────────┘    └───────────────┘    └─────────────┘             │
    │                                                                        │
    └───────────────────────────────────────────────────────────────────────┘
```

## Components

### Core Environment
| Component | Description |
|-----------|-------------|
| [iris_ma_env4.py](../iris_ma_env4.py) | Main environment implementation |
| [iris_ma_env4_cfg.py](../iris_ma_env4_cfg.py) | Configuration dataclass |

### Delay System V2
Realistic sensor delays with perspective-aware communication. See [delay_system_v2/doc/README.md](../delay_system_v2/doc/README.md).

| Perspective | Latency | Features |
|-------------|---------|----------|
| Ego (own state) | ~5ms | First-order lag only |
| Inter-agent | ~100ms | Lag + staleness + latency + dropout |

### Controllers
| Component | Purpose |
|-----------|---------|
| [PointMass](../controller/point_mass.py) | Velocity commands → thrust/torque |
| [GimbalStabilizer](../controller/gimbal_stabilizer.py) | Camera gimbal control |

### Safety
| Component | Purpose |
|-----------|---------|
| [SafetyManager](../safety/safety_manager.py) | Collision detection and TTC computation |

### Detection & Triangulation
| Component | Purpose |
|-----------|---------|
| [BBoxRaycaster](../bbox_raycaster/bbox_raycaster.py) | GPU-accelerated target detection |
| [Triangulation](../triangulation/triang_cov_reward_torch.py) | Multi-view 3D localization |

## Action Space

Per agent: **7 dimensions**
```
[vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
```

## Observation Space

Per agent: **47 dimensions** including:
- Ego state (position, orientation, velocities, gimbal angles, zoom)
- Detection info (bounding box, validity, age)
- Other agents' states (delayed/noisy)

## Reward Structure

| Reward Type | Scale | Description |
|-------------|-------|-------------|
| BBox Center | 60.0 | Target centering in image |
| BBox Size | 60.0 | Appropriate zoom level |
| Triangulation | 5.0 | Multi-agent localization quality |
| Collision | -100.0 | Penalty for inter-agent collision |
| TTC | -10.0 | Time-to-collision risk penalty |
| Action | -1.0 | Action magnitude penalty |

## Training

Supported RL frameworks:
- SKRL (MAPPO, MAPPO-RNN, PPO)
- RL-Games
- RSL-RL
- Stable-Baselines3

Example training command:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-IrisMA-v4 \
    --num_envs 512
```

## Curriculum Phases

| Phase | Focus | Steps |
|-------|-------|-------|
| 1 | Single-agent tracking | 10k-80k |
| 2 | Delay system introduction | 60k-150k |
| 3 | Multi-agent coordination | 120k-200k |
| 4 | Safety constraints | 180k-230k |
| 5 | Dynamics randomization | 40k-120k |

## File Structure

```
iris_ma5/
├── iris_ma_env4.py              # Main environment
├── iris_ma_env4_cfg.py          # Configuration
├── delay_system_v2/             # Realistic delays
├── controller/                  # Drone & gimbal control
├── safety/                      # Collision & TTC
├── bbox_raycaster/              # Target detection
├── triangulation/               # Multi-view localization
├── randomization/               # Domain randomization
├── curriculum/                  # Training curriculum
├── agents/                      # RL framework configs
├── visualization/               # Debug visualization
└── doc/                         # Documentation
```

## Frame Conventions

This section documents the coordinate frame conventions used throughout iris_ma5.

### Global Conventions

| Convention | Format | Description |
|------------|--------|-------------|
| Quaternion | `(w, x, y, z)` | Scalar-first format (Isaac Sim convention) |
| Euler Angles | `(roll, pitch, yaw)` | XYZ order, radians |
| Rotation Order | ZYX | Yaw → Pitch → Roll (extrinsic) |

### World Frame (W)

**Convention: ENU (East-North-Up)**

```
    Z (Up)
    │
    │
    │
    └───────── Y (North/Left)
   /
  /
 X (East/Forward)
```

- **X-axis**: East (or Forward in simulation)
- **Y-axis**: North (or Left in simulation)
- **Z-axis**: Up
- **Gravity**: `[0, 0, -9.81]` m/s²

### Body Frame (B) - Drone

**Convention: FLU (Forward-Left-Up)**

```
    Z_b (Up)
    │
    │    ┌────────┐
    │   /  Drone  /
    │  /   Body  /
    └────────── Y_b (Left)
   /
  /
 X_b (Forward)
```

- **X-axis**: Forward (nose direction)
- **Y-axis**: Left (port wing)
- **Z-axis**: Up (dorsal)
- Attached to the drone's center of mass
- Rotates with the drone body

**State Variables in Body Frame:**
- `body_linear_velocity_b`: Linear velocity in body frame
- `body_angular_velocity_b`: Angular velocity in body frame (roll_rate, pitch_rate, yaw_rate)
- `body_linear_acceleration_b`: Linear acceleration in body frame

### Gimbal Frame (G)

**Convention: FLU with ZYX rotation sequence**

The gimbal is mounted on the drone body and provides camera stabilization through three joints:

1. **Yaw Joint** (α): Rotation around body Z-axis
   - Positive: Counter-clockwise when viewed from above
   - Limits: Typically ±π rad

2. **Roll Joint**: Rotation around rotated X-axis (after yaw)
   - Used for horizon stabilization
   - Automatically computed to keep horizon level

3. **Pitch Joint** (β): Rotation around rotated Y-axis (after yaw and roll)
   - **Sign Convention**: Positive pitch = camera tilts DOWN, negative pitch = camera tilts UP
   - Default limits: [-70°, +70°] (can look 70° up and 70° down)
   - Configured in `gimbal_stabilizer_cfg.py`

**Pitch Sign Convention Reference:**

The authoritative source for pitch sign convention is `gimbal_stabilizer.py`:
```python
# From gimbal_stabilizer.py:102
gimbal_pitch = torch.atan2(-z, horizontal_dist)  # Negative z because down is positive pitch
```

Mathematical derivation:
- Given target at position (x, y, z) relative to drone
- Horizontal distance: `d_h = sqrt(x² + y²)`
- Vertical offset: `dz = z_target - z_drone`
- Pitch angle: `pitch = atan2(-dz, d_h)`

When target is BELOW drone (`dz < 0`): `-dz > 0`, so `pitch > 0` (positive, looking down)
When target is ABOVE drone (`dz > 0`): `-dz < 0`, so `pitch < 0` (negative, looking up)

**Rotation Composition:**
```
R_body_to_gimbal = R_yaw(α) @ R_roll @ R_pitch(β)
```

#### Gimbal Joint Ordering Convention: `[pitch, yaw, roll]`

Throughout iris_ma5, gimbal joint positions/velocities are stored in tensors with this ordering:

| Index | Joint | Description |
|-------|-------|-------------|
| 0 | pitch | Camera tilt angle (primary for target tracking) |
| 1 | yaw | Camera pan angle (primary for target tracking) |
| 2 | roll | Horizon stabilization (computed automatically) |

This convention is used in:
- `AgentStatesData.joint_positions_b`: `[N, 3]` tensor with `[pitch, yaw, roll]`
- `AgentStatesData.joint_velocities_b`: `[N, 3]` tensor with `[pitch_rate, yaw_rate, roll_rate]`
- All `set_joint_position_target()` calls in `iris_ma_env4.py`
- All gimbal joint extraction code

**Rationale**: Pitch and yaw are the primary control axes for camera pointing, while roll is typically used only for horizon stabilization. Placing pitch/yaw first aligns with camera-centric thinking and matches the expectations of `compute_camera_orientation_from_gimbal()` in `derived_field_computers.py`.

**Example Usage:**
```python
# Extracting gimbal angles from joint_positions_b
gimbal_pitch = state.data.joint_positions_b[:, 0:1]  # [N, 1]
gimbal_yaw = state.data.joint_positions_b[:, 1:2]    # [N, 1]
gimbal_roll = state.data.joint_positions_b[:, 2:3]   # [N, 1]

# Setting joint targets (target tensor and joint_ids must match order)
robot.set_joint_position_target(
    target=torch.stack([pitch_target, yaw_target, roll_target], dim=-1),
    joint_ids=[pitch_joint_id, yaw_joint_id, roll_joint_id]
)
```

### Camera Frame (C)

**Convention: Optical (RDF - Right-Down-Forward)**

The camera frame follows computer vision conventions:

```
       Z_c (Forward/Optical axis)
      /
     /
    /
   └───────── X_c (Right)
   │
   │
   │
   Y_c (Down)
```

- **X-axis**: Right (image u-direction)
- **Y-axis**: Down (image v-direction)
- **Z-axis**: Forward (optical axis, depth direction)

**Transformation from Gimbal (ENU) to Camera (RDF):**
```python
R_gimbal_to_camera = [
    [0,  -1,  0],   # X_cam = -Y_gimbal (Right = -Left)
    [0,   0, -1],   # Y_cam = -Z_gimbal (Down = -Up)
    [1,   0,  0]    # Z_cam =  X_gimbal (Forward = Forward)
]
```

**Camera Position Offset:**
- Specified in body frame via `camera_offset_position_b`
- Default offset quaternion: `[0.5, -0.5, 0.5, -0.5]` (represents the RDF rotation)

### Image/Pixel Coordinates

**Origin: Top-Left**

```
(0,0) ────────────────────── u (width)
  │
  │      Image Plane
  │
  │
  v (height)
```

- **u**: Horizontal pixel coordinate (0 to width-1)
- **v**: Vertical pixel coordinate (0 to height-1)

**Bounding Box Format: Normalized xywh**
```
bbox = [cx, cy, w, h]
```
- `cx`: Center x (normalized 0-1, where 0.5 = image center)
- `cy`: Center y (normalized 0-1, where 0.5 = image center)
- `w`: Width (normalized 0-1)
- `h`: Height (normalized 0-1)

**Camera Intrinsic Matrix (K):**
```
K = [fx,  0, cx]
    [ 0, fy, cy]
    [ 0,  0,  1]
```
Where:
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (typically image center)

### State Naming Conventions

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_w` | World frame | `body_position_w` |
| `_b` | Body frame | `body_angular_velocity_b` |
| `_c` | Camera frame | `points_camera` |
| `_g` | Gimbal frame | `R_bg` (body to gimbal) |

### Triangulation Geometry

The triangulation module uses the following convention:

```
           Target (X_w)
              ★
             /│\
            / │ \
           /  │  \
          /   │   \
    ray_1/    │    \ray_2
        /     │     \
       /      │      \
    Camera 1  │    Camera 2
      ○───────┴───────○
```

**Ray Direction:**
- Computed from bbox center using camera intrinsics
- Normalized to unit length
- Expressed in world frame

**Midpoint Method:**
- Finds the 3D point minimizing distance to all rays
- Returns validity flag for behind-camera cases

### Force/Torque Application

**Point Mass Controller:**
- Forces applied in body frame
- Torques applied around body-frame axes (roll, pitch, yaw)
- Gravity compensation in body frame when enabled

**External Force Application:**
```python
robot.set_external_force_and_torque(
    forces=force_body,      # [N, 1, 3] - in body frame
    torques=moment_body,    # [N, 1, 3] - in body frame
    body_ids=body_id
)
```

### Common Frame Transformations

| From | To | Function |
|------|-----|----------|
| World → Body | `quat_rotate_inverse(quat_w, vec_w)` |
| Body → World | `quat_rotate(quat_w, vec_b)` |
| Gimbal → Camera | Multiply by `R_gimbal_to_camera` |
| Camera → Image | `pixel = K @ (point_c / point_c.z)` |

### Important Notes

1. **USD Model Convention**: The IRIS drone USD model has a 180° roll offset. The controller accounts for this by targeting `roll=π` instead of `roll=0` for level flight.

2. **Gimbal Lock Prevention**: The gimbal stabilizer works directly in the gimbal base frame to avoid Euler angle singularities.

3. **Zoom Handling**: Camera intrinsics are scaled by zoom level:
   ```python
   K_zoomed[:, 0, 0] = K_base[:, 0, 0] * zoom_level  # fx
   K_zoomed[:, 1, 1] = K_base[:, 1, 1] * zoom_level  # fy
   ```

4. **Quaternion Operations**: Always use Isaac Lab's math utilities (`quat_mul`, `quat_inv`, `quat_rotate`) for consistency.

## Related Documentation

- [Delay System V2](../delay_system_v2/doc/README.md) - Detailed delay pipeline documentation
- [Safety Module](../safety/README.md) - Collision detection and TTC
- [Controller](../controller/README.md) - Point mass and gimbal control
