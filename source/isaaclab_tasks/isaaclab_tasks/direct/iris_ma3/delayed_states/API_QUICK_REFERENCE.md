# MultiAgentStateManager - Complete API Reference & Usage Guide

## Overview

The `MultiAgentStateManager` simplifies multi-agent environment development by internally managing:
- **Ground truth (GT) states**: Direct simulation states → for GT bbox detection
- **Delayed states**: GT + motion lag + detection latency (no noise) → **for rewards**
- **Delayed+noisy states**: Delayed + sensor noise → **for observations**
- **Received states**: States from other agents via communication channel → for multi-agent coordination

**Key Design Principle**: Single update, multiple outputs. Calling `update_gt_states()` automatically produces all three state versions.

---

## Table of Contents

1. [Import](#import)
2. [Initialization](#initialization)
3. [Setup](#setup)
4. [Per-Step Updates](#per-step-updates)
5. [State Retrieval](#state-retrieval)
6. [AgentStates Data Fields](#agentstates-data-fields)
7. [Curriculum Learning](#curriculum-learning)
8. [Dynamic Parameter Updates](#dynamic-parameter-updates)
9. [Statistics](#statistics)
10. [Reset](#reset)
11. [Complete Usage Example](#complete-usage-example)
12. [Best Practices](#best-practices)

---

## Import

```python
from isaaclab_tasks.direct.iris_ma3.delayed_states import (
    MultiAgentStateManager,
    AgentStates,
    MultiAgentStates,
)
```

---

## Initialization

### Initialize in Environment `__init__`

```python
from isaaclab_tasks.direct.iris_ma3.delayed_states import MultiAgentStateManager

class IrisMA3Env(DirectMARLEnv):
    def __init__(self, cfg: IrisMA3EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Initialize state manager
        self.state_manager = MultiAgentStateManager(
            possible_agents=cfg.possible_agents,  # ["drone_0", "drone_1"]
            num_envs=self.num_envs,
            num_joints_per_agent={agent: 3 for agent in cfg.possible_agents},  # [roll, pitch, yaw]
            num_targets_per_agent={agent: 1 for agent in cfg.possible_agents},
            dt=self.physics_dt,
            device=self.device,

            # Optional parameters with defaults:

            # Motion filter parameters
            motion_time_constant=0.1,          # Motion filter lag (seconds)
            gimbal_time_constant=0.03,         # Gimbal filter lag (seconds)

            # Detection parameters
            detection_fps=30.0,                # Detection frame rate (Hz)
            detection_mean_latency=0.05,       # Detection latency mean (seconds)
            detection_std_latency=0.02,        # Detection latency std (seconds)
            detection_failure_rate=0.0,        # Detection dropout rate [0, 1]

            # Communication parameters
            comm_mean_delay=0.1,               # Comm delay mean (seconds)
            comm_std_delay=0.03,               # Comm delay std (seconds)
            comm_throttle_period=0.0,          # Comm bandwidth limit (0 = no limit)
            comm_dropout_rate=0.05,            # Comm dropout rate [0, 1]

            # Noise parameters
            enable_noise=True,                 # Enable sensor noise
            position_noise_std=0.01,           # Position noise (m)
            orientation_noise_std=0.01,        # Orientation noise (rad)
            linear_velocity_noise_std=None,    # Auto: 10% of position_noise_std
            angular_velocity_noise_std=None,   # Auto: ori_std × pos_std
            linear_acceleration_noise_std=100e-6,  # Linear accel noise (m/s^2)
            gimbal_noise_std=0.01,             # Gimbal angle noise (rad)
            zoom_noise_std=0.01,               # Zoom level noise (relative)
            bbox_noise_std=1.0,                # Bbox noise (pixels)
            noise_seed=0,                      # Random seed (None for random)
            max_buffer_size=100                # Max history buffer size
        )
```

### Parameter Details

| Parameter | Type | Description |
|-----------|------|-------------|
| `possible_agents` | `list[str]` | Agent IDs (e.g., `["drone_0", "drone_1"]`) |
| `num_envs` | `int` | Number of parallel environments |
| `num_joints_per_agent` | `dict[str, int]` | Number of joints per agent (e.g., gimbal: 3) |
| `num_targets_per_agent` | `dict[str, int]` | Number of targets per agent |
| `dt` | `float` | Simulation timestep (seconds) |
| `device` | `torch.device` | Device for tensor allocation |

**Noise Formula Details**:
- `linear_velocity_noise_std`: Defaults to `0.1 * position_noise_std`
- `angular_velocity_noise_std`: Defaults to `orientation_noise_std * position_noise_std`
- All noise is **seeded** for reproducibility
- Noise is applied **after** filtering to ensure it doesn't get filtered out

---

## Setup

### Configure Camera

**Must be called before using camera-related features**:

```python
# Configure cameras for each agent
for agent_id, cam_cfg in self.agent_camera_cfgs.items():
    self.state_manager.set_camera_configs(
        agent_id=agent_id,
        width=cam_cfg.width,                # Image width (pixels)
        height=cam_cfg.height,              # Image height (pixels)
        focal_length=cam_cfg.spawn.focal_length,  # Focal length (meters)
        horizontal_aperture=cam_cfg.spawn.horizontal_aperture,  # Sensor width (meters)
        offset_position_b=torch.tensor([0.0, 0.0, 0.0], device=self.device),  # Camera offset from body
        offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Camera rotation offset (quat)
    )
```

---

## Per-Step Updates

### Typical Update Flow in `_compute_intermediate_values()`

```python
def _compute_intermediate_values(self):
    """Compute intermediate values shared by rewards and observations."""

    # 1. Update simulation time
    self.state_manager.update_time(self.step_dt)  # Or use default dt from init

    # 2. Update GT states for all agents
    for agent_id in self.cfg.possible_agents:
        robot = self._robots[agent_id]

        # Single call automatically produces GT, delayed, and delayed+noisy states
        self.state_manager.update_gt_states(
            agent_id=agent_id,
            body_position_w=robot.data.root_pos_w,           # [N, 3]
            body_orientation_w=robot.data.root_quat_w,       # [N, 4] - quat (w, x, y, z)
            body_linear_velocity_w=robot.data.root_lin_vel_w,# [N, 3]
            body_angular_velocity_w=robot.data.root_ang_vel_w,# [N, 3]
            body_combined_angular_velocity_w=None,           # Optional [N, 3]
            body_linear_acceleration_w=None,                 # Optional [N, 3]
            joint_positions_b=robot.data.joint_pos[:, :3],   # [N, J] - gimbal [roll, pitch, yaw]
            joint_velocities_b=robot.data.joint_vel[:, :3],  # Optional [N, J]
            zoom_level=self.zoom_level[agent_id],            # [N]
            env_idxs=None                                    # Optional - update all if None
        )

    # 3. Run GT bbox detection
    gt_states = self.state_manager.get_all_gt_states()

    # Prepare camera poses for bbox raycaster
    camera_poses = {}
    camera_intrinsics = {}
    for agent_id in self.cfg.possible_agents:
        agent_state = gt_states.agents[agent_id]
        camera_poses[agent_id] = (
            agent_state.data.camera_position_w,
            agent_state.data.camera_orientation_w
        )
        camera_intrinsics[agent_id] = agent_state.data.camera_intrinsics

    # Run bbox raycaster with GT camera poses
    self.bbox_raycaster.update(
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
        target_poses=(self.target.data.root_pos_w.unsqueeze(1),
                     self.target.data.root_quat_w.unsqueeze(1)),
        agent_poses=camera_poses,
        image_shapes={agent_id: (height, width) for agent_id in self.cfg.possible_agents},
        target_scale=self.target_scale
    )

    # 4. Process detections through pipeline (adds FPS throttle, latency, noise)
    for agent_id in self.cfg.possible_agents:
        bboxes_gt = self.bbox_raycaster.data.bboxes[agent_id]  # [N, T, 4] - xywh
        valid_gt = self.bbox_raycaster.data.valid_mask[agent_id]  # [N, T]

        self.state_manager.update_detections(
            agent_id=agent_id,
            bboxes_2d_gt=bboxes_gt,
            valid_mask_gt=valid_gt,
            env_idxs=None  # Optional
        )

    # 5. Process communication
    for agent_id in self.cfg.possible_agents:
        # Broadcast delayed states (no noise) to other agents
        self.state_manager.broadcast_state(
            sender_id=agent_id,
            state_keys=['position', 'camera_ray_directions_w', 'bboxes_2d_valid_mask'],
            has_source_data=None  # Optional [N] boolean mask
        )
```

### What Each Update Does

#### `update_time(time_increment)`

Updates the internal simulation time. Call once per step.

```python
state_manager.update_time(dt=0.01)  # Or use default dt from init
```

#### `update_gt_states()` - **Single Call, Triple Output**

**This single call automatically:**
1. Stores GT states
2. Applies motion filters (first-order lag)
3. Applies gimbal filters (first-order lag)
4. Stores delayed states (no noise)
5. Adds sensor noise
6. Stores delayed+noisy states
7. Updates camera poses for all state types
8. Updates camera intrinsics for all state types

```python
state_manager.update_gt_states(
    agent_id="drone_0",
    body_position_w=robot.data.root_pos_w,           # [N, 3] - required
    body_orientation_w=robot.data.root_quat_w,       # [N, 4] - required
    body_linear_velocity_w=robot.data.root_lin_vel_w,# [N, 3] - required
    body_angular_velocity_w=robot.data.root_ang_vel_w,# [N, 3] - required
    body_combined_angular_velocity_w=None,           # [N, 3] - optional
    body_linear_acceleration_w=None,                 # [N, 3] - optional
    joint_positions_b=robot.data.joint_pos[:, :3],   # [N, J] - optional
    joint_velocities_b=robot.data.joint_vel[:, :3],  # [N, J] - optional
    zoom_level=torch.ones(N, device=device),         # [N] - optional
    env_idxs=None                                    # Optional - update all if None
)
```

#### `update_detections()` - Process Bbox Detections

**This automatically:**
1. Stores GT bboxes
2. Applies FPS throttling
3. Applies detection dropout
4. Applies detection latency
5. Stores delayed bboxes (no noise)
6. Adds bbox pixel noise
7. Stores delayed+noisy bboxes
8. Updates camera ray directions for all state types

```python
state_manager.update_detections(
    agent_id="drone_0",
    bboxes_2d_gt=bboxes_gt,      # [N, T, 4] - xywh format (pixels)
    valid_mask_gt=valid_mask,    # [N, T] - boolean
    env_idxs=None                # Optional
)
```

#### `broadcast_state()` - Communication

Broadcast delayed states (no noise) to other agents via communication channel.

```python
state_manager.broadcast_state(
    sender_id="drone_0",
    state_keys=['position', 'camera_ray_directions_w', 'bboxes_2d_valid_mask'],
    has_source_data=None  # Optional [N] boolean mask - only broadcast where True
)
```

**Available state keys:**
- `'position'` → `body_position_w`
- `'orientation'` → `body_orientation_w`
- `'linear_velocity'` → `body_linear_velocity_w`
- `'angular_velocity'` → `body_angular_velocity_w`
- `'camera_position'` → `camera_position_w`
- `'camera_orientation'` → `camera_orientation_w`
- `'camera_intrinsics'` → `camera_intrinsics`
- `'bboxes_2d'` → `bboxes_2d`
- `'bboxes_2d_valid_mask'` → `bboxes_2d_valid_mask`
- `'camera_ray_directions_w'` → `camera_ray_directions_w`

If `state_keys=None`, broadcasts **all** keys.

#### `receive_other_agent_states()` - Receive Communication

Receive states from other agents (with comm delay and dropout).

```python
received = state_manager.receive_other_agent_states(receiver_id="drone_0")

# Returns: Dict[sender_agent_id -> Dict[state_key -> (data, valid_mask, data_age)]]
for sender_id, state_dict in received.items():
    position, pos_valid, pos_age = state_dict['position']
    # position: [N, 3]
    # pos_valid: [N] - boolean (True if data available and not dropped)
    # pos_age: [N] - time elapsed since data capture (seconds)
```

---

## State Retrieval

### Individual Agent States

```python
# Ground truth (no delay, no noise) - for GT bbox detection
gt_state = state_manager.get_gt_states(agent_id="drone_0")

# Delayed (lag + latency, no noise) - **FOR REWARDS**
delayed_state = state_manager.get_delayed_states(agent_id="drone_0")

# Delayed + noisy (lag + latency + noise) - **FOR OBSERVATIONS**
noisy_state = state_manager.get_delayed_noisy_states(agent_id="drone_0")
```

### All Agents

```python
# Returns MultiAgentStates object
gt_states = state_manager.get_all_gt_states()
delayed_states = state_manager.get_all_delayed_states()
noisy_states = state_manager.get_all_delayed_noisy_states()

# Access per-agent states
for agent_id in possible_agents:
    agent_state = delayed_states.agents[agent_id]
    position = agent_state.data.body_position_w  # [N, 3]
```

### State Types Summary

| State Type | Characteristics | Usage |
|------------|----------------|--------|
| **GT States** | Direct simulation states | Bbox raycasting, debugging |
| **Delayed States** | Lag + latency, **no noise** | **Reward computation** (fair comparison) |
| **Delayed+Noisy States** | Lag + latency + sensor noise | **Observations** (realistic sensors) |
| **Received States** | From communication channel (delay + dropout) | Multi-agent coordination |

---

## AgentStates Data Fields

Each `AgentStates` object has a `data` attribute with the following fields:

```python
agent_state = state_manager.get_delayed_states("drone_0")

# Body states (world frame)
agent_state.data.body_position_w              # [N, 3]
agent_state.data.body_orientation_w           # [N, 4] - quat (w, x, y, z)
agent_state.data.body_linear_velocity_w       # [N, 3]
agent_state.data.body_angular_velocity_w      # [N, 3]
agent_state.data.body_linear_acceleration_w   # [N, 3]
agent_state.data.body_combined_angular_velocity_w  # [N, 3] - body + gimbal

# Body states (body frame)
agent_state.data.body_linear_velocity_b       # [N, 3]
agent_state.data.body_angular_velocity_b      # [N, 3]
agent_state.data.body_linear_acceleration_b   # [N, 3]
agent_state.data.body_angular_acceleration_b  # [N, 3]
agent_state.data.body_combined_angular_velocity_b  # [N, 3]

# Joint states (gimbal)
agent_state.data.joint_positions_b            # [N, J] - gimbal [roll, pitch, yaw]
agent_state.data.joint_velocities_b           # [N, J]
agent_state.data.joint_accelerations_b        # [N, J]

# Camera states
agent_state.data.camera_position_w            # [N, 3]
agent_state.data.camera_orientation_w         # [N, 4] - quat
agent_state.data.camera_intrinsics            # [N, 3, 3]
agent_state.data.camera_zoom_level            # [N]

# Detection states
agent_state.data.bboxes_2d                    # [N, T, 4] - xywh (pixels)
agent_state.data.bboxes_2d_valid_mask         # [N, T] - boolean
agent_state.data.bboxes_2d_age                # [N, T] - time since detection (seconds)
agent_state.data.camera_ray_directions_w      # [N, T, 3] - from bbox centers
```

**Conventions**:
- All angles are in **radians**
- Quaternions use **(w, x, y, z)** convention
- Gimbal joint order is **[roll, pitch, yaw]** at indices **[0, 1, 2]**
- Bbox format is **[x, y, w, h]** in pixel coordinates

---

## Detection Age Usage

The `bboxes_2d_age` field tracks **time elapsed since each bbox was detected** (in seconds). This is useful for curriculum learning, staleness penalties, and monitoring detection pipeline performance.

### What Detection Age Represents

**Age = Current time - Detection capture time**

Includes:
- FPS throttling delay (e.g., 30 Hz detection rate)
- Processing latency (mean + std from config)
- Buffer delays

**Age is 0 when:**
- No detection received yet
- Detection was dropped/invalid
- Environment was just reset

### Basic Access

```python
# Get delayed+noisy states for observations
noisy_state = state_manager.get_delayed_noisy_states("drone_0")

# Access detection age for first target
bbox_age = noisy_state.data.bboxes_2d_age[:, 0]  # [N] - seconds

# Access all targets
all_ages = noisy_state.data.bboxes_2d_age  # [N, T]
```

### Using in Observations

```python
def _get_observations(self):
    noisy_states = state_manager.get_all_delayed_noisy_states()

    for agent_id in possible_agents:
        noisy_state = noisy_states.agents[agent_id]

        # Get detection with age
        bbox = noisy_state.data.bboxes_2d[:, 0, :]  # [N, 4]
        bbox_valid = noisy_state.data.bboxes_2d_valid_mask[:, 0:1]  # [N, 1]
        bbox_age = noisy_state.data.bboxes_2d_age[:, 0:1]  # [N, 1] - NEW!

        # Include in observation
        obs = torch.cat([
            # ... other observations ...
            bbox,        # [N, 4]
            bbox_valid,  # [N, 1]
            bbox_age,    # [N, 1] - Agent knows data freshness
        ], dim=-1)
```

### Penalizing Stale Detections in Rewards

```python
def _get_rewards(self):
    delayed_states = state_manager.get_all_delayed_states()

    for agent_id in possible_agents:
        delayed_state = delayed_states.agents[agent_id]

        bbox_valid = delayed_state.data.bboxes_2d_valid_mask[:, 0]  # [N]
        bbox_age = delayed_state.data.bboxes_2d_age[:, 0]  # [N]

        # Exponential decay penalty for stale detections
        max_age = 0.5  # Max acceptable age (seconds)
        staleness_penalty = torch.exp(-bbox_age / max_age)  # 1.0 → 0.37

        # Apply only when detection is valid
        detection_quality = torch.where(
            bbox_valid,
            staleness_penalty,
            torch.zeros_like(staleness_penalty)
        )

        reward = detection_quality * cfg.detection_quality_scale
```

### Curriculum: Gradually Increase Age Tolerance

```python
class IrisMA3Env(DirectMARLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.max_acceptable_age = 0.05  # Start with 50ms max age

    def update_curriculum(self, progress: float):
        """Gradually increase tolerance for old detections."""
        # 0.05s (easy) → 0.3s (hard)
        self.max_acceptable_age = 0.05 + progress * 0.25

    def _get_rewards(self):
        # Use curriculum-adjusted threshold
        staleness_penalty = torch.exp(-bbox_age / self.max_acceptable_age)
```

### Detection Age vs Communication Age

**Detection age** (`bboxes_2d_age`):
- Time since **local** bbox was detected
- Includes FPS throttle + detection latency

**Communication age** (from `receive_other_agent_states`):
- Time since **received** data was captured by **other agent**
- Includes detection latency + communication delay

```python
# Local detection age
bbox_age = noisy_state.data.bboxes_2d_age[:, 0]  # [N]

# Received detection age (from other agent)
received = state_manager.receive_other_agent_states("drone_0")
if "drone_1" in received:
    _, _, comm_age = received["drone_1"]['position']  # [N]
    # comm_age >= bbox_age (includes communication delay)
```

### Important Notes

- **Always check `bboxes_2d_valid_mask` before using age**
- Age is **0** when no data or detection failed
- Age **resets to 0** after environment reset
- Age is **per-target**: shape `[N, T]` matches `bboxes_2d`

---

## Curriculum Learning

### Set Noise Progress Scale

Gradually increase noise during training for curriculum learning:

```python
# Gradually increase noise during training (0.0 to 1.0)
state_manager.set_noise_progress_scale(progress=0.5)  # 50% of full noise

# Example: Linear curriculum over 1000 iterations
for iter in range(1000):
    progress = iter / 1000.0
    state_manager.set_noise_progress_scale(progress)
```

**What it does**: Scales all noise standard deviations by the given factor:
- `progress=0.0`: No noise (easy task)
- `progress=0.5`: Half noise (medium difficulty)
- `progress=1.0`: Full noise (realistic sensors)

**Applies to**:
- Position noise
- Orientation noise
- Velocity noise (linear & angular)
- Acceleration noise
- Gimbal noise
- Zoom noise
- Bbox pixel noise

---

## Dynamic Parameter Updates

### Update Detection Parameters

```python
state_manager.update_detection_params(
    detection_fps=torch.tensor([[20.0]], device=device).repeat(N, num_agents),
    detection_mean_latency=torch.tensor([[0.1]], device=device).repeat(N, num_agents),
    detection_std_latency=None,  # Leave unchanged if None
    detection_failure_rate=None
)
```

### Update Communication Parameters

```python
state_manager.update_comm_params(
    mean_latency=torch.tensor([[0.15]], device=device).repeat(N, num_agents),
    std_latency=torch.tensor([[0.05]], device=device).repeat(N, num_agents),
    throttle_period=None,  # Leave unchanged
    dropout_rate=None
)
```

### Curriculum Learning Example

```python
def update_curriculum(self, progress: float):
    """Update delay/noise parameters based on training progress."""

    # Gradually increase detection latency
    new_latency = 0.01 + progress * 0.09  # 0.01s → 0.1s
    self.state_manager.update_detection_params(
        detection_mean_latency=torch.full((self.num_envs, self.num_agents),
                                         new_latency, device=self.device)
    )

    # Gradually increase communication delay
    new_comm_delay = 0.05 + progress * 0.15  # 0.05s → 0.2s
    self.state_manager.update_comm_params(
        mean_latency=torch.full((self.num_envs, self.num_agents),
                               new_comm_delay, device=self.device)
    )
```

---

## Statistics

```python
stats = state_manager.get_statistics()

# Returns dict with:
# - 'current_time': [N] current simulation time
# - 'detection_stats': {f'agent_{idx}': {...stats...}}
# - 'comm_stats': {
#       'message_count': [N, num_agents, num_agents],
#       'dropout_count': [N, num_agents, num_agents],
#       'dropout_rate_effective': [N, num_agents, num_agents]
#   }
```

### Statistics Monitoring Example

```python
def log_statistics(self):
    """Log pipeline statistics for monitoring."""
    stats = self.state_manager.get_statistics()

    # Communication stats
    comm_stats = stats['comm_stats']
    dropout_rate = comm_stats['dropout_rate_effective'].mean()

    # Detection stats
    for agent_id in self.cfg.possible_agents:
        agent_idx = self.state_manager.agent_id_to_idx[agent_id]
        agent_stats = stats['detection_stats'][f'agent_{agent_idx}']
        throttle_count = agent_stats['throttle_count'].mean()

        print(f"{agent_id}: Dropout={dropout_rate:.2%}, Throttled={throttle_count}")
```

---

## Reset

```python
# Reset specific environments
state_manager.reset(env_ids=torch.tensor([0, 5, 10], device=device))

# Reset all environments
state_manager.reset(env_ids=None)
```

**What reset does**:
- Clears internal buffers (delay buffers, communication buffers)
- Resets filter states
- Resets time tracking
- Clears statistics

---

## Complete Usage Example

### 1. Initialization

```python
class IrisMA3Env(DirectMARLEnv):
    def __init__(self, cfg: IrisMA3EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Initialize state manager
        self.state_manager = MultiAgentStateManager(
            possible_agents=cfg.possible_agents,
            num_envs=self.num_envs,
            num_joints_per_agent={agent: 3 for agent in cfg.possible_agents},
            num_targets_per_agent={agent: 1 for agent in cfg.possible_agents},
            dt=self.physics_dt,
            device=self.device,
            motion_time_constant=cfg.motion_time_constant,
            gimbal_time_constant=cfg.gimbal_time_constant,
            detection_fps=cfg.detection_fps,
            detection_mean_latency=cfg.detection_mean_latency,
            detection_std_latency=cfg.detection_std_latency,
            detection_failure_rate=cfg.detection_failure_rate,
            comm_mean_delay=cfg.comm_mean_delay,
            comm_std_delay=cfg.comm_std_delay,
            comm_dropout_rate=cfg.comm_dropout_rate,
            enable_noise=cfg.enable_noise_in_observations,
            position_noise_std=cfg.pos_std,
            orientation_noise_std=cfg.ori_std,
            gimbal_noise_std=cfg.gimbal_std,
            bbox_noise_std=cfg.bbox_noise_std,
        )

        # Configure cameras for each agent
        for agent_id, cam_cfg in self.agent_camera_cfgs.items():
            self.state_manager.set_camera_configs(
                agent_id=agent_id,
                width=cam_cfg.width,
                height=cam_cfg.height,
                focal_length=cam_cfg.spawn.focal_length,
                horizontal_aperture=cam_cfg.spawn.horizontal_aperture,
                offset_position_b=torch.tensor([0.0, 0.0, 0.0], device=self.device),
                offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            )
```

### 2. Update States in `_compute_intermediate_values`

```python
def _compute_intermediate_values(self):
    """Compute intermediate values shared by rewards and observations."""

    # Update simulation time
    self.state_manager.update_time(self.step_dt)

    # Update GT states for all agents
    for agent_id in self.cfg.possible_agents:
        robot = self._robots[agent_id]

        self.state_manager.update_gt_states(
            agent_id=agent_id,
            body_position_w=robot.data.root_pos_w,
            body_orientation_w=robot.data.root_quat_w,
            body_linear_velocity_w=robot.data.root_lin_vel_w,
            body_angular_velocity_w=robot.data.root_ang_vel_w,
            joint_positions_b=robot.data.joint_pos[:, :3],
            zoom_level=self.zoom_level[agent_id]
        )

    # Run bbox detection with GT states
    gt_states = self.state_manager.get_all_gt_states()

    camera_poses = {}
    camera_intrinsics = {}
    for agent_id in self.cfg.possible_agents:
        agent_state = gt_states.agents[agent_id]
        camera_poses[agent_id] = (
            agent_state.data.camera_position_w,
            agent_state.data.camera_orientation_w
        )
        camera_intrinsics[agent_id] = agent_state.data.camera_intrinsics

    self.bbox_raycaster.update(
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
        target_poses=(self.target.data.root_pos_w.unsqueeze(1),
                     self.target.data.root_quat_w.unsqueeze(1)),
        agent_poses=camera_poses,
        image_shapes={(agent_id, (h, w)) for agent_id in self.cfg.possible_agents},
        target_scale=self.target_scale
    )

    # Process detections
    for agent_id in self.cfg.possible_agents:
        bboxes_gt = self.bbox_raycaster.data.bboxes[agent_id]
        valid_gt = self.bbox_raycaster.data.valid_mask[agent_id]

        self.state_manager.update_detections(
            agent_id=agent_id,
            bboxes_2d_gt=bboxes_gt,
            valid_mask_gt=valid_gt
        )

    # Process communication
    for agent_id in self.cfg.possible_agents:
        self.state_manager.broadcast_state(
            sender_id=agent_id,
            state_keys=['position', 'camera_ray_directions_w', 'bboxes_2d_valid_mask'],
            has_source_data=None
        )
```

### 3. Compute Rewards Using Delayed States (No Noise)

```python
def _get_rewards(self) -> Dict[str, torch.Tensor]:
    """Compute rewards using delayed states (no noise)."""

    self._compute_intermediate_values()

    rewards_dict = {}

    # Get delayed states (no noise) for fair reward computation
    delayed_states = self.state_manager.get_all_delayed_states()

    # Stack states for triangulation covariance
    N = self.num_envs
    C = len(self.cfg.possible_agents)

    robot_positions = torch.stack([
        delayed_states.agents[agent_id].data.body_position_w
        for agent_id in self.cfg.possible_agents
    ], dim=1)  # [N, C, 3]

    robot_quats = torch.stack([
        delayed_states.agents[agent_id].data.body_orientation_w
        for agent_id in self.cfg.possible_agents
    ], dim=1)  # [N, C, 4]

    gimbal_yaws = torch.stack([
        delayed_states.agents[agent_id].data.joint_positions_b[:, 2]
        for agent_id in self.cfg.possible_agents
    ], dim=1)  # [N, C]

    gimbal_pitches = torch.stack([
        delayed_states.agents[agent_id].data.joint_positions_b[:, 1]
        for agent_id in self.cfg.possible_agents
    ], dim=1)  # [N, C]

    camera_intrinsics = torch.stack([
        delayed_states.agents[agent_id].data.camera_intrinsics
        for agent_id in self.cfg.possible_agents
    ], dim=1)  # [N, C, 3, 3]

    bbox_valid = torch.stack([
        delayed_states.agents[agent_id].data.bboxes_2d_valid_mask
        for agent_id in self.cfg.possible_agents
    ], dim=1)  # [N, C, T]

    # Compute triangulation covariance
    Sigma_X, trace_cov, is_tri_valid = self._compute_triangulation_covariance(
        X_w=self.target.data.root_pos_w.unsqueeze(1),
        robot_positions=robot_positions,
        robot_quats=robot_quats,
        gimbal_yaws=gimbal_yaws,
        gimbal_pitches=gimbal_pitches,
        camera_intrinsics=camera_intrinsics,
        bbox_valid_mask=bbox_valid
    )

    # Compute per-agent rewards
    for i, agent_id in enumerate(self.cfg.possible_agents):
        delayed_state = delayed_states.agents[agent_id]

        # Bbox rewards
        bbox = delayed_state.data.bboxes_2d[:, 0, :]  # [N, 4]
        bbox_valid = delayed_state.data.bboxes_2d_valid_mask[:, 0]  # [N]

        bbox_center_reward = compute_bbox_center_reward(bbox, bbox_valid)
        bbox_size_reward = compute_bbox_size_reward(bbox, bbox_valid)

        # Triangulation quality reward
        tri_reward = compute_triangulation_reward(trace_cov, is_tri_valid)

        # Combine rewards
        total_reward = (
            bbox_center_reward * self.cfg.bbox_center_scale +
            bbox_size_reward * self.cfg.bbox_size_scale +
            tri_reward * self.cfg.triangulation_scale
        )

        rewards_dict[agent_id] = total_reward

    return rewards_dict
```

### 4. Compute Observations Using Delayed+Noisy States

```python
def _get_observations(self) -> Dict[str, torch.Tensor]:
    """Get observations using delayed+noisy states."""

    observations = {}

    # Get delayed+noisy states for realistic sensor simulation
    noisy_states = self.state_manager.get_all_delayed_noisy_states()

    for i, agent_id in enumerate(self.cfg.possible_agents):
        noisy_state = noisy_states.agents[agent_id]

        # Ego states (delayed + noisy)
        ego_pos = noisy_state.data.body_position_w  # [N, 3]
        ego_vel = noisy_state.data.body_linear_velocity_w  # [N, 3]
        ego_yaw = noisy_state.data.joint_positions_b[:, 2:3]  # [N, 1]

        # Detection (delayed + noisy)
        bbox = noisy_state.data.bboxes_2d[:, 0, :]  # [N, 4]
        bbox_valid = noisy_state.data.bboxes_2d_valid_mask[:, 0:1].float()  # [N, 1]
        bbox_age = noisy_state.data.bboxes_2d_age[:, 0:1]  # [N, 1] - time since detection
        zoom = noisy_state.data.camera_zoom_level.unsqueeze(-1)  # [N, 1]

        # Received states from other agents (communication)
        received = self.state_manager.receive_other_agent_states(receiver_id=agent_id)

        other_positions = []
        other_rays = []
        other_valid = []

        for sender_id in self.cfg.possible_agents:
            if sender_id == agent_id:
                continue

            if sender_id in received:
                # Extract received data with validity
                pos_data, pos_valid, _ = received[sender_id]['position']
                ray_data, ray_valid, _ = received[sender_id]['camera_ray_directions_w']
                det_data, det_valid, _ = received[sender_id]['bboxes_2d_valid_mask']

                other_positions.append(pos_data)
                other_rays.append(ray_data[:, 0, :])  # [N, 3]
                other_valid.append(det_valid[:, 0:1])  # [N, 1]
            else:
                # No data received
                other_positions.append(torch.zeros(self.num_envs, 3, device=self.device))
                other_rays.append(torch.zeros(self.num_envs, 3, device=self.device))
                other_valid.append(torch.zeros(self.num_envs, 1, device=self.device))

        # Concatenate observation
        obs = torch.cat([
            ego_pos,           # [N, 3]
            ego_yaw,           # [N, 1]
            ego_vel,           # [N, 3]
            bbox,              # [N, 4]
            bbox_valid,        # [N, 1]
            bbox_age,          # [N, 1]
            zoom,              # [N, 1]
            *other_positions,  # [N, 3] x (C-1)
            *other_rays,       # [N, 3] x (C-1)
            *other_valid,      # [N, 1] x (C-1)
        ], dim=-1)

        observations[agent_id] = obs

    return observations
```

### 5. Reset States

```python
def _reset_idx(self, env_ids: torch.Tensor | None):
    """Reset environments."""
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = self._robots[self.cfg.possible_agents[0]]._ALL_INDICES

    # Reset robots and targets
    # ...

    # Reset state manager
    self.state_manager.reset(env_ids)

    super()._reset_idx(env_ids)
```

---

## Best Practices

### Key Design Principles

1. **Single update, multiple outputs**: `update_gt_states()` produces GT, delayed, and delayed+noisy states in one call
2. **Delayed states for rewards**: Use delayed (no noise) for fair reward comparison across agents
3. **Delayed+noisy for observations**: Use delayed+noisy to match real sensor characteristics
4. **Communication uses delayed**: Agents broadcast delayed (no noise) states to avoid double-counting noise
5. **Automatic camera updates**: Camera poses and intrinsics are automatically computed for all state types

### Before vs After

**Before (Manual State Management)**:
- Environment has to manually store GT, delayed, delayed_noisy states separately
- Manually apply motion filters
- Manually add noise
- Manually process detections
- Manually handle communication
- Keep track of what goes where
- Result: **500+ lines** of boilerplate in `_compute_intermediate_values`

**After (MultiAgentStateManager)**:
- Call `update_gt_states()` with robot data
- Call `update_detections()` with bbox data
- Call `broadcast_state()` for communication
- Get delayed states for rewards
- Get delayed+noisy states for observations
- Result: **~100 lines** in `_compute_intermediate_values`, clean and readable

### Common Pitfalls

1. **Forgetting to set camera configs**: Call `set_camera_configs()` in `__init__` before using camera features
2. **Using GT states for rewards**: Use delayed states (no noise) for fair comparison
3. **Using delayed states for observations**: Use delayed+noisy states for realistic sensor simulation
4. **Broadcasting noisy states**: Broadcast delayed (no noise) states to avoid double-counting noise

### Testing

See [run_all_tests.py](tests/run_all_tests.py) for comprehensive test suite:

```bash
# Run all tests
python source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delayed_states/tests/run_all_tests.py

# Run only critical tests
python source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delayed_states/tests/run_all_tests.py --priority P0

# Run with coverage
python source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delayed_states/tests/run_all_tests.py --coverage
```

---

## API Method Reference Table

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `update_gt_states(agent_id, ...)` | Update GT states and automatically compute delayed/noisy versions | Every step in `_compute_intermediate_values` |
| `update_detections(agent_id, bboxes_gt, valid_gt)` | Process bbox detections through pipeline | After running bbox raycaster |
| `broadcast_state(sender_id, state_keys)` | Broadcast delayed states to other agents | For multi-agent coordination |
| `receive_other_agent_states(receiver_id)` | Get received states from other agents | In `_get_observations` |
| `get_delayed_states(agent_id)` | Get delayed states (no noise) | For reward computation |
| `get_delayed_noisy_states(agent_id)` | Get delayed+noisy states | For observations |
| `get_gt_states(agent_id)` | Get ground truth states | For GT bbox detection |
| `get_all_gt_states()` | Get all GT states | For GT bbox detection across all agents |
| `get_all_delayed_states()` | Get all delayed states | For multi-agent rewards |
| `get_all_delayed_noisy_states()` | Get all delayed+noisy states | For multi-agent observations |
| `set_noise_progress_scale(progress)` | Scale noise for curriculum learning | During training progression |
| `update_detection_params(...)` | Update detection parameters dynamically | For curriculum learning |
| `update_comm_params(...)` | Update communication parameters dynamically | For curriculum learning |
| `get_statistics()` | Get pipeline statistics | For monitoring/debugging |
| `reset(env_ids)` | Reset environments | In `_reset_idx` |

---

## Notes

- All angles are in **radians**
- Quaternions use **(w, x, y, z)** convention
- Gimbal joint order is **[roll, pitch, yaw]** at indices **[0, 1, 2]**
- Bbox format is **[x, y, w, h]** in pixel coordinates
- Time is tracked internally; just call `update_time()` each step
- Noise is applied **after** filtering to ensure it doesn't get filtered out
- **Noise is seeded** for reproducibility (use `noise_seed` parameter)
- **Noise formulas match iris_ma_env3.py**:
  - Linear velocity noise: 10% of position noise std
  - Angular velocity noise: ori_std × pos_std
  - Linear acceleration noise: 100e-6 (default)
  - All scaled by `noise_progress_scale` for curriculum learning
