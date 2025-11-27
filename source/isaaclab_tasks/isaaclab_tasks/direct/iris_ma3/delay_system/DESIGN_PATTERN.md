# Delay System Design Pattern (v2.2)

## Overview

This document describes the design pattern for simulating realistic time-shifting phenomena in multi-agent RL environments for sim-to-real transfer. The core pattern is a **stochastic sample-and-hold mechanism with dropout** that models latency, sampling rate discrepancy, and packet loss.

## v2.2 Architecture: Dual Pipeline with Noise-Before-Delay

### Key Principle: Noise MUST Be Applied BEFORE Delay

The delay system implements a **dual pipeline architecture** where:
1. **Noise is injected to raw sensor data FIRST**
2. **Then delays are applied to the noisy data**

This ordering is physically correct because:
- Noise represents sensor measurement errors at the point of sensing
- Delays represent processing/communication latency AFTER sensing
- Derived fields (camera pose, rays) computed from noisy delayed sensors preserve geometric consistency

### Dual Pipeline Architecture

The wrapper maintains **TWO separate DelaySystem instances**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiAgentDelaySystem                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Ground Truth States                                                     │
│        │                                                                 │
│        ├──────────────────────────────┬──────────────────────────────┐  │
│        ▼                              ▼                               │  │
│  ┌─────────────────────┐     ┌───────────────────────────────────┐   │  │
│  │  Clean Pipeline     │     │  Noisy Pipeline                   │   │  │
│  │  (for REWARDS)      │     │  (for OBSERVATIONS)               │   │  │
│  ├─────────────────────┤     ├───────────────────────────────────┤   │  │
│  │                     │     │  1. Inject noise to RAW sensors   │   │  │
│  │  GT → Delay System  │     │  2. GT+Noise → Delay System       │   │  │
│  │                     │     │                                   │   │  │
│  └─────────────────────┘     └───────────────────────────────────┘   │  │
│        │                              │                               │  │
│        ▼                              ▼                               │  │
│  Clean Delayed States          Noisy Delayed States                   │  │
│  (use in _get_rewards)         (use in _get_observations)             │  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### v2.1 Breaking Changes

**BREAKING CHANGE**: `bboxes_2d_valid_mask` has been **REMOVED** from the delay pipeline.

**Rationale**:
- The field caused confusion between frame-level validity and detection validity
- Propagating validity masks through delays is error-prone and unnecessary
- Users should validate bboxes AFTER all delay processing

**New Pattern**:
```python
# Get all states from agent's perspective using 'view' scheme
all_states = delay_system.get_all_states_for_observations("agent_0")

# Access noisy delayed states
ego_states = all_states["agent_0"]
other_states = all_states["agent_1"]

# Validate bboxes AFTER delay processing
bbox_valid = bbox_raycaster.validate_bbox(ego_states.data.bboxes_2d)  # [N, T]
```

## Core Algorithm

All time-shifting phenomena (latency, FPS throttling, dropout) share this common pattern:

```python
def sample_and_hold_with_dropout(current_time, data):
    sample_time_thres = sample_from_dist(period_config)  # uniform, normal, or constant
    is_period = current_time - last_sample_time > sample_time_thres
    is_dropout = sample_from_dist(dropout_config)

    if is_period and not is_dropout:
        last_sample_time = current_time
        last_sample = data
        return data
    else:
        return last_sample  # Hold stale data
```

**Key Properties:**
- **Sample-and-hold**: New data sampled at stochastic intervals, held between samples
- **Dropout**: Random sample failures (packet loss, frame drops)
- **Latency**: Samples delayed by processing time
- **Compounding**: Multiple stages chain together (detector -> communication)

## Simplified Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DelaySystem Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ground Truth States                                         │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Noise Injection (to RAW sensors only)           │    │
│  │     - Position, orientation, velocity               │    │
│  │     - Joint positions, bboxes, zoom                 │    │
│  └─────────────────────────────────────────────────────┘    │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  2. Delay Application (per-field)                   │    │
│  │     - Motion: First-order lag filter (100 Hz)       │    │
│  │     - Orientation: Quaternion SLERP filter          │    │
│  │     - Detection: Sample-and-hold (20 Hz, 300ms)     │    │
│  │     - Communication: Sample-and-hold (30 Hz, 100ms) │    │
│  └─────────────────────────────────────────────────────┘    │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  3. Derived Field Computation                       │    │
│  │     - camera_position_w (from body + offset)        │    │
│  │     - camera_orientation_w (from body + gimbal)     │    │
│  │     - camera_ray_directions_w (from bbox + cam)     │    │
│  └─────────────────────────────────────────────────────┘    │
│        ↓                                                     │
│  Delayed+Noisy AgentStates                                   │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  4. User Validation (OUTSIDE delay system)          │    │
│  │     bbox_valid = bbox_raycaster.validate_bbox(bbox) │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Agent Perspective Model

### Ego vs. Other Agents

Each agent has two views of states:
1. **Ego states** (self): Fast local processing with minimal communication delay
2. **Other agents' states** (received): Ego processing + inter-agent communication delay

**Example: 3-agent system**

```
Agent 0's perspective:
├── agent_0 (ego):    [motion_filter] → [detector] → [local_comm]      ← Fast
├── agent_1 (other):  [motion_filter] → [detector] → [inter_agent_comm] ← Slow
└── agent_2 (other):  [motion_filter] → [detector] → [inter_agent_comm] ← Slow
```

## AgentStatesData Fields

### Raw Sensor Fields (delay applied)

| Field | Shape | Description |
|-------|-------|-------------|
| `body_position_w` | [N, 3] | Body position in world frame |
| `body_orientation_w` | [N, 4] | Body orientation quaternion (w,x,y,z) |
| `body_linear_velocity_w` | [N, 3] | Linear velocity in world frame |
| `body_angular_velocity_w` | [N, 3] | Angular velocity in world frame |
| `body_linear_acceleration_w` | [N, 3] | Linear acceleration in world frame |
| `joint_positions_b` | [N, J] | Gimbal joint angles |
| `camera_zoom_level` | [N] | Optical zoom level |
| `bboxes_2d` | [N, T, 4] | Bounding boxes (x, y, w, h) |

### Derived Fields (computed from delayed raw sensors)

| Field | Shape | Description |
|-------|-------|-------------|
| `camera_position_w` | [N, 3] | Camera position in world frame |
| `camera_orientation_w` | [N, 4] | Camera orientation quaternion |
| `camera_ray_directions_w` | [N, T, 3] | Ray directions from bbox centers |
| `camera_ray_origins_w` | [N, T, 3] | Ray origins (camera position) |

### Static Fields (no delay)

| Field | Shape | Description |
|-------|-------|-------------|
| `camera_offset_position_b` | [N, 3] | Camera offset in body frame |
| `camera_offset_rotation_b` | [N, 4] | Camera rotation offset |
| `camera_base_intrinsics` | [N, 3, 3] | Camera intrinsics matrix K |

### Removed Fields (v2.1)

| Field | Reason |
|-------|--------|
| ~~`bboxes_2d_valid_mask`~~ | Users validate AFTER delay using `bbox_raycaster.validate_bbox()` |
| ~~`bboxes_2d_age`~~ | Use `timestamps` object instead |

## Usage Pattern

### Basic Usage (v2.2 API)

```python
from isaaclab_tasks.direct.iris_ma3.delay_system import (
    MultiAgentDelaySystem,
    AgentStates,
)

# Initialize
delay_system = MultiAgentDelaySystem(
    possible_agents=["agent_0", "agent_1", "agent_2"],
    num_envs=512,
    num_joints_per_agent={"agent_0": 2, "agent_1": 2, "agent_2": 2},
    num_targets_per_agent={"agent_0": 5, "agent_1": 5, "agent_2": 5},
    dt=0.01,
    device=torch.device("cuda:0"),
    enable_noise=True,
    position_noise_std=0.05,
    orientation_noise_std=0.02,
)

# Each simulation step:
delay_system.update_time()

# 1. Update ground truth states (from physics)
for agent_id in agent_ids:
    delay_system.update_gt_states(
        agent_id=agent_id,
        body_position_w=robot.body_position,
        body_orientation_w=robot.body_orientation,
        body_linear_velocity_w=robot.body_linear_velocity,
        body_angular_velocity_w=robot.body_angular_velocity,
        body_linear_acceleration_w=robot.body_linear_acceleration,
        body_combined_angular_velocity_w=combined_angular_velocity,
        joint_positions_b=gimbal.joint_positions,
        zoom_level=camera.zoom_level,
    )

    # 2. Update detection data
    delay_system.update_detections(
        agent_id=agent_id,
        bboxes_2d_gt=detector.bboxes,  # [N, T, 4]
    )

# 3. Get delayed states for REWARDS (clean, no noise)
for agent_id in agent_ids:
    all_reward_states = delay_system.get_all_states_for_rewards(agent_id)

    # Ego states
    ego_states = all_reward_states[agent_id]

    # Other agent states (with inter-agent comm delay)
    for other_id, other_states in all_reward_states.items():
        if other_id != agent_id:
            other_pos = other_states.data.body_position_w  # [N, 3]

# 4. Get delayed states for OBSERVATIONS (noisy)
for agent_id in agent_ids:
    all_obs_states = delay_system.get_all_states_for_observations(agent_id)

    # Ego states (noisy)
    ego_states = all_obs_states[agent_id]
    position = ego_states.data.body_position_w      # [N, 3]
    bboxes = ego_states.data.bboxes_2d              # [N, T, 4]
    ray_dirs = ego_states.data.camera_ray_directions_w  # [N, T, 3]

    # Validate bboxes AFTER delay processing
    bbox_valid = bbox_raycaster.validate_bbox(bboxes)  # [N, T]

    # Other agent states (noisy, with inter-agent comm delay)
    for other_id, other_states in all_obs_states.items():
        if other_id != agent_id:
            other_pos = other_states.data.body_position_w  # [N, 3]
```

### View Scheme (Recommended)

The **view scheme** provides a unified way to access all agent states from one agent's perspective:

```python
# For REWARDS: Get ALL agent states (clean, no noise)
all_reward_states = delay_system.get_all_states_for_rewards("agent_0")

# For OBSERVATIONS: Get ALL agent states (noisy)
all_obs_states = delay_system.get_all_states_for_observations("agent_0")

# Both return: Dict[agent_id -> AgentStates]
# - Ego agent: Fast local processing delays
# - Other agents: Slow inter-agent communication delays

# Access states directly
ego_states = all_obs_states["agent_0"]
other_states = all_obs_states["agent_1"]

# Direct access to AgentStates fields
other_pos = other_states.data.body_position_w      # [N, 3]
other_ori = other_states.data.body_orientation_w   # [N, 4]
other_bboxes = other_states.data.bboxes_2d         # [N, T, 4]
```

### Curriculum Learning

```python
# Gradually increase noise over training
progress = (current_step - noise_start) / (noise_end - noise_start)
delay_system.set_noise_progress_scale(progress)  # 0.0 to 1.0

# Adjust delay parameters during training
delay_system._delay_system.set_time_constants(
    motion=0.05,  # Faster motion filtering
    orientation=0.05,
    joints=0.02,
)

delay_system._delay_system.set_detection_latency_params(
    fps_mean=30.0,  # Higher FPS
    latency_mean=0.2,  # Lower latency
    dropout_prob=0.05,  # Less dropout
)
```

## Field Groups and Samplers

### Sampler Types

| Group | Sampler Type | Parameters |
|-------|-------------|------------|
| motion | FirstOrderLagSampler | time_constant: 0.1s |
| orientation | QuaternionFirstOrderLagSampler | time_constant: 0.1s |
| joints | FirstOrderLagSampler | time_constant: 0.05s |
| zoom | FirstOrderLagSampler | time_constant: 0.05s |
| detection_bbox | StochasticSampler | 20 Hz, 300ms latency, 10% dropout |
| camera_intrinsics | PassthroughSampler | No delay (static) |

### Noise Types

| Field | Noise Type | Description |
|-------|-----------|-------------|
| position | Gaussian | World-frame position noise |
| orientation | Quaternion perturbation | Small rotation noise via axis-angle |
| linear_velocity | Gaussian | Velocity measurement noise |
| angular_velocity | Gaussian | Gyroscope noise |
| joint_positions | Gaussian | Encoder noise |
| bboxes_2d | Realistic bbox noise | Scale-dependent, aspect-preserving |
| zoom | Gaussian | Zoom motor encoder noise |

## Best Practices

### 1. Bbox Validation

**ALWAYS** validate bboxes after delay processing:

```python
all_obs_states = delay_system.get_all_states_for_observations(agent_id)
ego_states = all_obs_states[agent_id]
bboxes = ego_states.data.bboxes_2d

# Validate using your bbox_raycaster
bbox_valid = bbox_raycaster.validate_bbox(bboxes)  # [N, T]

# Filter invalid bboxes
valid_bboxes = bboxes[bbox_valid]
# Or mask for computations
ray_dirs = ego_states.data.camera_ray_directions_w
ray_dirs_masked = ray_dirs * bbox_valid.unsqueeze(-1)
```

### 2. Reward vs Observation Separation

```python
# For REWARDS: Use clean delayed states (no noise)
all_reward_states = delay_system.get_all_states_for_rewards(agent_id)

# For OBSERVATIONS: Use noisy delayed states
all_obs_states = delay_system.get_all_states_for_observations(agent_id)

# Both methods return Dict[agent_id -> AgentStates] with proper
# ego vs inter-agent communication delays applied.
```

### 3. Timestamp Usage

```python
# Access timestamps for staleness tracking
if obs_states.timestamps is not None:
    motion_ts = obs_states.timestamps.timestamps['motion']
    detection_ts = obs_states.timestamps.timestamps['detection']

    motion_staleness = motion_ts.staleness  # [N]
    detection_staleness = detection_ts.staleness  # [N]

    # Include staleness in observations
    obs = torch.cat([
        obs_states.data.body_position_w,
        motion_staleness.unsqueeze(-1),
        detection_staleness.unsqueeze(-1),
    ], dim=-1)
```

## File Structure

```
delay_system/
├── __init__.py
├── DESIGN_PATTERN.md          # This document
├── API_REFERENCE.md           # API reference documentation
├── README.md                  # Quick start guide
├── usage_example.py           # Runnable example
│
├── agent_states.py            # AgentStates, AgentStatesData
├── delay_system.py            # Core DelaySystem
├── delay_system_cfg.py        # Configuration dataclasses
├── multi_agent_wrapper.py     # MultiAgentDelaySystem wrapper
│
├── stochastic_sampler.py      # StochasticSampler
├── specialized_samplers.py    # FirstOrderLag, Quaternion, Passthrough
├── sampler_chain.py           # SamplerChain composition
├── field_configs.py           # Field grouping and sampler factories
│
└── tests/
    ├── run_tests.py           # Test runner
    └── test_*.py              # Unit tests
```

## Key Design Principles

1. **Simplicity**: No valid_mask propagation through delays
2. **Separation**: Clean states for rewards, noisy states for observations
3. **Composability**: Samplers chain together for complex delay scenarios
4. **Per-field configuration**: Each field independently configurable
5. **Perspective-aware**: Ego vs. other agents have different delay profiles
6. **GPU-accelerated**: All operations batched across environments
7. **Type-aware**: Special handling for quaternions
8. **User validation**: Bbox validity computed AFTER delay, not propagated through

## Migration from v2.0/v2.1

### API Changes in v2.2

**⚠️ REMOVED FROM CODE** - The following methods no longer exist in the codebase:
- ~~`get_delayed_states()`~~ → use `get_all_states_for_rewards()`
- ~~`get_delayed_noisy_states()`~~ → use `get_all_states_for_observations()`
- ~~`receive_other_agent_states()`~~ → use `get_all_states_for_rewards()`
- ~~`receive_other_agent_states_noisy()`~~ → use `get_all_states_for_observations()`

```python
# OLD (v2.1):
ego_reward = delay_system.get_delayed_states(agent_id)
ego_obs = delay_system.get_delayed_noisy_states(agent_id)
other_reward = delay_system.receive_other_agent_states(agent_id)
other_obs = delay_system.receive_other_agent_states_noisy(agent_id)

# NEW (v2.2) - View scheme:
all_reward_states = delay_system.get_all_states_for_rewards(agent_id)
all_obs_states = delay_system.get_all_states_for_observations(agent_id)

# Access ego and other agents from the returned dict
ego_reward = all_reward_states[agent_id]
ego_obs = all_obs_states[agent_id]
for other_id in agent_ids:
    if other_id != agent_id:
        other_reward = all_reward_states[other_id]
        other_obs = all_obs_states[other_id]
```

### From v2.0

**⚠️ REMOVED FROM CODE** - `valid_mask_gt` parameter and `bboxes_2d_valid_mask` field no longer exist:

```python
# OLD (v2.0):
delay_system.update_detections(
    agent_id=agent_id,
    bboxes_2d_gt=bboxes,
    valid_mask_gt=valid_mask,  # REMOVE THIS
)

# NEW (v2.2):
delay_system.update_detections(
    agent_id=agent_id,
    bboxes_2d_gt=bboxes,
)

# Validate AFTER:
all_obs_states = delay_system.get_all_states_for_observations(agent_id)
bbox_valid = bbox_raycaster.validate_bbox(all_obs_states[agent_id].data.bboxes_2d)
```
