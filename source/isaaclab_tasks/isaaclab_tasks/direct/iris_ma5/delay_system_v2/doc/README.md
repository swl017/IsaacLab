# Multi-Agent Delay System V2

A field-based delay system for multi-agent reinforcement learning environments, built on top of the quadcopter's `DelaySystemV2` architecture.

## Overview

The `MultiAgentDelaySystemV2` provides realistic sensor delays and noise injection for multi-agent drone simulations. Key features include:

- **Dual Pipeline Architecture**: Separate clean (rewards) and noisy (observations) data paths
- **Perspective-Aware Delays**: Fast ego delays vs slow inter-agent communication delays
- **Field-Based Processing**: Per-field delay configurations with first-order lag, staleness, latency, and dropout
- **Arbitrary Agent Scaling**: Supports any number of agents without code changes
- **API Compatibility**: Drop-in replacement for iris_ma3 delay system

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MultiAgentDelaySystemV2                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │  GT States      │───▶│           Noise Injection                       │ │
│  │  (per agent)    │    │  (position, orientation, velocity, bbox, etc.)  │ │
│  └─────────────────┘    └──────────────┬──────────────────────────────────┘ │
│                                        │                                     │
│                         ┌──────────────┴──────────────┐                     │
│                         ▼                              ▼                     │
│              ┌──────────────────┐          ┌──────────────────┐             │
│              │  Clean Pipeline  │          │  Noisy Pipeline  │             │
│              │  (for rewards)   │          │ (for observations)│             │
│              └────────┬─────────┘          └────────┬─────────┘             │
│                       │                              │                       │
│         ┌─────────────┴─────────────┐  ┌────────────┴─────────────┐        │
│         ▼                           ▼  ▼                          ▼        │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │ Ego Fields  │           │ Other Fields│           │ Ego Fields  │       │
│  │ (fast path) │           │ (slow path) │           │ (fast path) │ ...   │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│                                                                              │
│  Field Naming: "{agent_id}.{field_name}.{perspective}"                      │
│  Example: "drone_0.body_position_w.ego"                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Field Types and Delay Configurations

| Field Category | Fields | Ego Delay | Inter-Agent Delay |
|----------------|--------|-----------|-------------------|
| Motion | position, velocity, acceleration | First-order lag (τ=5ms) | First-order lag + staleness (30Hz) + latency (100ms) + dropout (5%) |
| Orientation | body_orientation_w | First-order lag (τ=5ms) | First-order lag + staleness + latency + dropout |
| Joints | joint_positions_b, joint_velocities_b | First-order lag (τ=5ms) | First-order lag + staleness + latency + dropout |
| Detection | bboxes_2d | Staleness (30Hz) + latency (30ms) + dropout (5%) | Combined detector + communication delays |
| Zoom | camera_zoom_level | First-order lag (τ=5ms) | First-order lag + staleness + latency + dropout |

## Installation

The delay_system_v2 is part of the `isaaclab_tasks` package. No additional installation is required.

## Quick Start

```python
from isaaclab_tasks.direct.iris_ma5.delay_system_v2 import (
    MultiAgentDelaySystemV2,
    MultiAgentDelaySystemV2Cfg,
)

# Configure the delay system
cfg = MultiAgentDelaySystemV2Cfg(
    dt=0.01,  # 100 Hz simulation
    motion_time_constant=0.1,
    inter_agent_comm_latency_mean=0.1,  # 100ms inter-agent delay
    enable_noise=True,
    position_noise_std=0.05,
)

# Create delay system
delay_system = MultiAgentDelaySystemV2(
    cfg=cfg,
    possible_agents=["drone_0", "drone_1", "drone_2"],
    num_envs=512,
    num_joints_per_agent={"drone_0": 3, "drone_1": 3, "drone_2": 3},
    num_targets_per_agent={"drone_0": 2, "drone_1": 2, "drone_2": 2},
    device="cuda:0",
)

# Set camera configuration for each agent
for agent_id in ["drone_0", "drone_1", "drone_2"]:
    delay_system.set_camera_configs(
        agent_id,
        width=640, height=480,
        focal_length=24.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.0,
        offset_position_b=torch.tensor([0.0, 0.0, 0.1]),
        offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
    )
```

## Usage in Environment Step

```python
# In your environment's _pre_physics_step or _get_observations:

# 1. Advance time
delay_system.update_time(dt=self.step_dt)

# 2. Update ground truth states for each agent
for agent_id in self.possible_agents:
    delay_system.update_gt_states(
        agent_id=agent_id,
        body_position_w=robot.data.root_pos_w,
        body_orientation_w=robot.data.root_quat_w,
        body_linear_velocity_w=robot.data.root_lin_vel_w,
        body_angular_velocity_w=robot.data.root_ang_vel_w,
        body_linear_acceleration_w=computed_accel,
        body_combined_angular_velocity_w=combined_ang_vel,
        joint_positions_b=gimbal.data.joint_pos,
        zoom_level=camera_zoom,
    )

    # 3. Update detections (bounding boxes)
    delay_system.update_detections(
        agent_id=agent_id,
        bboxes_2d_gt=detected_bboxes,  # [N, T, 4] in (x, y, w, h) format
    )

# 4. Get delayed states for rewards (clean, no noise)
for ego_agent_id in self.possible_agents:
    reward_states = delay_system.get_all_states_for_rewards(ego_agent_id)
    ego_state = reward_states[ego_agent_id]
    other_states = {k: v for k, v in reward_states.items() if k != ego_agent_id}

    # Use for reward computation...

# 5. Get delayed states for observations (with noise)
for ego_agent_id in self.possible_agents:
    obs_states = delay_system.get_all_states_for_observations(ego_agent_id)
    # Use for observation tensor construction...
```

## Configuration Reference

### MultiAgentDelaySystemV2Cfg

#### Simulation
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 0.01 | Simulation timestep in seconds |

#### Motion Field Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `motion_time_constant` | float | 0.1 | Time constant for position/velocity fields (seconds) |
| `orientation_time_constant` | float | 0.1 | Time constant for quaternion orientation (seconds) |
| `joint_time_constant` | float | 0.05 | Time constant for gimbal joint fields (seconds) |
| `zoom_time_constant` | float | 0.1 | Time constant for camera zoom (seconds) |

#### Detection Field Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detection_fps_mean` | float | 30.0 | Mean detection FPS for staleness (Hz) |
| `detection_fps_std` | float | 5.0 | Std dev of detection FPS |
| `detection_latency_mean` | float | 0.03 | Mean detection latency (seconds) |
| `detection_latency_std` | float | 0.005 | Std dev of detection latency |
| `detection_dropout_rate` | float | 0.05 | Probability of detection dropout [0, 1] |

#### Ego Communication (Fast)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ego_comm_time_constant` | float | 0.005 | Time constant for ego's own state (seconds) |

#### Inter-Agent Communication (Slow)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inter_agent_comm_fps_mean` | float | 30.0 | Mean FPS for inter-agent staleness (Hz) |
| `inter_agent_comm_fps_std` | float | 5.0 | Std dev of inter-agent FPS |
| `inter_agent_comm_latency_mean` | float | 0.1 | Mean inter-agent latency (seconds) |
| `inter_agent_comm_latency_std` | float | 0.02 | Std dev of inter-agent latency |
| `inter_agent_comm_dropout_rate` | float | 0.05 | Probability of inter-agent dropout [0, 1] |

#### Noise Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_noise` | bool | True | Enable noise injection |
| `position_noise_std` | float | 0.0 | Position noise std (meters) |
| `orientation_noise_std` | float | 0.0 | Orientation noise std (quaternion components) |
| `linear_velocity_noise_std` | float | 0.0 | Linear velocity noise std (m/s) |
| `angular_velocity_noise_std` | float | 0.0 | Angular velocity noise std (rad/s) |
| `linear_acceleration_noise_std` | float | 0.0 | Linear acceleration noise std (m/s²) |
| `gimbal_noise_std` | float | 0.0 | Gimbal joint angle noise std (radians) |
| `bbox_noise_std` | float | 0.0 | Bounding box noise std (pixels) |
| `zoom_noise_std` | float | 0.0 | Zoom level noise std |

## API Reference

### MultiAgentDelaySystemV2

#### Constructor
```python
MultiAgentDelaySystemV2(
    cfg: MultiAgentDelaySystemV2Cfg,
    possible_agents: List[AgentID],
    num_envs: int,
    num_joints_per_agent: Dict[AgentID, int],
    num_targets_per_agent: Dict[AgentID, int],
    device: torch.device,
)
```

#### Properties
| Property | Type | Description |
|----------|------|-------------|
| `current_time` | `torch.Tensor` | Current simulation time per environment [N] |
| `gt_states` | `GTStatesAccessor` | Access to ground truth states (before delays) |
| `cfg` | `MultiAgentDelaySystemV2Cfg` | Configuration object |

#### Methods

##### `update_time(dt: Optional[float] = None)`
Advance simulation time for all delay systems.

##### `update_gt_states(...)`
Update ground truth states for an agent. See parameters in usage example.

##### `update_detections(agent_id: AgentID, bboxes_2d_gt: torch.Tensor)`
Update bounding box detections for an agent.
- `bboxes_2d_gt`: Shape [N, T, 4] in (x, y, w, h) pixel format

##### `get_all_states_for_rewards(ego_agent_id: AgentID) -> Dict[AgentID, AgentStates]`
Get clean delayed states (no noise) for reward computation.

##### `get_all_states_for_observations(ego_agent_id: AgentID) -> Dict[AgentID, AgentStates]`
Get noisy delayed states for observation computation.

##### `reset(env_ids: torch.Tensor, initial_gt_states: Optional[Dict] = None)`
Reset delay system for specified environments.

##### `set_camera_configs(...)`
Set camera intrinsics and mounting offset for an agent.

##### `set_noise_progress_scale(progress: float)`
Set noise scaling for curriculum learning. `progress=0` means no noise, `progress=1` means full noise.

## AgentStates Data Structure

Each agent's delayed state contains:

```python
@dataclass
class AgentStatesData:
    # Timestamps
    timestamp_sim_walltime: torch.Tensor  # [N]
    timestamp_motion: torch.Tensor        # [N]
    timestamp_detection: torch.Tensor     # [N]

    # Body motion (world frame)
    body_position_w: torch.Tensor         # [N, 3]
    body_orientation_w: torch.Tensor      # [N, 4] (w, x, y, z)
    body_linear_velocity_w: torch.Tensor  # [N, 3]
    body_angular_velocity_w: torch.Tensor # [N, 3]
    body_linear_acceleration_w: torch.Tensor  # [N, 3]

    # Body motion (body frame)
    body_linear_velocity_b: torch.Tensor  # [N, 3]
    body_angular_velocity_b: torch.Tensor # [N, 3]

    # Combined angular velocity (body + gimbal)
    body_combined_angular_velocity_w: torch.Tensor  # [N, 3]
    body_combined_angular_velocity_b: torch.Tensor  # [N, 3]

    # Joint states (gimbal)
    joint_positions_b: torch.Tensor       # [N, J]
    joint_velocities_b: torch.Tensor      # [N, J]

    # Camera geometry (derived)
    camera_position_w: torch.Tensor       # [N, 3]
    camera_orientation_w: torch.Tensor    # [N, 4]
    camera_ray_directions_w: torch.Tensor # [N, T, 3]
    camera_ray_origins_w: torch.Tensor    # [N, T, 3]
    camera_zoom_level: torch.Tensor       # [N]
    camera_base_intrinsics: torch.Tensor  # [N, 3, 3]

    # Detection
    bboxes_2d: torch.Tensor               # [N, T, 4]
```

## Derived Field Computation

Derived fields are automatically computed from delayed raw fields:

1. **Camera Position**: `camera_position_w = body_position_w + rotate(camera_offset_position_b, body_orientation_w)`

2. **Camera Orientation**: Composed from body orientation → gimbal rotation → camera offset

3. **Ray Directions**: Unprojected from 2D bbox centers using camera intrinsics and zoom level

These are computed **after** delays are applied, ensuring consistency between raw and derived fields.

## Testing

Run the test suite:
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/delay_system_v2/tests/run_tests.py
```

Expected output:
```
================================================================================
TEST SUMMARY
================================================================================
Total Tests: 22
Passed:      22 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Migration from iris_ma3

The API is designed to be compatible with iris_ma3. Key changes:

| iris_ma3 | iris_ma5 (V2) |
|----------|---------------|
| `MultiAgentDelaySystem` | `MultiAgentDelaySystemV2` |
| `MultiAgentDelaySystemCfg` | `MultiAgentDelaySystemV2Cfg` |
| Sampler-based delays | Field-based pipeline with `DelaySystemV2` |

The method signatures remain the same:
- `update_time()`
- `update_gt_states()`
- `update_detections()`
- `get_all_states_for_rewards()`
- `get_all_states_for_observations()`
- `reset()`
- `set_camera_configs()`
- `set_noise_progress_scale()`

## File Structure

```
delay_system_v2/
├── __init__.py                          # Package exports
├── multi_agent_delay_system_v2.py       # Main wrapper class
├── multi_agent_delay_system_v2_cfg.py   # Configuration dataclass
├── agent_states.py                      # AgentStates/AgentStatesData
├── derived_field_computers.py           # Camera geometry functions
├── doc/
│   └── README.md                        # This file
└── tests/
    ├── __init__.py
    ├── run_tests.py                     # Test runner
    └── README.md                        # Test documentation
```

## Dependencies

- `isaaclab_tasks.direct.quadcopter.delay_system`: Core `DelaySystemV2` implementation
- `isaaclab.utils.math`: Quaternion operations
- `torch`: Tensor operations

## DistributionCfg API

For custom field configurations, you can use `DistributionCfg` from the quadcopter delay_system:

```python
from isaaclab_tasks.direct.quadcopter.delay_system import DistributionCfg, FieldDelayCfg

# Constant value
dist = DistributionCfg(type="constant", value=100.0)

# Uniform distribution (samples from [10.0, 50.0] with mean 30.0)
dist = DistributionCfg(type="uniform", mean=30.0, half_range=20.0)

# Normal distribution
dist = DistributionCfg(type="normal", mean=30.0, std=5.0)

# Example: Custom field configuration with uniform sample rate
custom_cfg = FieldDelayCfg(
    staleness_enabled=True,
    sample_rate=DistributionCfg(type="uniform", mean=25.0, half_range=5.0),  # 20-30 Hz
    latency_enabled=True,
    latency=DistributionCfg(type="normal", mean=0.05, std=0.01),
)
```

## Known Limitations

1. **Quaternion filtering**: First-order lag on quaternions requires normalization after filtering, which may cause slight discontinuities for large orientation changes.

2. **Dropout behavior**: When dropout occurs, the previous held value is kept. This may not be physically accurate for fast-changing states.

3. **Derived field consistency**: Derived fields (camera position, ray directions) are computed from delayed raw fields, so they inherit any artifacts from the delay pipeline.
