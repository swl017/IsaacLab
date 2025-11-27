# Delay System API Reference (v2.2)

## Quick Links

- [MultiAgentDelaySystem](#multiagentdelaysystem) - Main wrapper class
- [DelaySystem](#delaysystem) - Core delay system
- [AgentStates](#agentstates) - State container
- [Samplers](#samplers) - Delay samplers
- [Configuration](#configuration) - Configuration classes

---

## Architecture Overview

The v2.2 delay system uses a **dual pipeline architecture**:

```
                                    ┌─────────────────────┐
                                    │    Raw GT States    │
                                    │  (from simulation)  │
                                    └──────────┬──────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
                       ▼                                               ▼
            ┌──────────────────┐                            ┌──────────────────┐
            │  Clean Pipeline  │                            │  Noisy Pipeline  │
            │  (no noise)      │                            │  (noise → delay) │
            └────────┬─────────┘                            └────────┬─────────┘
                     │                                               │
                     │                                    ┌──────────┴──────────┐
                     │                                    │   Inject Noise      │
                     │                                    │   (before delay)    │
                     │                                    └──────────┬──────────┘
                     │                                               │
                     ▼                                               ▼
            ┌──────────────────┐                            ┌──────────────────┐
            │  DelaySystem #1  │                            │  DelaySystem #2  │
            │  (clean)         │                            │  (noisy)         │
            └────────┬─────────┘                            └────────┬─────────┘
                     │                                               │
                     ▼                                               ▼
            ┌──────────────────┐                            ┌──────────────────┐
            │  REWARDS         │                            │  OBSERVATIONS    │
            │  get_all_states  │                            │  get_all_states  │
            │  _for_rewards()  │                            │  _for_observations()|
            └──────────────────┘                            └──────────────────┘
```

**Key Principle:** Noise is injected to raw sensor data BEFORE delays are applied.

---

## MultiAgentDelaySystem

Multi-agent delay system wrapper with dual pipeline architecture.

### Constructor

```python
MultiAgentDelaySystem(
    possible_agents: List[str],
    num_envs: int,
    num_joints_per_agent: Dict[str, int],
    num_targets_per_agent: Dict[str, int],
    dt: float,
    device: torch.device,
    enable_noise: bool = False,
    position_noise_std: float = 0.0,
    orientation_noise_std: float = 0.0,
    linear_velocity_noise_std: float = 0.0,
    angular_velocity_noise_std: float = 0.0,
    linear_acceleration_noise_std: float = 0.0,
    gimbal_noise_std: float = 0.0,
    bbox_noise_std: float = 0.0,
    zoom_noise_std: float = 0.0,
    noise_seed: int = 0,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `possible_agents` | `List[str]` | Agent identifiers (e.g., `["agent_0", "agent_1"]`) |
| `num_envs` | `int` | Number of parallel environments |
| `num_joints_per_agent` | `Dict[str, int]` | Mapping agent_id -> number of joints |
| `num_targets_per_agent` | `Dict[str, int]` | Mapping agent_id -> number of targets |
| `dt` | `float` | Physics timestep in seconds |
| `device` | `torch.device` | Device for tensors |
| `enable_noise` | `bool` | Enable observation noise |
| `*_noise_std` | `float` | Noise standard deviations |
| `noise_seed` | `int` | Random seed for reproducibility |

---

### Methods

#### update_time()

Advance simulation time by `dt`. **Must be called at start of each step.**

```python
def update_time(self, dt: float = None) -> None
```

---

#### set_camera_configs()

Set camera configuration for an agent.

```python
def set_camera_configs(
    self,
    agent_id: str,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    offset_position_b: torch.Tensor,  # [3]
    offset_rotation_b: torch.Tensor,  # [4] quaternion
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | Agent identifier |
| `width` | `int` | Image width (pixels) |
| `height` | `int` | Image height (pixels) |
| `focal_length` | `float` | Physical focal length (mm) |
| `horizontal_aperture` | `float` | Sensor width (mm) |
| `vertical_aperture` | `float` | Sensor height (mm) |
| `offset_position_b` | `Tensor[3]` | Camera offset in body frame |
| `offset_rotation_b` | `Tensor[4]` | Camera rotation offset (quaternion) |

---

#### set_noise_progress_scale()

Set curriculum progress for noise scaling.

```python
def set_noise_progress_scale(self, progress: float) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `progress` | `float` | Progress in [0.0, 1.0]. 0.0 = no noise, 1.0 = full noise |

**Example:**
```python
# Gradually increase noise over training
progress = (current_step - noise_start) / (noise_end - noise_start)
delay_system.set_noise_progress_scale(progress)
```

---

#### update_gt_states()

Update ground-truth states for an agent. States are fed to BOTH pipelines.

```python
def update_gt_states(
    self,
    agent_id: str,
    body_position_w: torch.Tensor,           # [N, 3]
    body_orientation_w: torch.Tensor,        # [N, 4]
    body_linear_velocity_w: torch.Tensor,    # [N, 3]
    body_angular_velocity_w: torch.Tensor,   # [N, 3]
    body_linear_acceleration_w: torch.Tensor,  # [N, 3]
    body_combined_angular_velocity_w: torch.Tensor,  # [N, 3]
    joint_positions_b: torch.Tensor,         # [N, J]
    zoom_level: torch.Tensor,                # [N]
) -> None
```

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `agent_id` | `str` | Agent identifier |
| `body_position_w` | `[N, 3]` | Body position in world frame |
| `body_orientation_w` | `[N, 4]` | Body orientation quaternion (w,x,y,z) |
| `body_linear_velocity_w` | `[N, 3]` | Linear velocity in world frame |
| `body_angular_velocity_w` | `[N, 3]` | Angular velocity in world frame |
| `body_linear_acceleration_w` | `[N, 3]` | Linear acceleration in world frame |
| `body_combined_angular_velocity_w` | `[N, 3]` | Body + gimbal angular velocity |
| `joint_positions_b` | `[N, J]` | Gimbal joint angles |
| `zoom_level` | `[N]` | Optical zoom level |

**Note:** Internally, this method:
1. Feeds clean GT to the clean pipeline
2. Injects noise to GT copy, then feeds to noisy pipeline

---

#### update_detections()

Update detection data (bounding boxes) for an agent.

```python
def update_detections(
    self,
    agent_id: str,
    bboxes_2d_gt: torch.Tensor,  # [N, T, 4]
) -> None
```

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `agent_id` | `str` | Agent identifier |
| `bboxes_2d_gt` | `[N, T, 4]` | Bounding boxes in (x, y, w, h) format |

**Note:** `valid_mask_gt` has been **REMOVED** in v2.1. Validate bboxes AFTER delay processing.

---

#### get_all_states_for_rewards() (RECOMMENDED)

Get ALL agent states for REWARD computation (clean, no noise).

```python
def get_all_states_for_rewards(self, ego_agent_id: str) -> Dict[str, AgentStates]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ego_agent_id` | `str` | The ego agent's perspective |

**Returns:** `Dict[agent_id -> AgentStates]`

- Ego agent: Fast local processing delays
- Other agents: Slow inter-agent communication delays

**Use for:** Reward computation, triangulation covariance, ground truth comparisons.

**Example:**
```python
# Get clean states from agent_0's perspective
reward_states = delay_system.get_all_states_for_rewards("agent_0")

# Ego states (fast local processing)
ego_states = reward_states["agent_0"]
ego_pos = ego_states.data.body_position_w  # [N, 3]

# Other agent states (slow inter-agent communication)
other_states = reward_states["agent_1"]
other_pos = other_states.data.body_position_w  # [N, 3]
```

---

#### get_all_states_for_observations() (RECOMMENDED)

Get ALL agent states for OBSERVATION computation (noisy).

```python
def get_all_states_for_observations(self, ego_agent_id: str) -> Dict[str, AgentStates]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ego_agent_id` | `str` | The ego agent's perspective |

**Returns:** `Dict[agent_id -> AgentStates]`

- Ego agent: Fast local processing delays + noise
- Other agents: Slow inter-agent communication delays + noise

**Use for:** Policy observations.

**Example:**
```python
# Get noisy states from agent_0's perspective
obs_states = delay_system.get_all_states_for_observations("agent_0")

# Ego states (noisy)
ego_obs = obs_states["agent_0"]
ego_pos = ego_obs.data.body_position_w  # [N, 3] with noise
ego_bboxes = ego_obs.data.bboxes_2d     # [N, T, 4] with noise

# Other agent states (noisy)
other_obs = obs_states["agent_1"]
other_pos = other_obs.data.body_position_w
other_bboxes = other_obs.data.bboxes_2d
```

---

#### reset()

Reset delay system for specified environments.

```python
def reset(self, env_ids: torch.Tensor) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `env_ids` | `Tensor[E]` | Environment indices to reset |

**Note:** Resets BOTH clean and noisy pipelines.

---

## DelaySystem

Core delay system (accessed via `delay_system._delay_system_clean` or `delay_system._delay_system_noisy`).

### Curriculum Learning Methods

#### set_time_constants()

Update first-order lag filter time constants.

```python
def set_time_constants(
    self,
    motion_tc: Optional[float] = None,
    orientation_tc: Optional[float] = None,
    joint_tc: Optional[float] = None,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `motion_tc` | `float` | Time constant for motion fields (seconds) |
| `orientation_tc` | `float` | Time constant for orientation (seconds) |
| `joint_tc` | `float` | Time constant for joint fields (seconds) |

---

#### set_detection_latency_params()

Update detection sampler parameters.

```python
def set_detection_latency_params(
    self,
    fps_mean: Optional[float] = None,
    latency_mean: Optional[float] = None,
    dropout_rate: Optional[float] = None,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `fps_mean` | `float` | Mean detection rate (Hz) |
| `latency_mean` | `float` | Mean processing latency (seconds) |
| `dropout_rate` | `float` | Frame dropout probability (0-1) |

---

#### set_comm_latency_params()

Update communication sampler parameters.

```python
def set_comm_latency_params(
    self,
    latency_mean: Optional[float] = None,
    dropout_prob: Optional[float] = None,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `latency_mean` | `float` | Mean network latency (seconds) |
| `dropout_prob` | `float` | Packet dropout probability (0-1) |

---

### Query Methods

#### get_ego_states()

Get ego states for an agent (fast local processing).

```python
def get_ego_states(self, agent_id: str) -> AgentStates
```

---

#### get_other_agent_states()

Get other agent's states from ego agent's perspective.

```python
def get_other_agent_states(
    self,
    ego_agent: str,
    other_agent: str,
) -> AgentStates
```

---

#### get_all_agent_states_for_ego()

Get all agent states from ego agent's perspective.

```python
def get_all_agent_states_for_ego(
    self,
    ego_agent: str,
) -> Dict[str, AgentStates]
```

**Returns:** `Dict[agent_id -> AgentStates]` where ego_agent has fast processing, others have slow communication.

---

## AgentStates

Container for agent state data.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `AgentStatesData` | State data container |
| `timestamps` | `AgentStatesTimestamps` | Optional timestamp tracking |

### AgentStatesData Fields

#### Motion States

| Field | Shape | Description |
|-------|-------|-------------|
| `body_position_w` | `[N, 3]` | Body position in world frame |
| `body_orientation_w` | `[N, 4]` | Body orientation quaternion |
| `body_linear_velocity_w` | `[N, 3]` | Linear velocity in world frame |
| `body_linear_velocity_b` | `[N, 3]` | Linear velocity in body frame |
| `body_angular_velocity_w` | `[N, 3]` | Angular velocity in world frame |
| `body_angular_velocity_b` | `[N, 3]` | Angular velocity in body frame |
| `body_combined_angular_velocity_w` | `[N, 3]` | Body + gimbal angular velocity (world) |
| `body_combined_angular_velocity_b` | `[N, 3]` | Body + gimbal angular velocity (body) |
| `body_linear_acceleration_w` | `[N, 3]` | Linear acceleration in world frame |
| `body_linear_acceleration_b` | `[N, 3]` | Linear acceleration in body frame |
| `body_angular_acceleration_b` | `[N, 3]` | Angular acceleration in body frame |

#### Joint States

| Field | Shape | Description |
|-------|-------|-------------|
| `joint_positions_b` | `[N, J]` | Gimbal joint angles |
| `joint_velocities_b` | `[N, J]` | Gimbal joint velocities |
| `joint_accelerations_b` | `[N, J]` | Gimbal joint accelerations |

#### Camera States

| Field | Shape | Description |
|-------|-------|-------------|
| `camera_offset_position_b` | `[N, 3]` | Camera offset in body frame |
| `camera_offset_rotation_b` | `[N, 4]` | Camera rotation offset |
| `camera_base_intrinsics` | `[N, 3, 3]` | Camera intrinsics matrix K |
| `camera_position_w` | `[N, 3]` | Camera position in world frame |
| `camera_orientation_w` | `[N, 4]` | Camera orientation quaternion |
| `camera_zoom_level` | `[N]` | Optical zoom level |

#### Detection States

| Field | Shape | Description |
|-------|-------|-------------|
| `bboxes_2d` | `[N, T, 4]` | Bounding boxes (x, y, w, h) |
| `camera_ray_directions_w` | `[N, T, 3]` | Ray directions from bbox centers |
| `camera_ray_origins_w` | `[N, T, 3]` | Ray origins (camera position) |

#### Timestamps

| Field | Shape | Description |
|-------|-------|-------------|
| `timestamp_sim_walltime` | `[N]` | Simulation time (seconds) |
| `timestamp_motion` | `[N]` | Last motion update time |
| `timestamp_detection` | `[N]` | Last detection update time |

---

## Samplers

### StochasticSampler

Sample-and-hold with stochastic period, latency, and dropout.

```python
StochasticSampler(
    config: SamplerConfig,
    num_envs: int,
    device: torch.device,
    dt: float,
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `update(data, t_current)` | Process data through sampler |
| `reset(env_ids)` | Reset sampler state |
| `update_config(period_mean, latency_mean, dropout_prob)` | Update parameters |

---

### FirstOrderLagSampler

First-order lag filter for continuous smoothing.

```python
FirstOrderLagSampler(
    num_envs: int,
    state_dim: int,
    time_constant: float,
    dt: float,
    device: torch.device,
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `update(data)` | Apply filter |
| `reset(env_ids, initial_data)` | Reset filter state |
| `update_config(time_constant)` | Update time constant |

---

### QuaternionFirstOrderLagSampler

First-order lag filter for quaternions using SLERP.

```python
QuaternionFirstOrderLagSampler(
    num_envs: int,
    time_constant: float,
    dt: float,
    device: torch.device,
)
```

Same methods as `FirstOrderLagSampler`.

---

### PassthroughSampler

No delay, instant updates (for static data).

```python
PassthroughSampler(
    num_envs: int,
    device: torch.device,
)
```

---

## Configuration

### DelaySystemCfg

Configuration for the delay system.

```python
@configclass
class DelaySystemCfg:
    # Simulation
    dt_sim: float = 0.01  # 100 Hz

    # Motion filter (first-order lag)
    motion_time_constant: float = 0.1
    orientation_time_constant: float = 0.1
    joint_time_constant: float = 0.05

    # Detection sampler
    detection_fps_mean: float = 20.0
    detection_fps_std: float = 5.0
    detection_latency_mean: float = 0.3
    detection_latency_std: float = 0.05
    detection_dropout_rate: float = 0.10

    # Ego communication (local, fast)
    ego_comm_latency: float = 0.005

    # Inter-agent communication
    inter_agent_comm_rate_min: float = 20.0
    inter_agent_comm_rate_max: float = 40.0
    inter_agent_comm_latency_mean: float = 0.1
    inter_agent_comm_latency_std: float = 0.02
    inter_agent_comm_dropout_rate: float = 0.05
```

---

### SamplerConfig

Configuration for stochastic samplers.

```python
@dataclass
class SamplerConfig:
    period_dist: DistributionConfig
    latency_dist: DistributionConfig
    dropout_dist: Optional[DistributionConfig] = None
```

---

### DistributionConfig

Configuration for probability distributions.

```python
@dataclass
class DistributionConfig:
    distribution_type: str  # "normal", "uniform", "constant"

    # For normal distribution
    mean: Optional[float] = None
    std: Optional[float] = None

    # For uniform distribution
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # For constant distribution
    value: Optional[float] = None
```

---

## Example: Complete Integration (v2.2)

```python
import torch
from isaaclab_tasks.direct.iris_ma3.delay_system import (
    MultiAgentDelaySystem,
    AgentStates,
)

# Initialize with noise enabled
delay_system = MultiAgentDelaySystem(
    possible_agents=["agent_0", "agent_1", "agent_2"],
    num_envs=512,
    num_joints_per_agent={aid: 2 for aid in ["agent_0", "agent_1", "agent_2"]},
    num_targets_per_agent={aid: 5 for aid in ["agent_0", "agent_1", "agent_2"]},
    dt=0.01,
    device=torch.device("cuda:0"),
    enable_noise=True,
    position_noise_std=0.05,
    orientation_noise_std=0.02,
    bbox_noise_std=5.0,
)

# Each step:
delay_system.update_time(dt)

# Curriculum progress (0.0 = no noise, 1.0 = full noise)
delay_system.set_noise_progress_scale(0.5)

# Update states for all agents
for agent_id in ["agent_0", "agent_1", "agent_2"]:
    delay_system.update_gt_states(
        agent_id=agent_id,
        body_position_w=positions,
        body_orientation_w=orientations,
        body_linear_velocity_w=velocities,
        body_angular_velocity_w=angular_velocities,
        body_linear_acceleration_w=accelerations,
        body_combined_angular_velocity_w=combined_angular_vel,
        joint_positions_b=joint_positions,
        zoom_level=zoom_levels,
    )
    delay_system.update_detections(
        agent_id=agent_id,
        bboxes_2d_gt=bboxes,
    )

# ============================================================
# View Scheme API (v2.2)
# ============================================================

# For REWARDS: Use clean states (no noise)
reward_states = delay_system.get_all_states_for_rewards("agent_0")
ego_clean = reward_states["agent_0"]
other_clean = reward_states["agent_1"]

# For OBSERVATIONS: Use noisy states
obs_states = delay_system.get_all_states_for_observations("agent_0")
ego_noisy = obs_states["agent_0"]
other_noisy = obs_states["agent_1"]

# Access data directly
ego_pos_clean = ego_clean.data.body_position_w      # [N, 3] clean
ego_pos_noisy = ego_noisy.data.body_position_w      # [N, 3] noisy

other_bboxes_clean = other_clean.data.bboxes_2d     # [N, T, 4] clean
other_bboxes_noisy = other_noisy.data.bboxes_2d     # [N, T, 4] noisy

# Validate bboxes AFTER delay (v2.1+)
bbox_valid = bbox_raycaster.validate_bbox(ego_noisy.data.bboxes_2d)

# ============================================================
# Curriculum Learning
# ============================================================

# Access internal delay systems for curriculum adjustments
# Note: Adjustments apply to BOTH clean and noisy pipelines
delay_system._delay_system_clean.set_time_constants(motion_tc=0.05)
delay_system._delay_system_noisy.set_time_constants(motion_tc=0.05)

delay_system._delay_system_clean.set_detection_latency_params(
    fps_mean=25.0,
    latency_mean=0.2,
    dropout_rate=0.05,
)
delay_system._delay_system_noisy.set_detection_latency_params(
    fps_mean=25.0,
    latency_mean=0.2,
    dropout_rate=0.05,
)

# Reset on episode end
delay_system.reset(env_ids_to_reset)
```

---

## Migration from v2.1 to v2.2

### API Changes

**⚠️ REMOVED FROM CODE** - The following methods no longer exist in the codebase:

| v2.1 Method (REMOVED) | v2.2 Replacement |
|----------------------|------------------|
| ~~`get_delayed_states(agent_id)`~~ | `get_all_states_for_rewards(agent_id)[agent_id]` |
| ~~`get_delayed_noisy_states(agent_id)`~~ | `get_all_states_for_observations(agent_id)[agent_id]` |
| ~~`receive_other_agent_states(ego_id)`~~ | `get_all_states_for_rewards(ego_id)` or `get_all_states_for_observations(ego_id)` |

### Migration Example

**Before (v2.1):**
```python
# Ego states
ego_clean = delay_system.get_delayed_states("agent_0")
ego_noisy = delay_system.get_delayed_noisy_states("agent_0")

# Other agent states
received = delay_system.receive_other_agent_states("agent_0")
other_pos = received["agent_1"]["position"][0]  # [N, 3]
```

**After (v2.2):**
```python
# All states in one call
reward_states = delay_system.get_all_states_for_rewards("agent_0")
obs_states = delay_system.get_all_states_for_observations("agent_0")

# Ego states
ego_clean = reward_states["agent_0"]
ego_noisy = obs_states["agent_0"]

# Other agent states
other_clean = reward_states["agent_1"]
other_noisy = obs_states["agent_1"]
other_pos = other_noisy.data.body_position_w  # [N, 3]
```

### Internal Changes

- Two separate `DelaySystem` instances (`_delay_system_clean`, `_delay_system_noisy`)
- Noise injected BEFORE delays (physically correct)
- Removed legacy wrapper methods

---

## Version History

### v2.2 (Current)

- **ADDED** Dual pipeline architecture (clean + noisy)
- **ADDED** `get_all_states_for_rewards()` - View scheme for rewards
- **ADDED** `get_all_states_for_observations()` - View scheme for observations
- **⚠️ REMOVED FROM CODE** `get_delayed_states()` - use `get_all_states_for_rewards()`
- **⚠️ REMOVED FROM CODE** `get_delayed_noisy_states()` - use `get_all_states_for_observations()`
- **⚠️ REMOVED FROM CODE** `receive_other_agent_states()` - use view scheme API
- **⚠️ REMOVED FROM CODE** `valid_mask_gt` parameter in `update_detections()`
- **CHANGED** Noise injection happens BEFORE delay processing

### v2.1

- **REMOVED** `bboxes_2d_valid_mask` from delay pipeline
- **SIMPLIFIED** `_agent_states_to_dict` - all fields always valid
- **ADDED** explicit bbox validation pattern post-delay
- **ADDED** curriculum learning hooks documentation

### v2.0

- **MOVED** noise injection to RAW sensors (before delays)
- **ADDED** derived field computation from noisy delayed inputs
- **ADDED** curriculum learning hooks (`set_time_constants`, etc.)

### v1.0

- Initial implementation
- Noise injection after delays (incorrect)
