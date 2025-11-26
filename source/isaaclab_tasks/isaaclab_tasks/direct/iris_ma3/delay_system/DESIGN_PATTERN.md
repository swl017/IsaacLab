# Delay System Design Pattern

## Overview

This document describes the design pattern for simulating realistic time-shifting phenomena in multi-agent RL environments for sim-to-real transfer. The core pattern is a **stochastic sample-and-hold mechanism with dropout** that models latency, sampling rate discrepancy, and packet loss.

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
- **Compounding**: Multiple stages chain together (detector → communication)

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

Agent 1's perspective:
├── agent_1 (ego):    [motion_filter] → [detector] → [local_comm]      ← Fast
├── agent_0 (other):  [motion_filter] → [detector] → [inter_agent_comm] ← Slow
└── agent_2 (other):  [motion_filter] → [detector] → [inter_agent_comm] ← Slow
```

**Key Insight:** Base processing (motion, detection) is the same for all agents. Only the final communication stage differs between ego and others.

## Data Structure Refinements

### AgentStatesData Cleanup

**Removed fields:**
- ~~`bboxes_2d_valid_mask`~~ - Confusion with frame bounds; invalid detections handled by zeros/NaNs
- ~~`bboxes_2d_age`~~ - Redundant; use timestamp comparison instead

**Timestamp fields:**
- `timestamp_sim_walltime` - Ground truth simulation time (s)
- `timestamp_motion` - When motion states were last updated (s)
- `timestamp_detection` - When detection data was last updated (s)

**Invalid data handling:**
- No explicit "valid mask" for detections
- Invalid/no detection represented by zeros, NaNs, or out-of-bounds values
- Agent learns to interpret stale/invalid data as part of observation

## Field Grouping

Fields grouped by timing characteristics:

```python
field_groups = {
    # Motion states: First-order lag (100 Hz), same filter
    'motion': [
        'body_position_w',
        'body_velocity_w', 'body_velocity_b',
        'body_angular_velocity_w', 'body_angular_velocity_b',
        'body_linear_acceleration_w', 'body_linear_acceleration_b',
        'body_angular_acceleration_b',
    ],

    # Orientation: Quaternion SLERP filtering (100 Hz)
    'orientation': [
        'body_orientation_w',
        'camera_orientation_w',
    ],

    # Joint states: First-order lag (100 Hz)
    'joints': [
        'joint_positions_b',
        'joint_velocities_b',
        'joint_accelerations_b',
    ],

    # Detection states: Detector sampling (20 Hz, 300ms latency, dropout)
    'detection': [
        'bboxes_2d',
        'camera_ray_directions_w',
        'camera_ray_origins_w',
    ],

    # Camera intrinsics: Static, no delay
    'camera_intrinsics': [
        'camera_width', 'camera_height',
        'camera_focal_length',
        'camera_horizontal_aperture', 'camera_vertical_aperture',
        'camera_offset_position_b', 'camera_offset_rotation_b',
        'camera_intrinsics',
        'camera_zoom_level',
    ],
}
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DelaySystem                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  DataStorage                                 │          │
│  │  - Ground truth state history buffer         │          │
│  │  - Indexed by time for delay retrieval       │          │
│  └──────────────────────────────────────────────┘          │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────┐          │
│  │  Base Processing (All Agents)                │          │
│  │  ┌────────────┐  ┌──────────────┐           │          │
│  │  │ Motion     │  │ Detection    │           │          │
│  │  │ Filter     │  │ Sampler      │           │          │
│  │  │ (100 Hz)   │  │ (20 Hz, 0.3s)│           │          │
│  │  └────────────┘  └──────────────┘           │          │
│  └──────────────────────────────────────────────┘          │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────┐          │
│  │  Communication Stage (Perspective-Dependent) │          │
│  │                                              │          │
│  │  ┌─────────────┐         ┌──────────────┐  │          │
│  │  │ Ego View    │         │ Others View  │  │          │
│  │  │ Local Comm  │         │ Inter-Agent  │  │          │
│  │  │ (fast, ~ms) │         │ Comm (slow)  │  │          │
│  │  │             │         │ (30Hz, 0.1s) │  │          │
│  │  └─────────────┘         └──────────────┘  │          │
│  └──────────────────────────────────────────────┘          │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────┐          │
│  │  Timestamp Tracking                          │          │
│  │  - Per-group timestamps                      │          │
│  │  - Staleness calculation                     │          │
│  │  - Latency accumulation                      │          │
│  └──────────────────────────────────────────────┘          │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────┐          │
│  │  Output: Delayed AgentStates                 │          │
│  │  - Perspective-specific delays applied       │          │
│  │  - Timestamp metadata per field group        │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Per-Field Sampler Architecture

**Key Design Decision: Each field has its own sampler instance**

**Rationale:**
- Independent time constants per field (e.g., position vs. velocity filtering)
- Independent dropout rates per field
- Flexible per-field configuration
- Can mix different sampler types (linear, quaternion, etc.)

**Structure:**
```python
DelaySystem:
    field_samplers: Dict[str, SamplerChain]
        # Example entries:
        "body_position_w" → SamplerChain([FirstOrderLagSampler])
        "body_orientation_w" → SamplerChain([QuaternionFirstOrderLagSampler])
        "bboxes_2d" → SamplerChain([DetectorSampler])

    perspective_samplers: Dict[str, StochasticSampler]
        "ego" → LocalCommSampler(fast)
        "other" → InterAgentCommSampler(slow)
```

## Timestamp Management

### Timestamp Hierarchy

```
Ground Truth (t_sim)
    ↓
Base Processing (t_base_sampled)
    ↓ (with base latency)
Base Available (t_base_available)
    ↓
Communication Processing (t_comm_sampled)
    ↓ (with comm latency)
Final Available (t_final_available)
```

### FieldTimestamp Class

```python
class FieldTimestamp:
    """Timestamp metadata for a field or field group"""

    t_captured: torch.Tensor      # [num_envs] When data was captured in sim
    t_sampled: torch.Tensor       # [num_envs] When last sampled (period-based)
    t_available: torch.Tensor     # [num_envs] When became available (after latency)
    t_current: torch.Tensor       # [num_envs] Current simulation time

    @property
    def staleness(self) -> torch.Tensor:
        """Time since data was captured (seconds)"""
        return self.t_current - self.t_captured

    @property
    def latency(self) -> torch.Tensor:
        """Total latency from capture to availability (seconds)"""
        return self.t_available - self.t_captured

    @property
    def age(self) -> torch.Tensor:
        """Time since data became available (seconds)"""
        return self.t_current - self.t_available
```

### Per-Group Timestamp Tracking

```python
class AgentStatesTimestamps:
    """Timestamp tracking for all AgentStates field groups"""

    timestamps: Dict[str, FieldTimestamp]
    # Example:
    # timestamps['motion'] = FieldTimestamp(...)
    # timestamps['detection'] = FieldTimestamp(...)
    # timestamps['orientation'] = FieldTimestamp(...)
```

## Sampler Chaining

### Chain Concept

```
Raw Data → Stage1 → Stage2 → ... → StageN → Output
           (t1)     (t2)             (tN)
```

**Each stage adds:**
- Sampling period (FPS throttling)
- Latency (processing delay)
- Dropout (packet loss)

**Timestamps accumulate through stages:**
- Total latency = sum of all stage latencies
- Final staleness = t_current - t_captured (initial)

### Example Chains

**Detection Pipeline (Ego View):**
```
Bboxes @ 100Hz → DetectorSampler(20Hz, 0.3s, 10%) → LocalCommSampler(fast) → Output
   t_capture        t_detect (300ms later)            t_ego (5ms later)

Total latency: ~305ms
```

**Detection Pipeline (Other Agent View):**
```
Bboxes @ 100Hz → DetectorSampler(20Hz, 0.3s, 10%) → InterAgentCommSampler(30Hz, 0.1s, 5%) → Output
   t_capture        t_detect (300ms later)            t_comm (100ms later)

Total latency: ~400ms
```

**Motion Pipeline:**
```
Position @ 100Hz → FirstOrderLag(tau=0.1) → Output
   t_capture          t_filtered (smooth)

No explicit latency, continuous filtering
```

### SamplerChain Implementation

```python
class SamplerChain:
    """Chain of samplers for a field"""

    field_name: str
    samplers: List[StochasticSampler]

    # Timestamp tracking
    t_captured: torch.Tensor
    stage_timestamps: List[torch.Tensor]

    def update(self, data: torch.Tensor, t_current: torch.Tensor) -> Tuple:
        """
        Pass data through sampler chain.

        Returns:
            (output_data, field_timestamp)
        """
        current_data = data
        self.t_captured = t_current.clone()
        self.stage_timestamps = [t_current.clone()]

        # Pass through each stage
        for sampler in self.samplers:
            current_data, stage_info = sampler.update(current_data, t_current)
            stage_time = t_current + sampler.accumulated_latency
            self.stage_timestamps.append(stage_time)

        # Build timestamp metadata
        field_ts = FieldTimestamp(
            t_captured=self.t_captured,
            t_sampled=self.stage_timestamps[-2],
            t_available=self.stage_timestamps[-1],
            t_current=t_current,
        )

        return current_data, field_ts
```

## Multi-Agent State Management

### DelaySystem Interface

```python
class DelaySystem:
    """Main delay system for multi-agent environment"""

    def update_agent_gt_states(self, agent_id: AgentID, states: AgentStates):
        """
        Record ground truth states for an agent.
        Called every step for each agent.
        """
        pass

    def get_ego_states(self, agent_id: AgentID) -> AgentStates:
        """
        Get ego states for an agent (fast local processing).

        Returns:
            AgentStates with ego perspective delays applied
        """
        pass

    def get_other_agent_states(self, ego_agent: AgentID, other_agent: AgentID) -> AgentStates:
        """
        Get other agent's states from ego agent's perspective.
        Includes inter-agent communication delays.

        Returns:
            AgentStates with full delays (base + inter-agent comm)
        """
        pass

    def get_all_agent_states_for_ego(self, ego_agent: AgentID) -> Dict[AgentID, AgentStates]:
        """
        Get all agent states from ego agent's perspective.

        Returns:
            Dict mapping agent_id → delayed states
            - ego_agent: Fast ego processing
            - other agents: Slow inter-agent communication
        """
        pass
```

### Usage Example

```python
# Initialize delay system for 3 agents
delay_system = DelaySystem(
    agent_ids=["agent_0", "agent_1", "agent_2"],
    num_envs=512,
    device="cuda:0",
)

# Each step: Update ground truth
for agent_id in agent_ids:
    gt_states = get_ground_truth_states(agent_id)
    delay_system.update_agent_gt_states(agent_id, gt_states)

# Get observations for agent_0
agent_0_obs = delay_system.get_all_agent_states_for_ego("agent_0")
# Returns:
# {
#   "agent_0": <ego states with fast processing>,
#   "agent_1": <states with slow inter-agent comm>,
#   "agent_2": <states with slow inter-agent comm>,
# }

# Get observations for agent_1
agent_1_obs = delay_system.get_all_agent_states_for_ego("agent_1")
# Returns:
# {
#   "agent_1": <ego states with fast processing>,    ← Different ego agent
#   "agent_0": <states with slow inter-agent comm>,
#   "agent_2": <states with slow inter-agent comm>,
# }
```

## Configuration Example

### Scenario Setup

- Simulation: 100 Hz
- Object detector: 20 Hz ± 5 Hz, 300ms ± 50ms latency, 10% dropout
- Navigation filter: 100 Hz, 100ms time constant (first-order lag)
- Local communication: ~5ms (minimal)
- Inter-agent communication: 30 Hz ± 10 Hz, 100ms ± 20ms latency, 5% dropout

### Config Definition

```python
from delay_system import (
    DelaySystemCfg,
    SamplerConfig,
    DistributionConfig,
)

@configclass
class MyDelaySystemCfg(DelaySystemCfg):
    # Simulation
    dt_sim: float = 0.01  # 100 Hz

    # Motion filter (first-order lag)
    motion_time_constant: float = 0.1  # 100ms
    orientation_time_constant: float = 0.1
    joint_time_constant: float = 0.05  # Faster for joints

    # Detection sampler
    detection_fps_mean: float = 20.0
    detection_fps_std: float = 5.0
    detection_latency_mean: float = 0.3  # 300ms
    detection_latency_std: float = 0.05  # 50ms
    detection_dropout_rate: float = 0.10  # 10%

    # Ego communication (local, fast)
    ego_comm_latency: float = 0.005  # 5ms, nearly instant

    # Inter-agent communication
    inter_agent_comm_rate_min: float = 20.0  # Hz
    inter_agent_comm_rate_max: float = 40.0  # Hz (30 ± 10)
    inter_agent_comm_latency_mean: float = 0.1  # 100ms
    inter_agent_comm_latency_std: float = 0.02  # 20ms
    inter_agent_comm_dropout_rate: float = 0.05  # 5%
```

### Sampler Instantiation

```python
# Motion samplers (per-field)
motion_samplers = {
    'body_position_w': FirstOrderLagSampler(
        num_envs=num_envs,
        state_dim=3,
        time_constant=cfg.motion_time_constant,
        dt=cfg.dt_sim,
        device=device,
    ),
    'body_velocity_w': FirstOrderLagSampler(
        num_envs=num_envs,
        state_dim=3,
        time_constant=cfg.motion_time_constant,
        dt=cfg.dt_sim,
        device=device,
    ),
    # ... other motion fields
}

# Orientation samplers
orientation_samplers = {
    'body_orientation_w': QuaternionFirstOrderLagSampler(
        num_envs=num_envs,
        time_constant=cfg.orientation_time_constant,
        dt=cfg.dt_sim,
        device=device,
    ),
}

# Detection sampler (shared config for all detection fields)
detection_sampler = StochasticSampler(
    config=SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="normal",
            mean=1.0 / cfg.detection_fps_mean,
            std=cfg.detection_fps_std / (cfg.detection_fps_mean ** 2),
        ),
        latency_dist=DistributionConfig(
            distribution_type="normal",
            mean=cfg.detection_latency_mean,
            std=cfg.detection_latency_std,
        ),
        dropout_dist=DistributionConfig(
            distribution_type="constant",
            value=cfg.detection_dropout_rate,
        ),
    ),
    num_envs=num_envs,
    device=device,
    dt=cfg.dt_sim,
)

# Communication samplers
ego_comm_sampler = StochasticSampler(
    config=SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="constant",
            value=cfg.dt_sim,  # No FPS throttling for local
        ),
        latency_dist=DistributionConfig(
            distribution_type="constant",
            value=cfg.ego_comm_latency,
        ),
        dropout_dist=None,  # No dropout for local comm
    ),
    num_envs=num_envs,
    device=device,
    dt=cfg.dt_sim,
)

inter_agent_comm_sampler = StochasticSampler(
    config=SamplerConfig(
        period_dist=DistributionConfig(
            distribution_type="uniform",
            min_value=1.0 / cfg.inter_agent_comm_rate_max,
            max_value=1.0 / cfg.inter_agent_comm_rate_min,
        ),
        latency_dist=DistributionConfig(
            distribution_type="normal",
            mean=cfg.inter_agent_comm_latency_mean,
            std=cfg.inter_agent_comm_latency_std,
        ),
        dropout_dist=DistributionConfig(
            distribution_type="constant",
            value=cfg.inter_agent_comm_dropout_rate,
        ),
    ),
    num_envs=num_envs,
    device=device,
    dt=cfg.dt_sim,
)
```

## Timestamp Usage in Observations

### Adding Staleness to Observations

```python
# Get delayed states for ego agent
ego_states = delay_system.get_ego_states(agent_id)
other_states = delay_system.get_other_agent_states(agent_id, other_agent_id)

# Extract timestamps
motion_ts = ego_states.timestamps['motion']
detection_ts = ego_states.timestamps['detection']

# Calculate staleness
motion_staleness = motion_ts.staleness  # [num_envs]
detection_staleness = detection_ts.staleness  # [num_envs]

# Include in observation
obs = torch.cat([
    ego_states.data.body_position_w,
    ego_states.data.bboxes_2d.flatten(start_dim=1),
    motion_staleness.unsqueeze(-1),      # Add staleness info
    detection_staleness.unsqueeze(-1),
], dim=-1)
```

### Staleness-Based Reward Shaping

```python
# Penalize stale detections
detection_staleness = detection_ts.staleness
staleness_penalty = -0.1 * torch.clamp(detection_staleness, max=2.0)

# Exponential decay confidence
detection_confidence = torch.exp(-detection_staleness / tau_decay)
confidence_reward = 0.5 * detection_confidence

total_reward = base_reward + staleness_penalty + confidence_reward
```

## File Structure

```
delay_system/
├── __init__.py
├── DESIGN_PATTERN.md              # This document
│
├── stochastic_sampler.py          # Core sampler + distribution configs
│   ├── DistributionConfig
│   ├── SamplerConfig
│   └── StochasticSampler
│
├── specialized_samplers.py        # Domain-specific samplers
│   ├── FirstOrderLagSampler
│   ├── QuaternionFirstOrderLagSampler
│   ├── PassthroughSampler (no delay)
│   └── ...
│
├── sampler_chain.py               # Chain composition
│   ├── SamplerChain
│   └── ComposableSampler
│
├── timestamp_manager.py           # Timestamp tracking
│   ├── FieldTimestamp
│   └── AgentStatesTimestamps
│
├── field_configs.py               # Field metadata & grouping
│   ├── field_groups
│   └── create_field_samplers()
│
├── delay_system.py                # Main orchestration
│   └── DelaySystem
│
├── delay_system_cfg.py            # Configuration dataclasses
│   └── DelaySystemCfg
│
└── agent_states.py                # AgentStates with timestamps
    ├── AgentStatesData
    ├── AgentStates
    └── AgentStatesTimestamps
```

## Key Design Principles

1. **Composability**: Samplers chain together for complex delay scenarios
2. **Per-field configuration**: Each field independently configurable
3. **Perspective-aware**: Ego vs. other agents have different delay profiles
4. **Timestamp-rich**: Comprehensive timing metadata for staleness tracking
5. **GPU-accelerated**: All operations batched across environments
6. **Type-aware**: Special handling for quaternions, complex data types
7. **No invalid masking**: Keep it simple; invalid = no detection or zeros
8. **Separation of concerns**: Base processing vs. communication delays

## Benefits for Sim-to-Real Transfer

1. **Realistic timing**: Models actual sensor/network behavior
2. **Stale data training**: Agent learns to handle outdated information
3. **Dropout robustness**: Agent adapts to missing data
4. **Latency awareness**: Agent compensates for delays
5. **Multi-agent coordination**: Realistic inter-agent communication constraints
6. **Observability**: Staleness info helps agent assess data quality

## Next Steps

1. Implement `stochastic_sampler.py` (core algorithm)
2. Implement `specialized_samplers.py` (first-order lag, quaternion)
3. Implement `sampler_chain.py` (composition)
4. Implement `timestamp_manager.py` (timing metadata)
5. Implement `field_configs.py` (field grouping)
6. Implement `delay_system.py` (orchestration)
7. Update `agent_states.py` (add timestamps)
8. Create configuration examples and tests
