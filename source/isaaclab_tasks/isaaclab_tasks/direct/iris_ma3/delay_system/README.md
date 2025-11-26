# Delay System for Multi-Agent RL

A GPU-accelerated delay system for realistic sim-to-real transfer in multi-agent reinforcement learning environments.

## Overview

This delay system simulates realistic time-shifting phenomena that occur in real-world robotics:
- **Sensor latency**: Processing delays (e.g., object detection: 300ms)
- **Sampling rate discrepancy**: FPS throttling (e.g., camera: 20Hz vs sim: 100Hz)
- **Communication delays**: Network latency and packet loss
- **Data staleness**: Temporal misalignment between agents

### Key Features

✅ **Stochastic sample-and-hold with dropout** - Core algorithm for all delays
✅ **Per-field configuration** - Independent delays for each state field
✅ **Multi-agent perspectives** - Ego (fast) vs other agents (slow) communication
✅ **Timestamp tracking** - Comprehensive timing metadata for staleness calculation
✅ **Sampler chaining** - Compose delays (detector → communication)
✅ **GPU-accelerated** - Batched operations across environments
✅ **Modular design** - Easy to extend with custom samplers

## Quick Start

### Installation

The delay system is part of the `iris_ma3` task. No additional installation needed.

### Basic Usage

```python
import torch
from delay_system import DelaySystem, DelaySystemCfg

# 1. Configure
cfg = DelaySystemCfg()
cfg.detection_fps_mean = 20.0  # 20 Hz detector
cfg.detection_latency_mean = 0.3  # 300ms latency
cfg.inter_agent_comm_latency_mean = 0.1  # 100ms inter-agent comm

# 2. Initialize
delay_system = DelaySystem(
    cfg=cfg,
    agent_ids=["agent_0", "agent_1", "agent_2"],
    num_envs=512,
    num_joints=2,
    num_targets=5,
    device=torch.device("cuda:0"),
)

# 3. Simulation loop
for step in range(num_steps):
    # Update ground truth states for each agent
    for agent_id in agent_ids:
        gt_states = get_ground_truth_states(agent_id)
        delay_system.update_agent_gt_states(agent_id, gt_states)

    # Query delayed states from agent_0's perspective
    agent_0_view = delay_system.get_all_agent_states_for_ego("agent_0")

    # agent_0_view["agent_0"] → fast ego processing
    # agent_0_view["agent_1"] → slow inter-agent comm
    # agent_0_view["agent_2"] → slow inter-agent comm

    # Advance time
    delay_system.step()
```

See [usage_example.py](usage_example.py) for a complete example.

## Architecture

### Core Algorithm

All time-shifting phenomena share this pattern:

```python
def sample_and_hold_with_dropout(current_time, data):
    sample_time_thres = sample_from_dist(period_config)
    is_period = current_time - last_sample_time > sample_time_thres
    is_dropout = sample_from_dist(dropout_config)

    if is_period and not is_dropout:
        last_sample_time = current_time
        last_sample = data
        return data
    else:
        return last_sample  # Hold stale data
```

### System Components

```
DelaySystem
├── Base Samplers (shared by all agents)
│   ├── Motion samplers (first-order lag)
│   ├── Orientation samplers (quaternion SLERP)
│   ├── Joint samplers (first-order lag)
│   ├── Detection sampler (stochastic, 20Hz, latency, dropout)
│   └── Passthrough samplers (static fields)
│
├── Communication Samplers (perspective-specific)
│   ├── Ego comm sampler (fast, ~5ms)
│   └── Inter-agent comm sampler (slow, 30Hz, 100ms, dropout)
│
└── Timestamp Manager
    └── Per-agent, per-group timestamps
```

### Field Groups

Fields are grouped by timing characteristics:

- **Motion** (`body_position_w`, `body_velocity_w`, etc.) - First-order lag
- **Orientation** (`body_orientation_w`, `camera_orientation_w`) - Quaternion SLERP
- **Joints** (`joint_positions_b`, `joint_velocities_b`, etc.) - First-order lag
- **Detection** (`bboxes_2d`, `camera_ray_directions_w`, etc.) - Stochastic sampler
- **Camera Intrinsics** (static) - Passthrough (no delay)

## Configuration

### DelaySystemCfg Parameters

```python
@configclass
class DelaySystemCfg:
    # Simulation
    dt_sim: float = 0.01  # 100 Hz

    # Motion filtering
    motion_time_constant: float = 0.1  # 100ms
    orientation_time_constant: float = 0.1
    joint_time_constant: float = 0.05

    # Detection
    detection_fps_mean: float = 20.0
    detection_fps_std: float = 5.0
    detection_latency_mean: float = 0.3  # 300ms
    detection_latency_std: float = 0.05
    detection_dropout_rate: float = 0.10  # 10%

    # Ego communication (fast)
    ego_comm_latency: float = 0.005  # 5ms

    # Inter-agent communication (slow)
    inter_agent_comm_rate_min: float = 20.0  # Hz
    inter_agent_comm_rate_max: float = 40.0  # Hz
    inter_agent_comm_latency_mean: float = 0.1  # 100ms
    inter_agent_comm_latency_std: float = 0.02
    inter_agent_comm_dropout_rate: float = 0.05  # 5%
```

### Custom Sampler Example

```python
from stochastic_sampler import StochasticSampler, SamplerConfig, DistributionConfig

# Create custom detector sampler
detector_config = SamplerConfig(
    period_dist=DistributionConfig(
        distribution_type="uniform",
        min_value=1.0/30,  # 30 Hz max
        max_value=1.0/10,  # 10 Hz min
    ),
    latency_dist=DistributionConfig(
        distribution_type="normal",
        mean=0.5,  # 500ms mean
        std=0.1,   # 100ms std
    ),
    dropout_dist=DistributionConfig(
        distribution_type="constant",
        value=0.20,  # 20% dropout
    ),
)

detector_sampler = StochasticSampler(
    config=detector_config,
    num_envs=num_envs,
    device=device,
    dt=0.01,
)
```

## Multi-Agent Perspectives

Each agent has two views of state data:

### Ego States (Self)
- **Fast local processing**
- Motion filter → Detector → Local comm (~5ms)
- Total latency: ~300-350ms

### Other Agent States (Received)
- **Slow inter-agent communication**
- Motion filter → Detector → Inter-agent comm (30Hz, 100ms)
- Total latency: ~400-450ms

**Example:**

```python
# Agent 0's perspective
agent_0_view = delay_system.get_all_agent_states_for_ego("agent_0")
# {
#   "agent_0": <ego states, fast>,
#   "agent_1": <other states, slow>,
#   "agent_2": <other states, slow>,
# }

# Agent 1's perspective (different ego!)
agent_1_view = delay_system.get_all_agent_states_for_ego("agent_1")
# {
#   "agent_1": <ego states, fast>,
#   "agent_0": <other states, slow>,
#   "agent_2": <other states, slow>,
# }
```

## Timestamp Usage

Timestamps enable staleness-aware observations and rewards:

```python
ego_states = delay_system.get_ego_states("agent_0")

# Access timestamps (if available)
if ego_states.timestamps is not None:
    motion_ts = ego_states.timestamps.timestamps['motion']
    detection_ts = ego_states.timestamps['detection']

    # Calculate staleness
    motion_staleness = motion_ts.staleness  # [num_envs]
    detection_staleness = detection_ts.staleness

    # Use in observations
    obs = torch.cat([
        ego_states.data.body_position_w,
        ego_states.data.bboxes_2d.flatten(start_dim=1),
        motion_staleness.unsqueeze(-1),
        detection_staleness.unsqueeze(-1),
    ], dim=-1)

    # Reward shaping based on staleness
    staleness_penalty = -0.1 * torch.clamp(detection_staleness, max=2.0)
```

## Testing

Run the test suite:

```bash
python test_delay_system.py
```

Tests cover:
- Stochastic sampler functionality
- First-order lag filtering
- Quaternion SLERP
- Sampler chains
- Timestamp tracking
- Full system integration

## File Structure

```
delay_system/
├── __init__.py                    # Package exports
├── README.md                      # This file
├── DESIGN_PATTERN.md              # Detailed design documentation
│
├── stochastic_sampler.py          # Core sampler + distributions
├── specialized_samplers.py        # FirstOrderLag, Quaternion, Passthrough
├── sampler_chain.py               # Chain composition
├── timestamp_manager.py           # Timestamp tracking
├── field_configs.py               # Field grouping & factory functions
├── delay_system.py                # Main orchestration
├── delay_system_cfg.py            # Configuration
├── agent_states.py                # AgentStates data structures
│
├── usage_example.py               # Complete usage example
└── test_delay_system.py           # Test suite
```

## Design Philosophy

1. **Composability**: Samplers chain together for complex pipelines
2. **Per-field control**: Each field independently configurable
3. **Perspective-aware**: Ego vs other agents have different delays
4. **Timestamp-rich**: Comprehensive timing metadata
5. **GPU-accelerated**: Batched across environments
6. **Type-aware**: Special handling for quaternions, complex types
7. **Simplicity**: No invalid masking; invalid = no detection or zeros

## References

- [DESIGN_PATTERN.md](DESIGN_PATTERN.md) - Complete design documentation
- [usage_example.py](usage_example.py) - Working code example
- [test_delay_system.py](test_delay_system.py) - Test suite

## Citation

If you use this delay system in your research, please cite:

```bibtex
@software{delay_system_2025,
  title={Delay System for Multi-Agent RL},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```
