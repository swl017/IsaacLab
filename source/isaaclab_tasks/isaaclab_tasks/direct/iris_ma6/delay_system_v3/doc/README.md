# Delay System V3

A simplified, unified delay system with configurable per-step/per-episode sampling and guaranteed timestamp-data synchronization.

## Key Features

### 1. Unified Architecture
V3 replaces V2's 4 separate delay systems (clean_ego, clean_other, noisy_ego, noisy_other) with a single `UnifiedDelaySystem`. Perspective (ego/other) and noise mode are specified at query time.

### 2. Timestamp-Data Coupling
**Fundamental invariant**: The returned timestamp is ALWAYS the capture time of the returned data. This ensures Age-of-Information (AoI) calculations are always correct:
```
AoI = t_current - returned_timestamp
```

### 3. Configurable Sampling Frequencies
Each parameter (latency, staleness, dropout) can be sampled at different frequencies:
- `per_step`: Resample every simulation step
- `per_episode`: Resample at episode start (all envs)
- `per_env_reset`: Resample only for reset environments

### 4. Curriculum Learning Support
Full curriculum API for gradual delay introduction:
- Mode `none`: No delay (pass-through)
- Mode `fixed`: Deterministic delay (no variance)
- Mode `random`: Random delay with staleness enabled
- Progress `[0, 1]`: Scales delay magnitude

## Architecture

```
delay_system_v3/
├── __init__.py                    # Public API exports
├── delay_cfg_v3.py               # Configuration dataclasses
├── sampling_strategies.py        # Per-step/per-episode samplers
├── field_storage.py              # Raw/noisy data storage
├── delay_pipeline_v3.py          # Core pipeline with timestamp guarantees
├── delay_system_v3.py            # UnifiedDelaySystem
├── multi_agent_wrapper.py        # Multi-agent orchestration
├── agent_states.py               # Agent state containers
├── derived_field_computers.py    # Derived field computation
├── doc/
│   └── README.md                 # This file
└── tests/
    ├── run_tests.py              # Test runner (33 tests)
    └── README.md                 # Test documentation
```

## Quick Start

### Basic Usage
```python
from isaaclab_tasks.direct.iris_ma6.delay_system_v3 import (
    MultiAgentDelaySystemV3,
    MultiAgentDelayCfgV3,
    create_random_delay_cfg,
)

# Create configuration
cfg = create_random_delay_cfg()

# Initialize system
delay_system = MultiAgentDelaySystemV3(
    cfg=cfg,
    possible_agents=["drone_0", "drone_1", "drone_2"],
    num_envs=64,
    device=torch.device("cuda"),
)

# Set curriculum mode
delay_system.set_delay_mode("random", progress=1.0)
delay_system.set_dropout_rate(0.05)

# In simulation loop:
# 1. Update ground truth
delay_system.update_ground_truth("drone_0", position=pos, velocity=vel, ...)

# 2. Get delayed states for rewards (clean, no noise)
reward_states = delay_system.get_all_states_for_rewards("drone_0")

# 3. Get delayed states for observations (noisy)
obs_states = delay_system.get_all_states_for_observations("drone_0")
```

### Configuration Presets
```python
# No delay (for baseline training)
cfg = create_no_delay_cfg()

# Fixed delay (for curriculum stage 2)
cfg = create_fixed_delay_cfg(ego_latency=0.05, other_latency=0.1)

# Random delay (for final stage)
cfg = create_random_delay_cfg()
```

### Custom Configuration
```python
from isaaclab_tasks.direct.iris_ma6.delay_system_v3 import (
    MultiAgentDelayCfgV3,
    UnifiedDelayCfgV3,
    PerspectiveCfg,
    DelayPipelineCfgV3,
    LatencyCfg,
    StalenessCfg,
    DropoutCfg,
    DistributionCfg,
    SamplingCfg,
)

cfg = MultiAgentDelayCfgV3(
    delay_cfg=UnifiedDelayCfgV3(
        dt=0.04,
        ego=PerspectiveCfg(
            pipeline=DelayPipelineCfgV3(
                latency=LatencyCfg(
                    enabled=True,
                    distribution=DistributionCfg(type="normal", mean=0.05, std=0.02),
                    sampling=SamplingCfg(frequency="per_episode"),
                    min_steps=2,
                ),
                staleness=StalenessCfg(
                    enabled=True,
                    fps_distribution=DistributionCfg(type="uniform", mean=25.0, half_range=5.0),
                ),
                dropout=DropoutCfg(
                    enabled=True,
                    probability=0.05,
                    sampling=SamplingCfg(frequency="per_step"),
                ),
            ),
        ),
        other=PerspectiveCfg(
            pipeline=DelayPipelineCfgV3(
                # Higher delay for other agents
                latency=LatencyCfg(
                    enabled=True,
                    distribution=DistributionCfg(type="normal", mean=0.1, std=0.05),
                ),
            ),
        ),
    ),
)
```

## Pipeline Stages

### 1. Staleness (FPS Limiting)
Simulates limited sensor update rates. Data is held until the detection period elapses.
- Configured via `fps_distribution` (e.g., 20-30 FPS)
- Only active in `mode="random"`

### 2. Latency (Communication Delay)
Simulates network/processing delays using circular buffers.
- Parallel buffers for data AND timestamp
- `min_steps=2` ensures CircularBuffer warmup
- Configurable via `distribution` (constant, normal, uniform)

### 3. Dropout (Missed Detections)
Simulates communication failures by holding previous data.
- Configurable probability per step
- Rate can be varied per episode

### 4. First-Order Lag (Optional)
Smooths sudden data changes. Does NOT affect timestamp.

## Curriculum API

```python
# Stage 1: No delay
delay_system.set_delay_mode("none", progress=1.0)

# Stage 2: Fixed delay at 50% magnitude
delay_system.set_delay_mode("fixed", progress=0.5)

# Stage 3: Random delay, full staleness
delay_system.set_delay_mode("random", progress=1.0)

# Add dropout
delay_system.set_dropout_rate(0.05)

# Scale observation noise
delay_system.set_noise_scale(1.0)
```

## Testing

Run the full test suite:
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/run_tests.py --headless
```

Expected output: 33/33 tests passing.

## Improvements Over V2

| Aspect | V2 | V3 |
|--------|----|----|
| Architecture | 4 separate systems | 1 unified system |
| Timestamp handling | Concatenated to data | Separate, parallel buffers |
| Sampling | Per-episode only | Per-step, per-episode, per-env-reset |
| Field naming | `{agent}.{field}.{perspective}` | `{agent}.{field}` + perspective at query |
| Warmup | Manual step counting | Built-in min_steps |
| Mode behavior | Staleness always disabled in fixed | Staleness only in random |
| Code size | ~3000 lines | ~2000 lines |
