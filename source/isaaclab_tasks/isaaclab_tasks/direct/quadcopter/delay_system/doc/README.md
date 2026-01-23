# Delay System Module

A modular delay system for simulating action and observation latency in reinforcement learning environments. This module provides two versions:

1. **Legacy (V1)**: Simple step-based delays using Isaac Lab's `DelayBuffer`
2. **Enhanced (V2)**: Full data bus architecture with per-field configuration, first-order lag, staleness, dropout, and noise injection

## Overview

The delay system simulates real-world latency and imperfections in robotic systems:

- **Random Latency**: Transport delay (communication, computation)
- **First-order Lag**: Dynamics delay (filter delay) with configurable time constant
- **Staleness**: Sensors reporting slower than control loop (sample-and-hold)
- **Dropout**: Communication packet drops
- **Fast Noise**: Per-step Gaussian noise (IMU, bbox inaccuracies)

---

# V1 (Legacy) API

The legacy API provides simple step-based delays:

- **Action Delay**: Time between control command issuance and physical execution (actuator latency, control loop delay)
- **Observation Delay**: Time between physical state and observation availability (sensor latency, communication delay)

## Components

### DelayCfg

Configuration for a single delay buffer.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import DelayCfg

# Fixed 10-step delay
delay_cfg = DelayCfg(enabled=True, min_delay=10, max_delay=10)

# Randomized delay between 5 and 15 steps
delay_cfg = DelayCfg(enabled=True, min_delay=5, max_delay=15, randomize_on_reset=True)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Whether to enable delay |
| `min_delay` | int | 0 | Minimum delay in simulation steps |
| `max_delay` | int | 0 | Maximum delay in simulation steps |
| `randomize_on_reset` | bool | True | Whether to randomize delays on reset |

### DelaySystemCfg

Container configuration for both action and observation delays.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import DelaySystemCfg, DelayCfg

# Action delay only
cfg = DelaySystemCfg(
    action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
)

# Both action and observation delays
cfg = DelaySystemCfg(
    action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
    observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
)
```

### DelaySystem

The main class that manages delay buffers and computes delayed data.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import DelaySystem, DelaySystemCfg, DelayCfg

cfg = DelaySystemCfg(
    action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
    observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
)

delay_system = DelaySystem(
    cfg=cfg,
    num_envs=4096,
    action_dim=4,
    observation_dim=12,
    device="cuda:0",
)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `reset(env_ids)` | Reset buffers and randomize delays for specified environments |
| `compute_delayed_action(action)` | Apply delay to actions |
| `compute_delayed_observation(observation)` | Apply delay to observations |
| `get_delay_info()` | Get delay statistics for logging |
| `get_action_delays()` | Get current action delays per environment |
| `get_observation_delays()` | Get current observation delays per environment |
| `set_action_delays(delays, env_ids)` | Manually set action delays |
| `set_observation_delays(delays, env_ids)` | Manually set observation delays |

## Integration with Environments

### In DirectRLEnv

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.quadcopter.delay_system import DelaySystem, DelaySystemCfg, DelayCfg


@configclass
class MyEnvCfg(DirectRLEnvCfg):
    # ... other config ...
    delay: DelaySystemCfg = DelaySystemCfg(
        action_delay=DelayCfg(enabled=True, min_delay=5, max_delay=10),
        observation_delay=DelayCfg(enabled=True, min_delay=2, max_delay=5),
    )


class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    def __init__(self, cfg: MyEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize delay system
        self._delay_system = DelaySystem(
            cfg=self.cfg.delay,
            num_envs=self.num_envs,
            action_dim=self.cfg.action_space,
            observation_dim=self.cfg.observation_space,
            device=self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        # Apply action delay
        delayed_actions = self._delay_system.compute_delayed_action(actions)
        # Use delayed_actions for physics...

    def _get_observations(self) -> dict:
        # Compute raw observations
        obs = self._compute_raw_observations()
        # Apply observation delay
        delayed_obs = self._delay_system.compute_delayed_observation(obs)
        return {"policy": delayed_obs}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # ... reset logic ...
        # Reset delay system
        self._delay_system.reset(env_ids)

        # Log delay info
        delay_info = self._delay_system.get_delay_info()
        for key, value in delay_info.items():
            self.extras["log"][f"Delay/{key}"] = value
```

## Converting Delay to Real Time

The delay is specified in **simulation steps**, not real time. To convert:

```python
# Given:
sim_dt = 0.01  # Simulation timestep (seconds)
decimation = 2  # Physics steps per action
delay_steps = 10  # Delay in simulation steps

# Real-world delay:
delay_seconds = delay_steps * sim_dt  # = 0.1 seconds = 100ms

# Or considering decimation:
delay_per_action = delay_steps / decimation  # = 5 action steps
```

## Typical Delay Values for Robotics

| System | Typical Latency | At 100Hz sim | At 50Hz sim |
|--------|-----------------|--------------|-------------|
| Camera processing | 20-50ms | 2-5 steps | 1-2 steps |
| IMU | 1-5ms | 0-1 steps | 0 steps |
| Actuator response | 10-30ms | 1-3 steps | 1-2 steps |
| Network (WiFi) | 5-50ms | 1-5 steps | 0-2 steps |
| Network (4G/LTE) | 30-100ms | 3-10 steps | 2-5 steps |

## Logging and Monitoring

The delay system provides logging information:

```python
# Get delay statistics
info = delay_system.get_delay_info()
# {
#     "action_delay_min": 5,
#     "action_delay_max": 10,
#     "action_delay_mean": 7.5,
#     "observation_delay_min": 2,
#     "observation_delay_max": 5,
#     "observation_delay_mean": 3.2,
# }

# Get per-environment delays
action_delays = delay_system.get_action_delays()  # Shape: (num_envs,)
obs_delays = delay_system.get_observation_delays()  # Shape: (num_envs,)
```

## Advanced Usage

### Manual Delay Control

```python
# Set all environments to same delay
delay_system.set_action_delays(5)

# Set specific environments
delay_system.set_action_delays(8, env_ids=[0, 1, 2])

# Set with tensor
custom_delays = torch.randint(3, 8, (num_envs,), device=device, dtype=torch.int)
delay_system.set_action_delays(custom_delays)
```

### Curriculum Learning with Delay

```python
# Start with no delay, gradually increase
def update_delay_curriculum(epoch: int):
    if epoch < 100:
        max_delay = 0
    elif epoch < 200:
        max_delay = epoch - 100  # Gradually increase 0 -> 100
    else:
        max_delay = 100

    delay_system.set_action_delays(
        torch.randint(0, max_delay + 1, (num_envs,), device=device, dtype=torch.int)
    )
```

## Testing

Run the test suite:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py
```

## File Structure

```
delay_system/
├── __init__.py          # Module exports
├── delay_cfg.py         # DelayCfg, DelaySystemCfg, FieldDelayCfg, etc.
├── delay_system.py      # DelaySystem class (V1)
├── delay_system_v2.py   # DelaySystemV2 class (V2)
├── data_bus.py          # DataBus class (V2)
├── delay_pipeline.py    # DelayPipeline class (V2)
├── derived_fields.py    # DerivedFieldComputer (V2)
├── doc/
│   ├── README.md        # This documentation
│   └── REQUIREMENTS.md  # Design requirements
└── tests/
    ├── __init__.py
    ├── run_tests.py     # Standalone test runner (V1 + V2)
    └── README.md        # Test documentation
```

---

# V2 (Enhanced) API

The enhanced V2 API provides a central data bus architecture with per-field delay configurations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA BUS                                       │
│  ┌─────────────────────┐        ┌─────────────────────┐                │
│  │   CLEAN BUFFER      │        │   NOISY BUFFER      │                │
│  │   (for rewards)     │        │   (for observations)│                │
│  │                     │        │                     │                │
│  │  raw_sensor_data    │        │  raw_sensor_data    │                │
│  │  no noise           │        │  + noise injected   │                │
│  └─────────────────────┘        └─────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
         ┌──────────────────┐       ┌──────────────────┐
         │  Apply Delays    │       │  Apply Delays    │
         │  per-field:      │       │  per-field:      │
         │  - First-order   │       │  - First-order   │
         │    lag           │       │    lag           │
         │  - Staleness     │       │  - Staleness     │
         │  - Random        │       │  - Random        │
         │    latency       │       │    latency       │
         │  - (no dropout)  │       │  - Dropout       │
         └──────────────────┘       └──────────────────┘
                    │                          │
                    ▼                          ▼
         ┌──────────────────┐       ┌──────────────────┐
         │ Compute Derived  │       │ Compute Derived  │
         │ Fields           │       │ Fields           │
         └──────────────────┘       └──────────────────┘
                    │                          │
                    ▼                          ▼
              REWARDS                   OBSERVATIONS
```

## V2 Components

### DistributionCfg

Configuration for stochastic distributions (used for sample rates, latencies, etc.).

```python
from isaaclab_tasks.direct.quadcopter.delay_system import DistributionCfg

# Constant value
dist = DistributionCfg(type="constant", value=100.0)

# Uniform distribution
dist = DistributionCfg(type="uniform", min_value=10.0, max_value=50.0)

# Normal distribution
dist = DistributionCfg(type="normal", mean=30.0, std=5.0)
```

### NoiseCfg

Configuration for Gaussian noise injection.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import NoiseCfg

noise = NoiseCfg(enabled=True, std=0.1)
```

### FieldDelayCfg

Per-field delay pipeline configuration with all imperfection types.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import FieldDelayCfg, DistributionCfg, NoiseCfg

# IMU with fast dynamics and small noise
imu_cfg = FieldDelayCfg(
    first_order_lag_enabled=True,
    time_constant=0.005,  # 5ms time constant
    noise=NoiseCfg(enabled=True, std=0.1),
)

# GPS with slower update rate and staleness
gps_cfg = FieldDelayCfg(
    first_order_lag_enabled=True,
    time_constant=0.1,  # 100ms time constant
    staleness_enabled=True,
    sample_rate=DistributionCfg(type="constant", value=10.0),  # 10 Hz
    noise=NoiseCfg(enabled=True, std=0.5),
)

# Camera with dropout and latency
camera_cfg = FieldDelayCfg(
    staleness_enabled=True,
    sample_rate=DistributionCfg(type="normal", mean=30.0, std=5.0),  # ~30 Hz
    dropout_enabled=True,
    dropout_prob=0.1,
    latency_enabled=True,
    latency=DistributionCfg(type="normal", mean=0.05, std=0.01),  # 50ms ± 10ms
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `first_order_lag_enabled` | bool | False | Enable first-order lag filtering |
| `time_constant` | float | 0.0 | Time constant τ in seconds |
| `staleness_enabled` | bool | False | Enable sample-and-hold |
| `sample_rate` | DistributionCfg | 100 Hz constant | Sample rate distribution |
| `latency_enabled` | bool | False | Enable transport latency |
| `latency` | DistributionCfg | 0 constant | Latency distribution in seconds |
| `dropout_enabled` | bool | False | Enable packet dropout |
| `dropout_prob` | float | 0.0 | Dropout probability [0, 1] |
| `noise` | NoiseCfg | disabled | Noise configuration |

### DelaySystemCfgV2

Enhanced delay system configuration.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import (
    DelaySystemCfgV2, FieldDelayCfg, DistributionCfg, NoiseCfg
)

cfg = DelaySystemCfgV2(
    dt=0.01,  # Simulation timestep
    use_enhanced_mode=True,
    field_configs={
        "imu_accel": FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.005,
            noise=NoiseCfg(enabled=True, std=0.1),
        ),
        "gps_position": FieldDelayCfg(
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="constant", value=10.0),
        ),
    },
)
```

### DelaySystemV2

The main enhanced delay system class.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import (
    DelaySystemV2, DelaySystemCfgV2, FieldDelayCfg, DistributionCfg, NoiseCfg
)

# Configure per-field delays
cfg = DelaySystemCfgV2(
    dt=0.01,
    use_enhanced_mode=True,
    field_configs={
        "imu_accel": FieldDelayCfg(
            first_order_lag_enabled=True,
            time_constant=0.005,
            noise=NoiseCfg(enabled=True, std=0.1),
        ),
        "gps_position": FieldDelayCfg(
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="constant", value=10.0),
            noise=NoiseCfg(enabled=True, std=0.5),
        ),
        "camera_bbox": FieldDelayCfg(
            staleness_enabled=True,
            sample_rate=DistributionCfg(type="normal", mean=30.0, std=5.0),
            dropout_enabled=True,
            dropout_prob=0.1,
        ),
    },
)

system = DelaySystemV2(
    cfg=cfg,
    num_envs=4096,
    device="cuda:0",
    field_dims={"imu_accel": 3, "gps_position": 3, "camera_bbox": 4},
)

# In environment step:
system.store("imu_accel", raw_imu)
system.store("gps_position", raw_gps)
system.store("camera_bbox", raw_bbox)

# For observations (noisy + delayed)
obs_imu = system.get_delayed_noisy("imu_accel")
obs_gps = system.get_delayed_noisy("gps_position")
obs_bbox = system.get_delayed_noisy("camera_bbox")

# For rewards (clean + delayed, no dropout)
reward_pos = system.get_delayed_clean("gps_position")

system.step()
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `store(field_name, data)` | Store raw sensor data to data bus |
| `get_delayed_clean(field_name)` | Get delayed clean data (for rewards) |
| `get_delayed_noisy(field_name)` | Get delayed noisy data (for observations) |
| `compute_derived_fields(use_clean)` | Compute derived fields from delayed data |
| `step(dt)` | Advance simulation time |
| `reset(env_ids)` | Reset for specified environments |
| `set_field_time_constant(field_name, tau)` | Update time constant (curriculum) |
| `set_field_dropout_rate(field_name, prob)` | Update dropout rate (curriculum) |

### DerivedFieldComputer

Compute derived fields from delayed raw sensor data.

```python
from isaaclab_tasks.direct.quadcopter.delay_system import (
    DerivedFieldComputer, DerivedFieldDef
)
import torch

# Define derived fields
def compute_velocity_magnitude(linear_velocity_b: torch.Tensor) -> torch.Tensor:
    return torch.norm(linear_velocity_b, dim=-1, keepdim=True)

velocity_mag_def = DerivedFieldDef(
    name="velocity_magnitude",
    sources=["linear_velocity_b"],
    compute_fn=compute_velocity_magnitude,
    dependency_level=1,
)

# Create computer
computer = DerivedFieldComputer([velocity_mag_def])

# Compute from delayed fields
delayed_fields = {"linear_velocity_b": delayed_vel}
all_fields = computer.compute_all(delayed_fields)
velocity_mag = all_fields["velocity_magnitude"]
```

## V2 Integration Example

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.quadcopter.delay_system import (
    DelaySystemV2, DelaySystemCfgV2, FieldDelayCfg, DistributionCfg, NoiseCfg
)


@configclass
class MyEnvCfg(DirectRLEnvCfg):
    # ... other config ...
    delay: DelaySystemCfgV2 = DelaySystemCfgV2(
        dt=0.01,
        use_enhanced_mode=True,
        field_configs={
            "body_position": FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=0.05,
                noise=NoiseCfg(enabled=True, std=0.01),
            ),
            "body_velocity": FieldDelayCfg(
                first_order_lag_enabled=True,
                time_constant=0.02,
                noise=NoiseCfg(enabled=True, std=0.05),
            ),
        },
    )


class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    def __init__(self, cfg: MyEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize V2 delay system
        self._delay_system = DelaySystemV2(
            cfg=self.cfg.delay,
            num_envs=self.num_envs,
            device=self.device,
            field_dims={"body_position": 3, "body_velocity": 3},
        )

    def _get_observations(self) -> dict:
        # Store raw sensor data
        self._delay_system.store("body_position", self._robot.data.root_pos_w)
        self._delay_system.store("body_velocity", self._robot.data.root_lin_vel_b)

        # Get delayed noisy observations
        obs_pos = self._delay_system.get_delayed_noisy("body_position")
        obs_vel = self._delay_system.get_delayed_noisy("body_velocity")

        return {"policy": torch.cat([obs_pos, obs_vel], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # Use clean delayed data for unbiased rewards
        clean_pos = self._delay_system.get_delayed_clean("body_position")
        # ... compute rewards ...

    def _post_physics_step(self):
        self._delay_system.step()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # ... reset logic ...
        self._delay_system.reset(env_ids)
```

## Key Design Principles

### Time Constant Invariance

The first-order lag uses `alpha = dt / (dt + tau)` which ensures the same physical response regardless of simulation rate:

```python
# Same tau produces same response at different dt
pipeline_100hz = DelayPipeline(cfg, num_envs, dim, dt=0.01, device)  # 100 Hz
pipeline_200hz = DelayPipeline(cfg, num_envs, dim, dt=0.005, device)  # 200 Hz
# Both converge to same value after same physical time
```

### Clean vs Noisy Separation

- **Clean path**: No dropout, used for unbiased reward computation
- **Noisy path**: All imperfections, used for observations fed to policy

### Noise Before Delay

Noise is injected into the data bus immediately (before delay stages) per the architecture in REQUIREMENTS.md.

## Typical Sensor Configurations

| Sensor | Time Constant | Sample Rate | Latency | Dropout | Noise |
|--------|---------------|-------------|---------|---------|-------|
| IMU | 0.001-0.01s | 100-1000 Hz | 1-5ms | 0% | 0.05-0.2 |
| GPS | 0.05-0.2s | 1-10 Hz | 50-200ms | 1-5% | 0.5-2.0 |
| Camera | 0.02-0.05s | 15-60 Hz | 20-100ms | 5-10% | varies |
| Barometer | 0.05-0.1s | 10-50 Hz | 10-30ms | 0% | 0.1-0.5 |
| Lidar | 0.01-0.02s | 10-20 Hz | 30-100ms | 2-5% | 0.02-0.1 |
