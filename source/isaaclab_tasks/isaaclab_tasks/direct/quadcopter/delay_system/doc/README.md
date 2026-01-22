# Delay System Module

A modular delay system for simulating action and observation latency in reinforcement learning environments. This module wraps Isaac Lab's `DelayBuffer` class with a flexible configuration system.

## Overview

The delay system simulates real-world latency in robotic systems:

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
├── delay_cfg.py         # DelayCfg, DelaySystemCfg
├── delay_system.py      # DelaySystem class
├── doc/
│   └── README.md        # This documentation
└── tests/
    ├── __init__.py
    ├── run_tests.py     # Standalone test runner
    └── README.md        # Test documentation
```
