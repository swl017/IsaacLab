# Delay System V3

A simplified, unified delay system with configurable per-step/per-episode sampling and guaranteed timestamp-data synchronization.

## ⚠ Known footgun: `LatencyCfg.min_steps`

`LatencyCfg.min_steps` **must stay ≥ 2**. Setting it to `0` destroys training
even in the `mode="none"` curriculum window where the latency stage is
short-circuited. Empirically, `min_steps=0` crashed a 50k-step bisect run
(`bisect/min_steps` — reward −6265 at 40k, pair_valid 0.03, sigma at the
2.0 ceiling; see
[`doc/experiments/2026-04-15_bisect_min_steps_vs_ticket029.md`](../../doc/experiments/2026-04-15_bisect_min_steps_vs_ticket029.md)).
The proximate mechanism is RNG-consumption drift at sampler / buffer init
(buffer depth and sampled-then-clamped step values differ between
`min_steps=0` and `min_steps=2` even when the latency stage itself is
bypassed), which lands episode-init randomization on a trajectory the
policy cannot recover from.

**Do not use `min_steps=0` to get "zero latency"** — even if the
`_apply_latency` read-after-write semantics are correct in isolation
(validated in `tests/test_min_steps_zero.py`). If you need pass-through,
call `set_delay_mode("none")` instead; that bypasses the latency stage
end-to-end.

### Curriculum-forgetting caveat

Using `mode="none"` for early curriculum and only later switching to
`fixed`/`random` creates a second hazard: the policy learned in `"none"`
mode has never encountered latency, and may catastrophically forget its
no-latency behavior (or simply fail to adapt) when the mode transition
fires. This is what the 120k-cliff in the post-mortem
(`2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md`)
captured. The long-term fix is to introduce latency gradually from step 0
with a non-zero `min_steps` floor (so the delayed channel exists but is
small, not absent), rather than flipping between "no latency at all" and
"full latency". Curriculum re-tuning for this is ticket 030 territory.

## Key Features

### 1. Unified Architecture
V3 replaces V2's 4 separate delay systems (clean_ego, clean_other, noisy_ego, noisy_other) with a single `UnifiedDelaySystem`. Perspective (ego/other) and noise mode are specified at query time.

### 2. Per-Agent Randomization
Each agent can have independent delay parameters sampled from configurable distributions:
- Communication delay (latency between agents)
- Detection delay (bbox staleness)
- Dropout rate (missed detections)
- Staleness FPS (sensor update rate)

Parameters are stored with shape `(num_envs, num_agents)` and scale with curriculum progress.

### 3. Configurable Reward States
Four reward computation modes via `RewardStateCfg`:
- `use_delay=False, use_noise=False`: Pure GT (privileged training)
- `use_delay=True, use_noise=False`: Delayed clean (default Dec-POMDP)
- `use_delay=False, use_noise=True`: GT with noise
- `use_delay=True, use_noise=True`: Full perception-aligned

### 4. Timestamp-Data Coupling
**Fundamental invariant**: The returned timestamp is ALWAYS the capture time of the returned data. This ensures Age-of-Information (AoI) calculations are always correct:
```
AoI = t_current - returned_timestamp
```

### 5. Configurable Sampling Frequencies
Each parameter (latency, staleness, dropout) can be sampled at different frequencies:
- `per_step`: Resample every simulation step
- `per_episode`: Resample at episode start (all envs)
- `per_env_reset`: Resample only for reset environments

### 6. Curriculum Learning Support
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
    ├── run_tests.py              # Test runner (55 tests)
    ├── test_per_agent.py         # Per-agent randomization tests
    ├── test_reward_modes.py      # Reward state mode tests
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

## Data Flow Architecture

### Block Diagram: Rewards vs Observations

```
                              ┌─────────────────────────────────────────────────────────────┐
                              │                    Ground Truth States                       │
                              │  (body_position_w, body_orientation_w, velocities, etc.)    │
                              └────────────────────────────┬────────────────────────────────┘
                                                           │
                                                           ▼
                              ┌─────────────────────────────────────────────────────────────┐
                              │                  update_ground_truth()                       │
                              │               Store raw + noisy versions                     │
                              └────────────────────────────┬────────────────────────────────┘
                                                           │
                         ┌─────────────────────────────────┴─────────────────────────────────┐
                         │                                                                   │
                         ▼                                                                   ▼
          ┌──────────────────────────────┐                              ┌──────────────────────────────┐
          │        Raw Storage           │                              │       Noisy Storage          │
          │   (no observation noise)     │                              │   (with observation noise)   │
          └──────────────┬───────────────┘                              └──────────────┬───────────────┘
                         │                                                             │
                         │ ◄── cfg.reward_state_cfg.use_delay ──►                      │ ◄── always use_delay=True
                         │                                                             │     for observations
    ┌────────────────────┴────────────────────┐                     ┌──────────────────┴───────────────────┐
    │                                         │                     │                                      │
    ▼                                         ▼                     ▼                                      ▼
┌─────────────┐                    ┌─────────────────────┐   ┌─────────────────────┐            ┌─────────────────────┐
│ use_delay   │                    │    use_delay        │   │    Delay Pipeline   │            │    Delay Pipeline   │
│   = False   │                    │      = True         │   │    (Ego Path)       │            │   (Other Path)      │
│ (GT timing) │                    │  ┌───────────────┐  │   │  ┌───────────────┐  │            │  ┌───────────────┐  │
└──────┬──────┘                    │  │ Staleness     │  │   │  │ Staleness     │  │            │  │ Staleness     │  │
       │                           │  │ (FPS limit)   │  │   │  │ (FPS limit)   │  │            │  │ (FPS limit)   │  │
       │                           │  └───────┬───────┘  │   │  └───────┬───────┘  │            │  └───────┬───────┘  │
       │                           │          ▼          │   │          ▼          │            │          ▼          │
       │                           │  ┌───────────────┐  │   │  ┌───────────────┐  │            │  ┌───────────────┐  │
       │                           │  │ Latency       │  │   │  │ Latency       │  │            │  │ Latency       │  │
       │                           │  │ (comm delay)  │  │   │  │ (per-agent)   │  │            │  │ (per-agent)   │  │
       │                           │  └───────┬───────┘  │   │  └───────┬───────┘  │            │  └───────┬───────┘  │
       │                           │          ▼          │   │          ▼          │            │          ▼          │
       │                           │  ┌───────────────┐  │   │  ┌───────────────┐  │            │  ┌───────────────┐  │
       │                           │  │ Dropout       │  │   │  │ Dropout       │  │            │  │ Dropout       │  │
       │                           │  │ (missed det)  │  │   │  │ (per-agent)   │  │            │  │ (per-agent)   │  │
       │                           │  └───────┬───────┘  │   │  └───────┬───────┘  │            │  └───────┬───────┘  │
       │                           └──────────┼──────────┘   └──────────┼──────────┘            └──────────┼──────────┘
       │                                      │                         │                                  │
       ▼                                      ▼                         ▼                                  ▼
┌──────────────────────────────────────────────────┐         ┌───────────────────────────────────────────────────┐
│           cfg.reward_state_cfg.use_noise         │         │           Always use_noise=True                   │
├────────────────────┬─────────────────────────────┤         │                                                   │
│   use_noise=False  │      use_noise=True         │         │                                                   │
│   (clean states)   │  (add noise to GT/delayed)  │         │                                                   │
└─────────┬──────────┴──────────────┬──────────────┘         └─────────────────────┬─────────────────────────────┘
          │                         │                                              │
          ▼                         ▼                                              ▼
    ┌─────────────────────────────────────┐                          ┌─────────────────────────────────────┐
    │       get_all_states_for_rewards()  │                          │    get_all_states_for_observations() │
    │                                     │                          │                                     │
    │  4 combinations:                    │                          │  Always: delayed + noisy            │
    │  • delay=F, noise=F → Pure GT       │                          │  Per-agent latency/dropout          │
    │  • delay=T, noise=F → Delayed clean │                          │                                     │
    │  • delay=F, noise=T → GT + noise    │                          │                                     │
    │  • delay=T, noise=T → Delayed noisy │                          │                                     │
    └─────────────────┬───────────────────┘                          └─────────────────┬───────────────────┘
                      │                                                                │
                      ▼                                                                ▼
           ┌─────────────────────┐                                        ┌─────────────────────┐
           │  Reward Computation │                                        │ Observation Building│
           │                     │                                        │                     │
           │  • Task reward      │                                        │  • State vector     │
           │  • CBF penalties    │                                        │  • BBox data        │
           │  • Collision costs  │                                        │  • AoI info         │
           └─────────────────────┘                                        └─────────────────────┘
```

### Reward State Modes

The `RewardStateCfg` controls how states are computed for reward calculation:

| use_delay | use_noise | Result | Use Case |
|-----------|-----------|--------|----------|
| False | False | Pure GT | Privileged training, baselines |
| True | False | Delayed clean | Standard Dec-POMDP (default) |
| False | True | GT + noise | Noise robustness without delay |
| True | True | Delayed + noisy | Full perception-aligned training |

### Per-Agent Delay Parameter Flow

```
                    ┌──────────────────────────────────────────┐
                    │          PerAgentDelayCfg               │
                    │  • max_comm_delay = 0.2s                 │
                    │  • max_detection_delay = 0.15s           │
                    │  • max_dropout_rate = 0.1                │
                    │  • min_fps = 10, max_fps = 60            │
                    └───────────────────┬──────────────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │       Curriculum Progress [0, 1]         │
                    │                                          │
                    │  progress=0.0: No delay/dropout          │
                    │  progress=0.5: Half max values           │
                    │  progress=1.0: Full max values           │
                    └───────────────────┬──────────────────────┘
                                        │
                                        ▼
            ┌───────────────────────────────────────────────────────────┐
            │              Per-Agent Independent Sampling               │
            │                                                           │
            │   Agent 0                Agent 1              Agent 2    │
            │   ┌──────────┐           ┌──────────┐         ┌──────────┐│
            │   │ latency: │           │ latency: │         │ latency: ││
            │   │  0.08s   │           │  0.15s   │         │  0.12s   ││
            │   │ dropout: │           │ dropout: │         │ dropout: ││
            │   │  0.03    │           │  0.07    │         │  0.05    ││
            │   │ fps:     │           │ fps:     │         │ fps:     ││
            │   │  25 Hz   │           │  18 Hz   │         │  30 Hz   ││
            │   └──────────┘           └──────────┘         └──────────┘│
            │                                                           │
            │   Shape: (num_envs, num_agents)                          │
            │   Each env×agent combination has independent params       │
            └───────────────────────────────────────────────────────────┘
```

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

Expected output: 55/55 tests passing.

Test categories:
- **Distribution/Sampling**: Parameter sampling strategies
- **Field Storage**: Data and timestamp storage
- **Pipeline**: Core delay pipeline with timestamp guarantees
- **Timestamp Sync**: AoI correctness verification
- **Curriculum Modes**: none/fixed/random mode behavior
- **Per-Agent Randomization**: Independent agent parameters (13 tests)
- **Reward Modes**: 4-mode reward state configuration (9 tests)

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
