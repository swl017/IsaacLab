# MAPPO-RNN Training System Guide

Complete guide for using the configurable MAPPO-RNN training system for both single-agent and multi-agent tasks in IsaacLab.

## Overview

The MAPPO-RNN system provides:

- ✅ **RNN with burn-in steps** - Proper hidden state warm-up
- ✅ **Episode masking** - Step-by-step GRU processing to prevent gradient contamination
- ✅ **Preprocessors** - RunningStandardScaler for state normalization
- ✅ **Bounded log std** - Prevents policy collapse
- ✅ **YAML configuration** - Easy hyperparameter tuning per task
- ✅ **Single & multi-agent** - Works for both paradigms
- ✅ **CLI overrides** - Change hyperparameters without editing files

## Quick Start

### Single-Agent Training

```bash
# Train Quadcopter with MAPPO-RNN
python scripts/reinforcement_learning/skrl/train_mappo_rnn.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --num_envs 1024 \
    --headless

# With custom hyperparameters
python scripts/reinforcement_learning/skrl/train_mappo_rnn.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --num_envs 1024 \
    --sequence_length 50 \
    --burn_in_steps 10 \
    --learning_rate 1e-3 \
    --rollouts 800 \
    --headless
```

### Multi-Agent Training

```bash
# Train multi-agent Iris environment
python scripts/reinforcement_learning/skrl/train_mappo_rnn.py \
    --task Isaac-Iris-MA2-Direct-Comm-v0 \
    --num_envs 400 \
    --headless
```

## File Structure

```
IsaacLab/
├── scripts/reinforcement_learning/skrl/
│   ├── train_mappo_rnn.py          # Main training script
│   └── mappo_rnn.py                 # MAPPO_RNN agent implementation
│
└── source/isaaclab_tasks/isaaclab_tasks/direct/
    └── quadcopter/
        ├── __init__.py              # Gym registration with config entry point
        └── agents/
            └── skrl_mappo_rnn_cfg.yaml  # Task-specific MAPPO-RNN config
```

## Configuration System

### 1. YAML Configuration File

Each task has a YAML config at `<task_dir>/agents/skrl_mappo_rnn_cfg.yaml`:

```yaml
# Example: skrl_mappo_rnn_cfg.yaml
seed: 42

models:
  policy:
    hidden_size: 256
    gru_num_layers: 2
    gru_hidden_size: 256
    initial_log_std: -0.5  # σ ≈ 0.6
    min_log_std: -5.0      # σ_min ≈ 0.007
    max_log_std: 0.7       # σ_max ≈ 2.0

  value:
    hidden_size: 256
    gru_num_layers: 2
    gru_hidden_size: 256

agent:
  # RNN-specific
  sequence_length: 50
  burn_in_steps: 10

  # PPO parameters
  rollouts: 800
  learning_epochs: 6
  mini_batches: 2
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 1.0e-3

  # Clipping
  grad_norm_clip: 0.5
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: true

  # Loss scales
  entropy_loss_scale: 0.01
  value_loss_scale: 1.0
  kl_threshold: 0.02

  # Preprocessors
  state_preprocessor: RunningStandardScaler
  shared_state_preprocessor: RunningStandardScaler
  value_preprocessor: RunningStandardScaler

  # Other
  time_limit_bootstrap: true
  mixed_precision: false

  experiment:
    directory: quadcopter_mappo_rnn
    wandb: false

trainer:
  timesteps: 400000
  headless: true
```

### 2. Environment Registration

In `<task_dir>/__init__.py`, add the config entry point:

```python
gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env:QuadcopterEnvCfg",
        # ... other entry points ...
        "skrl_mappo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_mappo_rnn_cfg.yaml",
    },
)
```

### 3. CLI Arguments

Override config values from the command line:

```bash
--sequence_length 50      # RNN sequence length
--burn_in_steps 10        # Burn-in steps
--learning_rate 1e-3      # Learning rate
--rollouts 800            # Rollouts before update
--max_iterations 5000     # Total training iterations
--experiment_name my_exp  # Custom experiment name
--seed 42                 # Random seed
```

## Key Features

### 1. Burn-in Steps

Burn-in allows the RNN hidden state to "warm up" before computing losses:

```python
# In YAML config
agent:
  sequence_length: 50
  burn_in_steps: 10  # First 10 steps are for warm-up, losses computed on last 40
```

**Benefits**:
- More stable training
- Better gradient quality
- Prevents poorly initialized hidden states from dominating losses

**Rule**: `burn_in_steps < sequence_length`

### 2. Episode Masking

The model zeros hidden states at episode boundaries:

```python
# In mappo_rnn.py:684-744
def compute_base(self, inputs):
    # Process sequence step-by-step
    for t in range(S):
        if need_reset:
            done_mask = dones[:, t]
            if done_mask.any():
                h[:, done_indices, :] = 0.0  # Zero hidden state at episode end
        out_t, h = self.gru(x[:, t:t+1, :], h)
```

**Benefits**:
- Prevents gradient contamination across episode boundaries
- Critical for RNN training stability

### 3. Bounded Log Std

Policy action noise is bounded to prevent collapse or explosion:

```python
# In YAML config
models:
  policy:
    initial_log_std: -0.5  # σ ≈ 0.6 (conservative exploration)
    min_log_std: -5.0      # σ_min ≈ 0.007 (prevents collapse)
    max_log_std: 0.7       # σ_max ≈ 2.0 (caps exploration)
```

### 4. Running Normalization

Input normalization improves convergence:

```python
# In YAML config
agent:
  state_preprocessor: RunningStandardScaler
  shared_state_preprocessor: RunningStandardScaler
  value_preprocessor: RunningStandardScaler
```

## Implementation Details

### Single-Agent Mode

For single-agent environments (like Quadcopter), the system automatically:

1. Wraps the environment in a single-agent dictionary:
   ```python
   possible_agents = ["agent_0"]
   observation_spaces = {"agent_0": env.observation_space}
   ```

2. Creates MAPPO-RNN agent with one agent
3. Trains using the MAPPO algorithm (which degenerates to PPO for single agent)

### Multi-Agent Mode

For multi-agent environments (like Iris), the system:

1. Detects `DirectMARLEnv` automatically
2. Creates shared policy/value networks for homogeneous agents
3. Manages separate memories per agent
4. Concatenates observations for centralized value function

## Hyperparameter Tuning Guide

### RNN Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `sequence_length` | 50 | 16-128 | Longer = more temporal context, higher memory |
| `burn_in_steps` | 10 | 0-30 | Higher = more stable, but fewer training samples |
| `gru_num_layers` | 2 | 1-3 | More layers = more capacity, slower training |
| `gru_hidden_size` | 256 | 128-512 | Larger = more capacity, higher memory |

### PPO Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `learning_rate` | 1e-3 | 1e-4 to 5e-3 | Lower = more stable for RNNs |
| `rollouts` | 800 | 400-1600 | More = less variance, slower updates |
| `mini_batches` | 2 | 2-8 | Lower = better sequence sampling |
| `learning_epochs` | 6 | 4-10 | More = more optimization per rollout |
| `grad_norm_clip` | 0.5 | 0.3-1.0 | Lower = more conservative for RNNs |

### Recommended Configurations

**Fast Prototyping** (lower quality, faster):
```yaml
sequence_length: 32
burn_in_steps: 5
rollouts: 400
learning_epochs: 4
num_envs: 2048
```

**High Quality** (better performance, slower):
```yaml
sequence_length: 64
burn_in_steps: 16
rollouts: 1600
learning_epochs: 8
num_envs: 4096
```

**RNN-Focused** (maximum temporal reasoning):
```yaml
sequence_length: 128
burn_in_steps: 32
gru_num_layers: 3
gru_hidden_size: 512
rollouts: 800
```

## Comparison: MAPPO_RNN vs SKRL PPO_RNN vs train_ppo_rnn.py

| Feature | MAPPO_RNN (This) | SKRL PPO_RNN | train_ppo_rnn.py |
|---------|------------------|--------------|------------------|
| **burn_in_steps** | ✅ 10 steps | ❌ Not supported | ❌ Not implemented |
| **Episode masking** | ✅ Step-by-step | ⚠️ Model dependent | ❌ Not implemented |
| **Preprocessors** | ✅ Auto-configured | ⚠️ Manual setup | ❌ Not configured |
| **Log std bounds** | ✅ Bounded | ⚠️ Model dependent | ❌ Unbounded |
| **YAML config** | ✅ Per-task | ❌ Code-based | ❌ Hardcoded |
| **Single-agent** | ✅ Supported | ✅ Native | ✅ Native |
| **Multi-agent** | ✅ Supported | ❌ Not supported | ❌ Not supported |
| **CLI overrides** | ✅ All hyperparams | ⚠️ Limited | ⚠️ Limited |

## Advanced Usage

### Custom Model Architecture

Edit the YAML to change network sizes:

```yaml
models:
  policy:
    hidden_size: 512        # Larger preprocessing network
    gru_num_layers: 3       # Deeper GRU
    gru_hidden_size: 512    # Larger GRU capacity
```

### Multiple Configs for Same Task

Create variants:

```bash
# Default config
agents/skrl_mappo_rnn_cfg.yaml

# Fast prototyping
agents/skrl_mappo_rnn_fast_cfg.yaml

# High quality
agents/skrl_mappo_rnn_hq_cfg.yaml
```

Then update `__init__.py`:

```python
gym.register(
    id="Isaac-Quadcopter-Direct-Fast-v0",
    # ...
    kwargs={
        "skrl_mappo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_mappo_rnn_fast_cfg.yaml",
    },
)
```

### Logging and Checkpoints

Results are saved to:

```
logs/skrl/<experiment_dir>/<timestamp>_<experiment_name>/
├── params/
│   ├── config.yaml          # Full configuration used
│   └── config.pkl           # Pickled config
├── checkpoints/             # Model checkpoints during training
├── agent_agent_0_final.pt   # Final trained model
└── training_config.pt       # Training metadata
```

## Troubleshooting

### Error: "burn_in_steps must be less than sequence_length"

**Solution**: Reduce `burn_in_steps` or increase `sequence_length`:

```bash
--burn_in_steps 10 --sequence_length 50
```

### Error: "Task does not have 'skrl_mappo_rnn_cfg_entry_point'"

**Solution**: Add the entry point to the task's `__init__.py`:

```python
gym.register(
    id="Isaac-YourTask-v0",
    # ...
    kwargs={
        "skrl_mappo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_mappo_rnn_cfg.yaml",
    },
)
```

### Poor Training Performance

**Check**:
1. Is `burn_in_steps` too high? (reduces effective training samples)
2. Is `sequence_length` too short? (RNN can't capture temporal dependencies)
3. Is `learning_rate` too high? (RNNs need lower LR than feedforward)
4. Is `mini_batches` too high? (breaks sequence structure)

**Solutions**:
- Increase `sequence_length` to 64 or 128
- Reduce `learning_rate` to 5e-4 or 3e-4
- Set `mini_batches` to 2 or 4
- Increase `num_envs` for more diverse data

### Out of Memory

**Solutions**:
- Reduce `num_envs`
- Reduce `sequence_length`
- Reduce `gru_hidden_size` or `gru_num_layers`
- Use `mixed_precision: true` in config

## Summary

You now have a complete, production-ready MAPPO-RNN training system that:

1. ✅ **Works out of the box** - Just run the command with your task name
2. ✅ **Highly configurable** - YAML configs + CLI overrides
3. ✅ **Single & multi-agent** - Automatic detection and handling
4. ✅ **All advanced features** - burn-in, episode masking, bounded exploration
5. ✅ **Extensible** - Easy to add new tasks or customize configs

The system is ready for your command:

```bash
python scripts/reinforcement_learning/skrl/train_mappo_rnn.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --num_envs 1024 \
    --headless
```

Enjoy training! 🚀
