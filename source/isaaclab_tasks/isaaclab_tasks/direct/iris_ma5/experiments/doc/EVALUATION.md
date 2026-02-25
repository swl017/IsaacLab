# Evaluation Pipeline

Post-training evaluation for IROS 2026 ablation experiments.

## Overview

The evaluation pipeline loads a trained checkpoint, runs policy rollouts under
full-difficulty curriculum conditions, and computes 8 paper metrics.
Results are saved as JSON and printed to console.

**Entry point**: `experiments/evaluate.py`

## 8 Paper Metrics

| # | Metric | Key | Description | Unit |
|---|--------|-----|-------------|------|
| 1 | trace(Sigma_X) | `trace_sigma_mean/std` | Mean triangulation uncertainty (trace of covariance matrix) over valid-triangulation steps | m^2 |
| 2 | Triangulation RMSE | `triangulation_rmse_mean/std` | Localization error: Euclidean distance between midpoint-method triangulation estimate and ground-truth target position | m |
| 3 | Task success rate | `task_success_rate` | Fraction of episodes where RMSE < 2 m for >= 80% of steps | 0-1 |
| 4 | Visibility ratio | `visibility_mean/std` | Mean fraction of agents with a valid bounding-box detection per step (0 = none see target, 1 = all agents see target) | 0-1 |
| 5 | Collision rate | `collision_rate_mean/std` | Total inter-agent collisions per episode | count |
| 6 | Convergence speed | `convergence_speed_mean/std` | Steps to first reach trace(Sigma_X) < 10.0 (-1 if never reached) | steps |
| 7 | Time-to-first-lock | `time_to_first_lock_mean/std` | Steps to first valid triangulation with trace < 50.0 (-1 if never reached) | steps |
| 8 | Tri valid ratio | `tri_valid_ratio_mean/std` | Time fraction with successful triangulation (>= 2 agents with valid detections and non-degenerate geometry) | 0-1 |

All metrics are reported as **mean +/- std** across completed episodes.

### Thresholds (configurable via `MetricTracker.__init__`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `success_rmse_threshold` | 2.0 m | Max RMSE for a step to count as "successful" |
| `success_time_fraction` | 0.8 | Fraction of successful steps needed for episode-level success |
| `convergence_trace_threshold` | 10.0 | trace(Sigma_X) below which convergence is declared |
| `first_lock_trace_threshold` | 50.0 | trace(Sigma_X) below which first lock is declared |

## Usage

### Single experiment evaluation

```bash
./isaaclab.sh -p .../experiments/evaluate.py \
    --experiment a3_noisy_reward \
    --checkpoint /path/to/agent_drone_0_final.pt \
    --num_episodes 100 \
    --num_envs 256 \
    --headless \
    --output results.json
```

### Greedy baseline (scripted policy, no checkpoint)

```bash
./isaaclab.sh -p .../experiments/evaluate.py \
    --experiment baseline_greedy \
    --num_episodes 100 \
    --output eval_greedy.json
```

### Delay sweep (runtime parameter override)

```bash
for delay in 0.0 0.05 0.1 0.15 0.2; do
    ./isaaclab.sh -p .../experiments/evaluate.py \
        --experiment a1_with_aoi \
        --checkpoint /path/to/best_agent.pt \
        --delay-override $delay \
        --output eval_delay_${delay}.json
done
```

### Batch evaluation (all experiments)

```bash
bash 2026-IROS/results/run_evaluations.sh
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | (required) | Experiment name from registry (e.g., `a1_with_aoi`, `a3_noisy_reward`) |
| `--checkpoint` | None | Path to `.pt` checkpoint file. Required for trained policies, omit for greedy baseline. |
| `--num_episodes` | 100 | Number of evaluation episodes to run |
| `--num_envs` | 256 | Number of parallel evaluation environments |
| `--output` | None | Output JSON path. If omitted, results are only printed. |
| `--headless` | True | Run without GUI |
| `--delay-override` | None | Override detection latency in seconds (e.g., `0.1` for 100 ms) |
| `--comm-delay-override` | None | Override communication latency in seconds |
| `--target-speed` | None | Override target speed in m/s |

## Checkpoint Formats

The evaluation script auto-detects two checkpoint formats:

### 1. `agent_drone_0_final.pt` (training-end checkpoint)

```python
{
    "policy_state_dict": ...,
    "value_state_dict": ...,
}
```

This format does **not** contain preprocessor (RunningStandardScaler) state.
The evaluator will attempt to load preprocessor state from a companion
`best_agent.pt` in a sibling `checkpoints/` directory. Without preprocessor
state, observations are not normalized and the policy produces garbage output.

**Search order for companion preprocessor file**:
1. `<checkpoint_dir>/checkpoints/best_agent.pt`
2. `<checkpoint_dir>/best_agent.pt`

### 2. `best_agent.pt` (SKRL multi-agent format)

```python
{
    "drone_0": {
        "policy": ...,
        "value": ...,
        "state_preprocessor": {"running_mean": ..., "running_variance": ..., "current_count": ...},
        "shared_state_preprocessor": ...,
        "value_preprocessor": ...,
        "optimizer": ...,
    },
    "drone_1": { ... },
}
```

This format is loaded via `agent.load()` which restores all state including
preprocessors. This is the preferred format when available.

### Preprocessor importance

The `RunningStandardScaler` normalizes observations to zero mean and unit
variance based on statistics collected during training. Without these saved
statistics, the policy receives un-normalized inputs and fails completely.

**Symptom of missing preprocessor**: 0% triangulation valid rate despite the
same policy achieving >80% during training.

## Architecture Selection

The evaluator auto-selects the correct model architecture based on
`exp_cfg.use_mlp_model`:

| Experiment | Architecture | Model classes |
|------------|-------------|---------------|
| a1_*, a3_*, a4_* (most) | MAPPO-RNN (GRU) | `MAPPORNNPolicy`, `MAPPORNNValue` |
| a2_mlp | MAPPO-MLP | `MAPPOMLPPolicy`, `MAPPOMLPValue` |
| baseline_gavin2024 | MAPPO-MLP | `MAPPOMLPPolicy`, `MAPPOMLPValue` |
| baseline_greedy | Scripted | `GreedyEquiangularPolicy` |

Agent overrides (e.g., `hidden_size: 256` for MLP) are applied automatically
from the experiment registry via `apply_agent_overrides()`.

## Evaluation Conditions

During evaluation, all curriculum phases are set to **full difficulty**
(`start_step=0`, `end_step=0`), meaning `_linear_progress()` returns 1.0 for
all dimensions. This ensures consistent evaluation regardless of the
curriculum schedule used during training.

This includes:
- Full tracking difficulty (target at max speed)
- Full safety penalties enabled
- Full coordination rewards
- Full delay, noise, and dropout (unless overridden by experiment config)

Experiments that disable delays (e.g., `a1_no_delay`, `a1_curriculum_no_delay`)
have their overrides applied *before* curriculum is set to full difficulty, so
their delay/noise values remain at zero.

## Data Flow

```
evaluate.py
  ├── Loads experiment config from registry (experiment_registry.py)
  ├── Applies env_overrides and agent_overrides
  ├── Sets all curriculum to full difficulty
  ├── Creates environment (Isaac Lab DirectMARLEnv)
  ├── Loads policy:
  │     ├── RNN: _load_rnn_policy() using mappo_rnn.py
  │     ├── MLP: _load_mlp_policy() using mappo_mlp.py
  │     └── Greedy: GreedyEquiangularPolicy (scripted)
  ├── Runs rollout loop:
  │     ├── policy.act(obs) → actions
  │     ├── env.step(actions) → obs, rewards, done
  │     ├── _collect_step_metrics(tracker, env)
  │     │     ├── triangulation_results_noisy[agent] → X_w_tri, trace, is_valid
  │     │     ├── target.data.root_pos_w → gt_target_pos
  │     │     ├── delay_system states → bbox_valid_mask
  │     │     └── safety_manager.get_collision_matrix() → collision_flags
  │     └── tracker.record_episode_end(done_envs)
  └── tracker.compute_final_metrics() → JSON output
```

### Key implementation details

**RMSE source**: Uses `triangulation_results_noisy` (observation/noisy path)
which stores the actual midpoint-method triangulated position at index 0.
The rewards-path `triangulation_results` stores ground-truth position at
index 0 (used for covariance reward computation), not an estimate.

**Collision detection**: Uses `safety_manager.get_collision_matrix()` which
returns `[N, A, A]` boolean tensor. Reduced to per-env flags via
`.any(dim=-1).any(dim=-1)`.

**Visibility**: Computed as `bbox_valid_mask.float().mean(dim=-1)` — the mean
fraction of agents with valid detections per step. For 2 agents: 0.0 (none),
0.5 (one), 1.0 (both).

## Output Format

The output JSON contains all metric means and standard deviations:

```json
{
  "trace_sigma_mean": 0.33,
  "trace_sigma_std": 1.09,
  "triangulation_rmse_mean": 0.56,
  "triangulation_rmse_std": 0.29,
  "task_success_rate": 0.547,
  "visibility_mean": 0.84,
  "visibility_std": 0.09,
  "collision_rate_mean": 0.77,
  "collision_rate_std": 3.75,
  "convergence_speed_mean": 76.5,
  "convergence_speed_std": 35.1,
  "time_to_first_lock_mean": 76.5,
  "time_to_first_lock_std": 35.1,
  "tri_valid_ratio_mean": 0.777,
  "tri_valid_ratio_std": 0.141,
  "num_episodes": 128,
  "experiment": "a3_noisy_reward",
  "checkpoint": "/path/to/agent_drone_0_final.pt"
}
```

## Files

| File | Purpose |
|------|---------|
| `evaluate.py` | Main evaluation script (CLI entry point) |
| `metrics/metric_tracker.py` | `MetricTracker` class — per-step accumulation + per-episode aggregation |
| `models/mappo_mlp.py` | MLP policy/value network definitions |
| `models/mappo_mlp_agent.py` | `MAPPO_MLP` agent class |
| `baselines/greedy_policy.py` | Scripted equiangular orbit policy |
| `experiment_registry.py` | Experiment definitions (overrides, architecture flags) |
| `env_overrides.py` | `apply_env_overrides()`, `apply_agent_overrides()` |
| `2026-IROS/results/run_evaluations.sh` | Batch script for all 7 experiments |
| `2026-IROS/results/eval_*.json` | Saved evaluation results |
