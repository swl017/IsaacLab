# Experiments Module Specification — iris_ma6

**Version**: 1.0
**Date**: 2026-03-22
**Status**: Draft
**Predecessor**: `iris_ma5/experiments/`

---

## 1. Overview

The experiments module provides reproducible experiment definition, training orchestration, and evaluation for the iris_ma6 multi-agent drone observation environment. It enables systematic ablation studies by expressing each experiment as a named set of overrides on top of the base `IrisMA6TestEnvCfg`.

### 1.1 Relationship to iris_ma5

The iris_ma5 experiments module defined 30+ experiments across groups A1-A4, baselines, and sweeps for IROS 2026. This spec adapts that architecture for iris_ma6 with the following changes:

| Aspect | iris_ma5 | iris_ma6 |
|--------|----------|----------|
| Base config | `IrisMAEnvCfg` | `IrisMA6TestEnvCfg` |
| Gym task ID | `Isaac-Iris-MA5-Direct-v0` | `Isaac-Iris-MA6-Direct-Test-v0` |
| Obs dim | `26 + 15*(n-1) + 6` | `24 + 6` (fixed, no inter-agent obs fields) |
| Action dim | 7 | 7 |
| Safety | SafetyManager (collision + TTC) | CBFManager (CPA penalty + deployment filter) |
| Delay | DelaySystemV2 | DelaySystemV3 |
| Controller | PointMass | PX4-style cascaded DroneController |
| Architecture ablation (A2) | RNN vs MLP | **Dropped** (RNN only) |
| Reward path (A3) | clean vs noisy | 2x2: delay(on/off) x noise(on/off) |
| New groups | -- | A5 (CBF), A6 (Controller), A7 (Task Reward Levels) |

### 1.2 Design Principles

- **Override-based**: Each experiment = base config + env_overrides + agent_overrides + seed
- **Registry-driven**: All experiments registered at import time; discoverable via CLI
- **Curriculum-aware**: Evaluation forces all curriculum to full difficulty (start=0, end=0)
- **Metric-first**: Evaluation pipeline defines paper metrics; experiments map to table rows

---

## 2. File Structure

```
iris_ma6/experiments/
├── __init__.py                 # Public API exports
├── experiment_cfg.py           # ExperimentCfg, ExperimentSuiteCfg dataclasses
├── experiment_registry.py      # Central registry + all experiment definitions
├── env_overrides.py            # apply_env_overrides(), apply_agent_overrides()
├── run_experiment.py           # Single experiment training CLI
├── run_suite.py                # Batch runner (sequential subprocess)
├── evaluate.py                 # Evaluation pipeline (metrics + trajectories)
├── frame_stack_wrapper.py      # MultiAgentFrameStackWrapper (reserved for future MLP)
├── metrics/
│   ├── __init__.py
│   ├── metric_tracker.py       # MetricTracker: per-episode metric aggregation
│   ├── timeseries_tracker.py   # TimeseriesTracker: per-timestep statistics
│   └── trajectory_recorder.py  # TrajectoryRecorder: XYZ + gimbal + zoom traces
└── baselines/
    ├── __init__.py
    └── greedy_policy.py        # GreedyEquiangularPolicy (7D action space)
```

---

## 3. Configuration Dataclasses

### 3.1 ExperimentCfg

```python
@configclass
class ExperimentCfg:
    # Identification
    name: str                          # Unique key (e.g., "a1_no_delay")
    description: str                   # Human-readable description
    group: str                         # Category (A1, A3, A4, A5, A6, A7, baseline, sweep)

    # Overrides
    env_overrides: Dict[str, Any]      # Dotted-path overrides for IrisMA6TestEnvCfg
    agent_overrides: Dict[str, Any]    # Dotted-path overrides for agent YAML dict

    # Training
    seeds: List[int] = [42, 123, 456]
    num_envs: int = 4096
    total_timesteps: int = 200000

    # Evaluation
    eval_num_envs: int = 256
    eval_episodes: int = 100
```

**Changes from iris_ma5**:
- Removed `use_mlp_model`, `frame_stack`, `frame_skip` (no MLP ablation)
- `total_timesteps` default matches iris_ma6 curriculum `all_end_step` (200k)
- `num_envs` default 4096 (validated for 2-agent with full physics on RTX 4090)

### 3.2 ExperimentSuiteCfg

```python
@configclass
class ExperimentSuiteCfg:
    name: str                          # Suite name (e.g., "iros2026_must")
    experiments: List[str]             # Registry keys
    parallel_seeds: bool = True        # Seeds on different GPUs
    base_log_dir: str = "logs/experiments"
```

Unchanged from iris_ma5.

---

## 4. Registry API

Same interface as iris_ma5:

```python
register_experiment(cfg: ExperimentCfg) -> None
get_experiment(name: str) -> ExperimentCfg
list_experiments(group: Optional[str] = None) -> list[str]

register_suite(cfg: ExperimentSuiteCfg) -> None
get_suite(name: str) -> ExperimentSuiteCfg
list_suites() -> list[str]
```

Observation dimension helper (simplified for iris_ma6):

```python
def _compute_obs_dim(enable_triangulation: bool = True) -> int:
    """24 base + 6 triangulation tail."""
    return 24 + (6 if enable_triangulation else 0)
```

---

## 5. Override Utilities

### 5.1 `apply_env_overrides(cfg: IrisMA6TestEnvCfg, overrides: Dict[str, Any]) -> IrisMA6TestEnvCfg`

Traverses dotted paths on the config object. Example:

```python
overrides = {"delay_system_params.dropout_prob": 0.0}
# Resolves to: cfg.delay_system_params.dropout_prob = 0.0
```

Raises `AttributeError` if path does not exist (fail-fast for typos).

### 5.2 `apply_agent_overrides(agent_cfg: dict, overrides: Dict[str, Any]) -> dict`

Creates intermediate dicts as needed. Same as iris_ma5.

### 5.3 Valid Override Paths

The following paths have been validated against `IrisMA6TestEnvCfg`:

**Delay System** (`delay_system_params.*`):
- `delay_system_params.ego_motion_latency_enabled` (bool)
- `delay_system_params.ego_motion_fol_tau` (float, seconds)
- `delay_system_params.ego_detection_latency_mean` (float, seconds)
- `delay_system_params.ego_detection_latency_std` (float, seconds)
- `delay_system_params.other_latency_mean` (float, seconds)
- `delay_system_params.other_latency_std` (float, seconds)
- `delay_system_params.staleness_fps_mean` (float)
- `delay_system_params.staleness_fps_range` (float)
- `delay_system_params.dropout_prob` (float, 0-1)
- `delay_system_params.noise_enabled` (bool)
- `delay_system_params.noise_position_std` (float, meters)
- `delay_system_params.noise_velocity_std` (float, m/s)
- `delay_system_params.noise_orientation_std` (float, radians)
- `delay_system_params.noise_bbox_std` (float, pixels)
- `delay_system_params.reward_use_delay` (bool)
- `delay_system_params.reward_use_noise` (bool)
- `enable_delay_system` (bool)

**Curriculum** (`curriculum.*`):
- `curriculum.noise_start_step`, `curriculum.noise_end_step`
- `curriculum.fixed_delay_start_step`, `curriculum.fixed_delay_end_step`
- `curriculum.random_delay_start_step`, `curriculum.random_delay_end_step`
- `curriculum.dropout_start_step`, `curriculum.dropout_end_step`
- `curriculum.tracking_start_step`, `curriculum.tracking_end_step`
- `curriculum.moving_target_start_step`, `curriculum.moving_target_end_step`
- `curriculum.coordination_start_step`, `curriculum.coordination_end_step`
- `curriculum.safety_start_step`, `curriculum.safety_end_step`
- `curriculum.dynamics_start_step`, `curriculum.dynamics_end_step`
- `curriculum.task_level_2_start_step`, `curriculum.task_level_2_end_step`
- `curriculum.task_level_3_start_step`, `curriculum.task_level_3_end_step`

**Reward / Task Level**:
- `use_noisy_rewards` (bool)
- `task_reward_level` (int, 1/2/3)
- `estimation_error_scale` (float)
- `estimation_error_temp` (float)
- `e2e_weight` (float, 0-1)
- `curriculum_task_levels` (bool)
- `triangulation_reward_scale` (float)
- `action_sum_penalty_scale` (float)
- `bbox_center_reward_scale` (float)
- `bbox_size_reward_scale` (float)

**CBF Safety** (`cbf_safety.*`):
- `cbf_safety.cpa_cfg.lambda_cbf` (float)
- `cbf_safety.cpa_cfg.D_s` (float, meters)
- `cbf_safety.cpa_cfg.gamma` (float)
- `cbf_safety.cpa_cfg.T` (float, seconds)
- `cbf_safety.enable_training_penalty` (bool)
- `cbf_safety.enable_collision_termination` (bool)
- `cbf_safety.collision_distance` (float, meters)

**Controller / Dynamics**:
- `gain_randomization.enabled` (bool)
- `gain_randomization.scale_range` (tuple)
- `gain_randomization.randomize_max_lin_vel` (bool)
- `max_lin_vel` (float, m/s)
- `max_lin_vel_min` (float, m/s)

**Camera / Observation**:
- `use_omnidirectional_cameras` (bool)
- `enable_triangulation` (bool)

---

## 6. Experiment Definitions

### 6.1 A1 — Delay Modeling

| Name | Description | Key Overrides |
|------|-------------|--------------|
| `a1_no_delay` | No delay, noise, or dropout | `enable_delay_system: False`, push all delay/noise/dropout curriculum beyond 999999 |
| `a1_stochastic_delay` | Full delay pipeline (default params) | Default config (no overrides) |
| `a1_with_aoi` | Full delay with AoI observation | Default + AoI appended to obs (implementation detail) |
| `a1_without_aoi` | Full delay without AoI observation | Default without AoI fields |

**Curriculum override pattern for "no delay"**:
```python
env_overrides = {
    "enable_delay_system": False,
    "curriculum.noise_start_step": 999999,
    "curriculum.noise_end_step": 999999,
    "curriculum.fixed_delay_start_step": 999999,
    "curriculum.fixed_delay_end_step": 999999,
    "curriculum.random_delay_start_step": 999999,
    "curriculum.random_delay_end_step": 999999,
    "curriculum.dropout_start_step": 999999,
    "curriculum.dropout_end_step": 999999,
}
```

### 6.2 A3 — Reward Path (Delay x Noise)

Four combinations testing how reward computation pipeline affects learned behavior.

| Name | Delay in Reward | Noise in Reward | Key Overrides |
|------|-----------------|-----------------|--------------|
| `a3_clean` | No | No | `delay_system_params.reward_use_delay: False`, `delay_system_params.reward_use_noise: False` |
| `a3_delay_only` | Yes | No | `delay_system_params.reward_use_delay: True`, `delay_system_params.reward_use_noise: False` (default) |
| `a3_noise_only` | No | Yes | `delay_system_params.reward_use_delay: False`, `delay_system_params.reward_use_noise: True` |
| `a3_noisy_delay` | Yes | Yes | `use_noisy_rewards: True` (uses observation pipeline for rewards) |

Note: `a3_delay_only` is the default (paper) configuration. The `use_noisy_rewards` flag routes reward computation through the full observation pipeline including delay+noise.

### 6.3 A4 — Covariance Reward Mode

| Name | Description | Key Overrides |
|------|-------------|--------------|
| `a4_analytical` | Multi-source FIM covariance (paper default, Level 1) | `task_reward_level: 1` (default) |
| `a4_analytical_delay` | **TODO**: Update with improved covariance model that accounts for delay-induced uncertainty | `task_reward_level: 1`, TBD covariance model overrides |

### 6.4 A5 — CBF Safety (NEW)

Tests the effect of CBF collision avoidance on learned behavior.

| Name | Description | Key Overrides |
|------|-------------|--------------|
| `a5_no_cbf` | No safety penalty | `cbf_safety.enable_training_penalty: False`, `cbf_safety.enable_collision_termination: False` |
| `a5_cbf_default` | Default CBF (lambda=1.0, D_s=2.0) | Default config |
| `a5_cbf_lambda_0.5` | Reduced penalty weight | `cbf_safety.cpa_cfg.lambda_cbf: 0.5` |
| `a5_cbf_lambda_2.0` | Increased penalty weight | `cbf_safety.cpa_cfg.lambda_cbf: 2.0` |
| `a5_cbf_lambda_5.0` | Strong penalty | `cbf_safety.cpa_cfg.lambda_cbf: 5.0` |
| `a5_cbf_radius_1.5m` | Tighter safety margin | `cbf_safety.cpa_cfg.D_s: 1.5`, `cbf_safety.collision_distance: 1.5` |
| `a5_cbf_radius_3.0m` | Wider safety margin | `cbf_safety.cpa_cfg.D_s: 3.0`, `cbf_safety.collision_distance: 3.0` |
| `a5_termination_only` | Collision terminates but no shaping | `cbf_safety.enable_training_penalty: False`, `cbf_safety.enable_collision_termination: True` |

### 6.5 A6 — Controller Robustness (NEW)

Tests gain randomization and velocity limits.

| Name | Description | Key Overrides |
|------|-------------|--------------|
| `a6_no_gain_rand` | No gain randomization | `gain_randomization.enabled: False` |
| `a6_gain_rand_default` | Default +-20% | Default config |
| `a6_gain_rand_40pct` | +-40% gain range | `gain_randomization.scale_range: (0.6, 1.4)` |
| `a6_max_vel_5` | Conservative velocity limit | `max_lin_vel: 5.0`, `max_lin_vel_min: 3.0` |
| `a6_max_vel_15` | Aggressive velocity limit | `max_lin_vel: 15.0`, `max_lin_vel_min: 8.0` |

### 6.6 A7 — Task Reward Levels (NEW)

Tests the multi-level reward curriculum (FIM -> GT-anchored -> E2E composite).

| Name | Description | Key Overrides |
|------|-------------|--------------|
| `a7_level1_fim` | FIM proxy only (sqrt(10/Tr)) | `task_reward_level: 1`, `curriculum_task_levels: False` |
| `a7_level2_gt_anchored` | GT-anchored estimation error | `task_reward_level: 2`, `curriculum_task_levels: False` |
| `a7_level3_composite` | Composite (GT + E2E blend) | `task_reward_level: 3`, `curriculum_task_levels: False`, `e2e_weight: 0.5` |
| `a7_curriculum_blend` | Progressive level transitions | `curriculum_task_levels: True` |
| `a7_e2e_weight_0.0` | Pure GT-anchored in Level 3 | `task_reward_level: 3`, `e2e_weight: 0.0` |
| `a7_e2e_weight_0.3` | Mostly GT-anchored | `task_reward_level: 3`, `e2e_weight: 0.3` |
| `a7_e2e_weight_0.7` | Mostly E2E | `task_reward_level: 3`, `e2e_weight: 0.7` |
| `a7_e2e_weight_1.0` | Pure E2E | `task_reward_level: 3`, `e2e_weight: 1.0` |
| `a7_high_temp` | Sharper reward falloff | `task_reward_level: 2`, `estimation_error_temp: 3.0` |
| `a7_low_temp` | Gentler reward falloff | `task_reward_level: 2`, `estimation_error_temp: 0.3` |

### 6.7 Baselines

| Name | Description | Notes |
|------|-------------|-------|
| `baseline_greedy` | Scripted equiangular orbit | Eval-only (no training). Agents maintain fixed-radius orbit with 2pi/N spacing. 7D action space: proportional velocity toward orbit point, gimbal tracks target, zoom = 0. |
| `baseline_no_triangulation` | BBox tracking only | `enable_triangulation: False`, `triangulation_reward_scale: 0.0` |

### 6.8 Sweeps

| Name | Sweep Variable | Values |
|------|---------------|--------|
| `sweep_agents_n2` | Agent count | `num_agents: 2` |
| `sweep_agents_n3` | Agent count | `num_agents: 3` |
| `sweep_agents_n4` | Agent count | `num_agents: 4` |
| `sweep_delay_200ms` | Comm latency | `delay_system_params.other_latency_mean: 0.2` |
| `sweep_delay_500ms` | Comm latency | `delay_system_params.other_latency_mean: 0.5` (default) |
| `sweep_delay_800ms` | Comm latency | `delay_system_params.other_latency_mean: 0.8` |
| `sweep_noise_0px` | BBox noise | `delay_system_params.noise_bbox_std: 0.0` |
| `sweep_noise_3px` | BBox noise | `delay_system_params.noise_bbox_std: 3.0` |
| `sweep_noise_7px` | BBox noise | `delay_system_params.noise_bbox_std: 7.0` (default) |
| `sweep_noise_14px` | BBox noise | `delay_system_params.noise_bbox_std: 14.0` |
| `sweep_target_speed_3` | Target speed | `max_lin_vel: 3.0`, `max_lin_vel_min: 2.0` |
| `sweep_target_speed_5` | Target speed | `max_lin_vel: 5.0`, `max_lin_vel_min: 3.0` |
| `sweep_target_speed_10` | Target speed | `max_lin_vel: 10.0`, `max_lin_vel_min: 5.0` (default) |

### 6.9 Experiment Suites

| Suite | Experiments | Purpose |
|-------|------------|---------|
| `iros2026_must` | A1 (no_delay, stochastic), A3 (all 4), A5 (no_cbf, default), A7 (level1, level2, curriculum_blend), baseline_greedy | Core paper results |
| `iros2026_should` | A4 (all), A5 (lambda sweep), A6 (no_rand, default), A7 (e2e_weight sweep) | Extended ablations |
| `iros2026_sweeps` | All sweep_* experiments | Generalization curves |
| `iros2026_full` | All registered experiments | Complete run |

---

## 7. Training Entry Points

### 7.1 `run_experiment.py`

CLI for running a single named experiment.

```bash
# Run experiment
./isaaclab.sh -p .../iris_ma6/experiments/run_experiment.py \
    --experiment a1_no_delay --seed 42

# List all experiments
./isaaclab.sh -p .../iris_ma6/experiments/run_experiment.py --list

# Resume from checkpoint
./isaaclab.sh -p .../iris_ma6/experiments/run_experiment.py \
    --experiment a3_clean --checkpoint /path/to/agent.pt
```

**Arguments**:
- `--experiment NAME` (required) — Registry key
- `--seed INT` — Override seed (default: first from experiment config)
- `--num_envs INT` — Override parallel env count
- `--task STR` — Gym task ID (default: `Isaac-Iris-MA6-Direct-Test-v0`)
- `--checkpoint PATH` — Resume training
- `--list` — Print registry and exit (no sim launch)
- `--headless` — No GUI (default: True)

**Workflow**:
1. Load experiment from registry
2. Create `IrisMA6TestEnvCfg`, apply `env_overrides`
3. Create gym environment
4. Load agent config YAML, apply `agent_overrides`
5. Initialize MAPPO-RNN agent (from `scripts/reinforcement_learning/skrl/mappo_rnn.py`)
6. Run SKRL SequentialTrainer

### 7.2 `run_suite.py`

Batch execution of experiment suites via subprocess.

```bash
# Run core experiments
./isaaclab.sh -p .../iris_ma6/experiments/run_suite.py \
    --suite iros2026_must --seeds 42,123,456

# Dry run (print commands)
./isaaclab.sh -p .../iris_ma6/experiments/run_suite.py \
    --suite iros2026_full --dry_run
```

Each (experiment, seed) pair launches as a separate subprocess calling `run_experiment.py`.

---

## 8. Evaluation Pipeline

### 8.1 CLI

```bash
# Standard evaluation
./isaaclab.sh -p .../iris_ma6/experiments/evaluate.py \
    --experiment a3_delay_only \
    --checkpoint /path/to/best_agent.pt \
    --num_episodes 1 --num_envs 4096 \
    --output results.json

# Parameter sweep (override at eval time)
./isaaclab.sh -p .../iris_ma6/experiments/evaluate.py \
    --experiment a3_delay_only \
    --checkpoint /path/to/best_agent.pt \
    --delay-override 0.2 --output results_200ms.json

# Greedy baseline (no checkpoint)
./isaaclab.sh -p .../iris_ma6/experiments/evaluate.py \
    --experiment baseline_greedy \
    --num_episodes 1 --num_envs 4096

# Trajectory recording
./isaaclab.sh -p .../iris_ma6/experiments/evaluate.py \
    --experiment a3_delay_only \
    --checkpoint /path/to/best_agent.pt \
    --record-trajectory --trajectory-envs 8 \
    --target-trajectory-mode linear \
    --output traj.json

# Action diagnostics: aggregate stats always emitted; per-step trace gated
./isaaclab.sh -p .../iris_ma6/experiments/evaluate.py \
    --experiment a3_delay_only \
    --checkpoint /path/to/best_agent.pt \
    --no-record-action-trace \
    --output diag.json
```

### 8.2 Runtime Overrides

| Flag | Type | Description |
|------|------|-------------|
| `--delay-override` | float | Detection latency (seconds) |
| `--comm-delay-override` | float | Communication latency (seconds) |
| `--target-speed` | float | Target max speed (m/s) |
| `--noise-override` | float | BBox pixel noise std |
| `--accel-override` | float | Target max acceleration (m/s^2) |
| `--detection-dropout-override` | float | Detection failure rate (0-1) |
| `--comm-dropout-override` | float | Communication dropout rate (0-1) |
| `--record-action-trace` / `--no-record-action-trace` | bool | Include per-step action trace arrays in `action_diagnostics` block (default on; aggregate stats always emitted) |

### 8.3 Evaluation Workflow

1. Load experiment from registry
2. Apply `env_overrides` + `agent_overrides`
3. Apply runtime parameter overrides (CLI flags)
4. **Force full difficulty**: Set all curriculum `start_step=0, end_step=0`
5. Create `IrisMA6TestEnv` via gym
6. Load policy checkpoint (auto-detect format)
7. Initialize MetricTracker, TimeseriesTracker (optional), TrajectoryRecorder (optional)
8. Rollout loop:
   a. `obs = policy.act(obs)` (with RNN hidden state)
   b. `obs, reward, terminated, truncated, info = env.step(actions)`
   c. Collect step metrics (see Section 8.4)
   d. On episode end: `tracker.record_episode_end(done_envs)`
9. `results = tracker.compute_final_metrics()`
10. Save JSON

### 8.4 Step Metric Collection

Each step, the evaluator must extract:

```python
def _collect_step_metrics(env, tracker, agent_ids):
    # 1. Triangulation uncertainty (trace of covariance)
    if env._triangulation_result_gt is not None:
        cov = env._triangulation_result_gt.covariance[:, 0, :, :]  # (N, 3, 3)
        trace_sigma = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        tri_valid = env._triangulation_result_gt.is_valid[:, 0]  # (N,)

    # 2. Triangulation RMSE (obs pipeline vs GT target)
    if env._triangulation_result_obs is not None:
        est_pos = env._triangulation_result_obs.position[:, 0, :]  # (N, 3)
        gt_pos = env._target_pos_w  # (N, 3)
        rmse = torch.norm(est_pos - gt_pos, dim=-1)  # (N,)

    # 3. Visibility (fraction of agents with valid bbox)
    num_valid = torch.zeros(env.num_envs, device=env.device)
    for agent_id in agent_ids:
        delayed_states = env._delay_system.get_all_states_for_observations(
            ego_agent_id=agent_id
        )
        bbox = delayed_states[agent_id].data.bboxes_2d[:, 0, :]
        num_valid += (bbox.abs().sum(dim=-1) > 1e-6).float()
    visibility = num_valid / len(agent_ids)

    # 4. Collisions
    gt_positions = torch.stack(
        [env._root_pos_w[a] for a in agent_ids], dim=1
    )  # (N, A, 3)
    collided = env.cbf_manager.check_collisions(gt_positions)  # (N,)

    # 5. CBF penalty (for cbf_violation_rate metric)
    cmd_vel = env.cmd_vel[:, :, 0:3]
    dt = env.cfg.sim.dt * env.cfg.decimation
    cbf_penalty = env.cbf_manager.compute_training_penalty(
        gt_positions, cmd_vel, dt
    )

    tracker.step(
        trace_sigma=trace_sigma,
        triangulated_pos=est_pos,
        gt_target_pos=gt_pos,
        bbox_valid_mask=visibility,
        collision_flags=collided,
        tri_valid=tri_valid,
        cbf_penalty=cbf_penalty,
    )
```

### 8.5 Action Diagnostics Output Schema

`evaluate.py` always emits `results["action_diagnostics"]` summarizing the per-axis policy command signal, recorded for the same `--trajectory-envs` envs sampled by the trajectory recorder. The implementation lives in [`experiments/metrics/action_trace_recorder.py`](../experiments/metrics/action_trace_recorder.py).

**Source signal**: `unwrapped.cmd_vel` of shape `(num_envs, num_agents, 7)`, read after `env.step()` returns.

**Axis layout** (v0 `iris_ma_env6_test.py` — body-frame `vx/vy/vz`):

| Index | Name | Stored in env as | Reported here in |
|------:|------|------------------|------------------|
| 0 | `vx_body` | m/s (post body-frame scaling) | m/s |
| 1 | `vy_body` | m/s | m/s |
| 2 | `vz_body` | m/s | m/s |
| 3 | `yaw_rate` | rad/s | rad/s |
| 4 | `gimbal_yaw_rate` | normalized [-1, 1] | rad/s (× `max_gimbal_rate`) |
| 5 | `gimbal_pitch_rate` | normalized [-1, 1] | rad/s (× `max_gimbal_rate`) |
| 6 | `zoom_rate` | normalized [-1, 1] | 1/s (× per-env `_max_zoom_rate`) |

**Important**: dims 4–6 are stored normalized in `cmd_vel`; the controller scales them downstream. The recorder applies the same scaling at record time so all reported stats are in physical units. Per-env zoom scale is captured at recorder construction (DR-perturbed). v1 (`iris_ma_env6_v1.py`) instead stores `vx/vy/vz` rotated to world frame; if used with v1, axis names should be relabeled (currently hard-coded for v0).

**JSON shape** (under `results["action_diagnostics"]`):

```json
{
  "step_dt_seconds": 0.04,
  "policy_rate_hz": 25.0,
  "axis_names": ["vx_body", "vy_body", "vz_body", "yaw_rate",
                 "gimbal_yaw_rate", "gimbal_pitch_rate", "zoom_rate"],
  "axis_units": ["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s", "1/s"],
  "sample_env_ids": [...],
  "agents": {
    "drone_0": {
      "action_rms":              {axis_name: float, ...},
      "action_delta_rms":        {axis_name: float, ...},
      "dominant_freq_hz":        {axis_name: float, ...},
      "dominant_freq_delta_hz":  {axis_name: float, ...},
      "trace": [[[...7 floats...], ...T_active steps...], ...one list per recorded env...]
    },
    "drone_1": {...}
  }
}
```

`trace` is omitted when `--no-record-action-trace` is passed.

**Stat definitions** (computed per (env, agent), then aggregated by `nanmean` across envs):

- `action_rms[axis] = sqrt(mean(a²))` over the active episode prefix.
- `action_delta_rms[axis] = sqrt(mean((a_t − a_{t−1})²)) × policy_rate_hz` — rate of change in physical units per second.
- `dominant_freq_hz[axis]`: peak bin of `torch.fft.rfft(a)` magnitude, DC zeroed, converted via `k / T × policy_rate_hz`.
- `dominant_freq_delta_hz[axis]`: same on the `Δa` signal.
- Constant signals (stddev < 1e-6 × max-abs) yield `NaN` for the freq fields to avoid reporting argmax over FP round-off noise.
- Episodes with fewer than 16 active steps yield `NaN` for the freq fields (insufficient resolution).

**Diagnostic interpretation**:

- `dominant_freq_delta_hz` near Nyquist (`policy_rate_hz / 2 = 12.5 Hz`) indicates step-by-step bang-bang oscillation on that axis.
- A large gap between `action_delta_rms` on the gimbal axes vs. drone-translation axes points to the gimbal weights (`action_weight[4:6]`, `action_delta_weight[4:6]`) being the right knob to tune, rather than the global `action_sum_penalty_scale` / `action_delta_penalty_scale`.
- Per-env `trace` enables post-hoc plotting (e.g. matplotlib step plot of `a_t` per axis) without re-running the eval.

### 8.6 Checkpoint Format Auto-Detection

Two formats supported:

**`best_agent.pt` (SKRL best)**:
```python
{"drone_0": {"policy": ..., "value": ..., "state_preprocessor": ...}, ...}
```

**`agent_drone_0_final.pt` (training end)**:
```python
{"policy_state_dict": ..., "value_state_dict": ...}
```

Evaluator searches for companion `best_agent.pt` to load preprocessor state when using the final checkpoint format.

---

## 9. Metrics System

### 9.1 Core Metrics (8, inherited from iris_ma5)

| # | Metric | Unit | Description |
|---|--------|------|-------------|
| 1 | `trace_sigma` | m^2 | Mean trace of triangulation covariance |
| 2 | `triangulation_rmse` | m | Localization error vs GT target |
| 3 | `task_success_rate` | 0-1 | Hierarchical: track_maintenance AND accuracy |
| 4 | `visibility_ratio` | 0-1 | Fraction of agents with valid bbox detection |
| 5 | `collision_rate` | count | Inter-agent collisions per episode |
| 6 | `convergence_speed` | steps | Steps to reach trace < 10 |
| 7 | `time_to_first_lock` | steps | Steps to first trace < 50 |
| 8 | `tri_valid_ratio` | 0-1 | Time fraction with >= 2 valid detections |

**Success definition** (hierarchical):
- Level 0 (Track Maintenance): `tri_valid_ratio >= 0.5`
- Level 1 (Accuracy): Among valid steps, `RMSE < 2m` for >= 80%
- Overall success = Level 0 AND Level 1

### 9.2 New Metrics (iris_ma6-specific)

| # | Metric | Unit | Description |
|---|--------|------|-------------|
| 9 | `cbf_violation_rate` | 0-1 | Fraction of steps with CBF penalty > 0 |
| 10 | `min_separation` | m | Minimum pairwise agent distance per episode |
| 11 | `estimation_error_gt` | m | L2 error of GT-anchored triangulation (Level 2) |
| 12 | `estimation_error_e2e` | m | L2 error of E2E triangulation (Level 3) |
| 13 | `controller_tracking_error` | m/s | Commanded vs actual velocity norm |

### 9.3 MetricTracker

Per-step accumulators (GPU tensors, shape `(N,)`):
- Trace: `_trace_sum`, `_trace_sq_sum`, `_trace_valid_count`
- RMSE: `_rmse_sum`, `_rmse_sq_sum`, `_rmse_valid_count`
- Visibility: `_visibility_steps`, `_visibility_sq_sum`
- Collision: `_collision_count`
- Convergence: `_first_lock_step`, `_converged_step`
- Success: `_success_steps`
- Track continuity: `_track_loss_count`, `_current_gap_length`, `_max_track_gap`
- CBF: `_cbf_violation_count`, `_min_separation`
- Estimation: `_est_error_gt_sum`, `_est_error_e2e_sum`

**API**:
```python
tracker = MetricTracker(num_envs, num_agents, device)
tracker.step(trace_sigma, triangulated_pos, gt_target_pos, ...)
tracker.record_episode_end(env_ids)
results = tracker.compute_final_metrics()  # -> Dict with mean/std/median/percentiles
```

### 9.4 TimeseriesTracker

Per-timestep statistics across parallel environments. Same pattern as iris_ma5.

### 9.5 TrajectoryRecorder

Records per-step positions for a sampled subset of environments.

**Data recorded per step**:
- Agent XYZ positions (world frame)
- Target XYZ position (world frame)
- Triangulated estimate XYZ
- Gimbal yaw/pitch per agent (new in ma6)
- Zoom level per agent (new in ma6)

Output: JSON-serializable dict for bird's-eye visualization.

---

## 10. Baselines

### 10.1 GreedyEquiangularPolicy

Scripted non-learning baseline. Maintains agents in equiangular formation around target.

**Action space** (7D):
- `vx, vy, vz`: Proportional velocity toward desired orbit position
- `yaw_rate`: Turn toward target
- `gimbal_yaw_rate`: Track target azimuth
- `gimbal_pitch_rate`: Track target elevation
- `zoom_rate`: 0 (no zoom control)

**Parameters**:
- Orbit radius: 10m (configurable)
- Angular spacing: `2*pi / num_agents`
- Velocity gain: proportional to position error, clamped to [-1, 1]

---

## 11. Frame-Stack Wrapper

Reserved for future MLP experiments. Port `MultiAgentFrameStackWrapper` from iris_ma5 with updated obs_dim (24 or 30 with triangulation). Not used by any currently defined experiment.

---

## 12. Calling Contract

### §12.1 Registry

- All experiments are registered at module import time (top-level in `experiment_registry.py`)
- Registration is idempotent within a process (duplicate name raises error)
- `get_experiment()` and `list_experiments()` are read-only, safe to call at any time

### §12.2 Override Application

- `apply_env_overrides()` mutates the config in-place and returns it
- Must be called BEFORE environment construction
- `__post_init__` on `IrisMA6TestEnvCfg` runs after override application (rebuilds delay_system from params)

### §12.3 Evaluation

- `evaluate.py` is a standalone script (AppLauncher at top)
- MetricTracker.step() must be called exactly once per policy step
- MetricTracker.record_episode_end() must be called for every done environment
- Curriculum is forced to full difficulty before rollout begins
