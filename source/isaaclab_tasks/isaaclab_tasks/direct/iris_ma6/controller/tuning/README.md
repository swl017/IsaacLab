# DroneController Auto-Tuning

Automatic parameter tuning for the iris_ma6 cascaded PID controller with **parallel parameter testing**, **oscillation detection metrics**, and **step response visualization**.

Each environment tests a different parameter set simultaneously, achieving ~Nx speedup where N is the number of parallel environments. The tuner uses the real `DroneController` (not a simplified approximation), so results directly predict training behavior.

## Quick Start

```bash
# Random search with 100 trials (100 parallel envs, one per trial)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless

# Grid search (~108 combinations, one env per combination)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless --search-mode grid

# Random search with 200 trials + step response plots for top 5
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless --num-trials 200 --top-k 5
```

All trials run in a single parallel batch — one Isaac Sim environment per parameter set.

## Architecture

The tuner evaluates the real 4-loop PX4-style cascade:

```
Velocity (PI) → Attitude (P) → Rate (PID) → Motor Dynamics → Forces/Torques
```

Per-env gains are set via `set_gains()` on each sub-controller, and `DroneController.step_policy()` is called each step. This includes motor dynamics, anti-windup, aerodynamics, and all other real controller features.

## Parameters Tuned

| Parameter | Description | Grid Values | Random Range |
|-----------|-------------|-------------|-------------|
| `Kp_vel_xy` | Velocity P-gain (XY) | [2.0, 3.0, 4.0] | [1.5, 5.0] |
| `Ki_vel_xy` | Velocity I-gain (XY) | [0.3, 0.5] | [0.2, 0.8] |
| `Kp_att_rp` | Attitude P-gain (roll/pitch) | [5.0, 6.5, 8.0] | [4.0, 10.0] |
| `Kp_rate_rp` | Rate P-gain (roll/pitch) | [0.10, 0.15, 0.20] | [0.08, 0.25] |
| `Ki_rate_rp` | Rate I-gain (roll/pitch) | [0.15, 0.25] | [0.1, 0.35] |
| `Kd_rate_rp` | Rate D-gain (roll/pitch) | 0.003 (fixed) | [0.001, 0.008] |

Z-axis and yaw gains are derived from XY/roll-pitch gains using PX4 ratios.

## Test Suite

Each parameter set is evaluated on three tests:

### 1. Hover Stability (3 seconds)
- **Drift Mean**: Average position error from initial position [m]
- **Drift Max**: Maximum position error [m]

### 2. Velocity Step Response (5 seconds, at 5 m/s and 10 m/s)
- **Settling Time**: Time to reach 95% of target velocity [s]
- **Overshoot**: Peak velocity above target [%]
- **Steady-State Error**: Final velocity error [m/s]
- **Oscillation Metrics** (see below)

### 3. Attitude Step Response (3 seconds, from 45°/45°/30° perturbation)
- **Recovery Time**: Time to reach <5° error [s]
- **Max Error**: Peak attitude error [deg]
- **Final Error**: Error at end of test [deg]
- **Oscillation Metrics** (see below)

## Oscillation Metrics

Computed on velocity error, attitude error, and rate error signals after 95% settling:

| Metric | Description | Method |
|--------|-------------|--------|
| **Damping Ratio** (ζ) | 0=undamped, 1=critically damped | Logarithmic decrement of first two peaks |
| **Zero Crossings** | Sign changes in error after settling | Direct count |
| **SS Amplitude** | Peak-to-peak in steady-state window | max - min |
| **Frequency** | Dominant oscillation frequency [Hz] | Mean half-period from zero crossings |

Six oscillation signals are measured:
- `vel_osc_5`, `vel_osc_10`: Velocity error at 5/10 m/s
- `att_osc`: Attitude error during recovery
- `rate_osc_vel5`, `rate_osc_vel10`, `rate_osc_att`: Rate error during each test

## Scoring

Combined score (lower is better), with configurable weights via `TuningScoreWeights`:

```
score = base_metrics_score + oscillation_penalty

base = w.hover_drift_mean * drift_mean
     + w.hover_drift_max * drift_max
     + w.vel_settling_time * settling_time
     + w.vel_overshoot * overshoot
     + w.vel_ss_error * ss_error
     + w.att_recovery_time * recovery_time
     + w.att_max_error * max_error
     + w.att_final_error * final_error

oscillation = mean over 6 signals of:
    w.oscillation_damping * (1 - damping_ratio)
  + w.oscillation_zero_crossings * zero_crossings
  + w.oscillation_ss_amplitude * ss_amplitude
```

Default weights penalize low damping ratio most heavily (5.0).

## Step Response Plots

The `--top-k` flag generates multi-panel PDF plots for the best K results:

```
tuning_results/step_responses/
├── trial_0042.pdf   # Best result
├── trial_0117.pdf   # 2nd best
└── ...
```

Each PDF contains 6 panels:
- **Row 1**: Velocity step response at 5 m/s and 10 m/s (actual vs target)
- **Row 2**: Attitude error during velocity step and attitude recovery
- **Row 3**: Rate error during velocity step and attitude recovery

Panel titles show oscillation metrics (ζ, zero crossings, amplitude, frequency).

## Output Files

Results are saved to `tuning_results/` (configurable via `--output-dir`):

| File | Description |
|------|-------------|
| `tuning_results_*.json` | Full results with all metrics including oscillation |
| `best_config_*.py` | Python config file ready to import |
| `step_responses/trial_*.pdf` | Step response plots for top-K results |

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--search-mode` | `random` | Search mode: `grid` or `random` |
| `--num-trials` | `100` | Number of trials (random mode). Each trial gets its own parallel env. |
| `--output-dir` | `tuning_results` | Output directory |
| `--top-k` | `10` | Number of top results to plot |
| `--headless` | `True` | Run without visualization |

Note: `num_envs` is set automatically to match the number of parameter sets (one env per trial).

## Using Tuned Parameters

```python
from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg

cfg = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(4.60, 4.60, 1.96),
        Ki_vel=(0.19, 0.19, 0.43),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(14.45, 14.45, 4.01),
    ),
    rate=RateControllerCfg(
        Kp_rate=(0.150, 0.150, 0.200),
        Ki_rate=(0.200, 0.200, 0.100),
        Kd_rate=(0.00300, 0.00300, 0.00000),
    ),
)
```

## Performance

All trials run in a single GPU-parallel batch. Wall-clock time is independent of trial count (limited by GPU memory, not sequential execution).

Each trial runs ~16s of sim time (3s hover + 5s vel@5 + 5s vel@10 + 3s attitude).

| Trials | Approx. Wall Time | GPU Memory |
|--------|-------------------|------------|
| 50 | ~20s | ~4 GB |
| 100 | ~20s | ~8 GB |
| 200 | ~20s | ~16 GB |
| 500 | ~25s | ~40 GB |
