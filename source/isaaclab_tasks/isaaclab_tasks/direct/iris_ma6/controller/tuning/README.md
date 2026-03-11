# DroneController Auto-Tuning

This module provides automatic parameter tuning for the iris_ma6 DroneController with **parallel parameter testing** - each environment tests a different parameter set simultaneously, making tuning N times faster.

## Quick Start

```bash
# Run grid search (default, ~540 combinations)
# With 64 parallel envs, tests 64 parameter sets simultaneously
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless

# Run random search with 100 trials (tests 64 in parallel by default)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless --search-mode random --num-trials 100

# Use more parallel environments for faster evaluation (128x speedup)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tuning/auto_tune.py --headless --num-envs 128
```

## Parallel Testing

The tuner uses **per-environment parameter testing**:
- Each of the `num_envs` parallel environments tests a DIFFERENT parameter set
- With `--num-envs 64`, 64 parameter sets are evaluated simultaneously
- Grid search with 540 combinations completes in ~9 batches (instead of 540 sequential trials)
- **Speedup**: ~Nx faster where N = num_envs

## Parameters Tuned

### Grid Search Ranges
| Parameter | Description | Grid Values |
|-----------|-------------|-------------|
| `Kp_vel_xy` | Proportional gain (XY) | [2.0, 3.0, 4.0, 5.0] |
| `Kp_vel_z` | Proportional gain (Z) | [1.5, 2.0, 3.0] |
| `Ki_vel_xy` | Integral gain (XY) | [0.3, 0.5, 0.8] |
| `Kp_att_rp` | Proportional gain (roll/pitch) | [6.0, 8.0, 10.0, 12.0] |
| `Kd_att_rp` | Derivative gain (roll/pitch) | [1.5, 2.5, 3.5] |

### Random Search Ranges
| Parameter | Description | Range |
|-----------|-------------|-------|
| `Kp_vel_xy` | Proportional gain (XY) | [1.5, 6.0] |
| `Kp_vel_z` | Proportional gain (Z) | [1.0, 4.0] |
| `Ki_vel_xy` | Integral gain (XY) | [0.1, 1.0] |
| `Ki_vel_z` | Integral gain (Z) | [0.1, 0.6] |
| `Kp_att_rp` | Proportional gain (roll/pitch) | [4.0, 15.0] |
| `Kp_att_y` | Proportional gain (yaw) | [2.0, 8.0] |
| `Kd_att_rp` | Derivative gain (roll/pitch) | [1.0, 5.0] |
| `Kd_att_y` | Derivative gain (yaw) | [0.5, 2.5] |

## Metrics Evaluated

### Hover Stability (3 seconds)
- **Drift Mean**: Average position error from initial position [m]
- **Drift Max**: Maximum position error [m]

### Velocity Tracking (2 seconds)
- **Settling Time**: Time to reach 95% of target velocity (3 m/s) [s]
- **Overshoot**: Peak velocity above target [%]
- **Steady-State Error**: Final velocity error [m/s]

### Combined Score
Lower is better:
```
score = 1.0 * drift_mean + 0.5 * drift_max + 2.0 * settling_time + 0.1 * overshoot + 5.0 * ss_error
```

## Output Files

Results are saved to `tuning_results/` (configurable via `--output-dir`):

1. **`tuning_results_YYYYMMDD_HHMMSS.json`**: Full results
   - Configuration used
   - Best parameters found
   - All trial results

2. **`best_config_YYYYMMDD_HHMMSS.py`**: Python config file
   - Ready to import and use
   ```python
   from tuning_results.best_config_20260311_123456 import TUNED_CONTROLLER_CFG
   ```

## Example Results

```
================================================================================
TUNING RESULTS
================================================================================
Total time: 245.3s (0.45s per trial)
Stable configurations: 498/540

Best configuration (score: 1.1684):
  Velocity Controller:
    Kp_vel: (4.60, 4.60, 1.96)
    Ki_vel: (0.19, 0.19, 0.43)
  Attitude Controller:
    Kp_att: (14.45, 14.45, 4.01)
    Kd_att: (1.13, 1.13, 0.88)

  Performance Metrics:
    Hover drift (mean): 0.0821 m
    Hover drift (max):  0.2134 m
    Velocity settling:  0.320 s
    Velocity overshoot: 8.2%
    Steady-state error: 0.0423 m/s
```

## Using Tuned Parameters

```python
from isaaclab_tasks.direct.iris_ma6.controller import DroneController, DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg

# Use tuned parameters
cfg = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(4.60, 4.60, 1.96),
        Ki_vel=(0.19, 0.19, 0.43),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(14.45, 14.45, 4.01),
        Kd_att=(1.13, 1.13, 0.88),
    ),
)

controller = DroneController(
    cfg=cfg,
    mass=1.5,
    gravity=9.81,
    num_envs=num_envs,
    device=device,
)
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--search-mode` | `grid` | Search mode: `grid` or `random` |
| `--num-envs` | `64` | Number of parallel environments (= batch size for parallel testing) |
| `--num-trials` | `100` | Number of trials (random mode only) |
| `--output-dir` | `tuning_results` | Output directory |
| `--headless` | `False` | Run without visualization |

## Performance

### Parallel Speedup

With parallel parameter testing:
- **Sequential mode (old)**: Each trial takes ~8 seconds → 540 trials = ~72 minutes
- **Parallel mode (new)**: Each batch tests N params simultaneously → 540/64 ≈ 9 batches = ~1.5 minutes

| num_envs | Grid Search Time (540 params) | Speedup |
|----------|-------------------------------|---------|
| 1 | ~72 minutes | 1x |
| 32 | ~2.5 minutes | 29x |
| 64 | ~1.5 minutes | 48x |
| 128 | ~0.8 minutes | 90x |

### GPU Memory Usage

Higher `num_envs` requires more GPU memory. Recommended settings:
- **8GB VRAM**: `--num-envs 32`
- **16GB VRAM**: `--num-envs 64`
- **24GB+ VRAM**: `--num-envs 128`

## Notes

### Console Output
Due to Isaac Sim's logging redirection in headless mode, console output may not appear in real-time. Results are always saved to the output directory regardless of console output.

### Isaac Sim Physics
The tuner uses the actual Isaac Sim environment (`Isaac-Iris-MA6-Direct-Test-v0`) for physics simulation. This provides realistic evaluation including:
- Full rigid body dynamics with gravity and inertia
- Rotor-level motor model with first-order lag
- Aerodynamic drag effects
- Gimbal dynamics

Each trial runs approximately 5-10 seconds (3s hover test + 2s velocity test + overhead).

### Stability Criteria
A configuration is considered "stable" if:
- Hover drift max < 2.0 m
- No NaN/Inf values in physics

## Extending the Tuner

To add new parameters or metrics:

1. Update `ParameterSet` dataclass in `auto_tune.py`
2. Modify `to_controller_cfg()` to apply new parameters
3. Add evaluation methods (e.g., `evaluate_attitude_tracking()`)
4. Update the scoring function in `evaluate()`
