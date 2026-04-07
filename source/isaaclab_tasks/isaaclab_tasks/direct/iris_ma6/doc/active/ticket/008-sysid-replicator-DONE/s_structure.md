# Stage S — Structure Outline (008-sysid-replicator)

## New files

### `controller/tuning/sysid_replicator.py`

Isaac Sim script (AppLauncher + argparse at top, like auto_tune.py).

```python
# --- AppLauncher boilerplate (before all other imports) ---
parser args: --sysid-dir, --output-dir, --num-trials (default 1024),
             --aero-level, --top-k

# --- Post-AppLauncher imports ---

@dataclass
class TargetTimeseries:
    """Resampled PX4 SITL target for one test."""
    t: np.ndarray           # (T,) time at 100 Hz [s]
    vel_x: np.ndarray       # (T,) forward velocity [m/s]
    yaw_rate: np.ndarray    # (T,) yaw rate [rad/s]
    att_error: np.ndarray   # (T, 3) attitude error [rad] (actual vs att_sp)
    rate_error: np.ndarray  # (T, 3) rate error [rad/s] (actual vs rate_sp)

@dataclass
class ReplicatorScoreWeights:
    """MSE weights per signal group."""
    velocity: float = 0.60
    attitude: float = 0.25
    rate: float = 0.15

class PX4TargetLoader:
    """Loads PX4 SITL CSVs and resamples to 100 Hz."""

    def __init__(self, sysid_dir: str | Path):
        # Loads all 5 drone test CSVs from sysid_dir

    def _load_csv(self, name: str) -> pd.DataFrame:
        # Read CSV, dedup timestamps, add relative time column t
        # Reorder quaternion columns xyzw → wxyz

    def _resample_to_100hz(self, df: pd.DataFrame) -> pd.DataFrame:
        # Linear interpolation from ~125 Hz to 100 Hz grid (dt=0.01s)

    def _compute_att_error(self, df: pd.DataFrame) -> np.ndarray:
        # att_error = quaternion_error(att_sp_quat, actual_quat)
        # Uses same formula as AttitudeController.compute_attitude_error

    def _compute_rate_error(self, df: pd.DataFrame) -> np.ndarray:
        # rate_error = rate_sp - ang_vel (both in body frame)

    def _detect_command_onset(self, df: pd.DataFrame, test_name: str) -> float:
        # Find t where step command begins (vel_x crosses threshold, etc.)

    def get_target(self, test_name: str) -> TargetTimeseries:
        # Returns resampled target for named test, trimmed from command onset

    @property
    def test_names(self) -> list[str]:
        # Returns available test names

    @property
    def test_durations(self) -> dict[str, float]:
        # Returns {test_name: duration_seconds} from loaded data


class SysidReplicator:
    """Gain sweep scored against PX4 SITL timeseries."""

    def __init__(self, num_envs: int, device: str, aero_level: int,
                 targets: dict[str, TargetTimeseries],
                 weights: ReplicatorScoreWeights):
        # Creates ParallelTuner (composition, not subclass)
        # Stores targets and weights

    def _run_velocity_test(self, duration: float, target_speed: float) -> dict:
        # Like ParallelTuner._run_velocity_test but with extended duration
        # Also records rate_setpoint (from controller._attitude output)
        # Returns: vel_history (T,N), att_error_history (T,N,3),
        #          rate_error_history (T,N,3)

    def _run_yaw_step_test(self, duration: float) -> dict:
        # NEW: yaw_rate_cmd=0.5 rad/s for 2s, then zero, total=duration
        # Records: yaw_rate_history (T,N), att_error_history (T,N,3),
        #          rate_error_history (T,N,3)

    def _run_impulse_recovery_test(self, duration: float) -> dict:
        # NEW: vel_cmd=5.0 m/s for 1s, then zero, total=duration
        # Records same signals as velocity test

    def _run_hover_test(self, duration: float) -> dict:
        # Like ParallelTuner hover but with extended duration
        # Records: drift_history (T,N)

    def evaluate_batch(self, params_list: list) -> tuple[list[float], list[dict]]:
        # Orchestrates all 5 tests at PX4-matched durations
        # Returns: (scores_per_env, per_env_histories)

    def _compute_timeseries_mse(self, iris_data: dict, target: TargetTimeseries,
                                 test_name: str) -> float:
        # Primary score: weighted MSE on vel_x/yaw (60%), att_error (25%), rate_error (15%)
        # Trims to common length, computes per-signal MSE, applies weights

    def _compute_metric_comparison(self, iris_data: dict, target: TargetTimeseries,
                                    test_name: str) -> dict:
        # Secondary: extracts settling_time, ss_error, damping_ratio from both
        # Returns: {metric_name: {iris_ma6: val, px4_sitl: val, diff_pct: val}}

    def close(self):
        # Delegates to ParallelTuner.close()


def generate_comparison_plots(best_histories: dict, targets: dict[str, TargetTimeseries],
                               output_dir: Path, dt: float):
    # For each test: overlay iris_ma6 best vs PX4 SITL on same axes
    # Rows: velocity/yaw_rate, attitude error, rate error
    # Saves PDF per test to output_dir

def export_px4_matched_config(best_params: ParameterSet, score: float,
                               comparison: dict, output_path: Path):
    # Writes PX4_MATCHED_CONTROLLER_CFG to Python file
    # Same format as auto_tune.py's best_config export

def main():
    # 1. Load PX4 targets via PX4TargetLoader(args_cli.sysid_dir)
    # 2. Generate 1024 random params via generate_random_params (imported from auto_tune)
    # 3. Create SysidReplicator
    # 4. Evaluate in batches (num_envs == num_trials)
    # 5. Find best, generate comparison plots
    # 6. Export PX4_MATCHED_CONTROLLER_CFG
    # 7. Save mismatch table to replicator_metrics.json
```

### `controller/tuning/tuning_results/px4_matched.py`

```python
# Auto-generated by sysid_replicator.py
# PX4 SITL-matched DroneController configuration
# Score: <mse_score>
# Generated: <timestamp>

from isaaclab_tasks.direct.iris_ma6.controller import DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.velocity_controller_cfg import VelocityControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.attitude_controller_cfg import AttitudeControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.rate_controller_cfg import RateControllerCfg

PX4_MATCHED_CONTROLLER_CFG = DroneControllerCfg(
    velocity=VelocityControllerCfg(
        Kp_vel=(...),
        Ki_vel=(...),
    ),
    attitude=AttitudeControllerCfg(
        Kp_att=(...),
    ),
    rate=RateControllerCfg(
        Kp_rate=(...),
        Ki_rate=(...),
        Kd_rate=(...),
    ),
)
```

### `sysid_output/` (directory)

Copied PX4 SITL CSVs (read-only input):
- `hover_drift.csv`
- `vel_step_5.csv`
- `vel_step_10.csv`
- `vel_impulse_recovery.csv`
- `yaw_step.csv`

### `sysid_output/analysis/` (directory, generated output)

- `replicator_metrics.json` — mismatch table + per-test MSE scores
- `replicator_hover_drift.pdf` — overlay plot
- `replicator_vel_step_5.pdf` — overlay plot
- `replicator_vel_step_10.pdf` — overlay plot
- `replicator_vel_impulse_recovery.pdf` — overlay plot
- `replicator_yaw_step.pdf` — overlay plot

## Modified files

### `controller/tuning/tuning_results/__init__.py`
- [add] `from .px4_matched import PX4_MATCHED_CONTROLLER_CFG` — re-export for convenience
- [existing] `TUNED_CONTROLLER_CFG` — no change

## Files NOT modified

- `controller/tuning/auto_tune.py` — no changes (imports only: `ParameterSet`, `generate_random_params`, `ParallelTuner`)
- `controller/drone_controller.py` — no changes
- `sysid_node.py`, `sysid_analyze.py` — no changes
