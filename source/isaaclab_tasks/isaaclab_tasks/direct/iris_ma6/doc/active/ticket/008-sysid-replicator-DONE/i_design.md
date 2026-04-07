# Design Document: Sysid Replicator

## Problem statement

iris_ma6's DroneController dynamics diverge from PX4 SITL across 7/8 comparison metrics (settling time +55%, SS error +50%, hover drift +58%, damping ratio -91%). The RL policy trained on iris_ma6 dynamics will fail at PX4 deployment. We need to tune iris_ma6 controller gains so its step responses match the PX4 SITL recordings from ticket-007, closing the dynamics gap from the controller side.

## Proposed approach

Build a new script `sysid_replicator.py` that reuses ParallelTuner's infrastructure (Isaac Sim env creation, batched DroneController, gain setting, physics stepping) but replaces the scoring objective. Instead of scoring against hand-crafted penalty weights, the replicator scores each candidate gain set by how closely its step-response timeseries matches the PX4 SITL recordings.

The replicator loads PX4 SITL CSVs (copied into `iris_ma6/sysid_output/`) as target data. For each candidate gain set, it runs the same test commands as sysid_node (hover, vel_step_5, vel_step_10, vel_impulse_recovery, yaw_step) through DroneController at 100 Hz physics rate, collecting velocity, attitude error, and rate error timeseries. Both timeseries are resampled to a common 100 Hz timebase via interpolation of the PX4 data. The primary score is weighted timeseries MSE (velocity 60%, attitude 25%, rate 15%) on the X-velocity and yaw channels. A secondary metric-based comparison (settling time, SS error, damping ratio deltas) produces the mismatch improvement table.

The gain search uses 1024 random candidates over the same 12-dimensional parameter space as auto_tune.py. The best-matching gain set is exported to a separate file alongside comparison PDFs overlaying iris_ma6 best vs PX4 SITL.

## Key interfaces and data flow

```
PX4 SITL CSVs (sysid_output/*.csv)
    │
    ▼
┌─────────────────────┐
│  PX4TargetLoader    │ ── Load CSV, reorder quat xyzw→wxyz,
│                     │    resample 125 Hz → 100 Hz,
│                     │    extract per-test target timeseries
└────────┬────────────┘
         │  target: dict[test_name → TargetTimeseries]
         ▼
┌─────────────────────────────────────────────────────┐
│  SysidReplicator                                     │
│                                                      │
│  Uses ParallelTuner's env/controller infrastructure: │
│  - _setup_batch(params) → set per-env gains          │
│  - _step_with_controller(v_cmd) → physics step       │
│  - _reset_to_hover() → clean initial state           │
│                                                      │
│  New test methods:                                   │
│  - _run_test(test_name, duration, cmd_fn)            │
│    → vel_history, att_error_history, rate_error_hist │
│  - _run_yaw_step_test(duration)     [NEW]            │
│  - _run_impulse_recovery_test(dur)  [NEW]            │
│                                                      │
│  Scoring:                                            │
│  - compute_timeseries_mse(iris_ma6, px4_target)      │
│    → weighted MSE (vel 60%, att 25%, rate 15%)       │
│  - compute_metric_comparison(iris_ma6, px4_target)   │
│    → mismatch table for reporting                    │
└────────┬────────────────────────────────────────────-┘
         │
         ▼
┌──────────────────────┐
│  Output               │
│  - PX4_MATCHED_CFG    │ → tuning_results/px4_matched.py
│  - comparison PDFs    │ → sysid_output/analysis/
│  - mismatch table     │ → sysid_output/analysis/replicator_metrics.json
└───────────────────────┘
```

**TargetTimeseries** per test contains:
- `t`: (T,) time array at 100 Hz
- `vel_x`: (T,) forward velocity
- `yaw_rate`: (T,) yaw rate (for yaw test)
- `att_error`: (T, 3) attitude error (computed from quat vs att_sp_quat)
- `rate_error`: (T, 3) rate error (computed from ang_vel vs rate_sp)

**Resampling**: PX4 CSVs at ~125 Hz sim-time are interpolated to 100 Hz grid (matching iris_ma6 physics dt=0.01s) using numpy linear interpolation. The iris_ma6 response is already at 100 Hz, so no resampling needed on that side.

**Test alignment**: Each PX4 CSV contains the full test recording. The replicator trims to the command-active window (e.g., vel_step starts at t=0 when command is applied). iris_ma6 tests start from hover and apply the command at t=0. Both are aligned at command onset.

**New tests to add** (not in auto_tune.py):
1. Yaw step: 0.5 rad/s for 2s, then zero, total 10s. Score on yaw rate channel.
2. Impulse recovery: 5.0 m/s for 1s, then zero, total 10s. Score on vel_x recovery.

**Cascade setpoint matching**: The PX4 CSVs include att_sp (quaternion) and rate_sp (body rates). For each iris_ma6 physics step, the replicator also records the DroneController's internal q_des (attitude setpoint) and rate_setpoint. The attitude error and rate error timeseries are computed identically on both sides: att_error = actual vs setpoint, rate_error = actual vs setpoint. This enables intermediate cascade signal matching per Q answer #5.

## What this does NOT include

- No modification to auto_tune.py (fork, don't modify)
- No modification to PX4 parameters, sysid_node, or sysid_analyze.py
- No changes to controller architecture, motor dynamics, or aerodynamic model
- No gimbal matching (out of scope — separate ticket)
- No policy retraining or reward function changes
- No multi-stage refinement (random search only for initial version)

## Open risks

1. **Sim-time alignment ambiguity**: PX4 SITL CSV timestamps may not start exactly at command onset. Need to detect the step onset in the PX4 data (e.g., first sample where vel_x command > 0) rather than assuming t=0 is command start.
2. **Cascade setpoint availability**: auto_tune.py currently extracts att_error and rate_error inline but does not store rate_setpoint as a separate tensor. The replicator will need to extract this from the controller internals (attitude_controller output → rate_setpoint).
3. **PX4 at 10 m/s slow rise**: PX4 SITL vel_10 settling_time is 9.99s with 0.68 m/s SS error (~7% at 10 m/s), which represents a slow but settled response. The replicator must reproduce this slow rise characteristic. If iris_ma6 cannot match this transient shape via gains alone (it may be a plant model difference, not just gains), the 10 m/s match quality may remain poor — indicating the dynamics gap is not fully closable by gain tuning.
4. **1024 candidates may be insufficient**: 12-dimensional space with 1024 samples is sparse. If best-match quality is poor, upgrade to multi-stage refinement per Q answer #12.
