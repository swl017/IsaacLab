# PX4 EKF2 State-Stream Lag — Spec

**Status**: Draft (ticket 041)
**Owner**: ticket 041 — measurement-only deliverable
**Consumers**: future "apply EKF2 state delay to iris_ma6 obs / controller" ticket; sim-to-real residual-gap analysis after ticket 040 (Pegasus physics parity).

This spec defines *what* the EKF2 state-lag characterization produces and the contract its consumers can rely on. It does NOT modify the env, controller, or delay system. See [active/ticket/041-px4-ekf-state-lag-measurement/ticket.md](active/ticket/041-px4-ekf-state-lag-measurement/ticket.md) for the motivation, scope boundary, and risk discussion.

## 1. Goal

Measure the per-channel time lag between **Pegasus ground-truth state** (the simulator's PhysX state, published with sim-time stamps by `ROS2Backend`) and **PX4 EKF2 estimates** (delivered through MAVROS) for a single Iris vehicle in Pegasus SITL. Produce a JSON describing the lag of each channel and a single-page-per-channel PDF report.

The output is **the contract** for a downstream ticket that will inject per-channel delay into the iris_ma6 training-time observation path. Lock the JSON schema (§4) first so downstream work can be written against it before this ticket finishes data collection.

## 2. Channels

Each channel has its own delay fit (no shared τ across channels — they live in different fusion paths).

| Channel | Truth source | EKF estimate (MAVROS) | Frame | Expected dominant latency |
|---|---|---|---|---|
| `position_xyz` | `<ns>/state/pose.pose.position` | `<ns>/mavros/local_position/pose.pose.position` | ENU (map) | `EKF2_GPS_DELAY = 110 ms` + fusion settle |
| `velocity_world_xyz` | `<ns>/state/twist_inertial.twist.linear` | `<ns>/mavros/local_position/velocity_local.twist.linear` | ENU (map) | GPS-driven, ~100 ms |
| `velocity_body_xyz` | `<ns>/state/twist.twist.linear` | `<ns>/mavros/local_position/velocity_body.twist.linear` | body FLU | GPS-driven, ~100 ms |
| `attitude_roll`, `attitude_pitch` | `quat→rpy(<ns>/state/pose.pose.orientation)` | `quat→rpy(<ns>/mavros/imu/data.orientation)` | ENU FLU | IMU-driven, ~10–20 ms |
| `attitude_yaw` | `quat→yaw(<ns>/state/pose.pose.orientation)` | `quat→yaw(<ns>/mavros/imu/data.orientation)` | ENU FLU | Mag-fused; check `EKF2_MAG_DELAY` |
| `body_rate_xyz` | `<ns>/state/twist.twist.angular` | `<ns>/mavros/imu/data.angular_velocity` | body FLU | IMU-driven, ~10 ms |
| `linear_acceleration_xyz` | `<ns>/state/accel.accel.linear` (inertial) — gravity-add to compare against IMU spec | `<ns>/mavros/imu/data.linear_acceleration` (body FLU, gravity-included spec) | rotate truth to body, add g, then compare | IMU-driven + gravity removal, ~10 ms |

**Quaternion handling.** Pegasus `state/pose.orientation` is the ROS quaternion convention `(qx, qy, qz, qw)` (scalar-last). MAVROS publishes the same `(qx, qy, qz, qw)` convention. Convert both to roll/pitch/yaw (XYZ Tait-Bryan, ZYX intrinsic equivalent — see `frame_conventions.md`) and fit per-Euler-axis. **Do not** fit a single attitude scalar — roll/pitch (high-bandwidth, IMU-fused) and yaw (low-bandwidth, mag-fused) have different physics.

**Body-frame vs world-frame angular velocity.** Pegasus `state/twist.twist.angular` is body-frame (matches MAVROS `imu/data.angular_velocity`). If a future Pegasus version regresses to world-frame, the recorder must rotate by the same-stamp orientation before fitting; validation gate §6 catches this (the fit would show ~0 lag in trivial-yaw maneuvers but ~360°/s in body-yaw maneuvers, an obvious tell).

**Linear-acceleration frame.** Pegasus `state/accel.accel.linear` is inertial-frame, gravity-removed (i.e., pure dv/dt). MAVROS `imu/data.linear_acceleration` is body-frame and gravity-included (the IMU specifies "spec force"). The analyzer rotates truth to body via the same-stamp orientation and adds `+g_body` before comparing magnitudes. Yields a delay number on the IMU integration channel, not on the inertial-frame acceleration — that is the channel the deployment-side controller actually consumes.

## 3. Time-base reconciliation

**The single most load-bearing correctness detail.** Get this wrong and the position-channel sanity gate (§6) will catch you.

1. All ROS nodes — recorder, Pegasus's `ROS2Backend`, MAVROS — must run with `use_sim_time=True`. Verified once at session start by `ros2 param get <node> use_sim_time` for each node in the stack.
2. PX4 SITL must run with **lockstep enabled** (`px4_mavlink_backend.enable_lockstep = True`, default in Pegasus). Lockstep ties PX4's internal clock to PhysX's stepping, so the MAVLink stream `HIL_SENSOR` and `HIGHRES_IMU` timestamps reflect sim-time, and MAVROS's `Header.stamp` follows.
3. **Use `Header.stamp` only, never receive time** (`rclpy.Clock.now()` at callback entry). MAVROS's `Header.stamp` carries PX4's time-of-validity — the t at which the EKF computed the estimate. Receive time adds ROS scheduling jitter on top.
4. Pegasus `state/*` stamps come from `node.get_clock().now()` inside `ROS2Backend.update_state` (sim-time-aware node). Same `Header.stamp` rule applies — log it as-is.
5. **Hover-drift recording (§5.1) is the null-test for time-base reconciliation.** With no excitation, all channel-fits should show ≤ 5 ms drift between streams (the prediction-window floor). If hover_drift shows 50 ms lag on the body-rate channel, the time-base is wrong; fix it before producing any other recording.

If lockstep is disabled (real-time mode), the recording is still valid but tagged `lockstep=false` in the JSON. Real-time-mode lag includes Linux scheduling jitter and is not bit-comparable to lockstep recordings; downstream consumers should prefer lockstep numbers.

## 4. Fit methods

Each channel runs **two** independent fits to bound the uncertainty:

### 4.1 `xcorr` — normalized cross-correlation

Resample truth(t) and EKF(t) onto a common 200 Hz grid via linear interpolation over the overlapping time range. Compute normalized cross-correlation; the lag that maximizes correlation is τ. Search range: `[-50 ms, +500 ms]` (negative lags allowed to detect time-base inversion; should never be the answer in practice).

`scipy.signal.correlate` + `scipy.signal.correlation_lags` is the primitive. Normalize each signal to zero-mean, unit-variance before correlating.

### 4.2 `step_50` — step-response 50%-crossing

For step-input maneuvers (vel_step_x, vel_step_z, yaw_step): identify each truth-step edge from the command profile timestamps (known); measure the time at which truth reaches 50% of the step magnitude (linear interp between adjacent samples); measure the same for the EKF estimate; delay = `t_50_ekf − t_50_truth`. Average across all step edges in the recording.

Robust when EKF settles to a different steady-state value than truth (e.g., calibration offset on velocity). Less sensitive to noise than xcorr peak-picking but only works on step-shaped inputs.

### 4.3 `chirp_phase_delay` — group-delay from cross-spectrum

For the body-rate channel under the chirp maneuver (§5.5): compute the cross-power spectrum of truth vs EKF; group delay at each frequency f is `phase(f) / (2π·f)`. Average group delay over the linear regime (0.5–2 Hz). Most accurate at high bandwidth.

If the group-delay-vs-frequency curve is flat ± 20% over [0.5, 2] Hz, constant delay is sufficient. If it varies > 50%, **escalate to a follow-up ticket** to fit a first-order or second-order channel model — do not silently widen the constant delay estimate to compensate.

### 4.4 Method selection per channel

| Channel | Primary | Secondary | Tertiary |
|---|---|---|---|
| `position_xyz` | xcorr (vel_step_x, vel_step_z) | step_50 (vel_step) | — |
| `velocity_world_xyz`, `velocity_body_xyz` | xcorr (vel_step_x, vel_step_z) | step_50 (vel_step) | — |
| `attitude_roll/pitch` | xcorr (vel_step_x, vel_step_z) | step_50 (vel_step) | chirp_phase_delay (chirp) |
| `attitude_yaw` | xcorr (yaw_step) | step_50 (yaw_step) | — |
| `body_rate_xyz` | chirp_phase_delay (chirp) | xcorr (vel_step, yaw_step) | — |
| `linear_acceleration_xyz` | xcorr (vel_impulse_recovery) | xcorr (chirp) | — |

Report all available methods per channel; flag disagreement > 30% as a result-of-interest in `notes`.

## 5. Excitation maneuvers

Each maneuver is **30 s**, with **3 trials** for stochastic averaging. Command sources: identical to the existing [controller/sysid_output/](../controller/sysid_output/) CSVs where applicable; new chirp/impulse profiles defined here.

### 5.1 `hover_drift` (passive baseline)
- Hold takeoff position 30 s. No velocity command (cmd_vel = 0).
- Purpose: bound noise floor; null-test for time-base reconciliation (§3.5).
- No delay fit — only `residual_rmse_after_compensation` at τ=0.

### 5.2 `vel_step_x` (primary fit for position / velocity)
- `cmd_vel.linear.x = +1.0, 0.0, −1.0, 0.0` in 5 s blocks → 4 transitions in 20 s, 10 s settle at end.
- Body-frame command (offboard_py uses `cmd_vel.linear.x` as body-forward).

### 5.3 `vel_step_z` (z-asymmetry validation per ticket 039)
- `cmd_vel.linear.z = +0.5, 0.0, −0.5, 0.0` in 5 s blocks (z-up is climb).
- Validates that PX4 MPC_Z_VEL_MAX_UP/DN asymmetry does not produce asymmetric EKF lag.

### 5.4 `yaw_step` (attitude / yaw fit)
- `cmd_vel.angular.z = +1.0 rad/s` for 0.5 s, hold 5 s, `−1.0 rad/s` for 0.5 s, hold 5 s; repeat. Yields ±0.5 rad heading excursions.

### 5.5 `chirp` (body-rate group-delay fit)
- `cmd_vel.linear.x = 0.5·sin(2π·f(t)·t)` with `f(t)` swept linearly from 0.1 → 3.0 Hz over 30 s.
- Excites body-rate via the cascaded controller; chirp_phase_delay reads the IMU group delay directly.

### 5.6 `vel_impulse_recovery` (acceleration channel)
- `cmd_vel.linear.x = ±0.5 m/s` square wave at 2 Hz for 5 s, then 25 s hover for recovery.
- Excites the d/dt(v) → acceleration channel sharply.

## 6. Validation gates (must pass before accepting a fit)

A fit that fails a gate is **dropped**, not silently widened. The channel's JSON entry gets `{"status": "skipped", "reason": "..."}` and the consuming ticket is notified.

| Gate | Bound | Rationale |
|---|---|---|
| Trials per maneuver | ≥ 3 | std-across-trials is meaningful |
| xcorr peak | ≥ 0.8 | below means truth and EKF are decorrelated → maneuver did not excite |
| `position_xyz` delay | 110 ± 30 ms | `EKF2_GPS_DELAY = 110 ms` default ± fusion settle |
| `body_rate_xyz` delay | 5 ≤ τ ≤ 30 ms | `EKF2_PREDICT_US = 10 ms` + IMU integration |
| `hover_drift` residual on all channels | ≤ 0.5σ of channel-typical magnitude | sanity that streams agree at rest |
| MAVROS `Header.stamp` monotonicity | strictly increasing | if not, MAVROS is publishing `now()` not time-of-validity — a known ROS-bridge bug |
| Chirp group-delay flatness over [0.5, 2] Hz | within ±20% of mean | otherwise escalate to higher-order model |

Position-channel gate failure is treated as **time-base misconfiguration**, not as a real result. Fix `use_sim_time` / lockstep before re-running.

## 7. JSON schema (locked contract for consumers)

```json
{
  "dataset_id": "YYYY-MM-DD_pegasus_iris_ekf_lag_vN",
  "schema_version": 1,
  "setup": {
    "px4_version": "v1.15.x (git sha)",
    "pegasus_version": "v4.2.0 (git sha)",
    "isaaclab_commit": "<sha>",
    "lockstep": true,
    "use_sim_time": true,
    "physics_dt_hz": 250,
    "control_rate_hz": 100,
    "num_trials_per_maneuver": 3,
    "maneuver_duration_s": 30,
    "namespace": "px4_1"
  },
  "px4_params_snapshot": {
    "EKF2_PREDICT_US": 10000,
    "EKF2_GPS_DELAY":  110.0,
    "EKF2_BARO_DELAY": 0.0,
    "EKF2_MAG_DELAY":  0.0,
    "EKF2_EV_DELAY":   0.0,
    "EKF2_IMU_POS_X":  0.0, "EKF2_IMU_POS_Y": 0.0, "EKF2_IMU_POS_Z": 0.0,
    "IMU_INTEG_RATE":  200,
    "MAV_SYS_HB_INTERVAL_MS": 100
  },
  "channels": {
    "position_xyz": {
      "status": "fit",
      "delay_mean_s": 0.115, "delay_std_s": 0.012,
      "delay_p50_s": 0.114, "delay_p95_s": 0.135, "delay_p99_s": 0.148,
      "residual_rmse_after_compensation": 0.03,
      "fit_method": "xcorr",
      "fit_method_secondary": "step_50",
      "fit_method_secondary_value_s": 0.118,
      "xcorr_peak": 0.94,
      "source_maneuvers": ["vel_step_x", "vel_step_z"],
      "per_axis": {"x": {...}, "y": {...}, "z": {...}}
    },
    "velocity_world_xyz": { "...": "same shape as position_xyz" },
    "velocity_body_xyz":  { "...": "same shape" },
    "attitude_roll":      { "...": "same shape (scalar channel)" },
    "attitude_pitch":     { "...": "same shape" },
    "attitude_yaw":       { "...": "same shape" },
    "body_rate_xyz":      { "...": "same shape; includes chirp_phase_delay tertiary" },
    "linear_acceleration_xyz": { "...": "same shape" }
  },
  "notes": [
    "Free-form remarks. E.g., 'velocity_z step shows 8% asymmetry between up and down — investigate in follow-up.'"
  ]
}
```

**Schema invariants** (consumers may rely on):
- `schema_version: 1` is stable; any breaking change bumps the integer.
- Every channel block has either `"status": "fit"` (all delay fields populated) or `"status": "skipped"` (only `reason` populated).
- Delays are seconds, floats. Negative delays are an error condition (would mean truth lags EKF — physically impossible); the fit script clamps to 0 and adds a note.
- `per_axis` is optional; the top-level `delay_mean_s` aggregates across axes (mean of per-axis means). Consumers that need per-axis should read `per_axis`.
- `source_maneuvers` is a list of maneuver IDs that contributed to the fit; useful for debugging if one fit looks anomalous.

## 8. Deliverable layout

```
controller/sysid_output/ekf_state_lag/
├── ekf_lag_recorder.py     # ROS2 node: records truth+EKF CSVs, drives cmd_vel for the requested maneuver
├── fit_ekf_lag.py          # Pure-Python: reads raw/*.csv, runs fits, writes ekf_state_lag.json + ekf_lag_report.pdf
├── run_measurement.sh      # Orchestrator: pre-flight checks, loop over 6 maneuvers × 3 trials, invoke fit
├── README.md               # Human summary (3 paragraphs, lab-notebook style)
├── ekf_state_lag.json      # Generated — the canonical fit (the "deliverable")
├── ekf_lag_report.pdf      # Generated — one page per channel
└── raw/
    ├── <maneuver>_trial<N>_<topic>.csv  # One file per (maneuver, trial, topic) — see §9
    └── px4_params_<timestamp>.json      # PX4 param snapshot captured at the start of the session
```

## 9. Recorder CSV schema (raw/ files)

One file per `(maneuver, trial, topic)` triple. Filename format: `<maneuver>_trial<N>_<topic_slug>.csv`. Example: `vel_step_x_trial0_state_pose.csv`.

Topic slugs:
- `state_pose`, `state_twist`, `state_twist_inertial`, `state_accel`
- `mavros_local_position_pose`, `mavros_local_position_velocity_local`, `mavros_local_position_velocity_body`, `mavros_imu_data`

Columns (one row per ROS message received during the maneuver):

| Topic | Columns after `stamp_s, recv_stamp_s` |
|---|---|
| `state_pose`, `mavros_local_position_pose` | `px, py, pz, qx, qy, qz, qw` |
| `state_twist` | `vx_body, vy_body, vz_body, wx_body, wy_body, wz_body` |
| `state_twist_inertial` | `vx_world, vy_world, vz_world` |
| `state_accel` | `ax_world, ay_world, az_world` |
| `mavros_local_position_velocity_local` | `vx_world, vy_world, vz_world, wx_world, wy_world, wz_world` |
| `mavros_local_position_velocity_body` | `vx_body, vy_body, vz_body, wx_body, wy_body, wz_body` |
| `mavros_imu_data` | `qx, qy, qz, qw, wx_body, wy_body, wz_body, ax_body, ay_body, az_body` |

`stamp_s` is the message's `Header.stamp` in seconds (sim-time). `recv_stamp_s` is the recorder's `get_clock().now()` at callback entry (also sim-time, since `use_sim_time=True`); the diff `recv_stamp_s − stamp_s` reveals any in-pipeline jitter and is sanity-checked by the analyzer (typical < 5 ms).

## 10. Calling contract

Per CLAUDE.md §"Stateful Component Rules":

### `EkfLagRecorder` (ROS2 node)
- **WRITE methods** (call exactly once per ROS message received):
  - `on_<topic>(msg)` callbacks. Each appends one row to its topic-specific CSV writer. Stateless beyond the file handle.
- **WRITE methods** (lifecycle):
  - `__init__(maneuver, trial, duration_s, ns, out_dir)` opens CSV writers, creates subscribers, creates the cmd_vel publisher, schedules the maneuver-command callback and the shutdown timer.
  - `start_maneuver()` — sets `t0 = get_clock().now()`, starts publishing the maneuver's `cmd_vel` profile via a 100 Hz timer.
  - `shutdown()` — closes CSV files, destroys the node. Called by the duration timer.
- **READ methods**: none — the recorder has no caller-facing query API.
- **Idempotency**: each subscriber callback is bound to a single topic; ROS guarantees no parallel calls for the same callback on the default executor. Per-step idempotency does not apply (the recorder runs in continuous mode, not on a sim-step boundary). Duplicate `Header.stamp` values are passed through unchanged — the analyzer dedupes if needed.

### `fit_ekf_lag.py` (pure-Python analyzer)
- **Pure functions**: `load_raw(dir, maneuver, trial, topic) -> DataFrame`, `resample_to_grid(df, t_grid, columns) -> DataFrame`, `xcorr_delay(truth, ekf, dt) -> (delay_s, peak)`, `step_50_delay(truth, ekf, step_edges) -> delay_s`, `chirp_group_delay(truth, ekf, dt, freq_band) -> (delay_s, flatness)`.
- **One entry point**: `python fit_ekf_lag.py --raw-dir raw/ --out-json ekf_state_lag.json --out-pdf ekf_lag_report.pdf`.
- **Idempotent**: identical input CSVs yield bit-identical JSON. (Numeric drift only from floating-point order; the script computes per-axis fits in a fixed order.)

### `run_measurement.sh`
- Sequenced pipeline; not idempotent (it overwrites `raw/*.csv` and the JSON/PDF). Re-running reproduces the JSON within ±10% per channel — the acceptance criterion.

## 11. Acceptance against the ticket

This spec satisfies ticket 041's "Deliverable" enumeration directly:
- `ekf_state_lag.json` with all 6 channels (§7).
- `ekf_lag_report.pdf` one page per channel (§8).
- `raw/*.csv` per maneuver (§9; 8 topics × 6 maneuvers × 3 trials = 144 files).
- `ekf_lag_recorder.py`, `fit_ekf_lag.py`, `run_measurement.sh` (§8, §10).
- No env / controller code changes (this is a measurement-only deliverable).
- Run reproducibility: `run_measurement.sh` documents the full launch + record + fit sequence; re-invocation reproduces the JSON within ±10% per channel (acceptance gate).

## 12. References

- Ticket: [active/ticket/041-px4-ekf-state-lag-measurement/ticket.md](active/ticket/041-px4-ekf-state-lag-measurement/ticket.md)
- PX4 EKF2 params: [PX4-Autopilot/src/modules/ekf2/ekf2_params.c](/home/usrg/IsaacPX4/PX4-Autopilot/src/modules/ekf2/ekf2_params.c)
- Pegasus state publishers: [PegasusSimulator/.../backends/ros2_backend.py](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/backends/ros2_backend.py)
- Sister ticket on comms / detection latency (different scope): [active/ticket/025-latency-measurement/ticket.md](active/ticket/025-latency-measurement/ticket.md)
- Sister ticket on Pegasus physics parity (different gap): [active/ticket/040-match-pegasus-physics-parameters/ticket.md](active/ticket/040-match-pegasus-physics-parameters/ticket.md)
- Existing PX4 SITL sysid CSVs (same maneuver names used here): [controller/sysid_output/](../controller/sysid_output/)
