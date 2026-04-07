# Stage R — Codebase Research Report (008-sysid-replicator)

## Module Inventory

### 1. auto_tune.py — Parallel Gain Sweep Infrastructure

**Path:** `controller/tuning/auto_tune.py` (1112 lines)

**Dataclasses:**
- `ParameterSet`: 12 gain fields (Kp/Ki_vel_{xy,z}, Kp_att_{rp,y}, Kp/Ki/Kd_rate_{rp,y})
- `OscillationMetrics`: damping_ratio, zero_crossings, ss_amplitude, frequency
- `TuningMetrics`: hover_drift, vel_settling/overshoot/ss_error, att_recovery, 6 oscillation metrics, is_stable, score
- `TuningScoreWeights`: 11 weight fields with defaults (hover ~1, velocity ~3-8, attitude ~0.02-1.0, oscillation ~5-10)
- `AttitudeTestConfig`: roll=20deg, pitch=20deg, yaw=15deg, duration=3s

**ParallelTuner class:**
- `__init__(num_envs, device, aero_level)`: Creates Isaac Sim env ("Isaac-Iris-MA6-Direct-Test-v0"), extracts physics_dt=0.01s, instantiates DroneController
- `_setup_batch(params_list)`: Sets per-env gains via `_velocity.set_gains()`, `_attitude.set_gains()`, `_rate.set_gains()` — tensors shape (N, 3)
- `_step_with_controller(v_cmd)`: One physics step through DroneController.step_policy(); extracts att_error via `attitude_controller.compute_attitude_error()`, rate_error = rate_setpoint - omega_body; returns (att_error, rate_error)
- `_reset_to_hover()`: Level attitude, 2m height, zero velocity/rates
- `_run_velocity_test(duration, target_speed)`: Forward velocity step, collects vel_history (T,N), att_error_history (T,N,3), rate_error_history (T,N,3)
- `_run_attitude_test(att_cfg)`: Tilted initial condition, recovery tracking, per-env failure detection
- `evaluate_batch(params_list)`: Orchestrates hover(3s) + vel_5(5s) + vel_10(5s) + att(3s), computes all metrics, returns (metrics_list, per_env_histories)

**Scoring formula:** Weighted sum of metric penalties. Unstable configs score 9999.0 (hover_drift_max>=2, NaN, att_failed, att_final_error>=15deg).

**Parameter generation:**
- `generate_random_params(num_trials)`: Random uniform within coupled ranges. Kp_vel_max = min(5.0, 2.0 + 10.0*Kp_rate_rp). Z/yaw derived from xy/rp by random scaling.

### 2. tuning_results/__init__.py — Current Gain Export

**Path:** `controller/tuning/tuning_results/__init__.py`

**Current TUNED_CONTROLLER_CFG (aero_level=3):**
- Velocity: Kp=(2.04, 2.04, 1.54), Ki=(1.30, 1.30, 0.78)
- Attitude: Kp=(5.21, 5.21, 1.93)
- Rate: Kp=(0.39, 0.39, 0.40), Ki=(0.18, 0.18, 0.09), Kd=(0.0145, 0.0145, 0.0)

**Format:** `DroneControllerCfg(velocity=VelocityControllerCfg(...), attitude=AttitudeControllerCfg(...), rate=RateControllerCfg(...))`

### 3. DroneController — Cascade Architecture

**Path:** `controller/drone_controller.py`

**step_policy() signature:**
```
step_policy(v_cmd(N,3), yaw_rate_cmd(N,), gimbal_yaw_rate(N,), gimbal_pitch_rate(N,), zoom_rate(N,),
            q_body(N,4 wxyz), v_body(N,3), omega_body(N,3), sim_dt, gimbal_joint_positions(N,3), physics_dt)
→ (F_body(N,3), tau_body(N,3), gimbal_targets, gimbal_vels, zoom_level(N,))
```

**Cascade flow:**
1. Velocity controller (once at sim_dt): v_cmd → q_des + thrust
2. Attitude controller (at inner_dt substeps): q_des + q_body → rate_setpoint
3. Rate controller (at inner_dt substeps): rate_setpoint + omega → tau + omega_cmd
4. Motor dynamics → F_body, tau_body

**Sub-controller set_gains APIs:**
- `_velocity.set_gains(Kp_vel=, Ki_vel=)` — (N,3) or (3,) or tuple
- `_attitude.set_gains(Kp_att=)` — (N,3) or (3,) or tuple
- `_rate.set_gains(Kp_rate=, Ki_rate=, Kd_rate=)` — (N,3) or (3,) or tuple

**Intermediate signal access:**
- `_attitude.compute_attitude_error(q_des, q_body)` → (N,3) radians
- Rate setpoint stored internally; auto_tune extracts rate_error = rate_sp - omega_body

### 4. Test Environment Config

**Path:** `iris_ma_env6_test_cfg.py`

- sim.dt = 0.01s (100 Hz physics)
- decimation = 4 (25 Hz policy)
- num_envs = 1024
- DroneController control_dt = 0.01s (inner loop at physics rate)

### 5. sysid_node.py — PX4 SITL Test Execution

**Path:** `/home/usrg/IsaacPX4/ros2_ws/src/offboard_py/offboard_py/sysid_node.py` (657 lines)

**Purpose:** ROS2 node commanding PX4 via MAVROS in OFFBOARD mode; records cascade telemetry to CSV.

**Test sequence (7 tests, sequential):**

| Test | Nominal Duration | Command |
|------|-----------------|---------|
| hover_drift | 10s | Zero velocity |
| vel_step_5 | 10s | 5.0 m/s forward (ENU +X) |
| vel_step_10 | 10s | 10.0 m/s forward (ENU +X) |
| vel_impulse_recovery | 10s | 5.0 m/s for 1s, then zero |
| yaw_step | 10s | 0.5 rad/s for 2s, then zero |
| gimbal_los_rate_step | 2s | Gimbal rate commands |
| gimbal_los_stabilization | 7s | LOS hold during drone disturbance |

Durations are nominal — i.e. relative to physics time. In sim this is sim-time; on real hardware this equals wallclock. The sysid_node timer runs at 100 Hz wallclock, but PX4 SITL physics runs at 125 Hz (dt=0.008s). The CSV `timestamp_s` column reflects sim-time from /clock, so the effective sample rate in the CSVs is governed by the sim-to-wallclock ratio, not the timer frequency.

**Recording rate:** Wallclock-driven timer at 100 Hz real-time (`update_rate` parameter). CSV `timestamp_s` sourced from /clock topic (sim time). Effective sim-time sample rate depends on sim-to-wallclock ratio; observed ~7300 rows per 10s nominal test suggests ~125 Hz in sim-time (consistent with PegasusSimulator's 125 Hz physics dt=0.008s).

**CSV columns (27):**
timestamp_s, pos_{xyz}, vel_{xyz}, quat_{xyzw}, ang_vel_{xyz}, gimbal_los_{az,el}_deg, gimbal_rpy_{r,p,y}_deg, att_sp_q{xyzw}, rate_sp_{xyz}, thrust_sp

**Quaternion convention:** xyzw in CSV (ROS convention). iris_ma6 uses wxyz.

**Cascade setpoint source:** MAVROS `target_attitude` topic (AttitudeTarget msg). att_sp is xyzw quaternion, rate_sp is body-frame rad/s, thrust is normalized 0-1.

### 6. sysid_analyze.py — Offline Metric Computation

**Path:** `/home/usrg/IsaacPX4/ros2_ws/src/offboard_py/offboard_py/sysid_analyze.py` (814 lines)

**CSV loading:** pandas read_csv, dedup timestamps (keep last), add relative time column `t = timestamp_s - min`.

**Metric definitions:**
- Settling time: first crossing of 95% target
- Overshoot: max(0, (peak - target)/target * 100) %
- SS error: |target - mean(last 0.5s)|
- Oscillation: log-decrement damping, zero-crossings, ss_amplitude, frequency from crossing intervals
- Hover drift: mean/max horizontal distance from initial position
- Yaw: target = 0.5 rad/s, command window = 0-2s

**Comparison logic:** For each metric, computes diff_pct = |sitl - ref| / |ref| * 100. PASS if diff_pct <= threshold*100 (default 20%).

**Output:** `sysid_metrics.json` (all metrics + comparison table) + per-test PDF plots (4-row: velocity, attitude, rate, thrust).

### 7. PX4 SITL Recorded Data

**Path:** `/home/usrg/IsaacPX4/sysid_output/` (7 CSV files, generated 2025-04-06)

**Row counts:** ~7250-7330 rows per 10s nominal test. The sysid_node wallclock timer fires at 100 Hz, but since PX4 SITL sim-time advances faster than wallclock, each wallclock tick captures more than 0.01s of sim-time. The CSV timestamps (sim-time) span ~10s nominal, sampled at an effective ~125 Hz in sim-time (matching PegasusSimulator physics dt=0.008s).

**Key PX4 SITL metrics (from sysid_metrics.json):**

| Metric | PX4 SITL Value |
|--------|---------------|
| hover_drift_mean | 0.053 m |
| hover_drift_max | 0.075 m |
| vel_5 settling_time | 2.73 s |
| vel_5 ss_error | 0.10 m/s |
| vel_5 damping_ratio | 0.0035 |
| vel_10 settling_time | 9.99 s |
| vel_10 ss_error | 0.68 m/s |
| vel_10 damping_ratio | 0.00019 |
| yaw settling_time | 0.14 s |
| yaw overshoot | 7.4% |

**iris_ma6 sysid_output directory:** Does not exist yet (needs to be created).

## Data Models

### Gain Representation
- `ParameterSet`: 12 scalar fields (6 independent dims + 6 derived via coupling ratios)
- `DroneControllerCfg`: Nested config with 3 sub-configs (VelocityControllerCfg, AttitudeControllerCfg, RateControllerCfg), each holding tuple[float, float, float] for (xy/rp, xy/rp, z/yaw)

### Timeseries Format
- auto_tune.py histories: torch tensors (T, N) or (T, N, 3) on GPU
- PX4 SITL CSVs: pandas DataFrames, 27 columns, ~100 Hz, timestamps in seconds

### Metric Format
- auto_tune.py: `TuningMetrics` dataclass per environment
- sysid_analyze.py: Separate dataclasses per test type (VelocityStepMetrics, HoverMetrics, YawStepMetrics)
- sysid_metrics.json: Nested dict with sitl_metrics, reference_metrics, comparison array

## Conventions Observed

1. **Quaternion conventions differ:** CSV uses xyzw (ROS), DroneController uses wxyz (Isaac Sim)
2. **Rate conventions differ:** sysid_node records at wallclock 100 Hz with /clock timestamps; auto_tune steps at physics 100 Hz (dt=0.01s)
3. **Test duration mismatch:** sysid_node uses 10s per test; auto_tune uses 3-5s per test
4. **Scoring paradigms differ:** auto_tune scores against hand-crafted penalty weights; sysid_analyze scores by metric-to-metric comparison with threshold
5. **Coordinate frames:** Both use ENU/world frame for velocity. Angular velocity in body frame.
6. **Gain coupling:** Z/yaw gains derived from XY/RP gains by random ratio in auto_tune (not independent search)

## Gaps and Inconsistencies

1. auto_tune.py has no yaw step test. sysid_node runs a yaw step (0.5 rad/s for 2s).
2. auto_tune.py has no impulse recovery test. sysid_node runs vel_impulse_recovery (5 m/s for 1s, then zero).
3. auto_tune.py does not record or compare cascade setpoints (att_sp, rate_sp, thrust). PX4 CSVs include these.
4. PX4 SITL att_sp quaternion convention (xyzw) differs from DroneController (wxyz).
5. sysid_analyze.py metric definitions (OscillationMetrics, VelocityStepMetrics) are defined independently from auto_tune.py's identically-named dataclasses — no shared code.
6. PX4 SITL data exists at `/home/usrg/IsaacPX4/sysid_output/` but not within iris_ma6 module directory.
7. auto_tune.py's `_step_with_controller` extracts att_error and rate_error but not rate_setpoint or att_setpoint as tensors — they are computed inline.
