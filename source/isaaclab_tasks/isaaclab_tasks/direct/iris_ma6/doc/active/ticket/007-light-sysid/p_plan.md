## Stage P — Implementation Plan

### Slice 1: sysid_node — core state machine + velocity tests

**End-to-end testable**: Node starts, arms drone, runs hover + velocity step tests, saves CSVs.

- Step 1.1: Create `sysid_node.py` with `SysidNode(Node)` — params, subscribers (odom, imu, /clock), publisher (cmd_vel_unstamped), service clients (arming, set_mode), callback caching, clock-driven 100 Hz state machine skeleton
- Step 1.2: Implement `TestPhase` enum, `TestSpec` dataclass, and `_run_state_machine()` — INIT → ARM → SETTLE → TEST → SETTLE → TEST → ... → DONE sequence with sim-time tracking
- Step 1.3: Implement `_arm_and_offboard()` — call MAVROS services, `_publish_velocity()` — publish Twist (ENU), continuous streaming at 100 Hz
- Step 1.4: Implement recording (`_record_sample`) and CSV writer (`_save_test_csv`) — timestamp, position xyz, velocity xyz, orientation quat, angular velocity xyz per row
- Step 1.5: Define test sequence for drone tests: hover_drift, vel_step_5, vel_step_10, vel_impulse_recovery, yaw_step — each with command_fn and duration
- Step 1.6: Add entry point in `setup.py`, add `mavros_msgs` to `package.xml`

**Test checkpoint**: Launch `isaac_sim.tmuxp.yaml` + MAVROS manually + `ros2 run offboard_py sysid_node`. Node arms px4_1, runs 5 drone tests sequentially with 10s settle between each, saves 5 CSVs to output dir.

### Slice 2: sysid_node — gimbal tests

**End-to-end testable**: Full 7-test sequence including gimbal.

- Step 2.1: Add gimbal subscribers (`gimbal_los_state_deg`, `gimbal_state_rpy_deg`) and publishers (`gimbal_cmd_los_rate`, `gimbal_cmd_los_world_deg`) to `SysidNode.__init__`
- Step 2.2: Add gimbal columns to CSV recording (gimbal_los_az_deg, gimbal_los_el_deg, gimbal_rpy_r_deg, gimbal_rpy_p_deg, gimbal_rpy_y_deg)
- Step 2.3: Implement gimbal LOS rate test — init → 0.5s pitch rate → hold 0.5s → 0.5s yaw rate → hold 0.5s. Short total duration, sub-phases within one TestSpec
- Step 2.4: Implement gimbal LOS stabilization test — command fixed LOS target → yaw drone → pulse x-velocity 1s (pitch disturbance) → pulse y-velocity 1s (roll disturbance), measure LOS drift
- Step 2.5: Append gimbal tests to `build_test_sequence()`

**Test checkpoint**: Full 7-test sequence runs end-to-end. CSVs include gimbal state columns. Gimbal responds to rate and position commands.

### Slice 3: isaac_sysid.tmuxp.yaml

- Step 3.1: Create `tmux/isaac_sysid.tmuxp.yaml` — window 1: MAVROS for px4_1 (`use_sim_time:=true`, params from `config/mavros/mavros_param_px4_1.yaml`), window 2: sysid_node (after sleep, sourcing ROS2 workspace)

**Test checkpoint**: `tmuxp load tmux/isaac_sysid.tmuxp.yaml` alongside `tmux/isaac_sim.tmuxp.yaml` — MAVROS connects, sysid_node runs full test sequence automatically.

### Slice 4: sysid_analyze — offline post-processing + comparison plots

- Step 4.1: Create `sysid_analyze.py` with `load_sysid_csv()`, argparse (`--sysid-dir`, `--output-dir`, `--threshold`)
- Step 4.2: Implement `generate_iris_ma6_reference()` — instantiate DroneController with tuned gains, run step_policy() for velocity/yaw tests matching sysid test commands, return timeseries dicts. Uses AppLauncher + Isaac Lab env (same pattern as auto_tune.py)
- Step 4.3: Implement `compute_metrics()` — settling time, overshoot, damping ratio, zero-crossings, SS amplitude, frequency. Reimplement from auto_tune.py definitions (avoid import dependency on Isaac Lab from a ROS2 script)
- Step 4.4: Implement `compute_gimbal_metrics()` — rate tracking lag, overshoot, settling time for LOS rate test; stabilization error (peak LOS drift) for stabilization test
- Step 4.5: Implement `plot_comparison()` — multi-panel PDF: reference vs SITL per test (velocity, attitude, rates vs time). Follow step_response_plotter.py visual style
- Step 4.6: Implement `mismatch_table()` — per-metric % deviation, pass/fail against threshold. Output JSON + stdout summary

**Test checkpoint**: Run on CSVs from slice 1-2 output. Generates comparison PDFs, prints mismatch table, outputs JSON. Pass/fail flags reflect ±20% threshold.
