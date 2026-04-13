# Stage P — Implementation Plan

Ticket: 023 — Offboard flight test harness + velocity step script

---

## Slice 1: Base class — subscriptions, timer, state machine (no recording)

- Step 1.1: Create `flight_test_harness.py` with imports, `HarnessPhase` enum, QoS profiles, and `FlightTestHarness.__init__` declaring all parameters (`vehicle_ns`, `update_rate`, `settle_duration`, `drift_limit`, `drift_kp`, `drift_vel_clamp`, `drift_return_threshold`, `altitude_margin`, `output_dir`, `config_file`)
- Step 1.2: Implement `_setup_subscribers` (mavros/state, odom, imu, att_target, gimbal_los, gimbal_rpy, /clock) and all subscriber callbacks (cache latest message)
- Step 1.3: Implement `_setup_publishers` (cmd_vel) and `_publish_velocity`, `_publish_zero_velocity`
- Step 1.4: Implement `_setup_services` (SetMode client)
- Step 1.5: Implement `_setup_timer` and `_timer_cb` dispatching to `_state_*` methods
- Step 1.6: Implement `_state_wait_offboard` — poll `_mavros_state` for armed + OFFBOARD, call `_capture_origin` and `_on_offboard_entered` on entry
- Step 1.7: Implement `_state_settle` — zero velocity (or drift correction), transition to TEST after `settle_duration`
- Step 1.8: Implement `_state_test` — call `_get_current_command`, publish velocity, transition to SETTLE when `_is_test_complete`
- Step 1.9: Implement `_state_abort` — zero velocity, `_request_posctl`, resume to SETTLE if offboard re-entered
- Step 1.10: Implement drift guard (`_capture_origin`, `_drift_exceeds_limit`, `_compute_drift_correction`) and `_check_altitude_margin`

**Test checkpoint:** Subclass with a dummy `_get_current_command` (returns zero) can be launched in SITL. Verify: node starts, waits for offboard, transitions through SETTLE → TEST → DONE when mode is switched. Mode loss triggers ABORT → POSCTL.

---

## Slice 2: CSV recording

- Step 2.1: Define `CSV_HEADER` (31 columns: 27 from sysid + `vel_sp_x`, `vel_sp_y`, `vel_sp_z`, `yaw_rate_sp`)
- Step 2.2: Implement `_record_sample(vel_sp)` — build 31-element row from cached topic data, append to `_record_buffer`
- Step 2.3: Implement `_save_csv(filename)` — write header + buffer to `{output_dir}/{filename}.csv`, clear buffer
- Step 2.4: Wire recording into `_state_test` (call `_record_sample` each tick) and `_state_abort` (continue recording)
- Step 2.5: Call `_save_csv` from `_on_test_complete` hook and on node shutdown

**Test checkpoint:** Run dummy subclass in SITL with offboard mode. Verify CSV file is created with correct 31-column header, rows accumulate during TEST phase, file is saved on completion.

---

## Slice 3: ROS2 bag recording + lifecycle cleanup

- Step 3.1: Implement `_start_bag_recording` — `subprocess.Popen(["ros2", "bag", "record", ...])` with auto-named output directory and curated topic list
- Step 3.2: Implement `_stop_bag_recording` — `terminate()` + `wait(timeout=5)` on subprocess
- Step 3.3: Wire bag start into `_state_wait_offboard` (on offboard entry) and bag stop into `_state_done` and `_state_abort` (on shutdown only, not on mode loss — keep recording)
- Step 3.4: Override `destroy_node` — call `_stop_bag_recording` in `finally` block for crash safety
- Step 3.5: Implement `_on_offboard_lost` default — log warning, do NOT stop bag

**Test checkpoint:** Run in SITL. Verify bag directory is created with test name prefix, bag contains expected topics, bag is closed cleanly on node shutdown and on Ctrl+C.

---

## Slice 4: Velocity step subclass + YAML config

- Step 4.1: Create `config/velocity_step.yaml` with default values (axes=[x,y,z], speeds=[3.0,5.0], hold=5s, settle=5s)
- Step 4.2: Create `flight_test_velocity_step.py` with `VelocityStep` dataclass and `FlightTestVelocityStep.__init__` loading config via ROS2 parameter
- Step 4.3: Implement `_load_config` — read YAML, validate fields
- Step 4.4: Implement `_expand_sequence` — generate full step list: for each axis, for each speed, `(axis, 0, settle) → (axis, +speed, hold) → (axis, 0, settle) → (axis, -speed, hold)`, ending with a final `(axis, 0, settle)`
- Step 4.5: Implement `_get_current_command` — index into expanded sequence based on elapsed time
- Step 4.6: Implement `_is_test_complete` — true when all steps exhausted
- Step 4.7: Implement `_on_test_complete` — save CSV with test name
- Step 4.8: Implement `_on_offboard_entered` — check altitude margin for z-axis, log sequence summary
- Step 4.9: Implement `main()` entry point
- Step 4.10: Add entry point to `setup.py`, update `feature_list.json`

**Test checkpoint:** Run `flight_test_velocity_step` in SITL with default YAML. Verify: correct velocity sequence published (x±3, x±5, y±3, y±5, z±3, z±5 with settle gaps), CSV contains matching setpoint columns, altitude check skips z-axis if too low.

---

## Slice 5: Composition example + final integration

- Step 5.1: Create `flight_test_harness_composition_example.py` showing pattern (a) — composition-based harness that accepts a Node instance
- Step 5.2: Add inline comments contrasting with pattern (b): where timers/subscribers are created, how lifecycle differs, pros/cons
- Step 5.3: Rebuild package (`colcon build --packages-select offboard_py`), verify entry point resolves
- Step 5.4: End-to-end SITL test: launch MAVROS + PX4 SITL, takeoff in position mode, switch to offboard, verify full velocity step sequence, switch back to position mode, verify ABORT + recording continuity

**Test checkpoint:** Composition example runs standalone. Full end-to-end SITL pass with velocity step subclass: mode detection → bag start → settle → velocity steps (all 3 axes) → done → bag stop → CSV saved.
