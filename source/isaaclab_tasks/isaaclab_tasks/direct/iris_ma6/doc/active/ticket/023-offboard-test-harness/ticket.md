## Ticket: Offboard flight test harness + velocity step script (A2)

**What**: Build a reusable offboard flight test harness and implement the velocity step response test script (`offboard_velocity_step.py`). The harness provides: offboard mode detection (wait for RC switch), auto ROS2 bag recording with test-name prefix, safety abort (return to position mode on timeout or error). The velocity step script is the first consumer and the highest-priority flight dynamics measurement.

**Why**: The sim-to-real checklist (ticket-021) identifies 7 test scripts needed for real-world data collection. All offboard flight tests share the same infrastructure: wait for mode switch, execute command sequence, record data, handle abort. Building the harness first avoids duplicating this across scripts. A2 (velocity step) is the highest-value single flight test — it simultaneously collects data for velocity controller validation, drag characterization (B1), gimbal dynamic response (D3), and feeds the model fitting pipeline (A6/ticket-027).

**Existing code to reuse**: `sysid_node.py` (656 lines) in `/home/usrg/IsaacPX4/ros2_ws/src/offboard_py/offboard_py/` already implements:
- MAVROS offboard state machine (INIT → RAMP_UP → ARM → TAKEOFF → SETTLE → TEST → DONE)
- CSV recording infrastructure (18-column format, `_record_buffer` + `_save_test_csv()`)
- Velocity step tests (`vel_step_5`, `vel_step_10`)
- QoS profiles, frame conversions, wallclock timer control loop
- Multi-vehicle namespace handling

**What sysid_node lacks** (and the harness must add):
1. **Waits for external offboard mode switch** — sysid_node arms and switches to offboard itself. The new harness should wait for the pilot to switch to offboard via RC, for safety.
2. **Auto ROS2 bag recording** — sysid_node only records CSV. The harness should trigger `ros2 bag record` with auto-naming (e.g., `A2_vel_step_x_run1_20260415T1030`).
3. **Return-to-origin pattern** — A2 needs 0→+v→0→-v to save space. sysid_node only does 0→+v→0.
4. **Configurable test sequences** — The harness should support declarative test sequences (list of velocity/duration pairs) rather than hardcoded tests.
5. **Setpoint logging** — Record velocity/attitude/rate setpoints alongside state for controller validation.

**Scope boundary**:
- DO: Create reusable harness class/module (mode detection, bag recording, safety, CSV recording)
- DO: Implement `offboard_velocity_step.py` using the harness
- DO: Add entry point to `setup.py`
- DO: Record both PX4 `.ulg` (pilot responsibility) and ROS2 bag (auto-triggered by script)
- DO NOT: Implement attitude (A3) or yaw rate (A4) scripts (ticket-024)
- DO NOT: Change existing sysid_node.py (it works for its purpose)
- DO NOT: Implement auto-arming or auto-takeoff (pilot handles this in position mode)

**Flight sequence**:
```
[Pilot] Position mode → Takeoff → Hover
[Pilot] Switch to Offboard mode
[Script] Detects offboard mode → starts ROS2 bag → waits 10s (stabilize)
[Script] Executes: 0 → +3 → 0 → -3 → 0 → +5 → 0 → -5 → ... (configurable)
[Script] Completes → stops ROS2 bag → publishes zero velocity
[Pilot] Switch to Position mode → Land
```

**Affected files**:
- NEW: `offboard_py/offboard_py/flight_test_harness.py` — reusable harness module
- NEW: `offboard_py/offboard_py/offboard_velocity_step.py` — A2 test script
- MODIFY: `offboard_py/setup.py` — add entry point
- MODIFY: `offboard_py/doc/active/feature_list.json` — add feature entry

**Acceptance criteria**:
- Script waits for offboard mode switch (does not arm or switch mode itself)
- Velocity step sequence executes with configurable speeds, axes, and hold durations
- ROS2 bag auto-records with test name + timestamp prefix
- CSV output includes: timestamp, position, velocity, attitude (quat), angular velocity, rate setpoints, velocity setpoints, actuator outputs
- On offboard mode loss (pilot switches away), script gracefully stops recording and publishes zero velocity
- Tested in PX4 SITL before real hardware

**Reference**: [checklist.md](../021-sim2real-measurement-checklist/checklist.md) items A2, B1, D3

**Flow**: Full QRISPY
