## Ticket: Light system identification — iris_ma6 vs PX4 SITL step response comparison

**What**: Run step response tests on PX4 SITL (via PegasusSimulator) and compare against iris_ma6's DroneController reference curves to quantify the dynamics mismatch. Output a mismatch summary table and comparison plots.

**Why**: Before deploying the trained policy in PegasusSimulator (sim-to-sim transfer), we need to verify that iris_ma6's controller dynamics are close enough to PX4's actual cascade. If the mismatch exceeds the ±20% gain randomization range, the policy may fail at deployment. This is the cheapest validation step — a ~200-line measurement script that reuses existing infrastructure on both sides.

**Evidence**:
- Analysis in ticket-005 §2.1 identifies this as P0.5 priority for sim-to-sim transfer
- iris_ma6 reference curves already exist from ticket-003 tuning (`controller/tuning/auto_tune.py` step response plotter)
- `mas_policy` bridge (observation_assembler, action_publisher, policy_loader) is already operational — the offboard interface exists
- Tuned gains from ticket-003 (aero3, Cd=0.03) define the reference: Kp_vel≈2.04, Ki_vel≈1.30, Kp_att≈5.21, Kp_rate≈0.39

**Scope**:
1. **PX4 SITL test script** (~200 lines): Send offboard commands to PX4 via MAVROS (`offboard_py`), record telemetry (`/mavros/local_position/odom`, `/mavros/imu/data`, gimbal state) (**CAUTION**: Timestamp from mavros topics are wall-clock based. Make sure to subscribe to simulation clock, or use simtime)
2. **Test suite**: Velocity steps (5 m/s, 10 m/s), attitude recovery (20° perturbation), hover drift (10s), yaw step (0.5 rad/s for 2s), gimbal tests (see below)
   - **Gimbal — LOS stabilization**: Command gimbal to hold a fixed world-frame LOS direction while the drone maneuvers (e.g., yaw step, velocity step). Measures stabilization error (LOS pointing drift) and disturbance rejection bandwidth.
   - **Gimbal — LOS rate command**: Command a step LOS rate (az/el) and measure gimbal angle response. Measures rate tracking lag, overshoot, settling time.
3. **Metric extraction**: Settling time, overshoot, damping ratio, effective tau_motor, effective Cd, effective max_tilt, effective gimbal lag — same metrics as `auto_tune.py`
4. **Comparison plots**: iris_ma6 reference vs PX4 SITL actual (velocity, attitude, rates vs time) per test
5. **Mismatch summary table**: Per-metric percentage deviation, pass/fail against ±20% threshold

**Scope boundary**:
- Do NOT modify iris_ma6 controller gains or architecture
- Do NOT modify PX4 parameters — measure the default SITL configuration
- Do NOT retrain the policy — this is measurement only
- If mismatch > 20%, document which parameters are off and by how much; corrective action is a separate ticket

**Launch environment**:
- `tmux/isaac_sim.tmuxp.yaml` — launches PegasusSimulator (Isaac Sim + PX4 SITL) + gimbal stabilizer (`multi_agent_los_rate.launch.py`) + QGroundControl
- The gimbal stabilizer (`ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py`) provides both LOS stabilization (position mode) and LOS rate command (rate mode) — both must be tested

**Affected modules / repos**:
- `ros2_ws/src/offboard_py` — reuse or extend for step response commands
- `ros2_ws/src/gimbal_stabilizer` — gimbal LOS rate controller (launched by tmuxp, provides the gimbal control interface)
- `controller/tuning/auto_tune.py` — reference curves and metric definitions to reuse
- `doc/active/ticket/005-sim-to-sim-transfer/analysis.md` §2.1 — authoritative spec for test matrix
- PegasusSimulator — launch environment for PX4 SITL tests

**Key references**:
- Test matrix: ticket-005 `analysis.md` §2.1, lines 167-176
- Tuned reference gains: `controller/tuning/tuning_results/__init__.py`
- Metric definitions: `controller/tuning/auto_tune.py` (TuningMetrics)
- Offboard control: `ros2_ws/src/offboard_py`
- Gimbal controller: `ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py` (rate mode + position mode)
- PegasusSimulator launch: `PegasusSimulator/launch/px4_multi_world_iris_gimbal3.isaac.py`
- tmuxp session: `tmux/isaac_sim.tmuxp.yaml`

**Acceptance criteria**:
1. Step response tests run on PX4 SITL for all 6 test cases in the matrix
2. Metrics extracted match the same definitions used in `auto_tune.py`
3. Comparison plots generated (PDF/PNG) showing reference vs actual per test
4. Mismatch summary table with per-metric % deviation and pass/fail flag
5. Decision documented: proceed with deployment OR list parameters needing correction

**Decision rule**:
- All metrics within ±20% → proceed with sim-to-sim deployment
- Any metric > 20% off → document gap, expand gain randomization range or update controller defaults (separate ticket)

**Reusability**: The same test scripts and metrics apply to sim-to-real validation — only the data source changes (PX4 SITL → real PX4 telemetry via MAVROS).

**Flow**: Full QRISPY
