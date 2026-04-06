## Design Document: PX4 SITL System Identification Node

### Problem statement

Before deploying the iris_ma6-trained policy in PegasusSimulator, we need to quantify how closely PX4 SITL's dynamics match iris_ma6's DroneController. A new ROS2 node sends step commands to a single PX4 instance via MAVROS (all ENU, matching the real-system interface), records telemetry, extracts the same oscillation metrics as `auto_tune.py`, and produces comparison plots and a pass/fail mismatch table against the ±20% gain randomization threshold.

### Proposed approach

**Interface: MAVROS-only, all ENU.** The sysid node does not touch `fmu/*` topics. It commands PX4 through MAVROS and reads telemetry from MAVROS. This matches the real-system interface exactly.

**MAVROS namespace**: `/px4_1/mavros/` (from `isaac_mavros.tmuxp.yaml`: `__ns:=/px4_1/mavros`). MAVROS connects to PX4 SITL instance 1 via `udp://:14541@localhost:14551`, `tgt_system: 2`.

**Commands** (publish/service):
- `/{ns}/mavros/setpoint_velocity/cmd_vel_unstamped` (geometry_msgs/Twist, ENU) — velocity setpoints
- `/{ns}/mavros/cmd/arming` (mavros_msgs/CommandBool) — arm/disarm service
- `/{ns}/mavros/set_mode` (mavros_msgs/SetMode) — set OFFBOARD mode

**Telemetry** (subscribe):
- `/{ns}/mavros/local_position/odom` (nav_msgs/Odometry, ENU/FLU) — position, velocity, orientation, angular velocity
- `/{ns}/mavros/imu/data` (sensor_msgs/Imu, ENU/FLU) — orientation quaternion, body angular velocity

**Gimbal** (subscribe/publish, under vehicle namespace `/{ns}/`):
- Subscribe: `/{ns}/gimbal_los_state_deg` (Vector3) — world-frame az/el (positive el = up)
- Subscribe: `/{ns}/gimbal_state_rpy_deg` (Vector3) — body-frame RPY, ENU convention
- Publish: `/{ns}/gimbal_cmd_los_rate` (Vector3) — x=az_rate, y=el_rate (rad/s, positive el = up)
- Publish: `/{ns}/gimbal_cmd_los_world_deg` (Vector3) — world-frame position target (degrees)

**Clock**: Subscribe `/clock` in sim (`use_sim_time: true`). Wallclock in real tests.

**Velocity-only offboard**: All PX4 commands are velocity setpoints via MAVROS. No position mode. No NED conversion in our node — MAVROS handles it. Stream at 100 Hz sim-time rate (matching iris_ma6 physics dt).

**State machine**: arm → hover (settle 10s) → test 1 (record) → hover (settle 10s) → test 2 → ... → done. No landing between tests.

**Test suite** (7 tests, velocity-only commands):
1. Hover drift — command [0,0,0] velocity, 10s
2. Velocity step 5 m/s — command [5,0,0] ENU, 10s
3. Velocity step 10 m/s — command [10,0,0] ENU, 10s
4. Velocity impulse recovery — command [5,0,0] for 1s → [0,0,0] for 9s, observe settling. *(Not directly comparable to auto_tune.py's attitude perturbation — adaptation is separate work.)*
5. Yaw step — yaw_rate 0.5 rad/s for 2s, then 0 for 8s
6. Gimbal LOS rate step — init → 0.5s pitch rate → hold 0.5s → 0.5s yaw rate → hold 0.5s (short to avoid joint limits)
7. Gimbal LOS stabilization — command fixed LOS target → yaw drone → pulse x-vel 1s (pitch disturbance) → pulse y-vel 1s (roll disturbance), measure LOS drift

**Recording**: Per-test CSV (timestamp, position xyz, velocity xyz, orientation quat, angular velocity xyz, gimbal LOS az/el, gimbal RPY). In-memory buffer, saved post-test.

**Post-processing**: Separate offline script (`sysid_analyze.py`, no ROS2). Reads CSVs, generates iris_ma6 reference by re-running DroneController.step_policy() with tuned gains, computes metrics, plots comparison PDFs, outputs mismatch table.

**Launch**: New `tmux/isaac_sysid.tmuxp.yaml` integrating MAVROS + sysid node.

### Key interfaces and data flow

```
                    ┌──────────────────────────────────┐
                    │     sysid_node (new ROS2 node)   │
                    │                                  │
  /clock ──────────►│  State machine:                  │
                    │  INIT → ARM → HOVER → TEST_N →   │
  mavros/local_     │  SETTLE → TEST_N+1 → ... → DONE │
  position/odom ───►│                                  │──► mavros/setpoint_velocity/
                    │  Telemetry recorder:             │    cmd_vel_unstamped (Twist ENU)
  mavros/imu/data ─►│  Buffers per-test timeseries     │
                    │                                  │──► mavros/cmd/arming (service)
  gimbal_los_       │  Gimbal commander:               │──► mavros/set_mode (service)
  state_deg ───────►│  LOS rate / position commands    │
                    │                                  │──► gimbal_cmd_los_rate (Vector3)
  gimbal_state_     │  CSV writer:                     │──► gimbal_cmd_los_world_deg (Vector3)
  rpy_deg ─────────►│  Saves per-test on completion    │
                    └──────────────────────────────────┘

         ┌────────────────────────────────────────────┐
         │  sysid_analyze.py (offline, no ROS2)       │
         │                                            │
         │  1. Load PX4 SITL CSVs                     │
         │  2. Generate iris_ma6 reference (rerun     │
         │     DroneController.step_policy with       │
         │     tuned gains, save timeseries)          │
         │  3. Compute metrics (reuse OscillationMetrics│
         │     definitions from auto_tune.py)         │
         │  4. Generate comparison PDFs               │
         │  5. Output mismatch table (JSON + stdout)  │
         └────────────────────────────────────────────┘
```

### What this does NOT include

- No modification to iris_ma6 controller gains or architecture
- No modification to PX4 parameters
- No retraining of the policy
- No changes to `offboard_control.py`
- No changes to `los_rate_controller.py`
- No rosbag2 recording (CSV is simpler)
- No adaptation of auto_tune.py attitude test (separate ticket)

### Open risks

1. **MAVROS IMU availability**: Gimbal stabilizer subscribes to `mavros/imu/data`. Verify PegasusSimulator ROS2Backend provides it, or ensure offboard_control.py republisher is enabled.
2. **Attitude perturbation comparability**: Velocity impulse ≠ direct attitude perturbation. Metrics not directly comparable to auto_tune.py for this test. Acknowledged — auto_tune.py adaptation is separate work.
3. **MAVROS offboard streaming rate**: PX4 requires continuous setpoint streaming. 100 Hz sim-time rate should be sufficient (matches PX4's expected 2+ Hz minimum by a large margin).
