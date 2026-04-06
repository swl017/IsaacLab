## Stage R — Codebase Research Report

### Module inventory

**1. `ros2_ws/src/offboard_py/`** — ROS2 offboard control package

- `offboard_control.py`: Main node. Subscribes to PX4 `fmu/out/*` topics (VehicleOdometry, VehicleStatus, VehicleLocalPosition, VehicleGlobalPosition), publishes `fmu/in/*` (OffboardControlMode, TrajectorySetpoint, VehicleCommand). Republishes to MAVROS-compatible topics (`mavros/local_position/odom`, `mavros/imu/data`) — currently commented out.
- Arming sequence: stream heartbeat + setpoint for 11 ticks → engage offboard → arm. All in `timer_callback` at 100 Hz.
- Velocity commands: `publish_velocity_setpoint(vx_enu, vy_enu, vz_enu, vyaw)` → TrajectorySetpoint with NED conversion: `[vy_enu, vx_enu, -vz_enu]`, `yawspeed = -vyaw`.
- Launch: `offboard.launch.py` reads `config/vehicles.yaml`, creates one node per vehicle with namespace.
- `vehicles.yaml`: 6 vehicles, namespaces `px4_1` through `px4_6`, each with `target_system` (2-7), position, yaw.
- **No simulation clock handling** — uses wall-clock.
- **No data logging** — only debug prints.
- Entry points: `offboard_control`, `thrust_and_rate_control`.
- **Direction mismatch for sysid reuse**: subscribes PX4 fmu/out, publishes MAVROS. Sysid needs the opposite (subscribe MAVROS telemetry, publish commands). New node required.

**2. `ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py`** — Gimbal LOS controller

- Subscribes: `gimbal_cmd_los_rate` (Vector3: x=az_rate, y=el_rate rad/s), `gimbal_cmd_los_world_deg` (Vector3: position targets), `mavros/imu/data` (Imu), `isaac_joint_states` (JointState, depth=50), `zoom_rate_cmd`, `zoom_level_set`, `/clock`.
- Publishes: `isaac_joint_commands` (JointState), `gimbal_state_rpy_deg`, `gimbal_los_state_deg`, `combined_ang_vel_w` (Vector3Stamped), `camera/zoom_level`, `camera/zoom_level_cmd`.
- Clock: subscribes `/clock` (best-effort QoS). Timestamp deduplication via `_prev_time`.
- Control modes: (a) Rate mode — Jacobian-inverse IK from integrated LOS rate. (b) Position mode — analytical IK from world az/el.
- Conventions: LOS positive elevation = up (ENU). Joint angles RPY = ENU.
- Namespacing: `multi_agent_los_rate.launch.py` creates per-vehicle namespaced nodes.

**3. `controller/tuning/auto_tune.py`** — Parallel controller auto-tuner

- Creates gym env, instantiates real DroneController per parallel env.
- Tests: velocity step (5 m/s, 10 m/s, 5s each), attitude recovery (20° perturbation, 3s), hover drift.
- Physics dt = 0.01s (100 Hz). Calls `controller.step_policy()` each step.
- Histories dict per test: `vel_history` (T,), `att_error_history` (T,3), `rate_error_history` (T,3) — torch tensors.
- **Raw timeseries NOT saved** — only passed to plotter, then discarded.
- Output: `tuning_results_{ts}.json` (metrics + params only), `best_config_{ts}.py`, PDFs `step_responses/trial_{idx:04d}.pdf`.

**4. `controller/tuning/step_response_plotter.py`** — Step response visualizer

- `plot_trial()` expects histories dict with keys `vel_5`, `vel_10`, `att`.
- 3×2 panel PDF: velocity responses, attitude errors, rate errors.
- Time axis: `t = np.arange(len(data)) * dt`.
- Annotates with oscillation metrics.

**5. `PegasusSimulator/launch/px4_multi_world_iris_gimbal3.isaac.py`** — Sim launcher

- 5 drones, model IrisGimbal3. Namespace: `px4_1` through `px4_5`.
- Per-drone: PX4MavlinkBackend (px4_autolaunch=True) + ROS2Backend.
- Publishes: `/clock`, `{ns}/isaac_joint_states`, `{ns}/clock`, sensor topics.
- Subscribes: `{ns}/isaac_joint_commands`.

**6. MAVROS configuration** — `config/mavros/mavros_param_px4_1.yaml`

- FCU URL: `udp://:14541@localhost:14551`, `tgt_system: 2`.
- Plugin denylist: `odometry`.
- MAVROS namespace (from tmuxp): `/px4_1/mavros`.
- Standard MAVROS topics: `mavros/setpoint_velocity/cmd_vel_unstamped` (Twist, ENU), `mavros/cmd/arming` (service), `mavros/set_mode` (service), `mavros/local_position/odom`, `mavros/imu/data`.

### Data models

| Data | Source | Format | Persistence |
|------|--------|--------|-------------|
| Controller metrics | auto_tune.py | TuningMetrics dataclass → JSON | `tuning_results_*.json` |
| Step response timeseries | auto_tune.py | torch tensors (T,) / (T,3) | **Not saved** — PDF only |
| Tuned gains | tuning_results/__init__.py | TUNED_CONTROLLER_CFG | Python source |
| PX4 telemetry | MAVROS | Odometry, Imu (ENU/FLU) | **Not recorded** |
| Gimbal state | los_rate_controller.py | Vector3 (degrees) | **Not recorded** |
| Sim clock | PegasusSimulator | `/clock` (rosgraph_msgs/Clock) | Not persisted |

### Conventions observed

- **Frames**: MAVROS topics in ENU/FLU. PX4 fmu topics in NED/FRD. Gimbal LOS: positive elevation = up. Gimbal RPY: ENU.
- **Quaternion**: ROS2 xyzw, iris_ma6 wxyz.
- **Namespace**: `px4_{id}` for vehicle topics, `px4_{id}/mavros` for MAVROS topics.
- **QoS**: PX4 fmu uses BEST_EFFORT + VOLATILE. Gimbal rate commands: best-effort. Position targets: reliable.
- **auto_tune.py**: Physics dt=0.01s (100 Hz). Velocity tests 5s. Attitude test 3s.

### Gaps and inconsistencies

1. **No data recording**: Neither offboard_py nor gimbal_stabilizer save telemetry. Sysid must implement recording.
2. **No sim clock in offboard_py**: Uses wall-clock. Sysid node must subscribe `/clock`.
3. **Raw timeseries not saved by auto_tune.py**: Must regenerate iris_ma6 reference curves.
4. **MAVROS IMU republishing commented out**: `offboard_control.py` lines ~193, ~209. Gimbal stabilizer needs `mavros/imu/data` — verify PegasusSimulator ROS2Backend provides it, or uncomment.
5. **offboard_control.py wrong direction for sysid**: Subscribes PX4, publishes MAVROS. Sysid needs opposite.
