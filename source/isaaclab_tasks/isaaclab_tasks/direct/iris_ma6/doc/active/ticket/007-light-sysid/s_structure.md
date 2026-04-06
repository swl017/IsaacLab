## Stage S — Structure Outline

### New files

**`ros2_ws/src/offboard_py/offboard_py/sysid_node.py`** — ROS2 sysid commander + recorder
- `class SysidNode(Node)`
  - `__init__(self)` — params: `vehicle_ns` (str), `output_dir` (str), `control_rate` (float, default 100.0), `settle_duration` (float, default 10.0)
  - Subscribers: `mavros/local_position/odom`, `mavros/imu/data`, `gimbal_los_state_deg`, `gimbal_state_rpy_deg`, `/clock`
  - Publishers: `mavros/setpoint_velocity/cmd_vel_unstamped`, `gimbal_cmd_los_rate`, `gimbal_cmd_los_world_deg`
  - Service clients: `mavros/cmd/arming`, `mavros/set_mode`
  - `_clock_callback(self, msg: Clock)` — update sim time, drive state machine at 100 Hz sim-time rate
  - `_odom_callback(self, msg: Odometry)` — cache latest odom
  - `_imu_callback(self, msg: Imu)` — cache latest IMU
  - `_gimbal_los_callback(self, msg: Vector3)` — cache latest gimbal LOS state (az/el deg)
  - `_gimbal_rpy_callback(self, msg: Vector3)` — cache latest gimbal RPY state (ENU deg)
  - `_run_state_machine(self)` — called at 100 Hz from clock callback; manages test sequence
  - `_arm_and_offboard(self)` — call arming + set_mode services
  - `_publish_velocity(self, vx, vy, vz, yaw_rate)` — publish Twist (ENU)
  - `_publish_gimbal_rate(self, az_rate, el_rate)` — publish Vector3 (rad/s, positive el = up)
  - `_publish_gimbal_los_target(self, az_deg, el_deg)` — publish Vector3 (world deg)
  - `_record_sample(self)` — append current state to active test buffer
  - `_save_test_csv(self, test_name: str)` — write buffer to CSV
- `class TestPhase(Enum)` — INIT, ARM, SETTLE, TEST, DONE
- `class TestSpec` — dataclass: `name` (str), `duration` (float), `command_fn` (Callable), `description` (str)
- `def build_test_sequence() -> list[TestSpec]` — returns the 7 test specs
- `def main()`

**`ros2_ws/src/offboard_py/offboard_py/sysid_analyze.py`** — Offline post-processing (no ROS2)
- `def load_sysid_csv(path: str) -> pd.DataFrame` — load recorded CSV
- `def generate_iris_ma6_reference(test_name: str, tuned_cfg, dt: float, duration: float) -> dict` — re-run DroneController.step_policy() for velocity/yaw tests, return timeseries dict
- `def compute_metrics(timeseries: np.ndarray, target: float, dt: float) -> OscillationMetrics` — settling time, overshoot, damping ratio, zero-crossings, SS amplitude, frequency
- `def compute_gimbal_metrics(los_timeseries: np.ndarray, command: np.ndarray, dt: float) -> dict` — rate tracking lag, overshoot, settling time, stabilization error
- `def plot_comparison(ref_data: dict, sitl_data: pd.DataFrame, test_name: str, output_path: str)` — multi-panel PDF
- `def mismatch_table(ref_metrics: dict, sitl_metrics: dict, threshold: float) -> dict` — per-metric % deviation, pass/fail
- `def main()` — argparse: `--sysid-dir`, `--output-dir`, `--threshold` (default 0.2)

**`tmux/isaac_sysid.tmuxp.yaml`** — Launch MAVROS + sysid node
- Window 1: MAVROS node for px4_1 (use_sim_time:=true, params from config/mavros/mavros_param_px4_1.yaml)
- Window 2: sysid_node with vehicle_ns=px4_1

### Modified files

**`ros2_ws/src/offboard_py/setup.py`**
- [add] entry point: `sysid_node` → `offboard_py.sysid_node:main`
- [add] entry point: `sysid_analyze` → `offboard_py.sysid_analyze:main`

**`ros2_ws/src/offboard_py/package.xml`**
- [add] dependency: `mavros_msgs`

### Unchanged files (referenced, not modified)

- `offboard_control.py` — no changes
- `los_rate_controller.py` — no changes
- `auto_tune.py` — no changes (metric definitions reused by reimplementation)
- `step_response_plotter.py` — no changes (style reference only)
