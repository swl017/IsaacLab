# Stage R — Codebase Research Report

Ticket: 023 — Offboard flight test harness + velocity step script

---

## Module Inventory

### offboard_py package (`/home/usrg/IsaacPX4/ros2_ws/src/offboard_py/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `offboard_control.py` | ~500 | PX4 uORB bridge (NED→ENU), state machine (ARM→TAKEOFF→OFFBOARD) |
| `thrust_and_rate_control_ros2.py` | ~250 | Attitude+thrust controller at 250 Hz via MAVROS |
| `stabilizer.py` | ~250 | `DroneStabilizingController` with PID gains |
| `sysid_node.py` | 656 | System ID test runner: 7-test sequence, CSV recording, auto-arm |
| `sysid_analyze.py` | ~800 | Post-processing of sysid CSV files |
| `rotations.py` | ~40 | NED/ENU and FRD/FLU rotation constants (no functions) |
| `px4/position_control.py` | ~450 | Python port of PX4 position controller |
| `px4/states.py` | ~200 | `State` dataclass with ENU↔NED conversion methods |
| `px4/numpy_transforms.py` | ~80 | Transform utilities |
| `latency_tools/` | 4 scripts | imu, detector, image_transport, datalink latency benchmarks |

### Entry Points (setup.py)

```
offboard_control    → offboard_py.offboard_control:main
thrust_and_rate_control → offboard_py.thrust_and_rate_control_ros2:main
sysid_node          → offboard_py.sysid_node:main
sysid_analyze       → offboard_py.sysid_analyze:main
imu_latency         → offboard_py.latency_tools.imu_latency:main
detector_latency_bench → offboard_py.latency_tools.detector_latency_bench:main
image_transport_latency → offboard_py.latency_tools.image_transport_latency:main
datalink_latency_ping → offboard_py.latency_tools.datalink_latency_ping:main
```

### Package Sub-packages Registered in setup.py

```python
packages=[package_name, package_name + '.latency_tools']
```

No `.px4` sub-package registered (works via implicit namespace or direct import).

---

## sysid_node.py — Detailed Anatomy

### Enums

```python
class TestPhase(Enum):
    INIT = auto()       # wait for MAVROS topics
    RAMP_UP = auto()    # stream 15 zero-velocity ticks (PX4 requirement)
    ARM = auto()        # request OFFBOARD + arm via services
    TAKEOFF = auto()    # climb at 3 m/s to target altitude
    SETTLE = auto()     # hover settle_duration between tests
    TEST = auto()       # run command function + record
    DONE = auto()       # publish zero velocity
```

### Dataclass

```python
@dataclass
class TestSpec:
    name: str
    duration: float          # seconds
    description: str
    command_fn: Callable[["SysidNode", float], None]
```

### QoS Profiles (module level)

```python
QOS_BEST_EFFORT = QoSProfile(reliability=BEST_EFFORT, durability=VOLATILE, history=KEEP_LAST, depth=1)
QOS_RELIABLE    = QoSProfile(reliability=RELIABLE,    durability=VOLATILE, history=KEEP_LAST, depth=10)
```

### ROS2 Subscriptions

| Topic | Msg Type | QoS | Callback |
|-------|----------|-----|----------|
| `/{ns}/mavros/state` | `MavrosState` | BEST_EFFORT | `_mavros_state_cb` |
| `/{ns}/mavros/local_position/odom` | `Odometry` | BEST_EFFORT | `_odom_cb` |
| `/{ns}/mavros/imu/data` | `Imu` | BEST_EFFORT | `_imu_cb` |
| `/{ns}/gimbal_los_state_deg` | `Vector3` | BEST_EFFORT | `_gimbal_los_cb` |
| `/{ns}/gimbal_state_rpy_deg` | `Vector3` | BEST_EFFORT | `_gimbal_rpy_cb` |
| `/clock` | `Clock` | BEST_EFFORT | `_clock_cb` |
| `/{ns}/mavros/setpoint_raw/target_attitude` | `AttitudeTarget` | BEST_EFFORT | `_att_target_cb` |

### ROS2 Publishers

| Topic | Msg Type | QoS |
|-------|----------|-----|
| `/{ns}/mavros/setpoint_velocity/cmd_vel` | `TwistStamped` | RELIABLE |
| `/{ns}/gimbal_cmd_los_rate` | `Vector3` | BEST_EFFORT |
| `/{ns}/gimbal_cmd_los_world_deg` | `Vector3` | RELIABLE |

### ROS2 Service Clients

| Service | Type |
|---------|------|
| `/{ns}/mavros/cmd/arming` | `CommandBool` |
| `/{ns}/mavros/set_mode` | `SetMode` |

### ROS2 Parameters

| Parameter | Default | Type |
|-----------|---------|------|
| `vehicle_ns` | `"px4_1"` | str |
| `output_dir` | `"/tmp/sysid"` | str |
| `update_rate` | `100.0` | float |
| `settle_duration` | `10.0` | float |
| `takeoff_altitude` | `15.0` | float |
| `takeoff_speed` | `3.0` | float |

### Timer

- Period: `1.0 / update_rate` (default 10 ms)
- Callback: `_timer_cb` → dispatches to `_state_*` based on `_phase`

### State Machine Transitions

```
INIT ─(mavros_state!=None && odom!=None)──→ RAMP_UP
RAMP_UP ─(ticks >= 15)────────────────────→ ARM
ARM ─(armed && mode=="OFFBOARD")──────────→ TAKEOFF
TAKEOFF ─(pos.z >= takeoff_altitude)──────→ SETTLE
SETTLE ─(elapsed >= settle_duration)──────→ TEST or DONE (if all tests run)
TEST ─(t >= test.duration)────────────────→ SETTLE
```

### CSV Schema (27 columns)

```
timestamp_s,
pos_x, pos_y, pos_z,                          # Odometry.pose.pose.position
vel_x, vel_y, vel_z,                          # Odometry.twist.twist.linear
quat_x, quat_y, quat_z, quat_w,              # Odometry.pose.pose.orientation
ang_vel_x, ang_vel_y, ang_vel_z,              # Imu.angular_velocity
gimbal_los_az_deg, gimbal_los_el_deg,         # gimbal_los_state_deg
gimbal_rpy_r_deg, gimbal_rpy_p_deg, gimbal_rpy_y_deg,  # gimbal_state_rpy_deg
att_sp_qx, att_sp_qy, att_sp_qz, att_sp_qw, # AttitudeTarget.orientation
rate_sp_x, rate_sp_y, rate_sp_z,             # AttitudeTarget.body_rate
thrust_sp                                     # AttitudeTarget.thrust
```

Missing values use `0.0` when topic not yet received.

### Test Sequence (hardcoded in `build_test_sequence()`)

| Test | Duration | Velocity (ENU) | Notes |
|------|----------|----------------|-------|
| `hover_drift` | 10s | (0, 0, 0) | Baseline |
| `vel_step_5` | 10s | (5, 0, 0) | Forward step |
| `vel_step_10` | 10s | (10, 0, 0) | Forward step |
| `vel_impulse_recovery` | 10s | 5 m/s for 1s then 0 | Impulse response |
| `yaw_step` | 10s | yaw_rate=0.5 for 2s then 0 | Yaw dynamics |
| `gimbal_los_rate_step` | 2s | Gimbal rate pulses (1 rad/s) | Gimbal dynamics |
| `gimbal_los_stabilization` | 7s | Drone shaking sequence | Gimbal rejection |

### Recording Pattern

```python
def _record_sample(self):
    # Builds 27-element row from cached topic data
    self._record_buffer.append(row)

def _save_test_csv(self, test_name):
    # Writes _record_buffer to {output_dir}/{test_name}.csv
    # Clears _record_buffer after save
```

### Arming Pattern

```python
def _request_offboard_and_arm(self):
    # 1. If not OFFBOARD mode: call SetMode("OFFBOARD")
    # 2. If not armed: call CommandBool(True)
    # Rate-limited: skips if < 50 ticks since last attempt
```

### main()

```python
def main(args=None):
    rclpy.init(args=args)
    node = SysidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        ...
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

---

## Conventions Observed

### Timer Pattern
- All nodes use wallclock timers (`create_timer`), not `/clock`-driven callbacks.
- `/clock` subscribed only for sim-time CSV timestamps.
- Decoupled pattern: subscribers cache latest message, timer reads cached values.

### Frame Convention
- MAVROS topics are ENU-FLU (MAVROS handles NED conversion).
- `offboard_control.py` uses raw px4_msgs in NED, converts manually.
- `sysid_node.py` uses MAVROS exclusively — all data already ENU-FLU.

### QoS Convention
- Subscribers: BEST_EFFORT for all sensor/state topics.
- Publishers: RELIABLE for commands, BEST_EFFORT for gimbal rate.

### Namespace Convention
- Single parameter `vehicle_ns` (default `"px4_1"`).
- MAVROS topics at `/{vehicle_ns}/mavros/...`.
- Gimbal topics at `/{vehicle_ns}/gimbal_*`.

### Node Naming
- `sysid_node.py`: node name = `"sysid_node"`, hardcoded.
- `offboard_control.py`: node name = `"offboard_control"`.

### CSV Convention
- One CSV per test, named `{test_name}.csv`.
- Buffer accumulates during test, flushed to disk at test end.
- Output directory created via `os.makedirs(exist_ok=True)`.

### Parameter Convention
- Declared via `self.declare_parameter(name, default)`.
- Retrieved via `self.get_parameter(name).get_parameter_value().string_value` (or `.double_value`).

---

## Data Models

### MavrosState Fields Used
- `.armed: bool` — vehicle arm state
- `.mode: str` — flight mode string (e.g., `"OFFBOARD"`, `"POSCTL"`)

### Odometry Fields Used
- `.pose.pose.position.{x,y,z}` — ENU position
- `.pose.pose.orientation.{x,y,z,w}` — ENU-FLU quaternion
- `.twist.twist.linear.{x,y,z}` — ENU velocity

### AttitudeTarget Fields Used
- `.orientation.{x,y,z,w}` — attitude setpoint quaternion
- `.body_rate.{x,y,z}` — rate setpoint (rad/s, body FLU)
- `.thrust` — normalized thrust (0–1)

### TwistStamped Publishing Pattern
```python
msg = TwistStamped()
msg.header.stamp = self.get_clock().now().to_msg()
msg.twist.linear.x = vx   # ENU forward
msg.twist.linear.y = vy   # ENU left
msg.twist.linear.z = vz   # ENU up
msg.twist.angular.z = yaw_rate
self._vel_pub.publish(msg)
```

---

## Gaps and Inconsistencies

1. **Two state detection mechanisms exist**: `offboard_control.py` uses px4_msgs `VehicleStatus` (uORB/uxrce); `sysid_node.py` uses `mavros_msgs/State`. These are different ROS2 interfaces to the same PX4 state.
   - **Resolution**: Use MAVROS only for the harness, EXCEPT if the actuator values are available only in uxrce.

2. **No existing node waits for external mode switch**: `sysid_node.py` actively requests OFFBOARD + ARM. `offboard_control.py` also requests arming and mode change.

3. **`setup.py` packages list**: `latency_tools` sub-package is registered, but `px4` sub-package is not. `px4` modules are imported directly (e.g., `from offboard_py.px4.states import State`).
   - **Resolution**: Add `px4` sub-package to setup.py only if needed later.

4. **No YAML-based test configuration**: All test sequences in `sysid_node.py` are hardcoded in `build_test_sequence()`.

5. **No bag recording infrastructure**: No existing code triggers `ros2 bag record`.

6. **No position hold logic**: `sysid_node.py` publishes velocity only. No position feedback to hold a specific location.
   - **Resolution**: Either set position setpoints or velocity control with position feedback (for drift guard).

7. **offboard_controller_spec.md** references a target MAVROS-based interface for `offboard_control.py`, but the actual code still uses px4_msgs/uORB.
   - **Resolution**: MAVROS is the correct interface for the harness.

8. **CSV header constant (CSV_HEADER) has 27 entries**, matching the 27 values appended per row in `_record_sample()`.

9. **Velocity setpoint columns not in CSV**: The CSV records attitude/rate/thrust setpoints (from AttitudeTarget feedback), but not the velocity command that was sent.
   - **Resolution**: Add velocity setpoint columns to CSV schema (the velocity command loop).

10. **No SetMode to POSCTL**: `sysid_node.py` only calls `SetMode("OFFBOARD")`. No code exists to switch to position control mode.
