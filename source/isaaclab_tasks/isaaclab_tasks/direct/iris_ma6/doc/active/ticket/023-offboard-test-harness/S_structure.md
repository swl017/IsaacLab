# Stage S — Structure Outline

Ticket: 023 — Offboard flight test harness + velocity step script

---

## New Files

### `offboard_py/offboard_py/flight_test_harness.py` — Base class (pattern b)

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List
import subprocess

from rclpy.node import Node
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import SetMode
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped, Vector3
from mavros_msgs.msg import AttitudeTarget
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile

class HarnessPhase(Enum):
    WAIT_OFFBOARD = auto()
    SETTLE = auto()
    TEST = auto()
    DONE = auto()
    ABORT = auto()

CSV_HEADER: list[str]  # 31-column header (27 from sysid + 4 vel setpoint)

QOS_BEST_EFFORT: QoSProfile
QOS_RELIABLE: QoSProfile

class FlightTestHarness(Node):
    """Base ROS2 node for offboard flight tests."""

    def __init__(self, node_name: str, test_name: str) -> None: ...

    # --- ROS2 setup (called from __init__) ---
    def _setup_subscribers(self) -> None: ...
    def _setup_publishers(self) -> None: ...
    def _setup_services(self) -> None: ...
    def _setup_timer(self) -> None: ...

    # --- Subscriber callbacks ---
    def _mavros_state_cb(self, msg: MavrosState) -> None: ...
    def _odom_cb(self, msg: Odometry) -> None: ...
    def _imu_cb(self, msg: Imu) -> None: ...
    def _att_target_cb(self, msg: AttitudeTarget) -> None: ...
    def _gimbal_los_cb(self, msg: Vector3) -> None: ...
    def _gimbal_rpy_cb(self, msg: Vector3) -> None: ...
    def _clock_cb(self, msg: Clock) -> None: ...

    # --- Command publishing ---
    def _publish_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None: ...
    def _publish_zero_velocity(self) -> None: ...

    # --- State machine ---
    def _timer_cb(self) -> None: ...
    def _state_wait_offboard(self) -> None: ...
    def _state_settle(self) -> None: ...
    def _state_test(self) -> None: ...
    def _state_done(self) -> None: ...
    def _state_abort(self) -> None: ...

    # --- Drift guard ---
    def _capture_origin(self) -> None: ...
    def _compute_drift_correction(self) -> Tuple[float, float, float]: ...
    def _drift_exceeds_limit(self) -> bool: ...

    # --- Safety ---
    def _request_posctl(self) -> None: ...
    def _check_altitude_margin(self, max_vz: float) -> bool: ...

    # --- Recording: CSV ---
    def _record_sample(self, vel_sp: Tuple[float, float, float, float]) -> None: ...
    def _save_csv(self, filename: str) -> None: ...

    # --- Recording: ROS2 bag ---
    def _start_bag_recording(self) -> None: ...
    def _stop_bag_recording(self) -> None: ...

    # --- Lifecycle ---
    def destroy_node(self) -> None: ...  # override to clean up bag subprocess

    # --- Hooks for subclasses ---
    def _on_offboard_entered(self) -> None: ...
    def _on_offboard_lost(self) -> None: ...
    def _get_current_command(self) -> Tuple[float, float, float, float]: ...
    def _is_test_complete(self) -> bool: ...
    def _on_test_complete(self) -> None: ...
```

### `offboard_py/offboard_py/flight_test_velocity_step.py` — A2 subclass

```python
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class VelocityStep:
    axis: str          # "x", "y", "z"
    speed: float       # m/s (signed)
    duration: float    # seconds

class FlightTestVelocityStep(FlightTestHarness):
    """Velocity step response test (A2)."""

    def __init__(self) -> None: ...

    # --- YAML loading ---
    def _load_config(self, config_path: str) -> None: ...
    def _expand_sequence(self) -> List[VelocityStep]: ...

    # --- Harness hooks ---
    def _on_offboard_entered(self) -> None: ...
    def _on_offboard_lost(self) -> None: ...
    def _get_current_command(self) -> Tuple[float, float, float, float]: ...
    def _is_test_complete(self) -> bool: ...
    def _on_test_complete(self) -> None: ...

def main(args=None) -> None: ...
```

### `offboard_py/offboard_py/flight_test_harness_composition_example.py` — Pattern (a) example

```python
class FlightTestHarnessComposition:
    """Composition-based harness (example only, not an entry point).

    Demonstrates how the same harness functionality works without
    inheritance. The caller's Node creates timers/subscribers and
    passes them through.
    """

    def __init__(self, node: Node, test_name: str) -> None: ...

    def setup(self) -> None: ...       # node.create_timer / create_subscription
    def timer_cb(self) -> None: ...    # called by the node's timer
    def get_phase(self) -> HarnessPhase: ...
    def record_sample(self, vel_sp: Tuple[float, float, float, float]) -> None: ...
    def save_csv(self, filename: str) -> None: ...
    def start_bag(self) -> None: ...
    def stop_bag(self) -> None: ...
    def destroy(self) -> None: ...     # cleanup bag subprocess
```

### `offboard_py/config/velocity_step.yaml` — Default config

```yaml
test_name: velocity_step
axes: [x, y, z]
speeds: [3.0, 5.0]
hold_duration: 5.0
settle_duration: 5.0
altitude_margin: 10.0
drift_limit: 5.0
drift_return_threshold: 1.0
drift_kp: 1.0
drift_vel_clamp: 2.0
```

---

## Modified Files

### `offboard_py/setup.py`

```python
# [add] entry point in console_scripts:
'flight_test_velocity_step = offboard_py.flight_test_velocity_step:main',
```

### `offboard_py/doc/active/feature_list.json`

```json
// [add] new feature entry:
{
  "id": "flight_test_harness",
  "name": "Reusable offboard flight test harness",
  "status": "in-progress",
  "notes": "Base class + velocity step (A2). MAVROS-only, YAML-driven sequences.",
  "last_updated": "2026-04-13"
}
```

---

## Not Modified

- `offboard_py/offboard_py/sysid_node.py` — unchanged
- `offboard_py/offboard_py/sysid_analyze.py` — compatibility deferred
- `offboard_py/package.xml` — no new ROS2 dependencies needed
