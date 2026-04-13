# Stage I — Design Document: Flight Test Harness + Velocity Step Script

## Problem Statement

All offboard flight tests (velocity step, attitude step, yaw rate, etc.) share identical infrastructure: wait for the pilot to switch to offboard mode, start recording, execute a command sequence, handle abort. Today `sysid_node.py` bundles this infrastructure with hardcoded tests and auto-arms the vehicle — unsafe for real hardware. A reusable base class extracts the shared infrastructure so each test script is a thin subclass declaring only its command sequence.

## Proposed Approach

The harness is a ROS2 Node base class (`FlightTestHarness`) that owns the full lifecycle: MAVROS subscriptions, wallclock timer, state machine, CSV recording, and ROS2 bag management. Test scripts subclass it and supply a YAML config file describing the command sequence. The first consumer is `flight_test_velocity_step.py` (checklist item A2).

The state machine differs from `sysid_node.py` in one fundamental way: the harness never arms or requests offboard mode. It sits in a WAIT_OFFBOARD state until it observes `mavros/state.mode == "OFFBOARD"` and `mavros/state.armed == True`. The pilot controls arming, takeoff, and mode switching. On mode loss (pilot switches away), the harness publishes zero velocity and calls `SetMode("POSCTL")` as a safety net, but keeps recording to capture the transition.

### Recording

Two parallel recording streams:

1. **CSV** — Same 27-column schema as `sysid_node.py` plus 4 velocity setpoint columns (`vel_sp_x`, `vel_sp_y`, `vel_sp_z`, `yaw_rate_sp`) for a total of 31 columns. Compatible with `sysid_analyze.py` (extra columns are ignored by existing analysis code that indexes by header name).

2. **ROS2 bag** — Started via `subprocess.Popen(["ros2", "bag", "record", ...])` when offboard mode is detected, stopped on test completion or shutdown. Auto-named: `{test_name}_{timestamp}`. Records a curated topic list (odom, imu, state, cmd_vel, attitude target, gimbal state).

### Drift Guard

During settle phases (zero-velocity hold between steps), the harness monitors position drift from a captured "origin" (position at offboard entry). If drift exceeds 5m, it switches from zero-velocity publishing to a simple P-controller on position error (`vel_cmd = Kp * (origin - current_pos)`, clamped to 2 m/s) until within 1m of origin, then resumes zero velocity. This avoids needing MAVROS position setpoints — stays within the velocity-only offboard interface.

### Altitude Safety

Before executing z-axis steps, the harness checks `current_altitude >= 10 + abs(max_vz_in_sequence)`. If insufficient, it logs a warning and skips the z-axis portion.

### Test Sequence Format (YAML)

```yaml
test_name: velocity_step
axes: [x, y, z]           # which axes to test
speeds: [3.0, 5.0]        # m/s magnitudes
hold_duration: 5.0         # seconds at each velocity
settle_duration: 5.0       # seconds at zero between steps
```

The harness expands this into the full sequence: for each axis, for each speed: `0 → +v → 0 → -v → 0`. Axes execute sequentially (all x speeds, then all y, then all z).

### Architecture Pattern (b) — Base Class (production)

```
FlightTestHarness(Node)          # Base class: state machine, recording, safety
    └── FlightTestVelocityStep   # Subclass: loads YAML, builds step sequence
```

The base class provides:
- `_on_offboard_entered()` — hook for subclass (start sequence)
- `_on_offboard_lost()` — hook for subclass (abort sequence)
- `_get_current_command() -> (vx, vy, vz, yaw_rate)` — called each timer tick during TEST phase
- `_is_test_complete() -> bool` — subclass signals completion

The subclass implements a simple index into its expanded step list, advancing based on elapsed time.

### Architecture Pattern (a) — Composition (example only)

A separate `FlightTestHarnessComposition` class (not a Node) that accepts a Node instance in its constructor. The Node must pass through `create_timer`, `create_subscription`, `create_publisher`, and `create_client` calls. Provided as `flight_test_harness_composition_example.py` — not an entry point, just a reference showing how the same functionality works without inheritance.

## Key Interfaces and Data Flow

```
                    ┌──────────────────────────────────┐
                    │     FlightTestHarness (Node)      │
                    │                                    │
  mavros/state ────►│  _mavros_state_cb ──► _phase      │
  mavros/odom  ────►│  _odom_cb ──► _odom (cached)      │
  mavros/imu   ────►│  _imu_cb ──► _imu (cached)        │
  att_target   ────►│  _att_target_cb ──► _att_target    │
  gimbal_*     ────►│  _gimbal_*_cb ──► _gimbal_*       │
  /clock       ────►│  _clock_cb ──► _sim_time           │
                    │                                    │
                    │  _timer_cb (100Hz wallclock)       │
                    │    ├─ WAIT_OFFBOARD: poll state    │
                    │    ├─ SETTLE: drift guard + hold   │
                    │    ├─ TEST: _get_current_command() │
                    │    └─ DONE: zero vel               │
                    │                                    │
                    │──► cmd_vel (TwistStamped)          │
                    │──► CSV buffer                      │
                    │──► bag subprocess                  │
                    └──────────────────────────────────┘
```

### State Machine

```
WAIT_OFFBOARD ─(armed && mode=="OFFBOARD")──► SETTLE (initial stabilize)
SETTLE ─(elapsed >= settle_duration)────────► TEST
TEST ─(step complete)──────────────────────► SETTLE (next step)
TEST ─(all steps done)────────────────────► DONE
SETTLE/TEST ─(mode != "OFFBOARD")─────────► ABORT (zero vel, SetMode POSCTL, keep recording)
ABORT ─(mode == "OFFBOARD" again)──────────► SETTLE (resume, but do NOT re-run completed steps)
```

### Files

| File | Action | Purpose |
|------|--------|---------|
| `offboard_py/flight_test_harness.py` | NEW | Base class |
| `offboard_py/flight_test_velocity_step.py` | NEW | A2 velocity step subclass |
| `offboard_py/flight_test_harness_composition_example.py` | NEW | Pattern (a) example, not an entry point |
| `config/velocity_step.yaml` | NEW | Default velocity step config |
| `setup.py` | MODIFY | Add `flight_test_velocity_step` entry point |
| `doc/active/feature_list.json` | MODIFY | Add feature entry |

## What This Does NOT Include

- Attitude step (A3), yaw rate (A4), or other test scripts (ticket-024)
- Changes to existing `sysid_node.py`
- Auto-arming or auto-takeoff
- uxrce/uORB actuator output recording (deferred; use `.ulg` for now)
- Position setpoint mode — drift guard uses velocity control with P-feedback

## Open Risks

1. **`SetMode("POSCTL")` service call timing**: If MAVROS service is slow or fails, the pilot is the backup. The harness publishes zero velocity regardless of service response.

2. **Bag subprocess cleanup**: If the node crashes, the bag subprocess may orphan. The harness should register a `finally`/`atexit` handler to terminate it.

3. **Drift guard P-controller tuning**: `Kp` and clamp values are initial guesses. May need tuning in SITL before real flight.

4. **`sysid_analyze.py` compatibility**: Extra CSV columns (vel_sp_*) added at the end. Existing analysis code that reads by column index (not name) would break. Need to verify `sysid_analyze.py` reads by header name.
