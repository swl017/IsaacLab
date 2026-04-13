# Stage Q — Questions (Resolved)

Ticket: 023 — Offboard flight test harness + velocity step script

## Resolved Decisions

| # | Question | Answer |
|---|----------|--------|
| 1 | MAVROS-only interface? | Yes, MAVROS only |
| 2 | Single-vehicle scope? | Yes |
| 3 | Offboard mode detection method | MAVROS `/mavros/state` topic |
| 4 | Detection trigger | MAVROS Armed state |
| 5 | Bag recording mechanism | `subprocess.Popen` |
| 6 | CSV schema | Match `sysid_node.py` 26-col format for `sysid_analyze.py` compatibility |
| 7 | Velocity setpoint interface | `cmd_vel` (TwistStamped) |
| 8 | Gimbal during A2 | Hold current world-frame pointing, record only |
| 9 | Safety abort | Switch to POSCTL mode, but do NOT stop recording |
| 10 | Output directory | Single directory (bags + CSVs together) |
| 11 | Harness architecture | (b) Base ROS2 node class for production use. Also provide (a) composition example for learning |
| 12 | Test sequence format | YAML config file (portable to Isaac Lab) |
| 13 | Return-to-origin | Zero velocity (drift OK within 5m radius); position hold if exceeds 5m |
| 14 | Step hold duration | 5s per velocity step, configurable but shared across steps |
| 15 | Settle between steps | Same 5s |
| 16 | Axes | All three axes in one run: vx+/-, vy+/-, vz+/-. Ensure >= 10m altitude margin |
| 17 | Actuator outputs | Not via MAVROS. If uORB/uxrce exposes a ROS2 topic, try it; otherwise defer to `.ulg`. Acceptable since uxrce not used at deploy |
| 18 | Entry point naming | `flight_test_*` series (e.g., `flight_test_velocity_step`) |

## Key Design Constraints (from answers)

- **Base class pattern**: Harness is a ROS2 Node base class; test scripts subclass it. Composition example also provided.
- **YAML-driven sequences**: Test sequences declared in YAML for portability.
- **Drift guard**: Monitor position drift from start; if > 5m, switch to position hold back to origin before continuing.
- **Altitude safety**: Before z-axis steps, verify altitude >= 10m + |max_vz_step| margin.
- **Recording continuity**: On mode loss / abort, switch to POSCTL but keep recording (captures the transition).
