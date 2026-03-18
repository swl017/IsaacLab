# Target Controller Module

## Purpose
Physics-based target movement with a behavior finite state machine.
Generates realistic evasive and patrol trajectories for surveillance targets,
with difficulty scaling via curriculum.

## Inputs
- Current target state: position `(N*T, 3)`, velocity `(N*T, 3)`, orientation `(N*T, 4)`
- Facility/waypoint position
- Interceptor (agent) positions and velocities
- Curriculum progress: `float` in [0.0, 1.0]
- Simulation timestep `dt`

## Outputs
- Body forces: `(N*T, 3)` applied to target rigid bodies
- Body torques: `(N*T, 3)` applied to target rigid bodies

## Dependencies
- `controller/` - Reuses `DroneController` for physics-based actuation

## Key Files
- `target_controller.py` - Top-level controller interface
- `behavior_fsm.py` - Finite state machine (approach, evade, linear, circular)
- `velocity_generators/` - Per-behavior velocity command generators
- `target_controller_cfg.py` - Configuration (speed limits, behavior weights)
- `tests/` - Unit tests
- `doc/target_movement_spec.md` - Specification document

## Spec
`doc/target_movement_spec.md`
