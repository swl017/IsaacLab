# Target Controller Documentation Index

This directory contains documentation for the physics-based target controller module.

## Documents

| Document | Description |
|----------|-------------|
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | Step-by-step guide for integrating TargetController into an environment |
| [../../doc/target_movement_spec.md](../../doc/target_movement_spec.md) | Full specification document (10 sections) |

## Quick Links

### Getting Started
- **New to the module?** Start with [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Need the full spec?** See [target_movement_spec.md](../../doc/target_movement_spec.md)

### Testing
- **Run tests:** `./isaaclab.sh -p .../target_controller/tests/run_tests.py`
- **Test docs:** [../tests/README.md](../tests/README.md)

## Module Overview

The `TargetController` provides physics-based target movement using the `DroneController` architecture:

```
TargetController
├── target_controller_cfg.py      # Configuration + behavior profiles
├── target_controller.py          # Main controller class
├── behavior_fsm.py               # FSM state management
└── velocity_generators/
    ├── base_generator.py         # Abstract base class
    ├── linear_mode.py            # Random direction changes
    ├── circular_mode.py          # Orbital flight
    ├── approach_mode.py          # Goal-directed approach
    └── evade_mode.py             # Reactive evasion
```

## Key Concepts

### Physics-Based vs Direct Velocity

| Aspect | iris_ma5 (Direct) | iris_ma6 (Physics) |
|--------|-------------------|-------------------|
| Method | `write_root_velocity_to_sim()` | `set_external_force_and_torque()` |
| Dynamics | Instantaneous | Realistic inertia |
| Attitude | Fixed | Tilts to accelerate |

### Velocity Modes

1. **Linear**: Random direction changes (timer-based)
2. **Circular**: Orbital flight around a center point
3. **Approach**: Goal-directed movement toward facility
4. **Evade**: Reactive evasion from interceptors

### Behavior Profiles

| Profile | Speed | Evasion | Description |
|---------|-------|---------|-------------|
| kamikaze | 1.5x | 0% | Fast, direct attack |
| standard | 1.0x | 50% | Balanced behavior |
| evasive | 1.0x | 90% | High evasion priority |
| stealth | 0.6x | 50% | Slow, low-altitude |

### FSM States

- **APPROACH**: Moving toward facility
- **EVADE**: Reactive evasion maneuver
- **DEAD**: Target intercepted
- **BREACH**: Target reached facility

## Curriculum Scaling

| Parameter | Progress 0 | Progress 1 |
|-----------|------------|------------|
| Max speed | 3 m/s | 12 m/s |
| Geofence | 50 m | 200 m |
| Update interval | 6-12 s | 2-4 s |
| Evasion agility | 0 | 1 (after 30%) |

## API Summary

```python
# Create controller
controller = TargetController(
    cfg=TargetControllerCfg(control_dt=0.01),
    mass=1.5,
    gravity=9.81,
    num_envs=4096,
    num_targets=10,
    device="cuda",
)

# Step controller (returns forces/torques)
F_body, tau_body = controller.step(
    current_position=pos,      # [N_env, N_target, 3]
    current_velocity=vel,      # [N_env, N_target, 3]
    current_quat=quat,         # [N_env, N_target, 4]
    current_angular_vel=omega, # [N_env, N_target, 3]
    facility_position=fac,     # [N_env, 3]
    interceptor_positions=int_pos,  # [N_env, N_def, 3]
    interceptor_roles=int_roles,    # [N_env, N_def]
    curriculum_progress=0.5,
    dt=0.01,
)

# Reset environments
controller.reset(env_ids)

# Access state
alive = controller.alive           # [N_env, N_target]
fsm_state = controller.fsm_state   # [N_env, N_target]
```

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-03-17 | 1.0 | Initial implementation |
