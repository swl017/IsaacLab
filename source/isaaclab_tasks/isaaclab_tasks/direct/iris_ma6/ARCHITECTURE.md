# iris_ma6 Architecture

Multi-agent drone observation environment for cooperative target tracking with triangulation.

## Module Dependency Graph

```
┌──────────────────────────────────────────────────────────┐
│            iris_ma_env6_test (Main Environment)          │
│            Orchestrates all modules below                │
└──────────────────┬───────────────────────────────────────┘
                   │ imports & calls
    ┌──────────────┼──────────────────────────────────┐
    │              │              │                    │
    ▼              ▼              ▼                    ▼
controller    delay_system_v3  cbf_safety       triangulation
    │
    │
    ▼              ▼              ▼                    ▼
target_      bbox_raycaster   initial_states    domain_
controller       _v2                            randomization
(uses controller)
                   ▼              ▼
              visualization    curriculum
              (debug only)     (config only)
```

## Directed Dependencies

Only one cross-module dependency exists outside the environment:

```
target_controller ──→ controller    (reuses DroneController)
```

All other modules are standalone. The environment (`iris_ma_env6_test.py`) is the sole integration point.

## Data Flow Per Step

```
Policy Output (7D action per agent: vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate)
       │
       ▼
_pre_physics_step()
  ├─ Scale actions to physical units
  ├─ CBF deploy filter (if enabled) → safe velocities
  └─ DroneController.step() → forces/torques + gimbal targets + zoom
       │
       ▼
_physics_step()
  ├─ Apply forces/torques to agent articulations
  ├─ TargetController.step() → target forces/torques
  └─ Simulation substeps
       │
       ▼
_post_physics_step()
  ├─ Read ground-truth state from simulation
  ├─ DelaySystem.update() → delayed AgentStates
  ├─ BBoxRayCasterV2.update() → 2D bboxes + occlusion
  └─ Triangulation on GT and delayed camera rays
       │
       ▼
_get_observations()                    _get_rewards()
  ├─ Delayed states (or GT)              ├─ Triangulation covariance quality
  ├─ Normalized bboxes                   ├─ CBF collision penalty
  └─ Per-agent obs vector                ├─ Bbox center/size rewards
                                         └─ Action regularization
       │
       ▼
_get_dones()
  ├─ Collision check (CBF threshold)
  ├─ Episode length timeout
  └─ NaN/Inf termination

_reset_idx(env_ids)
  ├─ InitialStates.generate(curriculum_progress)
  ├─ DomainRandomization.sample()
  └─ Reset delay system + reward accumulators
```

## Key Data Containers

| Container | Module | Shape | Description |
|-----------|--------|-------|-------------|
| AgentStates | delay_system_v3 | per-agent | Position, velocity, orientation, gimbal, camera intrinsics, timestamps |
| TriangulationResult | triangulation | (N, T, 3/3x3) | Estimated position, covariance, validity |
| BBoxRayCasterV2Data | bbox_raycaster_v2 | (N, C, T, 4) | 2D bboxes, occlusion flags |
| InitialStatesResult | initial_states | (N, A, 3) | Randomized starting configurations |

## Module Isolation

**Standalone** (no iris_ma6 internal dependencies):
controller, delay_system_v3, cbf_safety, triangulation, bbox_raycaster_v2,
initial_states, domain_randomization, visualization, curriculum, asset

**Has dependencies**:
target_controller → controller

## File Conventions

- `*_cfg.py` — Configuration dataclasses
- `CONTEXT.md` — Module routing contract (inputs, outputs, dependencies)
- `tests/` — Per-module test suites
- `doc/*_spec.md` — Authoritative specifications
