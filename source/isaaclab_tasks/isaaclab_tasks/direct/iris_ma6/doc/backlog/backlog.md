# iris_ma6 Backlog

This document tracks features and improvements that are planned but not yet implemented.

---

## Visual Propeller Spinning

**Priority:** Low
**Status:** Deferred
**Related files:** `iris_ma_env6_test.py`, `iris_gimbal2.py`, `iris_gimbal2.usda`

### Problem

Visual propeller spinning was attempted but caused drone instability. The approach used `write_joint_state_to_sim()` which overwrites ALL joint state including gimbal joints that are actively controlled via position targets.

### Attempted Solutions

1. **Velocity targets with zero-physics actuators**: Failed - velocity targets don't work with zero stiffness/damping actuators
2. **Direct joint state writing**: Caused instability - `write_joint_state_to_sim()` overwrites gimbal joint state

### Proper Solution

To implement visual-only propeller spinning without physics interference, the USD asset (`iris_gimbal2.usda`) needs to be modified:

1. **Option A: Kinematic propellers**
   - Set `physics:kinematicEnabled = true` on propeller rigid bodies
   - Set `physics:mass = 0` on propeller rigid bodies
   - This allows direct joint position control without physics simulation

2. **Option B: Remove RigidBodyAPI**
   - Remove the `RigidBodyAPI` entirely from propeller links
   - Keep only visual meshes
   - Use USD transform manipulation for rotation

3. **Option C: Separate visual mesh**
   - Create separate visual-only propeller prims
   - Rotate these via USD Xform operations
   - Keep physics propellers static/invisible

### Implementation Notes

- Motor angular velocities are available from `controller.motor_dynamics.omega` (N, 4) [rad/s]
- Propeller directions: M1, M2 spin CCW (+), M3, M4 spin CW (-)
- Update rate should be at policy frequency (25Hz), not physics rate (100Hz)

### References

- Isaac Sim USD documentation on kinematic bodies
- PhysX rigid body properties API

---

## Adaptive Lagrangian for CBF Penalty

**Priority:** Medium
**Status:** Planned
**Related files:** `cbf_safety/cbf_cfg.py`, `cbf_safety/cbf_manager.py`, `iris_ma_env6_test.py`

### Problem

Fixed `lambda_cbf` becomes mismatched as task reward evolves during training. At
different curriculum phases the optimal penalty-to-reward ratio shifts, making a
static weight either too weak (collisions) or too dominant (over-conservative).

### Proposed Solution

Replace fixed `lambda_cbf` with a learned Lagrange multiplier updated via dual
gradient ascent (MAPPO-Lagrangian / MACPO approach):

```python
lambda_cbf += lr_lambda * (collision_cost - threshold)
lambda_cbf = max(lambda_cbf, 0)
```

- Initialize at `lambda_cbf = 1.0`
- `lr_lambda ~ 0.001–0.01` (PID Lagrangian recommends this range, Stooke et al. ICML 2020)
- `threshold = 0` (zero tolerance for violations)

### References

- MACPO / MAPPO-Lagrangian (Gu et al., 2021, arxiv 2110.02793)
- PID Lagrangian (Stooke et al., ICML 2020)
- Barrier Functions Inspired Reward Shaping (arxiv 2403.01410)

---

## Per-agent Randomization

Add per-agent randomization layer on top of per-episode, per-env, per-step randomization

---

## Domain Randomization

**Priority:** Medium
**Status:** Spec Complete
**Spec:** [../domain_randomization_spec.md](../domain_randomization_spec.md)

Apply domain randomization for sim-to-real transfer including:
- Physics properties (mass, friction, scale)
- Camera parameters (focal length, resolution via computational simulation)
- Gimbal mount positions and dynamics

### Key Design Decisions

1. **Camera resolution randomization**: Implemented via computational crop+resize pipeline
   - Render at max resolution (1920x1080 Full HD)
   - Apply symmetric crops for FOV simulation
   - Resize to target resolution (1080p, 720p, 360p)
   - Maintain centered principal points and square pixels (16:9 aspect ratio)

2. **Randomization hierarchy**: Pre-startup → Per-env → Per-episode → Per-agent → Per-step

---

## Adverserial MARL

---

## Role-switching

Observer-Intercepter

---

## Ontology-based Mission Planning

Mission planning and task allocation

---

## Target Search / Reacquisition

**Priority:** Medium
**Status:** Deferred — observe if naturally learned first

### Problem

Currently, agents always reset with body yaw facing the target and gimbal pointed at the target.
This means the policy never needs to *find* a target that is out of frame — it only learns to
*maintain* tracking of a visible target.

In real deployment, targets may leave the FOV due to occlusion, aggressive maneuvers, or
communication-based handoff. The agent needs a search/reacquisition strategy.

### Action Items

1. After training converges with always-pointing resets, evaluate whether the trained policy
   can naturally reacquire a target that leaves the FOV mid-episode (it may learn this from
   the moving-target curriculum alone).
2. If not, introduce radar-like detection system which gives rough estimates of where the target is.