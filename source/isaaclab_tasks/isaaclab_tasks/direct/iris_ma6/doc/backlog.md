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

## Per-agent Randomization

Add per-agent randomization layer on top of per-episode, per-env, per-step randomization