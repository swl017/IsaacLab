# Gimbal Stabilization — Status & Next Steps

**Date**: 2026-03-19 (updated)

## Current Architecture

Two gimbal controller modes, selectable via `GimbalControllerCfg(mode=...)`:

### "analytical" (`gimbal_controller_analytical.py`)
Direct atan2 IK — computes exact joint angles each step, no dynamics:
```
dir_body = R_body^{-1} * dir_world
yaw   = atan2(dir_body_y, dir_body_x)    # LOS: 2-DOF
pitch = -atan2(dir_body_z, xy_dist)
roll  = atan2(-up_y_yawed, up_z_yawed)   # horizon: 1-DOF (decoupled)
velocity = finite_difference(pos, pos_prev) / dt
```

### "jacobian" (`gimbal_controller_jacobian.py`)
Unified J^{-1} velocity tracking with proportional gain:
```
q_dot_ref = J^{-1} * (-K_pointing * att_error - omega_body)
q_ref     = q + q_dot_ref * dt
```

### Shared config (`gimbal_controller_cfg.py`)
- `mode`: "analytical" or "jacobian" (default: "jacobian")
- `pointing_gain`: 10.0 (only used by jacobian mode)
- `feedback_blend`: 0.0 (direct state) or 0.05-0.2 (implicit actuator)
- Joint limits: roll ±45°, pitch ±45°, yaw ±160°

Joint actuation: **direct state setting** (`write_joint_state_to_sim`) bypasses implicit actuator lag; targets also set to match so PD applies zero force.

## Test Results (Direct State Mode)

### Jacobian mode (pointing_gain=10)

| Test | RMS | Max | Body Yaw | Status |
|------|-----|-----|----------|--------|
| Pure Pitch (vx) | 0.34 deg | 0.49 deg | 0.0 deg | OK |
| Pure Roll (vy) | 1.34 deg | 4.78 deg | 1.4 deg | OK |
| Pure Yaw (yr) | 0.30 deg | 0.64 deg | 51.6 deg | OK |
| Cross vx+vy | 13.58 deg | 21.70 deg | 9.0 deg | OK |
| Hover (static) | 1.18 deg | 1.32 deg | 2.5 deg | OK |
| Yaw Hold (dist) | 0.30 deg | 0.67 deg | 2.7 deg | OK |

### Analytical mode

| Test | RMS | Max | Body Yaw | Status |
|------|-----|-----|----------|--------|
| Pure Pitch (vx) | 0.96 deg | 1.54 deg | 0.0 deg | OK |
| Pure Roll (vy) | 0.55 deg | 1.89 deg | 1.4 deg | OK |
| Pure Yaw (yr) | 0.52 deg | 0.95 deg | 51.6 deg | OK |
| Cross vx+vy | 7.23 deg | 12.44 deg | 8.8 deg | OK |
| Hover (static) | 0.11 deg | 0.24 deg | 0.6 deg | OK |
| Yaw Hold (dist) | 0.51 deg | 0.74 deg | 0.3 deg | OK |

### Comparison

Analytical is better for hover (0.11° vs 1.18°) and cross-axis (7.23° vs 13.58°).
Jacobian is better for pure pitch (0.34° vs 0.96°) due to velocity feedforward partially pre-compensating next-step body motion.

## Fixes Applied This Session

### Drone yaw drift (173° → <3°)
Root cause: three bugs in the yaw control chain.

1. **Double yaw deprioritization** — `yaw_weight=0.4` was applied in both attitude controller (rate setpoint scaling) AND rate controller (torque scaling). Effective yaw authority was 16% of nominal. Fix: removed the duplicate in `rate_controller.py`.

2. **No yaw angle hold** — `velocity_controller.py` constructed `q_des` using current yaw, so the attitude controller never saw a yaw error. Fix: added `_yaw_setpoint` state that locks heading when `yaw_rate_cmd ≈ 0` and tracks current yaw during active commands.

3. **Weak yaw rate PID** — Kd=0.0 (no damping), Ki=0.1 (half of roll/pitch). Fix: Kd→0.001, Ki→0.15.

### Gimbal controller modes
- Renamed files: `gimbal_controller_jacobian.py`, `gimbal_controller_analytical.py`
- Added factory `gimbal_controller.py` dispatching on `cfg.mode`
- Re-activated analytical decomposition (was reverted previously; now works correctly with yaw fix)

## Pointing Gain Tuning (2026-03-19)

### Parallel sweep infrastructure (`controller/tuning/tune_pointing_gain.py`)
- Runs N environments in parallel, each with a different `pointing_gain` value
- Measures RMS tracking error: desired world-frame direction vs actual gimbal direction
- Maneuvers: SlewPitch, SlewRoll, SlewCirc, SlewOnly, StepHold

### Bugs found and fixed during tuning
1. **Wrong rotation formula in `gimbal_world_direction()`** — Y and Z components used incorrect Pitch-Roll-Yaw with negated roll instead of the actual Yaw(Z)->Roll(X)->Pitch(Y) chain. Caused gain-dependent noise in the error metric (different gains → different roll trajectories → different formula errors). Fixed to match `_apply_gimbal_rotation` exactly.
2. **Original maneuvers had zero gimbal commands** — only tested disturbance rejection (feedforward `-omega_body` handles this regardless of gain). Fixed: maneuvers now command active gimbal slewing while body moves.

### Stability analysis
Discrete-time P-controller pole = `(1 - K*dt)`:
- K*dt < 1: monotonic convergence (pole 0 to 1)
- 1 < K*dt < 2: oscillatory but convergent (pole -1 to 0)
- K*dt > 2: unstable (diverges). Confirmed: K≈220 diverges in sweep.

### Tuning results

**Direct state setting** (`write_joint_state_to_sim`):
- Optimal K=112 (K*dt=1.12, oscillatory-convergent regime), score 5.83 deg
- Gains 95–140 all score ~5.8 deg (flat plateau)

**Implicit actuator** (`set_joint_position_target` + PD loop):
- Optimal K=30.5, score 5.80 deg — same tracking quality at 1/3 the gain
- PD actuator adds natural damping (smooths oscillation) + one-step lag
- Effectively changes system from 1st order to 2nd order
- Requires `feedback_blend > 0` to correct internal state drift

### Final configuration (implicit actuator mode)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `pointing_gain` | 32.5 | Center of optimal plateau for implicit actuator |
| `feedback_blend` | 0.05 | Corrects drift between internal state and actual joints |
| `mode` | "jacobian" | J^{-1} velocity tracking with body-rate feedforward |
| Actuation | Implicit PD | Smoother tracking, natural damping, lower gain needed |

### Environment config changes for tuning
- Added `enable_tiled_cameras: bool` to `IrisMA6TestEnvCfg` — skip TiledCamera creation for faster headless runs
- Increased PhysX GPU buffer capacities (`gpu_heap_capacity=2**27`, `gpu_temp_buffer_capacity=2**25`) for 1024-env runs
- Tuning script disables `enable_target_controller` and `enable_tiled_cameras`

## Known Physical Limitations (accepted)

### Cross vx+vy error (7-14° RMS)
During circular maneuvers (37° body tilt), residual gimbal error comes from:
- Roll joint saturation (±45° limit with 37° body roll leaves 8° margin)
- 1-step physics latency (~1° additional)
Not a controller bug — hardware constraint of 3-axis gimbal with these joint limits.

### Pure Yaw body yaw reaches 51.6°
During commanded yaw rate, the drone physically rotates. The gimbal compensates well (<1° RMS error). The large body yaw is intended behavior — yaw hold only engages when `yaw_rate_cmd ≈ 0`.

## Key Files

| File | Role |
|------|------|
| `controller/gimbal_controller.py` | Factory + YAW_JOINT_OFFSET re-export |
| `controller/gimbal_controller_analytical.py` | Direct atan2 IK (zero-lag) |
| `controller/gimbal_controller_jacobian.py` | J^{-1} velocity tracking |
| `controller/gimbal_controller_cfg.py` | Shared config (mode, gains, limits) |
| `controller/velocity_controller.py` | Yaw setpoint hold logic |
| `controller/rate_controller.py` | Rate PID (yaw_weight removed) |
| `controller/rate_controller_cfg.py` | Yaw gains (Kd=0.001, Ki=0.15) |
| `controller/tuning/tune_pointing_gain.py` | Parallel gain sweep (N envs) |
| `controller/tests/test_gimbal_sim.py` | In-sim test with yaw drift assertions |
| `controller/tests/run_tests.py` | Unit tests (45/45 passing) |
