# Pegasus Physics Parity — Specification

**Status**: in progress (ticket 040)
**Owner**: iris_ma6
**Last updated**: 2026-05-25
**Companion ticket**: [doc/active/ticket/040-match-pegasus-physics-parameters/ticket.md](active/ticket/040-match-pegasus-physics-parameters/ticket.md)

## 1. Purpose

Make iris_ma6's open-loop rigid-body dynamics match the PegasusSimulator Iris model
when the controller is constructed in `"pegasus"` mode, so sim-to-sim transfer
between iris_ma6 and Pegasus (with PX4 SITL) tests a real plant alignment rather
than a fictional drone. Default ("default") mode is bit-exactly the pre-040
plant — Pegasus-mode is opt-in.

## 2. Source of truth — Pegasus values

All Pegasus values are taken from the in-tree copy at
`/home/usrg/IsaacPX4/PegasusSimulator/` as of 2026-05-25.

### 2.1 Motor / thrust curve (`QuadraticThrustCurve` defaults)

| Quantity | Symbol | Pegasus value | iris_ma6 default | Pegasus-mode override |
|---|---|---|---|---|
| Thrust coefficient | `k_f` | 8.54858e-6 N·s²/rad² | 1.2e-5 | **8.54858e-6** |
| Torque coefficient | `k_m` | 1.0e-6 Nm·s²/rad² | 1.4e-7 | **1.0e-6** |
| Max rotor speed | `omega_max` | 1100 rad/s | 5000 | **1100** |
| Motor lag | `tau_motor` | 0 (instantaneous) | 0.01 s | **1e-4** (effectively zero) |
| Min rotor speed | `omega_min` | 0 rad/s | 50 | **50** (kept; see §4.4) |
| Total `T_max` | — | 41.4 N (T/W ≈ 2.81 at 1.5 kg) | 1200 N (T/W ≈ 81.5) | **41.4 N** |
| Yaw authority ratio | `c = k_m/k_f` | 0.117 | 0.0117 | **0.117** (10× higher) |

### 2.2 Drag

| Quantity | Pegasus | iris_ma6 default | Pegasus-mode override |
|---|---|---|---|
| Drag formula | `F = -diag(d) · v_body` | `F = -½ρC_dA |v| v` (quadratic) | **linear_diag** |
| Drag coefficients | `(0.50, 0.30, 0.00)` (X, Y, Z body) | isotropic ~1.84e-3 | **(0.50, 0.30, 0.00)** |
| Wind / gust | none | Dryden (fidelity ≥ 2) | **off** |
| H-force / blade flap | none | fidelity 3 | **off** |

### 2.3 Mass / gravity / dt

| Quantity | Pegasus | iris_ma6 | Notes |
|---|---|---|---|
| Body mass | 1.5 kg | from USD | sanity-check USD mass at env reset |
| Gravity | 9.8 m/s² (IMU) | 9.81 m/s² | 0.1% gap — out of scope |
| Physics dt | 1/250 typical | 1/100 | out of scope (ticket 040 §A8) |

## 3. Mode-selector contract

Two cfg fields, both backward-compat (defaults preserve pre-040 numerics):

- `MotorDynamicsCfg.model: str ∈ {"default", "pegasus"}` (default: `"default"`).
  When `"pegasus"`, the cfg's `__post_init__` swaps `k_f, k_m, omega_max,
  tau_motor` to the Pegasus values listed in §2.1. `omega_min` and `arm_length`
  are not swapped (see §4.4).
- `AerodynamicsCfg.mode: str ∈ {"default", "pegasus"}` (default: `"default"`).
  When `"pegasus"`, the controller force-sets `drag_model = "linear_diag"` and
  skips the wind / gust / rotor-effects branches regardless of `fidelity_level`.

Selection is per-controller-construction. The env's `iris_ma_env6_test_cfg.py`
exposes a single top-level field:

- `physics_mode: str ∈ {"default", "pegasus"}` (default: `"default"`).
  When `"pegasus"`, the env propagates the mode into
  `drone_controller.motor.model = "pegasus"` and
  `drone_controller.aerodynamics.mode = "pegasus"` at env construction.

## 4. Architectural differences — documented, not eliminated

These structural mismatches remain even with parameters matched. Each is
**out of scope** for this ticket; reference the ticket for rationale.

1. **Force-application topology**. Pegasus applies four per-rotor `(0, 0, F_z)`
   forces at the four rotor link poses; iris_ma6 collapses to a single body
   wrench with algebraic pitch/roll via the X-config mixer. Identical at the
   rigid-body level if `arm_length` and rotor signs match the USD; PhysX-side
   inertia coupling will differ slightly.
2. **PX4 controller hierarchy is not replicated**. iris_ma6 has its own
   velocity → attitude → rate cascade; Pegasus + PX4 SITL has EKF2 +
   `mc_*_control` with integrator dynamics, anti-windup, notch filters. The
   sysid replicator matches closed-loop step responses; gains are not the
   PX4 parameters.
3. **Quaternion convention**. Pegasus = scalar-last `[qx, qy, qz, qw]`;
   iris_ma6 = scalar-first `[qw, qx, qy, qz]`. Bench data and the replicator
   already convert; not changed.
4. **`omega_min = 50`** is kept (not Pegasus's 0). Mixer's thrust-preserving
   clamp ([mixer.py:170-198](../controller/mixer.py#L170-L198)) needs a
   strictly-positive floor; with 0 the `total_headroom` denominator collapses
   in hover cases. Document as a known small bias.
5. **Sensor noise**. Pegasus injects IMU/GPS/baro/mag noise; iris_ma6 reads
   ground-truth state. Out of scope here; would be a DR ticket.
6. **Sim dt 1/100 vs Pegasus 1/250**. With `tau_motor ≈ 0` in Pegasus mode the
   gap shows up as a small ZOH error on rate commands; tolerated for this
   ticket. Follow-up ticket if post-fit residual stays > 10%.
7. **Yaw moment topology**. Pegasus applies a single body `tau_z = Σ k_m·ω²·dir`;
   iris_ma6's mixer derives `tau_z = (k_m/k_f) · (-T1-T2+T3+T4)`. Algebraically
   equivalent under the X-config sign convention; both are aligned after the
   `k_m` swap.

## 5. Domain-randomization knobs (mode-gated)

In `GainRandomizationCfg`:

- `k_f_scale_range: tuple[float, float] = (0.9, 1.1)` — rotor-constant variation.
- `k_m_scale_range: tuple[float, float] = (0.8, 1.2)` — propeller drag (looser).
- `drag_coefs_scale_range: tuple[float, float] = (0.7, 1.3)` — body-drag tolerance.

All three are **no-ops in `mode="default"`**. They activate only when the
controller is constructed with `physics_mode="pegasus"`. The pattern matches
the existing `tau_motor` / `max_zoom_rate` DR write paths in
`drone_controller.randomize_gains`.

## 6. Acceptance — pure-PyTorch contracts

(See ticket 040 §"Acceptance criteria" for the exact thresholds.)

- `mode="default"` rollout matches pre-040 bit-exactly (atol=1e-6, 200 steps).
- `model="pegasus"` motor: `omega_cmd = 1100` for 100 ms returns `omega ≈ 1100`
  within 0.1% (no first-order-lag tail).
- `model="pegasus"` thrust ceiling: `compute_body_wrench` at full throttle
  returns `F_z ≈ 41.4 N` (4 × 8.54858e-6 × 1100² = 41.36 N).
- `drag_model="linear_diag"`, `drag_coefs=(0.5, 0.3, 0.0)`, body velocity
  `(10, 0, 0)`: drag force `(-5, 0, 0) N`.
- USD body mass at env reset = 1.5 ± 1% (warning, not error).

## 7. Calling contract

Stateful invariants for the new Pegasus-mode code paths follow the existing
controller contract — see [controller/CONTEXT.md](../controller/CONTEXT.md):

- `MotorDynamics.step()` is **WRITE** (mutates `_omega`); call once per
  inner-loop substep. With `tau_motor ≈ 1e-4` the lag completes in one step.
- `AerodynamicEffects.compute_forces()` is **WRITE** when
  `fidelity_level >= 2 and mode == "default"` (advances Dryden gust state);
  in `mode == "pegasus"` it is **READ** (no internal state mutation).
- Mode selection is **CONFIG**: read once at controller `__init__`, never
  re-read per step. To switch modes, reconstruct the controller.

## 8. Out of scope (recap)

1. Per-rotor force application topology.
2. PX4 controller-hierarchy port into iris_ma6.
3. Quaternion convention change.
4. Sensor noise (IMU/GPS/baro/mag).
5. Sim dt drop from 1/100 → 1/200.
6. Gravity 9.81 → 9.8.
7. `omega_min` drop 50 → 0.
8. Pegasus-truth CSV capture and replicator re-run — done in a follow-up
   session (ticket 040 step 9), not this slice.
