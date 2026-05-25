## Ticket 040 — Match iris_ma6 physics parameters to PegasusSimulator (sim-to-sim parity)

**Status**: Open
**Created**: 2026-05-25
**Target env**: v0 ([iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) + [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py)).
**Affected modules**: [controller/motor_dynamics_cfg.py](../../../../controller/motor_dynamics_cfg.py), [controller/aerodynamics.py](../../../../controller/aerodynamics.py), [controller/aerodynamics_cfg.py](../../../../controller/aerodynamics_cfg.py), [controller/mixer.py](../../../../controller/mixer.py), [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py).
**Source of truth**: [PegasusSimulator/extensions/.../iris.py](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/multirotors/iris.py), [QuadraticThrustCurve](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/thrusters/quadratic_thrust_curve.py), [LinearDrag](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/dynamics/linear_drag.py), [Multirotor](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/multirotor.py), [PX4MavlinkBackendConfig](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/backends/px4_mavlink_backend.py), [iris.usd](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/).
**Prior sysid status**: [sysid_output/analysis/replicator_metrics.json](../../../../controller/sysid_output/analysis/replicator_metrics.json) — 8192-trial gain sweep against PX4 SITL CSVs found a best score 0.0170 but still leaves **26% vel-settling-time gap, 94% vel-ss-error gap, 133% yaw-settling-time gap**. Gains alone cannot close the gap; the physics parameters must move.

**What**: Re-fit iris_ma6's open-loop motor + drag + mass parameters so the rigid-body dynamics match Pegasus's Iris model. Add a Pegasus-mode flag to `AerodynamicsCfg` to switch from the current quadratic-body-drag + Dryden-gust + rotor-effects pipeline to Pegasus's diagonal-linear body-drag formula. Document (but do not eliminate) the residual architectural gaps (force-application topology, controller hierarchy, sensor noise) so we know what remains as DR and what's structurally different.

**Why**: ticket 032 (zoom), mas/035 (gimbal rate), and the deployment-side gimbal_stabilizer port already converged on Pegasus / SITL behavior as the deployment surrogate. The body dynamics never went through the same alignment — iris_ma6 is configured as an "aggressive racing-drone-class" model (cfg docstring), with 82:1 T/W, symmetric 10 ms motor lag, k_m/k_f = 0.012, and a quadratic body-drag formula. Pegasus's Iris is ~2.8:1 T/W, no motor lag at all on the thrust path, k_m/k_f ≈ 0.117 (default coefficients), and linear body drag with anisotropic per-axis coefficients `[0.50, 0.30, 0.00]`. Until these are aligned, every sim-to-sim transfer test runs against a fictional drone.

**Blocked on**: nothing.

**Depends on**: nothing strict; consumes [ticket 008's](../008-sysid-replicator-DONE/) infrastructure. Re-runs the replicator after the parameter change.

### Truth values — PegasusSimulator Iris defaults (2026-05-25 codebase)

These are the **values iris_ma6 should match** in its `siyi_a8`-equivalent "pegasus" aero mode. All values come from the Pegasus repo as it stands today (BSD-3 Marcelo Jacinto, 2023–2024).

#### Mass and geometry

| Quantity | Pegasus value | Pegasus source | iris_ma6 today |
|---|---|---|---|
| Body mass | **1.5 kg** | [iris_gimbal3.usda:110](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_gimbal3.usda) `float physics:mass = 1.5` | Whatever USD sets (must be 1.5 for hover init in [drone_controller.py:199](../../../../controller/drone_controller.py#L199) to be correct) |
| Per-rotor link mass | **0.001 kg** (visual stub) | iris_gimbal3.usda:1782/1820/1858 | n/a — iris_ma6 applies body wrench, no per-rotor rigidbodies |
| Inertia | from USD (xform on `/body`) | iris_gimbal3.usda | from same USD (shared asset path) |
| Gravity | **9.8 m/s²** (Pegasus IMU constants assume this) | sensors/imu.py:39 | 9.81 m/s² (env default) — **0.1% gap; tolerate** |

#### Motor / thrust curve (`QuadraticThrustCurve` defaults; Iris overrides NONE of these)

| Quantity | Pegasus value | Pegasus source | iris_ma6 today |
|---|---|---|---|
| `num_rotors` | **4** | quadratic_thrust_curve.py:33 | 4 (mixer.py X-config) |
| `rotor_constant` (= `k_f`) | **8.54858e-6 N·s²/rad²** per rotor | quadratic_thrust_curve.py:36 | 1.2e-5 (**1.4× larger**) |
| `rolling_moment_coefficient` (≈ `k_m`) | **1.0e-6 Nm·s²/rad²** per rotor | quadratic_thrust_curve.py:40 | 1.4e-7 (**7× smaller**) |
| `rot_dir` | **[-1, -1, +1, +1]** (M1/M2 CCW, M3/M4 CW) | quadratic_thrust_curve.py:44 | matches via [mixer.py:36-39](../../../../controller/mixer.py#L36-L39) |
| `min_rotor_velocity` | **0 rad/s** per rotor | quadratic_thrust_curve.py:48 | 50 (idle to keep mixer headroom) |
| `max_rotor_velocity` | **1100 rad/s** per rotor | quadratic_thrust_curve.py:51 | 5000 (**4.5× larger**) |
| **Motor first-order lag** | **None — instantaneous** `_velocity[i] = clip(input_reference)` | quadratic_thrust_curve.py:91-94 | τ = 0.01 s (symmetric, [motor_dynamics_cfg.py:38](../../../../controller/motor_dynamics_cfg.py#L38)) |
| Per-rotor T at ω_max | 8.54858e-6 × 1100² = **10.34 N** | derived | 1.2e-5 × 5000² = 300 N (**29× larger**) |
| Total T_max | **41.4 N** → T/W ≈ **2.81** at 1.5 kg | derived | 1200 N → T/W ≈ **81.5** |

#### Drag (`LinearDrag` configured for Iris)

| Quantity | Pegasus value | Pegasus source | iris_ma6 today |
|---|---|---|---|
| Drag form | **Linear in body velocity**: `F = -diag(d) · v_body` | linear_drag.py:62 | **Quadratic**: `F = -½ ρ C_d A · \|v\| · v` ([aerodynamics.py:144-148](../../../../controller/aerodynamics.py#L144-L148)) |
| Drag coefficients | **`[0.50, 0.30, 0.00]`** (X, Y, Z body) | iris.py:30 | ρ=1.225, C_d=0.03, A=0.1 → factor 1.84e-3, isotropic-quadratic ([aerodynamics_cfg.py:28-35](../../../../controller/aerodynamics_cfg.py#L28-L35)) |
| Anisotropic axes | **Yes** (forward 0.5, lateral 0.3, vertical 0.0) | iris.py:30 | No (isotropic) |
| Frame | Body (FLU) per Pegasus state.linear_body_velocity | linear_drag.py:59 | Body (FLU) ([aerodynamics.py:118](../../../../controller/aerodynamics.py#L118)) — frame consistent |
| Wind / gusts | **None** (Drag.update reads body velocity only) | linear_drag.py:45-63 | Dryden-like gust at fidelity ≥ 2 ([aerodynamics.py:152-167](../../../../controller/aerodynamics.py#L152-L167)) |
| Per-rotor disk drag | **None** at vehicle level | n/a | At fidelity 3: `F_H = -k_H · Σω · v_xy` ([aerodynamics.py:194-197](../../../../controller/aerodynamics.py#L194-L197)) |
| Blade flapping moment | **None** | n/a | At fidelity 3: `τ_x = -k_flap · v_y`, `τ_y = k_flap · v_x` ([aerodynamics.py:202-203](../../../../controller/aerodynamics.py#L202-L203)) |
| Vertical force scaling | **None** | n/a | None |

#### Wrench application topology

| Aspect | Pegasus | iris_ma6 today |
|---|---|---|
| Per-rotor force | **Yes** — `apply_force([0,0,F_z], "/rotor_i")` for i ∈ {0..3} | **No** — single mixed wrench |
| Body torque | **Yaw only** — `apply_torque([0,0,rolling_moment], "/body")` (sum of per-rotor `k_m·ω²·dir`) | Full `[τ_x, τ_y, τ_z]` from mixer, applied via `set_external_force_and_torque` |
| Pitch/roll moments | **Geometric** — emerge from per-rotor force lever arms via USD pose | **Algebraic** — `τ_x = d(-T1+T2+T3-T4)`, `τ_y = d(-T1+T2-T3+T4)` with `d = L/√2` ([mixer.py:90-100](../../../../controller/mixer.py#L90-L100)) |
| Drag application | At `/body` link | At rigidbody root via `set_external_force_and_torque` |

#### PX4 MAVLink backend (Pegasus's path to motor commands)

| Quantity | Pegasus value | Source |
|---|---|---|
| `input_offset` | `[0, 0, 0, 0]` | px4_mavlink_backend.py:230 |
| `input_scaling` | `[1000, 1000, 1000, 1000]` | px4_mavlink_backend.py:231 |
| `zero_position_armed` | `[100, 100, 100, 100]` | px4_mavlink_backend.py:120 |
| Mapping | `ω_i [rad/s] = (controls_i + 0) × 1000 + 100` for `controls_i ∈ [0, 1]` | px4_mavlink_backend.py:173 |
| Effective ω range | **[100, 1100] rad/s** | derived |

#### Physics step

| Aspect | Pegasus | iris_ma6 |
|---|---|---|
| Physics dt | Configurable (`set_world_settings(physics_dt=...)`); examples use 1/250 → 1/500 | **1/100 = 10 ms** ([iris_ma_env6_test_cfg.py:358](../../../../iris_ma_env6_test_cfg.py#L358)) |
| IMU update rate | **250 Hz** default ([sensors/imu.py:45](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/sensors/imu.py)) | n/a — env reads state directly |
| Sensor noise | **Yes** — IMU (gyro density 3.4e-4, accel 4e-3), GPS, baro, mag with bias models | None on state |

### Background — what iris_ma6 has today

- **Motor model**: first-order lag (10 ms symmetric) + `T = k_f·ω²` + `Q = k_m·ω²`, body-wrench via 4×4 mixer ([motor_dynamics.py](../../../../controller/motor_dynamics.py), [mixer.py](../../../../controller/mixer.py)).
- **Aero**: quadratic body drag + Dryden gust (fidelity 2) + H-force / blade flapping (fidelity 3), default `fidelity_level=3`.
- **Controller chain**: policy → velocity controller → attitude controller → rate controller → mixer → motor lag → wrench. All on the iris_ma6 side; no PX4 in the loop during training.
- **Force application**: `robot.set_external_force_and_torque(F_body, τ_body)` at [iris_ma_env6_test.py:901](../../../../iris_ma_env6_test.py#L901).
- **Sysid replicator**: 8192-trial gain sweep against PX4 SITL step-response CSVs ([controller/tuning/sysid_replicator.py](../../../../controller/tuning/sysid_replicator.py)). Targets in [controller/sysid_output/*.csv](../../../../controller/sysid_output/). Best metrics gap **still 26-94-133%** ([replicator_metrics.json](../../../../controller/sysid_output/analysis/replicator_metrics.json)).
- **DR**: `GainRandomizationCfg` ([gain_randomization_cfg.py](../../../../controller/gain_randomization_cfg.py)) randomizes controller gains, tau_motor, max_zoom_rate, etc. There is no current DR knob on `k_f`, `k_m`, `omega_max`, or drag coefs.

### Architectural differences — document, do not (in this ticket) eliminate

These are **structural mismatches** that even matched parameters cannot close. Each is annotated with whether/how it might be addressed later (and is therefore **out of scope for this ticket**).

1. **Force-application topology**. Pegasus applies four per-rotor `(0, 0, F_z)` forces at the four rotor link poses; pitch/roll moments emerge geometrically. iris_ma6 collapses to a single body wrench with algebraic pitch/roll via the X-config mixer. **Effect**: identical at the rigid-body level *if* `arm_length` and rotor signs match the USD geometry, but Pegasus picks up PhysX-side inertia coupling that the body-wrench path skips. **Out of scope** — verify the arm length matches USD; do not re-architect to per-rotor application.
2. **Per-rotor air drag and blade flapping live in different places.** PX4-Gazebo's `gazebo_motor_model.cpp` applies per-rotor disk drag and rolling moments using each rotor's velocity perpendicular to its spin axis. Pegasus does NOT do this (its motor plugin port dropped that path). iris_ma6 has body-level analogs at fidelity 3. **For Pegasus parity, fidelity 3 contributions should be OFF.** This ticket adds a fidelity mode that disables them.
3. **Controller hierarchy is fundamentally different.**
   - In iris_ma6: policy → velocity → attitude → rate → mixer → motor. All training-time.
   - In Pegasus + PX4 SITL: simulator state → PX4 EKF2 → PX4 mc_pos_control / mc_att_control / mc_rate_control → mixer → MAVLink → PegasusBackend → motor.
   PX4 has its own anti-windup, integrator dynamics, saturation, and notch filters that iris_ma6's controller does not replicate. The sysid replicator handles this by matching closed-loop step responses, but the open-loop gains and limits in PX4 (`MC_*_P`, `MPC_*`, `MC_ROLLRATE_P`, etc.) are NOT iris_ma6's controller gains. **Out of scope** — re-running the existing replicator after this ticket should give a much smaller residual since the *plant* will be closer.
4. **Quaternion convention.** Pegasus uses `[qx, qy, qz, qw]` (scalar-last). iris_ma6 uses `[qw, qx, qy, qz]` (scalar-first, per [IsaacLab/CLAUDE.md](../../../../../../../../CLAUDE.md) and [iris_ma6/CLAUDE.md](../../../../CLAUDE.md)). Bench data and replicator already handle this; flag it as a known boundary. **Out of scope** — do not change.
5. **Frame conventions.** Pegasus is ENU world, FLU body. iris_ma6 likewise. PX4 internally is NED/FRD; conversion happens in the MAVLink bridge. **No action needed**.
6. **Sensor noise.** Pegasus injects Gaussian + bias-walk + turn-on-bias on IMU, GPS, baro, mag. iris_ma6 reads ground-truth state. Adding sensor noise is a DR task, not a physics-parameter task. **Out of scope**.
7. **Yaw moment comes from a "rolling moment" sum, not per-motor reaction torques.** In Pegasus, `τ_z = Σ rolling_moment_coef[i] · ω[i]² · rot_dir[i]` is applied as a single body torque. In iris_ma6's mixer, `τ_z = (k_m/k_f) · (-T1-T2+T3+T4)` — algebraically equivalent only if signs and magnitudes are aligned. **In scope** — verify and align `k_m`.
8. **Physics step rate**. iris_ma6 = 100 Hz (10 ms). Pegasus typical = 250 Hz (4 ms). With matched motor lag (τ=0 in Pegasus mode), the 10 ms / 4 ms gap manifests as a small ZOH error on rate commands. **Out of scope** — accept 10 ms; if it shows up in sysid residual, the follow-up is to drop iris_ma6 sim dt to 1/200.

### Design choices (decided up front; do not re-litigate during implementation)

1. **Match Pegasus default `QuadraticThrustCurve` numerics exactly.**
   - `k_f = 8.54858e-6` (Pegasus default; same as PX4 Gazebo plugin `kDefaultMotorConstant`).
   - `k_m = 1.0e-6` (Pegasus default `rolling_moment_coefficient`). Note: this changes the mixer's `c = k_m/k_f` from 0.0117 to **0.117** (10× yaw authority per thrust). This is the biggest single change in this ticket.
   - `omega_max = 1100` (Pegasus default).
   - `omega_min = 0` (Pegasus default; iris_ma6 uses 50 today for mixer headroom — see decision 4 on whether to keep 0 or 50).

2. **Disable motor first-order lag in Pegasus-parity mode**. Pegasus's `QuadraticThrustCurve.update` is instantaneous. Set `tau_motor = 1e-4` (effectively zero) — keeps the first-order-lag code path active and DR-able, but with negligible lag. **Do not** delete the lag code; it stays for the "training default" mode (see decision 6).

3. **Linear body drag formula in Pegasus-parity mode**. Add a new `drag_model: str` field to `AerodynamicsCfg`:
   - `"quadratic"` (default — current behavior): `F = -½ ρ C_d A · |v| · v` (reads `rho`, `C_d`, `A`).
   - `"linear_diag"`: `F = -diag(d_x, d_y, d_z) · v_body` (reads new field `drag_coefs: tuple[float, float, float] = (0.50, 0.30, 0.00)`).
   The default stays `"quadratic"` so this ticket's landing is a no-op for any current training run / yaml. The new mode is opt-in.

4. **Keep `omega_min = 50` in Pegasus-parity mode** (NOT Pegasus's 0). Rationale: iris_ma6's mixer with thrust-preserving clamping ([mixer.py:170-198](../../../../controller/mixer.py#L170-L198)) needs strictly-positive minimum thrust to allocate yaw moments at low collective thrust without saturating the algorithm. Pegasus avoids this by per-rotor geometric force application. **Document this as a known small bias**; do not change to 0. If you do change to 0, the `allocate` path's `thrust_min = k_f · omega_min²` becomes 0, and the headroom-redistribution logic divides by ~0 in trivial hover cases.

5. **Wind / gust / rotor effects OFF in Pegasus-parity mode**. Add a new `AerodynamicsCfg.fidelity_level` value `"pegasus"` (string, not int — distinct from `0|1|2|3`) that:
   - Engages `drag_model = "linear_diag"` regardless of the int level.
   - Forces `_v_wind_mean = 0`, `_gust = 0` (no wind path).
   - Skips the H-force / blade flapping branch (`_compute_rotor_effects` is not called).
   - Reads `drag_coefs` from cfg.
   Implementation: keep the existing `fidelity_level: int = 3` field for backward compatibility; add `mode: str = "default"` ∈ {`"default"`, `"pegasus"`}. When `mode == "pegasus"`, the int level is ignored. (See decision 6 on the backward-compat selector pattern, identical to ticket 032's `cfg.zoom.model = "first_order" | "siyi_a8"`.)

6. **Backward-compatible mode selector**. Same pattern as ticket 032: add `mode: str = "default"` to `AerodynamicsCfg`. Default preserves bit-exact pre-040 behavior. New value `"pegasus"` engages all of this ticket's plant changes (motor numbers, no-lag, linear drag, no wind, no rotor effects). Selection is per-controller-construction (cfg-time), not per-step. Env's `_init_<task>_v0` (or equivalent) chooses the mode; yaml configs override via `agent_cfg.cfg_overrides`.

7. **DR knobs**. Three new fields in [GainRandomizationCfg](../../../../controller/gain_randomization_cfg.py), each multiplicative on the configured nominal — all gated on `mode == "pegasus"` (no-op in `"default"` mode):
   - `k_f_scale_range = (0.9, 1.1)` — captures rotor-constant variation across builds.
   - `k_m_scale_range = (0.8, 1.2)` — captures propeller drag variability (looser since `k_m` is harder to measure).
   - `drag_coefs_scale_range = (0.7, 1.3)` — looser still; body drag dominates at higher speeds and is sensitive to attitude / wind.
   No DR on `omega_max` (firmware constant) or `tau_motor` (already DR'd in `gain_randomization_cfg.py`).

8. **Mass match is a USD-level concern**. iris_ma6's USD must report `1.5 kg` body mass. If it doesn't, fix the USD or call `robot.write_root_mass_to_sim(1.5)` in env reset. **Acceptance test** in slice 1 reads `robot.data.default_mass` and asserts == 1.5 ± 1%.

9. **Sim dt stays at 100 Hz** (out of scope per architectural-diff §8). If the post-fix replicator residual is still > 10%, follow-up ticket can bump to 1/200.

10. **`gravity = 9.81` stays** (out of scope; Pegasus uses 9.8 in IMU bias spec but the actual gravity in Isaac Sim is set per-scene, and iris_ma6's scene uses 9.81. 0.1% gap, ignore).

### Workflow

1. **Write `pegasus_physics_parity_spec.md`** — per [iris_ma6/CLAUDE.md](../../../../CLAUDE.md) "Specs live in doc/*_spec.md". Document the truth table above (the spec is the authoritative source; this ticket is the execution plan).

2. **Edit [controller/motor_dynamics_cfg.py](../../../../controller/motor_dynamics_cfg.py)** — add `model: str = "default"`, NEW Pegasus-parity defaults gated under that mode. Concretely:
   ```python
   model: str = "default"                # NEW; "default" preserves pre-040 numbers
   # When model == "pegasus", these get used:
   k_f_pegasus: float = 8.54858e-6
   k_m_pegasus: float = 1.0e-6
   omega_max_pegasus: float = 1100.0
   tau_motor_pegasus: float = 1.0e-4
   # When model == "default", existing k_f, k_m, omega_max, tau_motor are used.
   ```
   Or simpler: keep one set of fields, expose a `set_pegasus_defaults()` helper that swaps them in at `__init__`. (Implementer's choice — both are fine; the goal is *no overwriting of existing default values*.)

3. **Edit [controller/aerodynamics_cfg.py](../../../../controller/aerodynamics_cfg.py)** — add:
   ```python
   mode: str = "default"                          # NEW; "default" | "pegasus"
   drag_model: str = "quadratic"                  # NEW; "quadratic" | "linear_diag"
   drag_coefs: tuple[float, float, float] = (0.50, 0.30, 0.00)  # NEW; used when drag_model == "linear_diag"
   ```
   When `mode == "pegasus"`, the controller force-sets `drag_model = "linear_diag"`, `fidelity_level = 1`, and bypasses wind/gust/rotor-effects branches regardless of the int level.

4. **Edit [controller/aerodynamics.py](../../../../controller/aerodynamics.py)** — three changes:
   - In `__init__`, if `cfg.mode == "pegasus"`: precompute `self._drag_diag = torch.diag(torch.tensor(cfg.drag_coefs))`.
   - In `compute_forces`, switch on `cfg.drag_model`: dispatch to a new `_compute_drag_linear(v_rel_body)` method that returns `F = -(diag(d) @ v_body.T).T`. Existing `_compute_drag` (quadratic) is unchanged.
   - In `compute_forces`, when `cfg.mode == "pegasus"`, force `v_wind = 0` and skip `_compute_rotor_effects`.

5. **Edit [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py)** — add a top-level cfg field `physics_mode: str = "default"` (default preserves pre-040). When `"pegasus"`, the env wires `drone_controller_cfg.motor_dynamics.model = "pegasus"` and `drone_controller_cfg.aerodynamics.mode = "pegasus"` at env construction. **No yaml change required** for default training runs.

6. **Edit [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py)** — at env reset, sanity-check the USD's actual body mass against `cfg.expected_body_mass` (NEW field, default 1.5 kg) and log a warning if it diverges by > 1%. This catches USD-asset drift early.

7. **Edit [controller/gain_randomization_cfg.py](../../../../controller/gain_randomization_cfg.py)** — add the three NEW DR scale ranges from design 7. Implement the multiplicative DR writes in [controller/drone_controller.py](../../../../controller/drone_controller.py)'s `randomize_gains` path (mirror the existing `tau_motor` / `max_zoom_rate` DR pattern).

8. **Tests** at [controller/tests/](../../../../controller/tests/):
   - **`test_pegasus_parity_physics_stages.py`** — pure-PyTorch unit test:
     - **Motor model**: with `model="pegasus"`, step `omega_cmd = 1100` for 100 ms. Assert `omega ≈ 1100` (no lag) within 0.1%. Compare to `model="default"`: `omega` reaches only `1100 · (1 - exp(-0.1/0.01)) ≈ 1100 · 0.9999` (because τ=10 ms, but at 100 ms the lag is also done — pick a shorter window, 10 ms: default reaches `1100 · 0.632`, pegasus reaches `1100 · 1.0`).
     - **Thrust ceiling**: `model="pegasus"` total T at ω_max = 4 × 8.54858e-6 × 1100² ≈ 41.4 N. Assert `compute_body_wrench()` returns ≈ (0, 0, 41.4 N).
     - **Yaw authority**: `model="pegasus"` at hover with mixer's `c = k_m/k_f = 0.117`. Drive τ_z = 0.1 Nm and confirm the resulting motor differential matches Pegasus's `aloc_matrix` predictions within 1%.
     - **Linear drag**: with `drag_model="linear_diag"`, `drag_coefs=(0.5, 0.3, 0.0)`, body velocity `(10, 0, 0)`. Assert drag force `(-5, 0, 0) N`. Compare to quadratic at same velocity: `~(-0.092, 0, 0) N`.
     - **Default mode bit-exactness**: with `mode="default"`, model produces identical output to pre-040 controller across a 200-step random-control rollout (atol=1e-6). This is the hard backward-compatibility gate.
   - **`test_pegasus_mass_sanity.py`** — env-side: construct the env with `physics_mode="pegasus"`, query `robot.data.default_mass`, assert `1.5 ± 0.015 kg`.

9. **Generate Pegasus truth CSVs** — re-target the sysid replicator at Pegasus instead of PX4 SITL:
   - Record step-response traces (hover_drift, vel_step_5, vel_step_10, vel_impulse_recovery, yaw_step) from a single-vehicle Pegasus Iris with the PX4 SITL backend, identical command profiles to the existing [controller/sysid_output/*.csv](../../../../controller/sysid_output/). Save to `controller/sysid_output_pegasus/`.
   - Run `controller/tuning/sysid_replicator.py --sysid-dir .../sysid_output_pegasus --aero-level 0 --num-trials 8192` with `physics_mode="pegasus"` active. The existing replicator code does not need to know about the mode — it sees only the resulting closed-loop trajectories.

10. **Iterative test harness** per [/home/usrg/IsaacPX4/IsaacLab/CLAUDE.md](../../../../../../../../CLAUDE.md) §"Iterative Test-Fix Workflow":
    - Run `./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py > .../tests/test_result.txt 2>&1`.
    - Append per-iteration error analysis to `controller/tests/error_log.txt`.
    - Loop until tests pass + existing `controller/tests/` suite stays green.

### Scope boundary

- **DO**: align motor `k_f`, `k_m`, `omega_max`, `tau_motor` to Pegasus defaults in a new `model="pegasus"` mode. Default mode preserves pre-040 behavior bit-exactly.
- **DO**: add linear-diagonal body drag formula and Pegasus drag coefficients `(0.50, 0.30, 0.00)` behind a new `aerodynamics.mode="pegasus"`.
- **DO**: disable wind / gust / blade flapping / H-force in Pegasus mode.
- **DO**: add DR knobs for `k_f`, `k_m`, `drag_coefs` gated on Pegasus mode.
- **DO**: re-target the existing sysid replicator at fresh Pegasus-truth CSVs and validate the closed-loop fit.
- **DO**: assert USD body mass = 1.5 kg at env reset.
- **DO NOT**: change the force-application topology (body-wrench → per-rotor). Out of scope; would require rewriting [iris_ma_env6_test.py:901](../../../../iris_ma_env6_test.py#L901) and the mixer.
- **DO NOT**: add IMU / GPS / baro / mag noise to the env. Sensor noise is a separate DR ticket if needed.
- **DO NOT**: replicate PX4's controller hierarchy inside iris_ma6. The sysid replicator handles closed-loop matching; the controllers stay as-is.
- **DO NOT**: change the quaternion convention. wxyz throughout iris_ma6.
- **DO NOT**: drop `omega_min` to 0. Mixer's `allocate` path needs strictly-positive floor.
- **DO NOT**: change sim dt from 1/100. If post-fix residual is still high, follow-up ticket.
- **DO NOT**: touch gravity (9.81 vs Pegasus's 9.8 IMU constant — 0.1% gap, not worth a ticket).
- **DO NOT**: enable Pegasus mode by default. Opt-in only; `physics_mode: str = "default"` at env cfg.

### Acceptance criteria

- **All tests pass.** New pure-PyTorch unit tests in `controller/tests/`; USD-mass sanity test passes for the iris_ma6 USD.
- **Bit-exact regression** at `mode="default"` vs pre-040 controller on a 200-step random-control rollout (atol=1e-6). Locked in by `test_pegasus_parity_physics_stages.py::test_default_mode_unchanged`.
- **Pegasus-mode sysid residual** improves substantially over the PX4-SITL baseline:
  - Median `score` in [replicator_metrics.json] should drop from current **0.076** by at least **50%** (i.e., ≤ 0.038) with `physics_mode="pegasus"` and Pegasus truth CSVs.
  - vel-step-5 settling time gap drops from **26.4%** to **< 15%**.
  - vel-step-5 ss-error gap drops from **94%** to **< 30%**.
  - yaw-settling gap drops from **133%** to **< 50%**.
  These are pass thresholds, not targets — closing more is better.
- **No regression** in existing `controller/tests/run_tests.py` suite, especially `test_dynamics_curriculum_fix.py` which exercises `randomize_gains`.
- **No regression** in existing default-mode training runs. Sanity-check: launch a 10k-step default-mode training run and confirm episode-return curve matches an archived reference within ±5%.

### Risk

Medium.

1. **Yaw-authority jump (10× higher k_m/k_f)** in Pegasus mode could destabilize a policy that learned tiny yaw via thrust differential. Mitigation: this mode is opt-in. Initial Pegasus-mode runs should expect a re-train from scratch (or at minimum a warm-start fine-tune), not a checkpoint resume from default-mode runs. **Add to the spec doc** that resuming a default-mode checkpoint in Pegasus mode is not supported.
2. **Thrust ceiling drop (1200 N → 41.4 N, factor of 29)** removes the policy's ability to make arbitrarily aggressive recoveries. Aggressive trained policies will hit motor saturation in Pegasus mode that they never saw in default mode. **Mitigation**: this is the *point* of the ticket — exposing the real T/W to the policy. Saturation handling is the policy's job.
3. **Mixer headroom at low collective thrust** with `omega_min = 50` and `k_m/k_f = 0.117`: the mixer's thrust-preserving clamp may behave differently. **Mitigation**: covered by the new unit tests; verify the mixer's `allocate` output stays within `[k_f·omega_min², k_f·omega_max²]` per rotor across a random sweep of thrust/moment commands.
4. **Sysid Pegasus-truth CSV capture is new infrastructure.** Mitigation: piggyback on the existing replicator's CSV format ([sysid_output/*.csv](../../../../controller/sysid_output/)). The Pegasus side just needs to log the same columns (`t, vel_x, vel_y, vel_z, att_q[w,x,y,z], rate_xyz`) during the same step-response profiles. A 2-3 hour scripting task; not architectural.

### Coupling

- **mas/035, mas/036, ticket 032**: gimbal and zoom paths already converged on Pegasus / SITL behavior. This ticket extends the same convergence pattern to the body dynamics.
- **ticket 003** (controller tuning): the existing controller tuning targets PX4 SITL CSVs and PID gains. After this ticket, the plant is closer to Pegasus, so a re-tune may be needed if PX4-SITL-tuned gains over/underdrive the new Pegasus-mode plant. **Sequence**: do this ticket first; ticket 003 follow-up re-runs gain tuning against the new plant.
- **ticket 005** (sim-to-sim transfer): this ticket directly enables ticket 005 by reducing the plant gap.
- **ticket 007** (light sysid), **ticket 008-DONE** (sysid replicator): consumes the existing replicator infrastructure verbatim with new truth data.
- **ticket 027** (model fitting): independent. Body dynamics is separate from the gimbal/zoom fits.
- **ticket 039** (asymmetric z-velocity envelope): independent. Action-scaling envelope is upstream of motor dynamics.
- **Curriculum cadence** ([curriculum_spec.md](../../../curriculum_spec.md)): no change. Pegasus mode does not introduce new curriculum knobs (DR ranges are constants, not curriculum-gated).

### Affected files

**New**:
- NEW: [doc/pegasus_physics_parity_spec.md](../../../pegasus_physics_parity_spec.md) — truth table, four-line force/drag pipeline diagram, DR / mode-selector hooks, architectural-difference catalog.
- NEW: [controller/tests/test_pegasus_parity_physics_stages.py](../../../../controller/tests/test_pegasus_parity_physics_stages.py)
- NEW: [controller/tests/test_pegasus_mass_sanity.py](../../../../controller/tests/test_pegasus_mass_sanity.py)
- NEW: [controller/sysid_output_pegasus/](../../../../controller/sysid_output_pegasus/) — directory with Pegasus-truth step-response CSVs (hover_drift, vel_step_5, vel_step_10, vel_impulse_recovery, yaw_step).
- NEW (optional, parallels [compare_gimbal.py](../../../../controller/sysid_output/gimbal/compare_gimbal.py)): [controller/sysid_output_pegasus/compare_body.py](../../../../controller/sysid_output_pegasus/compare_body.py) — single-page PDF of iris_ma6 vs Pegasus step responses.

**Edits**:
- EDIT: [controller/motor_dynamics_cfg.py](../../../../controller/motor_dynamics_cfg.py) — add `model: str = "default"` and the Pegasus-parity values (gated under `model == "pegasus"`).
- EDIT: [controller/aerodynamics_cfg.py](../../../../controller/aerodynamics_cfg.py) — add `mode`, `drag_model`, `drag_coefs`.
- EDIT: [controller/aerodynamics.py](../../../../controller/aerodynamics.py) — add `_compute_drag_linear`; gate wind/gust/rotor-effects branches on `cfg.mode != "pegasus"`.
- EDIT: [controller/motor_dynamics.py](../../../../controller/motor_dynamics.py) — read the gated parameters at `__init__`; no algorithmic change.
- EDIT: [controller/gain_randomization_cfg.py](../../../../controller/gain_randomization_cfg.py) — add `k_f_scale_range`, `k_m_scale_range`, `drag_coefs_scale_range`.
- EDIT: [controller/drone_controller.py](../../../../controller/drone_controller.py) — `_nominal_gains` reads the gated parameters; `randomize_gains` writes the new DR scales when `mode == "pegasus"`.
- EDIT: [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py) — add `physics_mode: str = "default"` and `expected_body_mass: float = 1.5`.
- EDIT: [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) — at env construction, propagate `physics_mode` to controller / aero cfgs; at reset, sanity-check USD body mass.
- EDIT: [controller/tuning/sysid_replicator.py](../../../../controller/tuning/sysid_replicator.py) — accept a `--physics-mode` CLI flag that forwards to env construction (does NOT change the replicator algorithm).
- EDIT: [doc/active/feature_list.json](../../feature_list.json) — flip the `controller` feature's notes to mention the Pegasus-parity mode (does not flip status).
- EDIT: [doc/sim-to-real/modeling_checklist.md](../../../sim-to-real/modeling_checklist.md) — mark body-dynamics-parity item in progress with link to this ticket.

### References

- [PegasusSimulator/.../vehicles/multirotors/iris.py](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/multirotors/iris.py) — IrisConfig; canonical drag coefficients `[0.50, 0.30, 0.00]`.
- [PegasusSimulator/.../thrusters/quadratic_thrust_curve.py](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/thrusters/quadratic_thrust_curve.py) — `k_f = 8.54858e-6`, `k_m = 1.0e-6`, `ω_max = 1100`, no first-order lag.
- [PegasusSimulator/.../dynamics/linear_drag.py](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/dynamics/linear_drag.py) — linear-diagonal drag formula `F = -diag(d) · v_body`.
- [PegasusSimulator/.../vehicles/multirotor.py:101-150](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/multirotor.py) — per-rotor force application topology and force/drag dispatch order.
- [PegasusSimulator/.../backends/px4_mavlink_backend.py:155-173](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/backends/px4_mavlink_backend.py) — `(controls × 1000) + 100` → rad/s mapping.
- [PegasusSimulator/.../sensors/imu.py](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/sensors/imu.py) — sensor noise model (FYI; out of scope here).
- [PegasusSimulator/.../assets/Robots/Iris/iris_gimbal3.usda](/home/usrg/IsaacPX4/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/) — `<mass>1.5</mass>`.
- [PX4-Autopilot/.../iris.sdf:355-417](/home/usrg/IsaacPX4/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf) — PX4-Gazebo motor params (for triangulation against Pegasus; Pegasus is the truth for this ticket).
- [iris_ma6 sysid_output/analysis/replicator_metrics.json](../../../../controller/sysid_output/analysis/replicator_metrics.json) — current PX4-SITL replicator residuals.
- [iris_ma6 controller/tuning/sysid_replicator.py](../../../../controller/tuning/sysid_replicator.py) — existing sysid harness; reused verbatim.
- [iris_ma6 controller/motor_dynamics.py](../../../../controller/motor_dynamics.py), [controller/aerodynamics.py](../../../../controller/aerodynamics.py), [controller/mixer.py](../../../../controller/mixer.py) — current plant.
- Ticket 032 (zoom response model) — same mode-selector pattern (`model: "first_order" | "siyi_a8"`).
- mas/035, mas/036, mas/037 (gimbal rate + dead-time + zoom) — same architectural-alignment pattern, prior tickets.

**Flow**: Medium. Three load-bearing changes: (1) motor numerics in a new mode, (2) linear drag formula behind the mode flag, (3) Pegasus-truth CSV capture for replicator re-run. Other items are plumbing. Estimated 3 commits, with vertical slices: (a) cfg + motor + mixer-check tests; (b) aero linear-drag + mode-gated DR + tests; (c) Pegasus-truth CSV capture + replicator re-run + acceptance metric.
