# Gimbal Oscillation Diagnosis Spec — ticket 049

**Status**: authoritative for ticket 049 deliverables.
**Scope**: measurement + localization only. No fix, no gain re-tune (see ticket 049 §Scope boundary).
**Owner artifacts**: `controller/sysid_output/gimbal/oscillation_diagnosis/`.

This spec defines *what* the ticket-049 infrastructure captures and *how* it localizes the
train↔deploy divergence that produces a growing gimbal oscillation at "pointing at target".
The ticket.md holds the hypotheses and the why; this spec holds the signal contract, scenario
protocols, alignment rules, and the divergence-localization algorithm.

---

## 0. Pre-flight config audit (static, done before any run)

Several ticket-049 hypotheses reference deploy config values that the ticket text assumed from an
older revision of the controller. The **as-shipped** values (read 2026-06-11 from
`ros2_ws/src/gimbal_stabilizer/config/los_rate_config.yaml` and `los_rate_controller.py`) are:

| Knob | Ticket-049 assumption | As-shipped (2026-06-11) | Effect on hypothesis |
|---|---|---|---|
| `control_trigger` | `joint_states` (250 Hz, may stall) | **`clock`** | **H3 reshaped**: control fires on `/clock`, decoupled from joint_states publish. Inter-sample dt is set by `/clock` cadence + the `dt < 1/update_rate → return` skip in `_run_control_loop_simtime` (los_rate_controller.py:485). Still must be measured. |
| `imu_source` | `mavros` (EKF2, wall-clock, 20–40 ms lag) | **`state`** | **H1 reshaped**: `omega_body` comes from Pegasus `state/twist.angular` (ground-truth, sim-time, body-FLU), NOT mavros. So the deployed body-rejection input is *not* MAVLink/EKF-lagged. The residual lag is sample-and-hold between `state/twist` publish rate and the control rate. H1 becomes "state/twist sample-hold lag ≠ train's 15 ms injection", a much smaller effect. We still record `mavros/imu/data` in parallel to quantify what H1 *would* be if `imu_source` were flipped. |
| `gimbal_controller_mode` | (train) `jacobian` vs (deploy) `combined` | **`jacobian`** both sides | Earlier `combined` drift (see compare_gimbal.py REPORT) is resolved; control law now matches. **H6/law-mismatch** less likely. |
| `servo_rate_limit` | 6.0 rad/s (≈344°/s) | **56.5 rad/s** | **H8 unlikely** at "pointing": clip only bites at \|att_error\| > 56.5/32.5 ≈ 1.74 rad ≈ 100°, far from a pointing condition. |
| `feedback_blend` | 0.05, blends actual joint back into integrator | **declared in yaml, UNUSED in code** | **H4 statically disproven**: `los_rate_controller._run_control` never reads `feedback_blend`; the `_az_world` integrator is corrected only by anti-windup limit checks (los_rate_controller.py:537-556), never by actual-joint blending. We still record `feedback_blend_residual = _az_world − (quat→az_world)` to *confirm* no drift, but the mechanism in H4 cannot fire. |
| `rate_loop_tau_yaw_s` / `_pitch_s` | train 0.0995 / 0.0954 | **0.0995 / 0.0954** | **H5 τ in sync.** |
| `max_gimbal_rate` / `rate_loop_max_rate_per_axis` | train π rad/s | **1.28 rad/s** (both yaml) | **H5 partial mismatch**: the rate-loop axis saturation (1.28) and the `azimuth_rate = cmd*max_gimbal_rate` scaling differ from training's π. At "pointing" (cmd≈0) this scaling is not exercised, but record it. |
| `pitch_limits_deg` | ±45 | **[−25, +45]** asymmetric | Pointing usually mid-range; record for completeness. |
| `yaw_limits_deg` | ±45 | **[−360, +360]** | Yaw essentially unlimited deploy-side. |

**Consequence for the diagnostic**: H1, H3, H4 as *originally framed* are weakened or disproven by
config alone. The live-capture diagnostic is therefore re-weighted toward **H2** (joint-inner-loop
PD lag — present in deploy Pegasus articulation, absent in train's `compute_control` which assumes
the articulation tracks the position target) and **H7** (PX4 attitude limit-cycle feeding a *real*
body oscillation into `state/twist`, which the gimbal then rejects with phase lag). The capture
still records every signal so the data — not the static audit — makes the final call.

---

## 1. Signal contract

All signals captured **per control step**, both sides, in SI units (rad, rad/s). Frames: body =
FLU, world = ENU. Joint triplets are ordered **[yaw, roll, pitch]** in the *internal* controller
convention (YAW_JOINT_OFFSET already subtracted; USD sign flips already undone) so the two sides are
directly comparable. The CSV column names below are the canonical schema — `train_recorder.py` and
`deploy_recorder.py` MUST emit exactly these names so `compare_oscillation.py` can pair them blind.

| CSV column | Units | Train source | Deploy source | Hypothesis |
|---|---|---|---|---|
| `t_sim` | s | `self._sim_time` (env) | `msg.header.stamp` / `/clock` | alignment |
| `t_recv` | s | same as t_sim (synchronous) | node clock at callback | H3 jitter |
| `policy_cmd_yaw_rate` | norm [-1,1] | `gimbal_yaw_rate_cmd` (env action[:,4]) | `/<ns>/gimbal_cmd_los_rate.x` | entry point |
| `policy_cmd_pitch_rate` | norm [-1,1] | `gimbal_pitch_rate_cmd` (action[:,5]) | `/<ns>/gimbal_cmd_los_rate.y` | entry point |
| `rateloop_yaw_rate` | rad/s | `gimbal_rate_loop.omega_actual[:,0]` | `az_rate_eff` (diag pub) | H5 |
| `rateloop_pitch_rate` | rad/s | `gimbal_rate_loop.omega_actual[:,1]` | `el_rate_eff` (diag pub) | H5 |
| `omega_cmd_x/y/z` | rad/s | `omega_cmd` (jacobian diag) | `omega_cmd` (diag pub) | pointing cmd |
| `omega_body_x/y/z` | rad/s | `omega_body` arg (root_ang_vel_b) | `_body_angular_velocity_b` (diag pub) | **H1** |
| `omega_combined_x/y/z` | rad/s | `omega_combined` (jacobian diag) | `omega_combined` (diag pub) | J⁻¹ input |
| `qdot_ref_yaw/roll/pitch` | rad/s | `qdot_ref` (jacobian diag, pre-clamp) | `_qdot` pre-clamp (diag pub) | algo output |
| `qdot_clip_yaw/roll/pitch` | rad/s | = qdot_ref (no servo clip train-side) | `_qdot` post-clamp (diag pub) | **H8** |
| `joint_cmd_yaw/roll/pitch` | rad | controller `pos_targets` (internal) | `_yaw/_roll/_pitch` (diag pub) | last stop |
| `joint_act_yaw/roll/pitch` | rad | `robot.data.joint_pos[gimbal_idx]` (internal) | `/<ns>/isaac_joint_states` (de-offset) | **H2** |
| `az_world` | rad | `gimbal.azimuth_world` | `_az_world` (= `gimbal_los_state_deg.x`·π/180) | LOS integrator |
| `el_world` | rad | `gimbal.elevation_world` | `_el_world` (= `gimbal_los_state_deg.y`·π/180) | LOS integrator |
| `az_world_from_quat` | rad | `body_to_world_gimbal_angles(joint_act, q_body)` | same, computed in recorder | **H4** confirm |
| `feedback_blend_residual` | rad | n/a (0) | `az_world − az_world_from_quat` | **H4** confirm |
| `q_body_w/x/y/z` | wxyz | `robot.data.root_quat_w` | `state/pose` (xyzw→wxyz) or `mavros/imu/data` | **H6** |
| `control_dt` | s | `physics_dt` (= sim.dt, constant) | `dt` passed to `_run_control` (diag pub) | **H3** |

Derived post-hoc by `compare_oscillation.py` (not stored raw): `vehicle_omega_xy_power_5_20Hz`
(FFT band power of `omega_body_x`,`omega_body_y`) → **H7**.

**Joint-convention normalization** (so the two sides overlay): deploy `_run_control` reads
`actual_yaw = joint['yaw'] − YAW_JOINT_OFFSET`, `actual_roll = −joint['roll']`,
`actual_pitch = −joint['pitch']` (los_rate_controller.py:493-495). The deploy recorder, when reading
`isaac_joint_states` directly, applies the **same** transform so `joint_act_*` is in the internal
frame on both sides. The diag publisher emits already-internal values.

---

## 2. Test scenarios

Each scenario isolates a hypothesis class. Run deploy first (capture CSV), then replay the *same*
policy-command sequence into train (capture CSV). See ticket.md §Test scenarios for the predicted
outcomes. Durations chosen so the oscillation (if present) has time to grow.

| # | Scenario | Deploy driver | Train driver | Isolates |
|---|---|---|---|---|
| **S1** | Pure hover, policy cmd = 0 | hold hover, `gimbal_cmd_los_rate=0`, 60 s | `num_envs=1`, zero gimbal action, hover init, 60 s | body-rejection-only (H1/H2/H7) |
| **S2** | Hover + sinusoid cmd (yaw 0.1 rad/s, 0.5 Hz) | publish sinusoid on `gimbal_cmd_los_rate`, 30 s | replay identical sinusoid | LOS integration / rate-loop fidelity (H5) |
| **S3** | Hover + small body excitation (5 Hz, ±0.5°) | brief stick disturbance OR record natural oscillation (Risk #2 fallback) | mirror with `_apply_external_torque` OR replay deploy `omega_body` (S6) | H1 + H7 |
| **S4** | Step cmd (yaw +0.5 rad 1 s then 0) | step on `gimbal_cmd_los_rate`, 10 s | replay step | H5 + H2 step response |
| **S5** | Replay deploy command sequence into train | (deploy run from S1/S3) | feed recorded `policy_cmd_*` into train | A/B at controller boundary |
| **S6** | Open-loop replay deploy `omega_body` into train | (deploy run from S1/S3) | feed recorded `omega_body_*` as the env's body ang-vel | does train oscillate on deploy's body input? (H1/H7) |

S5/S6 consume the CSVs from S1/S3 — they are *analysis* replays, run train-side only with the
deploy trace as input. They need no fresh deploy run.

---

## 3. Alignment + divergence-localization algorithm (`compare_oscillation.py`)

### 3.1 Time-alignment
- Sync channel = `policy_cmd_yaw_rate` (+ `policy_cmd_pitch_rate`). Both sides see the same command.
- Resample both CSVs onto a common uniform grid at `min(dt_train, dt_deploy)` via linear interp on
  `t_sim`. Cross-correlate the sync channel; shift deploy by the integer-sample lag that maximizes
  correlation; report residual sub-sample offset. Alignment must land within ±1 timestep
  (acceptance criterion).

### 3.2 Per-signal residual
For each paired column `c`: `resid_c(t) = deploy_c(t) − train_c(t)` after alignment. Report
RMS residual, max\|residual\|, and the cumulative residual `∫resid_c dt` (a growing cumulative on an
upstream signal localizes the entry point of the divergence).

### 3.3 First-divergence localization
Order the signals along the control chain:
```
policy_cmd → rateloop → omega_cmd → omega_body → omega_combined
           → qdot_ref → qdot_clip → joint_cmd → joint_act → az_world
```
For each signal compute a per-signal **noise floor** σ_c = std of `resid_c` over the first 0.5 s of
S1 (where, by construction, nothing should diverge). The **first-divergence signal** is the
earliest signal in chain order whose `|resid_c(t)|` exceeds `3·σ_c` for a sustained window
(≥ 5 consecutive samples) AND whose exceedance precedes the oscillation onset. That signal's
position in the chain is where the divergence enters.

### 3.4 Delay + gain fit per link
For each upstream→downstream pair on the chain, fit `out(t) = α · in(t − τ) + ε` by grid-search over
τ ∈ [0, 50 ms] (sub-sample via the resampled grid) minimizing residual variance, then LS for α.
Report `(α, τ, R²)` per link, per side. A link whose deploy `τ` exceeds train `τ` by > 1 control
period, or whose `α` differs by > 10%, is the lag/gain culprit.

### 3.5 Stability margin
From the rate-loop + joint-inner-loop fit, build the open-loop transfer of the body-rejection path
and evaluate gain/phase at the oscillation frequency `f_osc` (peak of the `joint_act` PSD in
5–20 Hz). Phase margin < 30° at `f_osc` confirms the instability mechanism. Report `f_osc`, the
loop gain at `f_osc`, and the phase margin.

### 3.6 Outputs
- `gimbal_oscillation_diagnosis.json`: per-scenario × per-signal RMS/max residual, `noise_floor`,
  `first_divergence_signal`, `fitted_links` (α,τ,R² per link per side), `f_osc`, `phase_margin`,
  `h7_band_power` (train vs deploy), and a top-level `root_cause` field
  (one of the hypothesis IDs, or `"not_localized_multi_cause"` with a residual-power breakdown).
- `oscillation_report.pdf`: per-scenario overlay plots (train blue, deploy red) for every signal,
  residual subplots, the marked first-divergence point, a (α,τ) table, and the H7 spectral panel.

---

## 4. Instrumentation contract

### 4.1 Train side (`train_recorder.py` + gated capture in `gimbal_controller_jacobian.py`)
- Off by default. Enabled by env-var `IRIS_MA6_GIMBAL_DIAG=1`.
- When enabled, `GimbalController._diag_capture=True` is set, which stashes `omega_cmd`,
  `omega_combined`, `qdot_ref`, `att_error` into `_diag_*` tensors inside `compute_control` (cloned,
  read-only side effect). When disabled these branches are not taken → **bit-exact** pre-049 output.
- The env, when the env-var is set, instantiates a `GimbalDiagRecorder`, sets the controller flag,
  and appends one row per `_apply_action` from a preallocated ring buffer; flushes to CSV on env
  close or after `IRIS_MA6_GIMBAL_DIAG_STEPS` steps. Single-env (`num_envs=1`) is required for
  diagnostic runs to avoid GPU↔CPU sync cost and keep one clean trace.

### 4.2 Deploy side (`los_rate_controller.py` diag publishers + `deploy_recorder.py`)
- New ROS2 param `diagnostic_publish: bool = False`. When true, `los_rate_controller` publishes the
  internal-state Vector3Stamped/Float64 topics under `gimbal_diag/*` on **BEST_EFFORT, depth 1** QoS,
  **after** `_publish_joint_commands` (out of the critical control path). When false → no publishers
  created, zero overhead, bit-exact pre-049 behavior.
- `deploy_recorder.py` subscribes to the policy/joint/state topics + `gimbal_diag/*`, forces
  `use_sim_time=true`, writes one CSV per vehicle per scenario, mirrors `ekf_lag_recorder.py` shape.

### 4.3 Risk controls (ticket.md §Risk)
- Diag publishers deferred to after the joint command (R#1). H3 measurement (`control_dt`) verifies
  the publishers did not perturb the loop: `control_dt` distribution with/without `diagnostic_publish`
  must be statistically identical.
- S3 fragile injection → fallback to recording the *naturally emerging* oscillation (R#2).
- `state/twist` frame sanity check at record start: yaw the vehicle, expect `omega_body_z > 0` (R#3).

### 4.4 Calling Contract (per CLAUDE.md requirement)

| Component / method | Call frequency | Idempotent? | Lifecycle placement | Stateful invariants |
|---|---|---|---|---|
| `GimbalController.compute_control` diag stash | once per `_apply_action` (per sim step) | yes — overwrites `_diag_*` each call, no accumulation | inside `compute_control`, gated by `_diag_capture` | `_diag_*` are last-step snapshots; not advanced state |
| `GimbalDiagRecorder.record(row)` | once per `_apply_action` | **WRITE** — appends to ring buffer; call exactly once per step | env `_apply_action`, after controller step | ring buffer monotonic; never called from rewards/obs hooks |
| `GimbalDiagRecorder.flush()` | once at end (close or step cap) | idempotent — flushing twice writes same rows then truncates buffer | env close / step-cap | flush clears buffer; safe to call on partial run |
| `los_rate_controller` diag publishers | once per `_run_control` | publish-only, no state mutation | end of `_run_control`, after joint cmd | reads `_diag_*` snapshots set earlier in the same `_run_control` |
| `LOSRateController._run_control` | once per `/clock` (trigger='clock') | already guarded by `dt < 1/update_rate → return` | unchanged | diag stash (`_diag_omega_cmd` etc.) written inside `_run_jacobian_ik`, read by publisher in same call |
| `deploy_recorder` subscriber callbacks | per incoming msg | append-only CSV write | node spin | one CSV per (vehicle, scenario); sim-time stamps |

The diag stash on both sides is a **READ-snapshot** of values already computed by the control law —
it never feeds back into the control computation, so it cannot change controller behavior.

---

## 5. Reuse map

| Existing | Reused as |
|---|---|
| `compare_gimbal.py` | open-loop controller-law baseline; `compare_oscillation.py` extends it to closed-loop live capture + alignment/fit. |
| `ros2_los_math.py` | numpy port of the deploy law; used by `compare_oscillation.py` for the S5/S6 train-side replay sanity check without booting ROS2. |
| `ekf_lag_recorder.py` | ROS2 recorder shape (sim-time, `remove_ros_args`, per-topic CSV writer table, sim-time shutdown); `deploy_recorder.py` mirrors it. |
| `fit_ekf_lag.py` | the `(α, τ)` grid-search + cross-correlation alignment pattern; `compare_oscillation.py` reuses the fit kernel. |

---

## 6. Acceptance (mirrors ticket.md)

1. All 6 scenarios recorded both sides, sim-time aligned within ±1 timestep.
2. JSON populated: per-scenario per-signal RMS residual, first-divergence signal, fitted (α,τ) per
   link, phase margin at `f_osc`.
3. Root cause localized to a signal pair, or explicitly `not_localized_multi_cause` with residual-
   power breakdown per hypothesis.
4. PDF with overlay plots + summary table.
5. Follow-up ticket opened with proposed fix referencing this JSON.
6. No env/controller default-path behavior change when `IRIS_MA6_GIMBAL_DIAG` unset /
   `diagnostic_publish=false`.
7. Single `run_all_scenarios.sh` reproduces the full diagnostic from clean state.
