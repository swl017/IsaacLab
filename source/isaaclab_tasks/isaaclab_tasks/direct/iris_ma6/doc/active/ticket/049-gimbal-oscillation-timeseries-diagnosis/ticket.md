## Ticket 049 — Gimbal oscillation at deploy: timeseries comparison diagnosis

**Status**: OPEN — instrumentation + analysis harness LANDED + validated on real rosbags (2026-06-11). **Phenomenon does not currently reproduce**: the `net_width` deploy bags show no divergence (gimbal stable; likely mitigated by ticket-047/048 policy command smoothing). Ticket held open pending a divergence-containing recording (pre-mitigation checkpoint) to localize the original root cause.
**Created**: 2026-05-29
**Type**: Diagnostic (measurement + localization; the fix is a follow-up ticket once the root cause is identified).
**Symptom**: At deploy (Pegasus + PX4 SITL via mas stack), the gimbal oscillates **while pointing at the target** and the oscillation **grows over time** until divergence. Training does not exhibit this.
**Target setup**: Training env — [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) with [px4_matched_pegasus](../../../../controller/tuning/tuning_results/px4_matched_pegasus.py) gains, ticket 040 plant, ticket 041/042 EKF lag. Deploy — [ros2_ws/src/gimbal_stabilizer/los_rate_controller.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py) on Pegasus + PX4 SITL routed through `mas`.

**What**: Capture time-aligned timeseries of the gimbal control chain on **both** the training env and the deploy stack under identical scenarios. Localize where the two diverge (which signal, at what time-delay, with what gain mismatch). Static code comparison closed many sim2sim gaps (tickets 040, 041, 042) but did not find this oscillation — only a live-data diff will.

**Why**: The growing oscillation has the signature of a **positive-feedback limit cycle** in the gimbal body-motion compensation path:
```
omega_combined = omega_cmd - omega_body          # train + deploy share this formula
qdot_ref      = J^{-1} * omega_combined          # gimbal joint velocities
```
At "pointing at target" the policy command `omega_cmd ≈ 0`, so the entire gimbal motion comes from the body-motion rejection term `-omega_body`. If `omega_body` arrives lagged (mavros / EKF / network) AND the rejection-loop has insufficient phase margin, the wrong-phase correction torques the gimbal, perturbs the vehicle through reaction or sensor coupling, and the loop closes positive. Static code diff cannot find this because the sign of the closed-loop pole flips with lag, and lag is a runtime quantity not a code constant.

**Blocked on**: nothing. Existing infra ([compare_gimbal.py](../../../../controller/sysid_output/gimbal/compare_gimbal.py)) does open-loop controller-law comparison; we extend it to closed-loop live capture.

**Depends on**:
- mas/035 (rate loop), mas/036 (dead-time), ticket 041 (EKF lag values) — all landed; this ticket consumes their numerics.
- [los_rate_controller.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py) deploy side, [gimbal_controller_jacobian.py](../../../../controller/gimbal_controller_jacobian.py) train side.

### Hypotheses (ranked by likelihood; each maps to specific signals to compare)

The diagnostic plan below is designed to **distinguish among these** rather than assume any one. Each row predicts which signal pair will diverge first.

| # | Hypothesis | Mechanism | Signal that diverges first |
|---|---|---|---|
| **H1** | **EKF lag on `omega_body` > what training models** | Train injects 15 ms ang-vel lag (ticket 041 SITL value); real mavros may deliver 20–40 ms because of MAVLink scheduling + ROS2 hop on top of EKF2. Body-rejection input is wrong-phase. | `omega_body[mavros]` vs `omega_body[Pegasus state/twist]` — cross-correlation peak ≠ 15 ms |
| **H2** | **Joint-inner-loop not modeled in train** | Train commands joint positions and assumes the articulation tracks instantly; deploy's Pegasus articulation has its own PD (joint stiffness + damping) that adds 1–2 control cycles of extra phase lag on top of `omega_body` lag. | `gimbal_joint_yaw_actual - gimbal_joint_yaw_cmd` non-zero in deploy, ~0 in train |
| **H3** | **Update-rate mismatch** | Train fires every 10 ms (100 Hz physics). Deploy yaml defaults to 250 Hz but `control_trigger="joint_states"` ties it to the joint_states publish rate, which may stall when isaac_sim is busy. Effective control dt jitters → discrete-time pole moves outside unit disk. | Inter-sample interval histogram on the deploy command publish |
| **H4** | **`feedback_blend = 0.05` in deploy not present in train** | Deploy's [los_rate_controller.py:265](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py#L265) blends actual joint position back into the integrator (`feedback_blend = 0.05`). If actual joint lags command by more than 1/0.05 = 20 cycles, the feedback term winds up. Train has no equivalent. | Deploy `az_world_integrator` drifts from `(quat→az_world)` over time |
| **H5** | **Rate-loop τ mismatch** | Train uses `tau_yaw_s=0.0995, tau_pitch_s=0.0954` (mas/035 fit). Deploy yaml at [config/los_rate_config.yaml](../../../../../../../../ros2_ws/src/gimbal_stabilizer/config/los_rate_config.yaml) may differ if the yaml has not been kept in sync. | Step-response 63% time on `omega_cmd → omega_actual` differs by > 5 ms |
| **H6** | **Sign convention or quaternion frame flip** | Pegasus state quaternion = xyzw, Isaac Lab = wxyz. The two paths read different `q_body` representations; mismatch in the world→body transform produces correctly-magnituded but wrong-frame compensation. | `(quat→yaw)[train]` and `(quat→yaw)[deploy]` differ in trajectory shape, not just lag |
| **H7** | **Vehicle attitude limit cycle from PX4 alone** | PX4 attitude loop has its own small steady-state oscillation (especially with the new pegasus-tuned gains' higher Kp_rate). Train does not generate this oscillation because iris_ma6's controller is different from PX4. Gimbal body-rejection on a non-existent input on train side → no oscillation; on deploy side, the input IS oscillating → gimbal rejection is correct but acting on a real disturbance. | `omega_body` in deploy hover has 5–20 Hz power that train hover lacks |
| **H8** | **Servo rate saturation under-compensation** | Deploy `servo_rate_limit = 56.5 rad/s` (yaml). When body motion drives `omega_combined` near this limit, the J^{-1}-derived `qdot` saturates, breaking the linearization. | `qdot_pre_clip` vs `qdot_post_clip` differs in deploy when oscillation is present |

H1 + H2 + H7 are mutually compatible and may compound. H6 is unlikely (would have shown in tickets 040 / compare_gimbal.py results) but worth checking. H8 should only fire at large excitation, not at "pointing at target".

### Signals to capture (per timestep, both sides)

All in body-FLU and world-ENU frames as appropriate. Sample rate: physics rate (10 ms) for train; deploy publish rate (10 ms for 100 Hz, or 4 ms for 250 Hz).

| Signal | Train source | Deploy source | Notes |
|---|---|---|---|
| `policy_cmd_yaw_rate`, `policy_cmd_pitch_rate` | `self._gimbal_cmd_yaw_rate` (normalized [-1, 1]) | `/px4_N/gimbal_cmd_los_rate` (Vector3) | Common entry point |
| `post_rateloop_yaw_rate`, `post_rateloop_pitch_rate` | After `GimbalRateLoop.step` | After `dynamics.GimbalRateLoop.step` in [los_rate_controller.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py) | Saturation + first-order lag applied |
| `omega_cmd` (body frame, rad/s) | rate_loop output × max_gimbal_rate | rate_loop output × max_gimbal_rate | Physical units |
| `omega_body` (body frame, rad/s) | `self.robot.data.root_ang_vel_b` (ground truth) — or lagged via ticket 042 buffer when active | `mavros/imu/data.angular_velocity` (default) or `state/twist.angular` (alt) | **H1**: the load-bearing signal |
| `omega_combined` | `omega_cmd - omega_body` | `omega_cmd - omega_body` | J^{-1} input |
| `qdot_ref` (joint velocities, pre-clip) | `_compute_jacobian_inverse_times_omega` output | same | Algorithm output before any saturation |
| `qdot_post_clip` | n/a (no servo limit in train) | After `servo_rate_limit` clip | **H8** check |
| `gimbal_joint_yaw_cmd`, `gimbal_joint_pitch_cmd`, `gimbal_joint_roll_cmd` | Setpoint written to articulation | `/px4_N/isaac_joint_commands.position` | Last stop before plant |
| `gimbal_joint_yaw_actual`, `gimbal_joint_pitch_actual`, `gimbal_joint_roll_actual` | `self._robots[a].data.joint_pos[:, gimbal_idx]` | `/px4_N/isaac_joint_states.position` (echo from Pegasus) | **H2** check |
| `az_world_integrator`, `el_world_integrator` | `self._gimbal._az_world` (internal state) | `self._az_world` (los_rate_controller internal) | LOS integrator state |
| `az_world_from_quat`, `el_world_from_quat` | derived from `q_body × q_gimbal × forward_vec` | derived from same on deploy | **H4** check |
| `q_body` (vehicle quaternion, wxyz) | `self.robot.data.root_quat_w` | `mavros/imu/data.orientation` (wxyz) or `state/pose.orientation` (xyzw, converted) | **H6** check via convention audit |
| `vehicle_omega_body_xy_power_5_20Hz` | FFT power in 5–20 Hz band | same | **H7** check |
| `control_cycle_dt_actual` | `self.physics_dt` (constant) | `now() - prev_callback_t` (jittery) | **H3** check |
| `feedback_blend_residual` | n/a | `_az_world_integrator - (quat→az_world)` | **H4** check |

For multi-vehicle deploy, capture all of the above per vehicle (subscribe to all `/px4_N/...` topics).

### Test scenarios

Each scenario isolates a hypothesis class. Run **both train and deploy** under each; capture all signals above; align by the policy command timestamp.

| # | Scenario | What it isolates | Predicted outcome if all hypotheses are FALSE |
|---|---|---|---|
| **S1** | **Pure hover, policy cmd = 0, target at fixed bearing** | Body-rejection-only path (no LOS integration). | Gimbal joints flat; `omega_combined ≈ 0`; no oscillation either side. |
| **S2** | **Hover + manual sinusoid policy cmd** (yaw 0.1 rad/s, 0.5 Hz) | LOS integration path; tests rate-loop fidelity. | Train and deploy LOS angles overlay within 5%. |
| **S3** | **Hover + small body excitation** (PX4 attitude oscillating at 5 Hz / ±0.5° amplitude — inject via brief stick disturbance in deploy; mirror with `_apply_external_torque` in train) | H1 + H7: stress the body-rejection lag. | If both reject cleanly, gimbal joints stay flat; if H1 active, deploy shows growing oscillation, train does not. |
| **S4** | **Step policy cmd** (yaw +0.5 rad for 1 s then 0) | H5 + H2: rate-loop and joint-inner-loop step response. | 63% rise time within 5 ms of each other; no overshoot beyond cfg saturation. |
| **S5** | **Replay deploy command sequence into train** | Direct A/B at the controller boundary. | Identical gimbal joint trajectory in train. If train is calm and deploy oscillates on **the same input**, the divergence is downstream of the policy → in the body-rejection / joint plant / EKF lag region. |
| **S6** | **Open-loop replay deploy `omega_body` into train** | Isolates whether train's gimbal controller would oscillate given deploy's body input. | If train oscillates too on deploy's lagged `omega_body`, H1/H7 confirmed (train was just getting cleaner input). |

### Comparison method

1. **Time-alignment**: use the `policy_cmd_*` signal as the synchronization channel. Both sides see the same command; cross-correlate to align ±1 timestep.
2. **Per-signal residual**: for each pair, compute `(deploy - train)` per timestep after alignment. Plot residual time series + cumulative residual.
3. **Localize divergence**: find the **first signal in the chain** whose residual exceeds 3σ of its own measurement noise floor. The divergence enters the system there.
4. **Fit delay + gain mismatch**: for each upstream→downstream signal pair (`omega_cmd → omega_combined`, `omega_combined → qdot_ref`, etc.), fit `output(t) = α · input(t - τ) + ε`. Report `(α, τ)` per pair. Identifies the link with the lag/gain mismatch.
5. **Stability margin**: from the rate-loop + joint-inner-loop fit, compute open-loop gain and phase at 5–20 Hz (the band where the oscillation lives). Phase margin < 30° at the resonant frequency → instability cause confirmed.

### Deliverables

- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/gimbal_oscillation_diagnosis.json` — per-scenario, per-signal, per-side time-series statistics + the localized divergence source.
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/oscillation_report.pdf` — multi-page report: per-scenario overlay plots (train trace blue, deploy trace red), residual plots, identified divergence point, fitted (α, τ) per link.
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/raw/*.csv` — raw per-scenario, per-side CSVs.
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/train_recorder.py` — env-side per-step CSV logger. Instruments [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) without changing its behavior (toggle via env-var or cfg flag).
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/deploy_recorder.py` — ROS2 node that subscribes to all relevant topics and writes per-vehicle CSVs.
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/compare_oscillation.py` — analysis script.
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/run_all_scenarios.sh` — wraps the 6 scenarios on both sides.
- **NEW**: `controller/sysid_output/gimbal/oscillation_diagnosis/README.md`.

### Workflow

1. **Write [doc/gimbal_oscillation_diagnosis_spec.md](../../../gimbal_oscillation_diagnosis_spec.md)** — signal list, scenario protocols, alignment rules, divergence-localization algorithm.

2. **Audit existing infra reuse**:
   - [compare_gimbal.py](../../../../controller/sysid_output/gimbal/compare_gimbal.py) does open-loop controller-law comparison — perfect baseline; this ticket extends to closed-loop with physics.
   - Ticket 041's [ekf_lag_recorder.py](../../../../controller/sysid_output/ekf_state_lag/ekf_lag_recorder.py) is the deploy-side ROS2 recorder pattern — reuse the same shape (sim-time stamps, ApproximateTimeSynchronizer).
   - Ticket 040's plant-mode + ticket 041's EKF lag are **active during these tests** — not toggled off, because the question is "given the matched plant + measured EKF lag, why does deploy still oscillate?"

3. **Instrument train side** ([train_recorder.py](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/train_recorder.py)):
   - Hook into env's `_pre_physics_step` and `_apply_action`.
   - Per step, append a row to an in-memory ring buffer; on scenario end (or via a `--dump-after-steps` arg), flush to CSV.
   - Toggle via `IRIS_MA6_GIMBAL_DIAG=1` env-var (off by default — no behavior change for normal runs).
   - Single-env mode (`num_envs=1`) to avoid GPU-CPU sync overhead for diagnostic recording.

4. **Instrument deploy side** ([deploy_recorder.py](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/deploy_recorder.py)):
   - Subscribe to: `/px4_N/gimbal_cmd_los_rate`, `/px4_N/isaac_joint_commands`, `/px4_N/isaac_joint_states`, `/px4_N/mavros/imu/data`, `/px4_N/state/pose`, `/px4_N/state/twist`, `/px4_N/gimbal_state_rpy_rad`, `/px4_N/gimbal_los_state_deg`.
   - Use sim-time stamps (`use_sim_time=true`). One CSV per vehicle per scenario.
   - **Critical**: also subscribe to `los_rate_controller`'s internal state via a NEW diagnostic publisher added to that node (one Vector3 per state we can't observe from outside — `_az_world`, `_el_world`, `qdot_ref_pre_clip`, `feedback_blend_residual`).

5. **Run 6 scenarios on both sides** (workflow):
   - Bring up Pegasus + PX4 SITL + mas stack.
   - Run scenario S1 (deploy) for 60 s → capture deploy CSVs.
   - Stop deploy. Boot train env with `IRIS_MA6_GIMBAL_DIAG=1 num_envs=1` and the same policy commands replayed → capture train CSVs.
   - Repeat for S2–S6.

6. **Run [compare_oscillation.py](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/compare_oscillation.py)**:
   - Loads paired CSVs, time-aligns, computes residuals, identifies first-divergence signal, fits (α, τ) per link.
   - Emits JSON + PDF.

7. **Triage**: read the JSON's `first_divergence_signal` and `fitted_lags` fields → match against the hypothesis table above → write a one-page "root cause" markdown.

8. **Open a follow-up ticket** for the fix. Likely candidates:
   - **Smith predictor on body-rejection path** if H1 confirmed.
   - **Joint-inner-loop model added to training-time gimbal_rate_loop** if H2 confirmed.
   - **Increase deploy update rate or fix `control_trigger`** if H3 confirmed.
   - **Disable `feedback_blend` or shorten its time constant** if H4 confirmed.
   - **Re-sync rate-loop yaml constants** if H5 confirmed.

### Acceptance criteria

- **All 6 scenarios recorded on both sides** with sim-time alignment within ±1 timestep.
- **JSON deliverable populated** with: per-scenario per-signal RMS residual, first-divergence signal, fitted (α, τ) per upstream→downstream pair, phase margin at the resonant frequency.
- **Root-cause identified** to a specific signal pair OR explicitly marked "not localized — multi-cause" with the residual-power breakdown showing the contribution of each hypothesis.
- **PDF report generated** with overlay plots for each scenario and a summary diagnostic table.
- **Follow-up ticket created** with the proposed fix, referencing this ticket's JSON.
- **No env / controller code changes in this ticket** beyond the opt-in `train_recorder.py` instrumentation (env-var gated).
- **Reproducible**: a single `run_all_scenarios.sh` invocation reproduces the full diagnostic from a clean state.

### Scope boundary

- **DO**: capture timeseries on both sides, time-align, compute residuals, localize the first-divergence signal, fit (α, τ) per link.
- **DO**: instrument train side with an opt-in CSV logger (env-var gated) and deploy side with a ROS2 recorder + minimal diagnostic-publish additions to [los_rate_controller.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py).
- **DO**: run all 6 scenarios; report the first-divergence signal per scenario.
- **DO**: open a follow-up ticket with the proposed fix once root cause is localized.
- **DO NOT**: implement the fix in this ticket. This is diagnosis only.
- **DO NOT**: re-tune controller gains. The current pegasus-matched gains are validated by sysid_replicator and pass ticket-040 acceptance.
- **DO NOT**: change training behavior outside the opt-in instrumentation. The env's default code path must produce bit-exact pre-049 outputs when `IRIS_MA6_GIMBAL_DIAG` is unset.
- **DO NOT**: characterize the camera / image / detection paths. Out of scope; this is gimbal-mechanical only.
- **DO NOT**: characterize the policy (RNN state, action distribution). Policy is the SAME on both sides; the divergence is downstream.
- **DO NOT**: re-measure EKF lag (ticket 041 already did, at the time of this writing the JSON is canonical). If H1 wins, the follow-up may rerun ticket 041's measurement against the current MAVROS version to check for drift, but that's a separate ticket.

### Risk

Low (it's a measurement ticket).

1. **`los_rate_controller.py` diagnostic publishers are intrusive code in deployed ROS2 node** — risk: adding publishers changes timing. Mitigation: publish on `BEST_EFFORT` QoS with depth 1; defer publishing until after the joint command is issued (out of the critical control path). Verify with H3 measurement (control_cycle_dt unchanged).
2. **Scenario S3's body excitation injection is fragile** — manual stick disturbance is hard to reproduce. Mitigation: alternate path — instead of injecting, *record naturally occurring oscillation* by running the policy until oscillation emerges (it does, per the user observation), then post-trim the relevant window.
3. **Pegasus state/twist's angular_velocity frame** — body-frame per [los_rate_controller.py:184-186](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py#L184-L186) docstring, but worth a sanity check at recording time (rotating the vehicle yaw → expecting ω_z > 0).

### Coupling

- **mas/035, mas/036, ticket 032** — same rate-loop + dead-time architecture both sides; this ticket asks whether they're *behaving* the same.
- **ticket 040 / 041 / 042** — current best sim2sim alignment; this ticket finds what they missed.
- **ticket 025** — measures comms/detection latency; not directly relevant here (gimbal control runs on the FCU stack, not detection).
- **gimbal_stabilizer / los_rate_controller** — deploy-side under-test; this ticket adds diagnostic publishers + comparison harness.

### Affected files

**New** (all diagnostic infra; no env/controller-default behavior changes):
- NEW: [doc/gimbal_oscillation_diagnosis_spec.md](../../../gimbal_oscillation_diagnosis_spec.md) — signal table, scenarios, alignment + analysis spec.
- NEW: [controller/sysid_output/gimbal/oscillation_diagnosis/train_recorder.py](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/train_recorder.py)
- NEW: [controller/sysid_output/gimbal/oscillation_diagnosis/deploy_recorder.py](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/deploy_recorder.py)
- NEW: [controller/sysid_output/gimbal/oscillation_diagnosis/compare_oscillation.py](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/compare_oscillation.py)
- NEW: [controller/sysid_output/gimbal/oscillation_diagnosis/run_all_scenarios.sh](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/run_all_scenarios.sh)
- NEW: `controller/sysid_output/gimbal/oscillation_diagnosis/gimbal_oscillation_diagnosis.json` — generated.
- NEW: `controller/sysid_output/gimbal/oscillation_diagnosis/oscillation_report.pdf` — generated.
- NEW: `controller/sysid_output/gimbal/oscillation_diagnosis/raw/*.csv` — 12 raw recordings (6 scenarios × 2 sides).
- NEW: [controller/sysid_output/gimbal/oscillation_diagnosis/README.md](../../../../controller/sysid_output/gimbal/oscillation_diagnosis/README.md).

**Edits** (minimal, gated):
- EDIT: [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) — add `if os.environ.get("IRIS_MA6_GIMBAL_DIAG") == "1": ...` hooks in `_pre_physics_step` and `_apply_action` that call into `train_recorder.py`. Off by default; bit-exact pre-049 behavior when unset.
- EDIT: [ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py) — add diagnostic publishers (Vector3 / Float64) for internal states (`_az_world`, `_el_world`, `qdot_ref_pre_clip`, `feedback_blend_residual`, `omega_combined`). Publishers off by default via a `diagnostic_publish: bool = False` ROS2 param.

### References

- [iris_ma6 controller/gimbal_controller_jacobian.py:210-212](../../../../controller/gimbal_controller_jacobian.py#L210-L212) — `omega_combined = omega_cmd - omega_body` definition.
- [ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/los_rate_controller.py) — deploy gimbal controller; parameters `feedback_blend`, `imu_source`, `update_rate`, `control_trigger`, `gimbal_controller_mode`.
- [ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/dynamics.py](../../../../../../../../ros2_ws/src/gimbal_stabilizer/gimbal_stabilizer/dynamics.py) — deploy-side mas/035 GimbalRateLoop port; should match training-side `gimbal_rate_loop.py`.
- [ros2_ws/src/gimbal_stabilizer/config/los_rate_config.yaml](../../../../../../../../ros2_ws/src/gimbal_stabilizer/config/los_rate_config.yaml) — deploy-time τ / max_rate / dead_time values; **audit against training-side defaults**.
- [controller/sysid_output/gimbal/compare_gimbal.py](../../../../controller/sysid_output/gimbal/compare_gimbal.py) — open-loop controller-law comparison (reusable harness pattern; this ticket extends to closed-loop).
- [controller/sysid_output/ekf_state_lag/ekf_lag_recorder.py](../../../../controller/sysid_output/ekf_state_lag/ekf_lag_recorder.py) — ROS2 recorder pattern; deploy_recorder.py mirrors its shape.
- Ticket 040: [040-match-pegasus-physics-parameters/ticket.md](../040-match-pegasus-physics-parameters/ticket.md) — plant parity (active for these tests).
- Ticket 041: [041-px4-ekf-state-lag-measurement/ticket.md](../041-px4-ekf-state-lag-measurement/ticket.md) — `ekf_state_lag.json` (15/18 ms — H1 challenges this).
- Ticket 042: [042-per-channel-ekf-latency-iris-ma6/ticket.md](../042-per-channel-ekf-latency-iris-ma6/ticket.md) — per-channel application of ticket 041 values.

**Flow**: Medium. Three load-bearing pieces: (a) train-side instrumentation hooks (must be opt-in, bit-exact off); (b) deploy-side diagnostic publishers on `los_rate_controller.py` (must not perturb the live control loop); (c) the alignment + divergence-localization algorithm in `compare_oscillation.py`. Estimated 1 commit + 1 day of bench time across both sides + 0.5 day of analysis. Output is a diagnosis pointing at one of the 8 hypotheses; the fix is a separate ticket.

---

### Execution trail

**2026-06-11 — instrumentation + analysis harness LANDED (no live data yet)**

Built all infra deliverables ahead of GPU availability (ticket-048 wide run held the GPU; ETA ~19:37):
- Spec: [doc/gimbal_oscillation_diagnosis_spec.md](../../../gimbal_oscillation_diagnosis_spec.md) — signal contract, scenarios, alignment + divergence-localization algorithm, **§0 pre-flight config audit**, §4.4 calling contract.
- Train: `controller/sysid_output/gimbal/oscillation_diagnosis/train_recorder.py` (env-var `IRIS_MA6_GIMBAL_DIAG`-gated) + gated `_diag_capture` snapshot in `controller/gimbal_controller_jacobian.py` + gated hooks in `iris_ma_env6_test.py` (`__init__` + `_apply_action`). Default path bit-exact (capture under `if self._diag_capture`).
- Deploy: `diagnostic_publish` ROS2 param + 8 `gimbal_diag/*` publishers in `ros2_ws/.../los_rate_controller.py`, emitted AFTER the joint command (Risk #1), BEST_EFFORT depth-1, fully gated. `deploy_recorder.py` ROS2 node (sim-time, per-vehicle CSV, optional gimbal-cmd driver for S1/S2/S4).
- Analysis: `compare_oscillation.py` (pure numpy/matplotlib) — align→residual→first-divergence→(α,τ) fit→H7 band power→JSON+PDF. **Smoke-tested end-to-end** with a synthetic H1/H7 trace: correctly localized divergence to `omega_body`, f_osc=8.0 Hz, voted H1, emitted JSON+PDF.
- Runner + README: `run_all_scenarios.sh`, `README.md`.

**Static findings (spec §0) — several hypotheses pre-answered by the as-shipped config (yaml + code read 2026-06-11):**
- **H4 statically disproven**: `feedback_blend=0.05` is declared in yaml but **never referenced** in `los_rate_controller.py`; the `_az_world` integrator is corrected only by anti-windup, never by actual-joint blending. Recorder still logs `feedback_blend_residual` to confirm no drift.
- **H1 reshaped**: deployed `imu_source: "state"` ⇒ `omega_body` is Pegasus `state/twist` ground truth (sim-time), NOT mavros/EKF-lagged. The H1-as-framed (20–40 ms MAVLink lag) is not the deployed path; residual is sample-and-hold only. Recorder logs `mavros/imu/data` in parallel (aux cols) to quantify what H1 *would* be.
- **H3 reshaped**: `control_trigger: "clock"` (not `joint_states`) ⇒ control fires on `/clock`, not tied to joint_states publish. Still measure `control_dt`.
- **H5 τ in sync** (0.0995/0.0954 both sides), but `max_gimbal_rate` differs (train π vs deploy 1.28) — not exercised at "pointing".
- **H8 unlikely**: `servo_rate_limit=56.5` (was 6.0) clips only above ~100° error.
- ⇒ diagnostic now re-weighted toward **H2** (deploy articulation PD lag absent in train) and **H7** (PX4 attitude limit-cycle feeding a real body oscillation into `state/twist`). Live data makes the final call.

**Pending (needs GPU + deploy stack)**: bring up Pegasus+PX4 SITL+mas with `los_rate_controller diagnostic_publish:=true`, run S1–S4 both sides via `run_all_scenarios.sh`, then S5/S6 replays, then write the one-page root-cause md + open the follow-up fix ticket.

**2026-06-11 (cont.) — rosbag path added + validated; phenomenon does not reproduce**

User supplied recorded rosbags (`/home/usrg/mas/bag/bag_20260611_*_t048_net_width_{wide256,baseline64,mid128}`). Pivoted the deploy data source to offline bag processing — more robust + reproducible than live capture, and needs no controller rebuild:
- NEW `controller/sysid_output/gimbal/oscillation_diagnosis/bag_to_csv.py`: rosbag2 .db3 → canonical-schema deploy CSV. Reads every observable signal; **recomputes** the internal-only signals (`omega_cmd`, `omega_combined`, `qdot_ref`) offline via the exact control law in `ros2_los_math.py` from recorded inputs (`az/el_world`, `q_body`, internal actual joints, `omega_body`). Recompute **validated** vs recorded `isaac_joint_commands.velocity`: mean|Δ| 0.011–0.081 rad/s across 3 vehicles (faithful). `control_dt` derived from `gimbal_state_rpy_deg` publish-stamp deltas (no diag publisher needed).
- Processed all 3 vehicles of the wide bag → `raw/S0_wide_deploy_px4_{1,2,3}.csv` (kept as the **current stability baseline**).

**Findings from the bags (deploy-only analysis):**
- **NO divergence / no limit cycle.** Gimbal motion is bounded: px4_1 pitch std 4.0°→6.6° over 25 s (mild, bounded) at **~0.3–0.5 Hz** (target-tracking band, NOT the hypothesized 5–20 Hz). px4_2/px4_3 pitch flat or decaying; `omega_body` stable on all three. The originally-reported growing oscillation is **not present in current config**.
- **Attribution (evidence, not proof)**: not the servo clip (as-shipped 56.5 rad/s is *looser* than the old 6.0). Most likely the **ticket-047/048 policy command smoothing** (action_delta_penalty −12→−24 + slew clip → ~10× smoother `action_delta`, per progress log) which stops exciting the body-rejection loop. Matches user's "slew rate clip helped".
- **H3 confirmed latent**: deploy control runs at **~50 Hz with ~18% dt jitter** (median 20 ms, std 3.7 ms, max 30 ms) vs training's fixed **100 Hz**. Real sim2sim gap; benign now, would bite if loop gain rises.
- **H4 confirmed dead** in live data: `feedback_blend_residual` ≈ 0 with no drift (consistent with the static finding that `feedback_blend` is unused).
- **No deploy code behavior change from this ticket**: the `los_rate_controller.py` diag publishers are gated (`diagnostic_publish=false` default) AND were never built into the recording node (absent from `ros2_ws/install/`; bags carry no `gimbal_diag/*`).

**To localize the original root cause**, a bag recorded with the **pre-047 checkpoint** (or with the policy slew clip disabled) is required — the harness will process it the same way. Until such data exists, 049 stays open.
