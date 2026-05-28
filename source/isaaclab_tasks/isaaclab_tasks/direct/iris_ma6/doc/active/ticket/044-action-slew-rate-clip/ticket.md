## Ticket 044 — Per-channel slew-rate clip on raw actions (PX4-aligned)

**Status**: Slice 0 complete (2026-05-28) — empirical Δa probe landed. Decision: ship PX4-strict (B1) in Slice 1; expect Slice 2 regression to trigger Slice 4 task-difficulty calibration follow-up.
**Created**: 2026-05-27
**Type**: Implementation + empirical sizing + A/B validation.
**Target setup**: iris_ma6 training (MAPPO-RNN baseline, post-043 default with `enable_prev_action_obs=True, enable_action_lowpass=False`); sim-to-sim transfer to PegasusSimulator + PX4 SITL as the load-bearing case.
**Deliverable**:
1. Empirical Slice-0 analysis: per-channel `|Δaction|` distribution from a trained `prev_action_only` checkpoint sampled at 4 curriculum points (env-step settings 39k / 79k / 119k / 199k via `use_debug_initial_step`). Output: percentile table per channel × per curriculum point, alongside PX4-derived reference values.
2. Env-side patch: per-channel hard slew clip applied to raw actions in `_pre_physics_step` *upstream* of the `_actions` assignment, so reward (`action_sum`, `action_delta`) and prev-action obs all see the same constrained signal.
3. Fix the pre-existing `_last_actions[env_ids].zero_()` no-op (advanced-index copy → assignment).
4. New tracker metrics: `cmd_vel_delta` (mean |Δ cmd_vel| per policy step, the t043-deferred metric) and `slew_saturation_frac` per-channel (fraction of steps where the clip is active).
5. Single feature flag `enable_action_slew_clip` on `IrisMaEnv6TestCfg`, plus six per-channel `action_slew_*` knobs. Flag-off produces bit-exact pre-patch behavior.
6. Bit-exact regression test that locks the flag off → produces a config tree identical to pre-patch.
7. Short A/B training run (200k steps × 1 seed × 2 configs: post-043 default vs slew-on with PX4-strict defaults) confirming smoothness improves without collision_fraction regressing.
8. One-page experiment writeup in `doc/experiments/`.

**What**: Address the residual action-jerkiness gap left by ticket 043. The `prev_action_only` config (ticket 043's diagnostic-turned-default) gave clear task-quality gains (+9% bbox_center, −44% tracking_lost at 200k vs the t040/042 baseline) but produced essentially flat `total_rms` smoothness vs baseline (≈−2%, within seed noise). The `_full` configuration (LP+obs) failed the task-quality bar catastrophically (−40% reward, −31% bbox_center at the 296k step that the late training reached) because (a) τ=0.08 s introduced too much applied-command lag for agile bbox tracking, and (b) the `action_delta` reward operates on raw `_actions` while the LP attenuates downstream, creating a perverse incentive where the policy must issue larger / jittery raw commands to drive the filter. Both failure modes are documented in [`2026-05-26_ticket043_prev_action_lowpass.md`](../../../experiments/2026-05-26_ticket043_prev_action_lowpass.md) and in the t043 run review.

The architectural response is a **hard per-channel slew limit on raw actions**, placed upstream of the `_actions` assignment in `_pre_physics_step`. Three properties this design has that the LP did not:

| Property | LP (t043_full) | Slew clip (this ticket) |
|---|---|---|
| Operates on the same signal the reward penalizes | ❌ (`_actions` raw, filter downstream) | ✅ (`_actions` post-clip) |
| Doesn't introduce lag on steady commands | ❌ (first-step attenuation α≈0.39) | ✅ (steady command passes through unchanged) |
| Constrains rate-of-change directly | indirectly (via EMA attenuation) | directly (Δa ≤ δ_max by construction) |

Because the clip is applied **before** `_actions` is stored, `action_delta` reward = `‖_actions[t] - _actions[t-1]‖` ≤ δ_max by construction. Reward and constraint align — same principle that fixed ticket 043's slice-1 unit test pattern, generalized.

**Why PX4-aligned δ_max defaults**: The principle of sim-to-sim and sim-to-real parity (ticket 040 plant identification; ticket 041 EKF-lag characterization) extends to action-rate envelopes. PX4's MC position controller rate-limits the velocity setpoint through `MPC_ACC_HOR`, `MPC_ACC_HOR_MAX`, `MPC_ACC_UP_MAX`, `MPC_ACC_DOWN_MAX` ([mc_pos_control_params.c:599–638](../../../../../../../../PX4-Autopilot/src/modules/mc_pos_control/mc_pos_control_params.c#L599-L638)). A policy trained without rate-limited actions can issue setpoint trajectories that PX4 will refuse to execute at deployment, opening a sim-to-real gap. The slew clip closes that gap on the env side.

**PX4 reference values** (default cruise, see also airframe overrides in [`6002_draco_r`](../../../../../../../../PX4-Autopilot/ROMFS/px4fmu_common/init.d/airframes/6002_draco_r), [`4020_holybro_px4vision_v1_5`](../../../../../../../../PX4-Autopilot/ROMFS/px4fmu_common/init.d/airframes/4020_holybro_px4vision_v1_5)):

| PX4 param | Value | Meaning |
|---|---|---|
| `MPC_ACC_HOR` | 3.0 m/s² | Cruise horizontal accel |
| `MPC_ACC_HOR_MAX` | 5.0 m/s² | Maximum horizontal accel |
| `MPC_ACC_UP_MAX` | 4.0 m/s² | Maximum vertical accel up |
| `MPC_ACC_DOWN_MAX` | 3.0 m/s² | Maximum vertical accel down |
| `MPC_JERK_MAX` | 8.0 m/s³ | Max jerk (Δ-accel limit, out of v1 scope) |

Converting to env action-space at `dt = sim.dt × decimation = 0.04 s` with our scales (`max_lin_vel = 10 m/s`, `max_vel_z_up = 3.0`, `max_vel_z_dn = 1.5`):

| Action channel | δ_max (PX4-strict) | Derivation | Physical interpretation |
|---|---|---|---|
| vx, vy | **0.020** | `MPC_ACC_HOR_MAX × dt / max_lin_vel = 5 × 0.04 / 10` | 5 m/s² peak horizontal accel |
| vz | **0.053** | `MPC_ACC_UP_MAX × dt / max_vel_z_up = 4 × 0.04 / 3` (worst-case at sign flip under asymmetric envelope) | 4 m/s² peak vertical accel up |
| yaw_rate | **0.30** | No PX4 acceleration analog (PX4 limits the *rate*, not the *rate of rate*); preserves ~3-step full-range reversal | Allows agile turning |
| gimbal_yaw_rate | **0.40** | n/a (gimbal not PX4-controlled — SIYI A8 native limits) | Allows fast gimbal tracking |
| gimbal_pitch_rate | **0.40** | same | same |
| zoom_rate | **0.20** | Visual stability matters more than zoom response time | Slowest channel |

These are **PX4-strict**: a policy under δ_xy = 0.020 takes 50 policy steps (~2 s) to traverse the full velocity envelope from rest, exactly matching PX4's position-controlled behavior. Slice 0 empirically validates whether the trained `prev_action_only` policy already operates within this envelope; if it routinely exceeds it, the policy is operating in a regime that won't transfer to PX4-flown hardware.

### Patch summary

1. **`iris_ma_env6_test_cfg.py`** — new cfg block under `IrisMaEnv6TestCfg`:
   ```python
   # --- ticket 044: per-channel slew-rate clip on raw actions ---
   enable_action_slew_clip: bool = True
   """When True, hard-clip the per-channel Δ between this step's policy
   action and the previous step's stored `_actions` to ±`action_slew_*`,
   applied *before* `_actions` is assigned. This bounds Δa ≤ δ_max by
   construction; `action_delta` reward and prev-action obs see the same
   constrained signal. Flag-off path is bit-exact to pre-patch."""

   # Per-channel slew limits in action units [-1, 1] per policy step.
   # PX4-derived defaults assuming dt = sim.dt × decimation = 0.04 s.
   action_slew_vel_xy: float = 0.020          # MPC_ACC_HOR_MAX = 5 m/s², envelope 10 m/s
   action_slew_vel_z: float = 0.053           # MPC_ACC_UP_MAX = 4 m/s², worst-case envelope 3 m/s
   action_slew_yaw_rate: float = 0.30         # no PX4 analog; ~3-step full reversal
   action_slew_gimbal_yaw_rate: float = 0.40  # gimbal not PX4-controlled
   action_slew_gimbal_pitch_rate: float = 0.40
   action_slew_zoom_rate: float = 0.20
   ```
   No `__post_init__` change (obs_dim unchanged).

2. **`iris_ma_env6_test.py`** —
   - Add precomputed tensor in `__init__` (alongside `_action_slew_max` mirror of cfg):
     ```python
     self._action_slew_max = torch.tensor([
         self.cfg.action_slew_vel_xy,
         self.cfg.action_slew_vel_xy,
         self.cfg.action_slew_vel_z,
         self.cfg.action_slew_yaw_rate,
         self.cfg.action_slew_gimbal_yaw_rate,
         self.cfg.action_slew_gimbal_pitch_rate,
         self.cfg.action_slew_zoom_rate,
     ], dtype=torch.float32, device=self.device)
     ```
   - In `_pre_physics_step` ([line 815](../../../iris_ma_env6_test.py#L815)), insert the clip *between* the existing `[-1, 1]` clamp and the `_actions` assignment:
     ```python
     action = torch.clamp(action, min=-1.0, max=1.0)
     if self.cfg.enable_action_slew_clip:
         lo = self._last_actions[agent_id] - self._action_slew_max
         hi = self._last_actions[agent_id] + self._action_slew_max
         action = torch.clamp(action, min=lo, max=hi)
         # _last_actions ∈ [-1, 1] from the prev step's clamp, so the
         # composition naturally stays inside [-1, 1] within ±δ_max.
     self._actions[agent_id] = action.clone()
     ```
     This implicitly capacitor-clips the post-reset first action: with `_last_actions[env_ids] = 0`, the first action after reset is bounded to `[-δ_max, +δ_max]`. **This is the intended behavior** — the drone shouldn't request a full-magnitude command from rest in one step. Depends on the bug fix below.
   - In `_reset_idx` ([line 3115](../../../iris_ma_env6_test.py#L3115)), fix the pre-existing `_last_actions` reset no-op:
     ```python
     # BEFORE: self._last_actions[agent_id][env_ids].zero_()  # operates on copy!
     # AFTER:
     self._last_actions[agent_id][env_ids] = 0.0
     ```
     `tensor[long_tensor].method_inplace()` operates on the advanced-index copy, not the underlying buffer. The assignment form uses `__setitem__` and IS in-place. Benign for `action_delta` reward (cross-episode delta is meaningless anyway), fatal for the slew clip (first action after reset would otherwise be pinned near the previous episode's last action).
   - Add `cmd_vel_delta` tracker metric in `_get_rewards`: mean of `|cmd_vel[t] - cmd_vel_prev[t-1]|` per policy step, per agent. Logged as `Info / Action_Smoothness/cmd_vel_delta_drone_X`. Mirrors the existing `_action_delta_sq_acc` pattern; requires a `self._cmd_vel_prev` buffer (similar to `_last_actions`) updated at the end of each policy step. **This is the metric the t043 ticket promised but didn't add** — adding it here also retrofits the LP-on case if anyone ever revisits ticket 043 Slice 3.
   - Add per-channel `slew_saturation_frac` tracker: fraction of policy steps where the slew clip was active for each channel. Logged as `Info / Action_Smoothness/slew_saturation_drone_X_ch_Y`. Set to 0 when flag is off.

3. **No changes to CBF, delay system, controller, observations, or any other module.** The slew clip is upstream of `cmd_vel` and downstream of the policy output — a single insertion point.

### Slices

**Slice 0 — empirical |Δaction| distribution at 4 curriculum points × dual checkpoints.**
- Script: `experiments/scripts/probe_action_delta_distribution.py` (new).
  - Args: `--checkpoint <path>`, `--debug_step <int>`, `--episodes <int>` (default 10), `--num_envs <int>` (default 1024), `--output_csv <path>`.
  - AppLauncher template; loads env with `cfg.use_debug_initial_step=True, cfg.debug_initial_step=<step>`, loads a MAPPO-RNN checkpoint (model + preprocessor states, same loader as `play_iris_mappo_rnn.py`), rolls out `episodes × steps_per_episode` policy steps in deterministic-mean mode across `num_envs` parallel envs, captures `_actions[t] - _actions[t-1]` per agent per channel, writes per-channel `{p50, p90, p95, p99}` to CSV.
  - `episodes × num_envs = 10 × 1024 = 10240` episode-equivalents per probe call → ~5.12M Δa samples per channel per agent. Wall-time per call ≈ 8 min (5000 steps × 4 sim substeps × ~10 ms/step on a 3090). Total 7-call probe budget: ~1 wall-hour.
- **Dual-checkpoint design** — at each curriculum point, probe BOTH (a) the **nearest-greater training checkpoint** (captures the policy in its near-natural state at that curriculum) and (b) the **final checkpoint** (captures what the fully-trained policy would do if thrown back into that curriculum). The pair shows the *developmental trajectory* of the action distribution per curriculum phase.

  | Curriculum point (`debug_step`) | Probe checkpoints |
  |---|---|
  | 39,000  | `agent_40000.pt`  + `agent_200000.pt` |
  | 79,000  | `agent_80000.pt`  + `agent_200000.pt` |
  | 119,000 | `agent_120000.pt` + `agent_200000.pt` |
  | 199,000 | `agent_200000.pt` (single — 200k is already the final checkpoint) |

  All checkpoints sourced from `logs/skrl/iris_ma6/2026-05-26_22-27-56_..._ticket043_prev_action_obs_true_action_lowpass_false/checkpoints/`. 7 probe calls total.

- Wrapper: `experiments/scripts/run_action_delta_probe.bash` — runs the 7 probe calls sequentially, writes per-call CSVs into a timestamped output dir, then aggregates them into one comparison report (`action_delta_probe_summary.csv` keyed by `(checkpoint, debug_step, agent, channel)`).

- Decision points (read from the aggregate):
  - For each `(channel, debug_step, checkpoint)` triple, compare the **empirical p90** against the **PX4-strict δ_max**.
  - **Case A (empirical p90 ≤ PX4-strict at all 4 step settings, both checkpoints)**: the trained policy already operates within the PX4 envelope across the curriculum. Ship PX4-strict defaults as v1. Slice-2 A/B should produce minimal regression.
  - **Case B (empirical p90 > PX4-strict at some step settings)**: the policy is requesting Δa outside the PX4 envelope. Two paths:
    - (B1) Ship PX4-strict anyway and accept the regression — sim2real-positive but task quality may suffer. Slice 2 quantifies the cost.
    - (B2) Loosen defaults to e.g. 2× PX4-strict (airframe-override-aligned: δ_xy = 0.040, δ_z = 0.107) and re-run Slice 2.
  - **Case C (empirical p90 ≪ PX4-strict)**: the policy is naturally smooth. The slew clip is a guardrail, not a constraint. Ship PX4-strict; expect saturation_frac to be very low everywhere.
  - **Diagnostic from dual-checkpoint comparison**: if `agent_40000.pt @ 39k` has much higher p90 than `agent_200000.pt @ 39k`, the late-stage policy has learned to be smoother *even in the acquisition regime*. That's a useful prior for the slew-clip's expected behavior. The opposite (final p90 > early p90) would indicate late-stage curriculum is *teaching* the policy to be jerkier — relevant for sizing decisions.

- No code edits to the env are required for Slice 0 (the script only reads checkpoints). This is a pure analysis pass.

**Slice 1 — patch + bit-exact regression unit test.**
- Implements the cfg block, env-side state, the `_pre_physics_step` clip insertion, the `_last_actions` reset bug fix, the `cmd_vel_delta` and `slew_saturation_frac` trackers.
- Unit test under `tests/test_action_slew_clip_v1.py`. Categories:
  - Cfg defaults: `enable_action_slew_clip=True`, per-channel `action_slew_*` match spec.
  - Bit-exact flag-off regression: with `enable_action_slew_clip=False`, `_actions = clamp(action, -1, 1)` exactly as pre-patch; `cmd_vel` byte-equal to action × scale; checked-in golden hash on a fixed-seed env step.
  - Slew enforcement: with `enable_action_slew_clip=True`, random action sequence → `‖_actions[t] - _actions[t-1]‖_∞ ≤ δ_max` everywhere.
  - Post-reset first-action bounded: after `_reset_idx(env_ids)`, `_last_actions[env_ids] == 0` (validates the bug fix), and the first `_pre_physics_step` produces `|_actions[env_ids]| ≤ δ_max`.
  - Composition with `[-1, 1]` clamp: with `_last_actions = 0.95` and δ = 0.30, the effective range is `[0.65, 1.0]` (slew bound at the bottom, magnitude bound at the top).
  - Reset zero semantics: explicit test that `_last_actions[env_ids]` is in fact zero post-reset (catches regression of the no-op bug).
  - `slew_saturation_frac` metric: with an action sequence that always saturates the clip, saturation_frac → 1; with an always-zero action, saturation_frac → 0.
  - `cmd_vel_delta` metric: matches `mean(|cmd_vel[t] - cmd_vel[t-1]|)` computed offline from the same rollout.
- AppLauncher template per [iris_ma6/CLAUDE.md](../../../CLAUDE.md) "Generating Tests for Functional Modules".

**Slice 2 — short A/B training run.**
- Two configs, same RNG seed (42), same curriculum, 200k steps each (~24 wall-h):
  - `validation_action_slew_short_baseline`: `enable_action_slew_clip=False`, `enable_prev_action_obs=True`. The current default (post-043).
  - `validation_action_slew_short_treatment`: `enable_action_slew_clip=True`, `enable_prev_action_obs=True`. The shipping candidate.
- Add to `experiments/registry.py`.
- Metrics tracked:
  - **Smoothness (load-bearing)**: `action_delta`, `action_sum`, `total_rms`, plus the new `cmd_vel_delta` and `slew_saturation_frac`.
  - **Task quality**: `bbox_center`, `bbox_size`, `pair_valid_rate`, `all_invalid_rate`, `tracking_lost_fraction`, `triangulation_rmse`.
  - **Safety (load-bearing for slew/CBF interaction)**: `collision_per_env`, `collision_fraction`, `cbf_penalty`.
  - **Training health**: mean reward (last-10 avg), KL band stability, σ trace.
- Acceptance bar:
  - `treatment` reduces `action_delta` (sum, last-10 avg) by ≥30% vs `baseline` at 200k.
  - `treatment` task-quality regressions within ±5% of baseline on bbox_center, bbox_size, pair_valid_rate, triangulation_rmse.
  - `treatment` `collision_fraction` not regressed by more than 2× vs baseline. **This is the load-bearing safety gate** — a tight slew clip can starve CBF emergency-braking authority. A 2× regression on `collision_fraction` (e.g., 0.001 → 0.002) is borderline acceptable given the smoothness gain; >2× is a hard stop.
  - `slew_saturation_frac` on at least one velocity channel (vx, vy, vz) is in the band [0.05, 0.40] — too low means the clip is irrelevant (loosen δ), too high means the clip is starving the policy (loosen δ).

**Slice 3 — δ_max sensitivity sweep (conditional, sequential).**
- Only execute if Slice 2 fails the task-quality bar (>5% regression) while clearing the safety bar.
- 100k-step runs at δ_xy ∈ {0.020 (PX4-strict, Slice-2 value), 0.040 (airframe-override 2×), 0.080 (4×)}; δ_z scaled proportionally; gimbal/zoom fixed at default.
- Acceptance: pick the smallest δ_xy that hits the smoothness bar without >5% task-quality regression.
- Note: if Slice 0 finds Case B and we shipped Slice 2 with PX4-strict, Slice 3 is the natural follow-up.

**Slice 4 — task-difficulty calibration (conditional, follow-up ticket).**
- Triggered by Slice 2 catastrophic task regression under PX4-strict δ_max. Slice 0 results (2026-05-28) confirm the trained policy operates 9–15× over PX4-strict on velocity channels (vy p90 = 0.295 vs δ_xy = 0.020; vx p90 = 0.224); shipping PX4-strict predicts severe reward collapse.
- **Design philosophy**: rather than loosen the slew clip to match the policy (which would defeat the sim-to-real motivation), dial down the task so the policy *can* be smooth enough to operate within PX4 envelopes. If the trained policy needs 15× the PX4 jerk envelope to track the target, the task is asking for behavior that cannot transfer to PX4-flown hardware.
- Candidate task-difficulty levers (separate ticket):
  - Lower `max_lin_vel` (currently 10 m/s) — e.g., to 5–7 m/s. Directly reduces vel-command magnitude → smaller required Δa.
  - Slow target motion (curriculum end_step shifted out, or peak speed reduced).
  - Soften bbox_center / bbox_size reward weights (less reactive demand from the tracker).
  - Loosen initial-state cylinder (less aggressive spawn distribution).
  - Reduce DR aggressiveness (FOV / mass / gimbal jitter), which currently forces over-correction.
- **Out of scope for 044 itself.** The ticket completes when Slice 1+2 land cleanly. Slice 4 spawns as a separate ticket if needed.

### Method (training-time validation)

1. **Slice 0**: load the prev_action_only checkpoint from `2026-05-26_22-27-56_..._ticket043_prev_action_obs_true_action_lowpass_false`, run the 4-point probe, decide between Cases A/B1/B2/C, finalize δ_max defaults.
2. **Slice 1 pre-flight**: unit test passes; smoke-test env construction with flag on/off; confirm cmd_vel_delta and slew_saturation_frac log to TB.
3. **Slice 2 launch**: branch from current default-cfg state (post-043), apply patch, register experiments, launch two runs. Tensorboard pull at 40k, 80k, 120k, 200k for early-stop on obvious safety regression.
4. **Decision rule**: `treatment` ships if it hits the smoothness bar without regressing task quality and without exceeding the 2× `collision_fraction` ceiling. If smoothness clears but task quality regresses, defer to ticket 002 reward retuning (don't roll back 044). If safety regresses, Slice 3 sweep is mandatory before shipping.

### Acceptance criteria

| Criterion | Result |
|---|---|
| Patch landed on `iris_ma_env6_test_cfg.py` and `iris_ma_env6_test.py` per §"Patch summary" | pending |
| `_last_actions[env_ids].zero_()` no-op bug fixed | pending |
| `cmd_vel_delta` and `slew_saturation_frac` trackers added to `_get_rewards` | pending |
| `enable_action_slew_clip=False` produces bit-exact pre-patch `_actions` + `cmd_vel` for a fixed seed | unit test green (Slice 1) |
| Post-reset first action bounded to `[-δ_max, +δ_max]` (validates `_last_actions` reset fix) | unit test green (Slice 1) |
| Slice-0 probe produces 4-point × 7-channel percentile table; defaults set | analysis output (Slice 0) |
| `treatment` Slice-2 run reduces `action_delta` ≥30% vs `baseline` at 200k | tensorboard |
| `treatment` task-quality within ±5% of baseline | tensorboard + writeup |
| `treatment` `collision_fraction` ≤ 2× baseline | tensorboard |
| `treatment` `slew_saturation_frac` ∈ [0.05, 0.40] on at least one velocity channel | tensorboard |
| `doc/experiments/<date>_ticket044_action_slew_clip.md` writeup landed | written |
| No regression in any existing experiment in the registry (spot-check on a representative one) | confirmed |

### Scope boundary

- **DO**: insert the slew clip in `_pre_physics_step` upstream of `_actions`; fix the `_last_actions` reset bug; add `cmd_vel_delta` and `slew_saturation_frac` trackers; validate via Slice-0 → Slice-1 → Slice-2 pipeline; document.
- **DO**: ship `enable_action_slew_clip=True` as the new cfg default once Slice 2 passes; legacy behavior opt-in via flag-off.
- **DO**: use PX4-strict δ_max defaults (vel_xy=0.020, vel_z=0.053) for Slice 2. The sim-to-sim / sim-to-real motivation makes PX4 alignment the principled starting point. Slice 3 sweeps looser only if Slice 2 fails.
- **DO NOT**: retune reward weights (`action_delta_penalty_scale`, bbox weights, etc.). That is ticket 002's scope and should follow 044, not precede it.
- **DO NOT**: add a soft-saturation slew (tanh smoothing). Hard `torch.clamp` matches the existing `[-1, 1]` magnitude clamp and is the standard pattern.
- **DO NOT**: add a curriculum on δ_max. Constant slew limit in v1 (per design decision); reconsider only if Slice 2 shows acquisition-phase regression.
- **DO NOT**: implement a `Δ²action` (jerk) limit equivalent to `MPC_JERK_MAX`. Out of scope for v1; tracked as a future possible ticket.
- **DO NOT**: revisit ticket 043's LP. The slew clip is an architectural alternative, not an addition. If a future ticket wants to combine them, that's a separate decision; 044 ships the slew alone.
- **DO NOT**: change controller gains, EKF lag values, asymmetric-z envelope, or curriculum.

### Risk

Low to medium.

1. **CBF emergency-braking starvation (load-bearing safety risk).** A tight slew clip limits the policy's deceleration authority. PX4-strict δ_xy = 0.020 allows full velocity reversal (10 m/s → -10 m/s) in 100 steps = 4 s, which is too slow for worst-case high-speed-closing CBF avoidance scenarios. Mitigation: Slice 2's `collision_fraction ≤ 2× baseline` is the hard gate. If breached, Slice 3 sweep with looser δ_xy (0.040, 0.080) is mandatory. The aligned-units chosen also mean a deployed PX4 controller would *also* refuse to execute faster maneuvers — the slew clip doesn't introduce a sim-to-real gap, it surfaces an existing one.

2. **Gradient starvation when policy persistently saturates the clip.** `torch.clamp` has zero gradient outside the band. PPO already has zero-gradient regions in its clipped surrogate and converges fine, but a channel that saturates >50% of the time stops getting directional signal. Mitigation: `slew_saturation_frac` per-channel metric is the diagnostic; if any channel sits >50%, loosen that channel's δ.

3. **PX4-strict defaults are too aggressive for the current trained policy** (Slice-0 outcome Case B). If the prev_action_only policy routinely operates outside the PX4 envelope, Slice 2 will show a regression. Mitigation: the Slice-0 analysis happens *before* Slice 2 launch precisely to surface this — decision tree is documented.

4. **Asymmetric z envelope interaction.** With `enable_asymmetric_z_envelope=True` (default), action[2] is scaled by `max_vel_z_up=3.0` (action>0) or `max_vel_z_dn=1.5` (action<0). δ_z = 0.053 (PX4-strict) bounds the worst-case Δvz at sign-flip transitions: `|Δaction[2]| × max(3, 1.5) = 0.053 × 3 = 0.16 m/s/step ≈ 4 m/s² = MPC_ACC_UP_MAX`. For non-sign-flipping transitions, the effective slew is tighter (0.053 × 1.5 = 0.08 m/s/step ≈ 2 m/s² for sustained descend). This asymmetry is a feature, not a bug — it matches PX4's own asymmetric acceleration envelope.

5. **Pre-existing `_last_actions` reset bug surfaces.** Fixed in this ticket; the unit test explicitly validates post-reset `_last_actions[env_ids] == 0`. Cross-episode `action_delta` reward distribution shifts very slightly post-fix (delta no longer carries previous-episode tail) — minor training-curve perturbation, documented in the Slice-2 writeup.

6. **CBF training-penalty alignment**. CBF's training penalty in `_get_rewards` reads `self.cmd_vel[:, :, 0:3]` ([iris_ma_env6_test.py:1769](../../../iris_ma_env6_test.py#L1769)). With the slew clip upstream of `_actions`, `cmd_vel` reflects the limited command — penalty is on what was actually issued. Aligned by the same principle that fixed the t043 LP misalignment.

7. **CBF deploy-filter interaction (not v1 critical, documentation only).** The hard deploy filter ([cbf_safety/deploy_filter.py](../../../cbf_safety/deploy_filter.py)) is currently `enable_deployment_filter=False` in training. When deployed (filter on), it intercepts `cmd_vel` and may modify it for safety. Because the slew clip pre-smooths `cmd_vel` (reduced magnitude / smoothed rate), the deploy filter **should** activate less often — the `filter_active_fraction` metric is the diagnostic. This is a side effect, not a bug; documented for whoever flips `enable_deployment_filter=True` later.

### Coupling

- **Ticket 040** (Pegasus physics parity, landed) — sister sim2sim ticket; the slew limits are derived against ticket 040's plant + matched-gain controller envelope.
- **Ticket 041** (PX4 EKF lag measurement, landed) — the rate-envelope analog of ticket 041's latency-envelope analysis. Same sim2sim alignment philosophy applied to a different signal axis.
- **Ticket 042** (per-channel EKF latency, landed) — orthogonal; ticket 042 operates on the observation path (input to the policy), 044 operates on the action path (output from the policy). They stack additively.
- **Ticket 043** (prev-action-in-obs, landed; LP deprecated to opt-in) — 044 supersedes the LP as the smoothness-mechanism default. `enable_prev_action_obs=True` (kept on default) provides the obs channel for the GRU to anticipate the slew, helping the policy learn to operate within the clip.
- **Ticket 002** (reward weight retuning) — follow-up. After 044 lands, `action_delta_penalty_scale` and bbox weights should be revisited against the new action geometry.
- **Experiments framework** — adds two experiments (`validation_action_slew_short_baseline`, `_treatment`) to the registry; existing pipeline unchanged.

### Affected files

**Edits**:
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — new cfg block (flags + per-channel δ defaults).
- [iris_ma_env6_test.py](../../../iris_ma_env6_test.py) — `__init__` adds `_action_slew_max` constant + `_cmd_vel_prev` buffer + `_slew_saturation_acc` accumulator; `_pre_physics_step` inserts the slew clip; `_reset_idx` fixes the `_last_actions` reset bug + zeros the new buffers; `_get_rewards` logs `cmd_vel_delta` and `slew_saturation_frac`.

**New**:
- `tests/test_action_slew_clip_v1.py` — Slice 1.
- `experiments/scripts/probe_action_delta_distribution.py` — Slice 0 probe.
- `experiments/scripts/run_action_delta_probe.bash` — Slice 0 wrapper.
- `experiments/registry.py` — add `validation_action_slew_short_{baseline, treatment}` (Slice 2).
- `doc/experiments/<date>_ticket044_action_slew_clip.md` — Slice 2 writeup.

### References

- [iris_ma_env6_test.py:815](../../../iris_ma_env6_test.py#L815) — `_pre_physics_step` (action → cmd_vel mapping; slew clip insertion point).
- [iris_ma_env6_test.py:338](../../../iris_ma_env6_test.py#L338) — `_last_actions` buffer init.
- [iris_ma_env6_test.py:1769](../../../iris_ma_env6_test.py#L1769) — CBF training penalty reads `cmd_vel` (interaction point).
- [iris_ma_env6_test.py:1893](../../../iris_ma_env6_test.py#L1893) — `_last_actions[agent_id] = self._actions[agent_id].clone()` (per-step update).
- [iris_ma_env6_test.py:3115](../../../iris_ma_env6_test.py#L3115) — pre-existing `_last_actions[env_ids].zero_()` no-op bug (this ticket fixes it).
- [PX4-Autopilot/src/modules/mc_pos_control/mc_pos_control_params.c:599](../../../../../../../../PX4-Autopilot/src/modules/mc_pos_control/mc_pos_control_params.c#L599) — PX4 MC position controller acceleration params (`MPC_ACC_HOR_MAX`, `MPC_ACC_UP_MAX`, `MPC_ACC_DOWN_MAX`, `MPC_JERK_MAX`).
- [PX4-Autopilot/ROMFS/.../airframes/4020_holybro_px4vision_v1_5](../../../../../../../../PX4-Autopilot/ROMFS/px4fmu_common/init.d/airframes/4020_holybro_px4vision_v1_5) — airframe override example (2× cruise defaults).
- [exp 037 doc](../../../experiments/2026-05-22_037_critic_priv_obs.md) — original `action_delta`-growing-more-negative evidence motivating the smoothness work.
- [exp 043 writeup](../../../experiments/2026-05-26_ticket043_prev_action_lowpass.md) — t043 results and the LP failure mode that motivated the slew alternative.
- [iris_ma6/CLAUDE.md](../../../CLAUDE.md) — test-writing conventions for Slice 1.
- Literature: PX4 documentation on position-controller acceleration limits; Hwangbo et al. (RAL 2017) and Kaufmann et al. (Nature 2023) on rate-limited action spaces in quadrotor sim-to-real pipelines.

**Flow**: Low to medium. Slice 0 is a clean read-only analysis (~30 min wall + 1 h compute). Slice 1 patch is mechanically straightforward (a buffer, a clamp insertion, a one-character bug fix, two new metrics). Risk concentrates in Slice 2's safety gate — a tight slew clip could starve CBF avoidance. Estimated 3–4 days: (a) Slice 0; (b) Slice 1 patch + tests; (c) Slice 2 launch (~24 wall-h); (d) analyze + writeup; (e) optionally Slice 3.
