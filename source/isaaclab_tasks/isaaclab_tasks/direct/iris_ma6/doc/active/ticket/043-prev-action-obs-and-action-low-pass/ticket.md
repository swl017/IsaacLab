## Ticket 043 — Prev-action-in-obs + first-order low-pass on cmd_vel

**Status**: Proposed
**Created**: 2026-05-26
**Type**: Implementation + A/B validation.
**Target setup**: iris_ma6 training (MAPPO-RNN baseline); sim-to-sim transfer to PegasusSimulator + PX4 SITL as the load-bearing case.
**Deliverable**:
1. Env-side patch adding (a) the previous applied velocity/rate command (in physical units) to the per-agent ego observation, and (b) a per-channel first-order low-pass on `cmd_vel` between the policy output and the controller input.
2. Per-channel time-constant cfg knobs on `IrisMaEnv6TestCfg` with one feature flag for each change (LP and prev-action) so the pre-patch behavior remains opt-in for regression.
3. Bit-exact regression test that locks both feature flags off → produces a config tree identical to pre-patch.
4. Short A/B training run (200k steps × 1 seed × 3 configs: baseline, prev-action-only, prev-action + LP) that confirms the smoothness improvement is real and task quality is preserved.
5. One-page experiment writeup in `doc/experiments/`.

**What**: Address the two architectural drivers of jerky actions identified in the 2026-05-26 review of `iris_ma_env6_test.py` (see prior session memo / experiment doc [`2026-05-26_01-38-37_..._ticket040_042_pegasus_matched_params.md`](../../../experiments/2026-05-26_01-38-37_mappo_rnn_torch_02c34fe802_ticket040_042_pegasus_matched_params.md)):

| Change | What is added | Where in code today | Architectural effect |
|---|---|---|---|
| **Prev-action in obs (per-agent ego)** | 7 dims (vx, vy, vz [m/s], yaw_rate [rad/s], gimbal yaw rate, gimbal pitch rate, zoom rate [all normalized]) of the *applied* command from the previous policy step. | `_last_actions` exists at [iris_ma_env6_test.py:338](../../../iris_ma_env6_test.py#L338) but is consumed **only** by the `action_delta` reward at [line 1744](../../../iris_ma_env6_test.py#L1744). Not in `_get_observations` ([line 2028](../../../iris_ma_env6_test.py#L2028)). | Gives the GRU a direct, in-channel reference for action continuity. Today the policy has to *infer* its previous command from the EKF-lagged state (18 ms attitude / 15 ms ang-vel after ticket 042). With prev-action in obs, action continuity becomes architecturally regularized rather than purely reward-mediated (the `action_delta` reward is losing the tug-of-war with bbox tracking — see [exp 037](../../../experiments/2026-05-22_037_critic_priv_obs.md): `action_delta` grows more negative over training while reward also drops). |
| **First-order low-pass on `cmd_vel`** | Per-channel τ (time constants in seconds), discretized exactly as `α = 1 − exp(−dt/τ)`. Applied in `_pre_physics_step` after the raw action→cmd_vel scaling, before `_apply_action` consumes `cmd_vel`. | `_pre_physics_step` ([line 815](../../../iris_ma_env6_test.py#L815)) writes `cmd_vel` directly from the scaled action with no filtering. Then `_apply_action` ([line 853](../../../iris_ma_env6_test.py#L853)) reads `cmd_vel` every sim step (4×, ZOH inside the decimation window). | Turns the 40 ms step-staircase at the controller input into a ramped trajectory. The PegasusSimulator-matched gain set (ticket 040) was tuned for smooth setpoint trajectories, not 40 ms steps. The LP also makes "previous applied command" a meaningful obs channel even if the policy itself outputs a step. |

**Physical-units convention for prev-action obs.** The obs channel is the **applied filtered command** (`cmd_vel_filt`), not the raw normalized action. Rationale:
- The per-env, per-agent curriculum + DR jitter on `_max_lin_vel` means a normalized action of `0.6` corresponds to a different m/s in different envs. Feeding physical units makes the channel semantically uniform across the batch.
- At deployment the policy issues commands in m/s through MAVROS; conditioning on the *same physical quantity* it produces closes a representational gap to real flight.
- The legged-locomotion convention (Kaufmann et al. 2023; Rudin et al. 2021; Hwangbo et al. 2017) follows the same pattern — joint-position targets / body-rate commands are fed in physical units, not normalized.
- For gimbal/zoom channels [4-6] which are already passed normalized into the gimbal/zoom controllers, the "physical" representation **is** the normalized value — those channels pass through unchanged.

**Low-pass parameterization.** Per-channel τ (seconds), with discretization
```
dt = sim.dt * decimation              # 0.04 s today (25 Hz policy)
α[ch] = 1.0 - exp(-dt / τ[ch])        # exact continuous-equivalent pole
cmd_vel_filt[ch] = α[ch] * cmd_vel_raw[ch] + (1 - α[ch]) * cmd_vel_filt[ch]
```
The `1 − exp(−dt/τ)` form is preferred over the Euler `dt/(τ+dt)` because it stays stable for `dt > τ` and is the exact discrete match to a continuous first-order pole. Sampling-rate invariance: changing `decimation` or `sim.dt` does not change the filter's continuous-time response. Per-channel τ allows the gimbal/zoom to be tuned independently of translational vel (a stiff gimbal loop can tolerate a tighter τ than the body-frame velocity loop).

**Defaults** (proposed for first pass; tunable):
- `tau_vel_xy_s = 0.08` (cutoff ≈ 2 Hz) — below the inner-loop velocity-controller bandwidth, above visible jerk frequencies in current rollouts.
- `tau_vel_z_s = 0.08`
- `tau_yaw_rate_s = 0.08`
- `tau_gimbal_yaw_rate_s = 0.04` — gimbal does not couple into body attitude, can be faster.
- `tau_gimbal_pitch_rate_s = 0.04`
- `tau_zoom_rate_s = 0.10` — zoom should be the smoothest channel (visual stability matters more than zoom response time).

**On observation-side low-pass (NOT in scope).** Considered and rejected for this ticket. The ego state already goes through 18/15 ms EKF lag (ticket 042); double-filtering compounds lag. Intermittent inter-agent updates are already handled architecturally by the `data_age` and `bbox_age` channels in the obs, which give the GRU an explicit staleness signal it can condition on — that is the correct mechanism for handling skipped/stale updates, not a downstream smoother on hold-last-value signals. If post-training analysis shows jerk that *correlates with* `bbox_empty` / high-`data_age` frames, the right intervention is in the delay system (gating/imputation) or `delay_system_v3`, not an obs LP. Documented here so the next session does not re-debate it.

**Why**: The 037 experiment doc records `action_delta` growing more negative across training (i.e., commands becoming jerkier) while episode reward also falls, indicating the smoothness penalty is losing the tug-of-war with the bbox tracking reward. The architectural review (this conversation) confirmed two unaddressed design gaps in the env: (1) no `a_{t-1}` channel anywhere in the policy input, despite `_last_actions` being maintained for the reward; (2) zero-order-hold of a step-changing setpoint into a velocity-controller cascade tuned for smooth setpoints (ticket 040's pegasus-matched gains). Both gaps are well-documented in the RL-quadrotor literature as smoothness drivers; prev-action-in-obs is essentially standard practice (Hwangbo+RAL'17, Kaufmann+Nature'23 (Swift), Molchanov+IROS'19, plus the legged side: Rudin+CoRL'21, Lee+SciRobotics'20, Margolis+Agrawal). An env-side first-order LP on the command is also widely used in sim-to-real quad pipelines. Reward retuning (action_delta weight up, bbox weight down) is a complementary fix but does not address the *architectural* gap and is deferred — once these architectural drivers are removed, the reward weights can be re-tuned against the right action geometry.

**Blocked on**: nothing.

**Depends on**:
- [ticket 040 (proposed/in-flight)](../040-match-pegasus-physics-parameters/) — pegasus plant + matched gains. The LP τ defaults are sensible against ticket 040's gain set; if 040 retunes gains further, τ may need a revisit but the parameterization is unchanged.
- [ticket 042 (proposed)](../042-per-channel-ekf-latency-iris-ma6/) — per-channel EKF lag. 043's prev-action-in-obs channel is logically independent of 042 (it does not flow through the EKF lag path), but the value of the channel comes precisely from its zero-lag observability of the most recent applied command, which is what makes it useful against the EKF-lagged ego state.

**Distinct from**:
- [ticket 002](../002-reward-weight-tuning/) — reward weight retuning (e.g., raising `action_delta_penalty_scale`). 043 is the architectural prerequisite; reward retuning should follow on top, not precede.
- CAPS-style temporal smoothness regularizer (out-of-scope, not yet a ticket) — gradient-level smoothness loss inside the MAPPO update. Considered as a third lever; defer until 043's outcome is known.
- Δ-action accumulation (out-of-scope) — switching the action space to Δv with an env-side integrator. Less common in quad RL literature; not pursued unless 043's outcome is insufficient.

### Patch summary

1. **`iris_ma_env6_test_cfg.py`** — new cfg block under `IrisMaEnv6TestCfg`:
   ```python
   # --- ticket 043: action smoothness ---
   enable_prev_action_obs: bool = True
   """When True, append the previous applied filtered command (7D, physical units
   for vel/yaw_rate, normalized for gimbal/zoom) to each agent's ego observation.
   Bumps per-agent observation dim by +7."""

   enable_action_lowpass: bool = True
   """When True, apply a per-channel first-order low-pass to cmd_vel before the
   controller consumes it. The unfiltered raw command is preserved in
   ``_actions`` for the action-delta reward to remain on the policy's raw output."""

   # Per-channel time constants (seconds). Discretization: alpha = 1 - exp(-dt/tau)
   # with dt = sim.dt * decimation. Sampling-rate invariant.
   action_lowpass_tau_vel_xy_s: float = 0.08
   action_lowpass_tau_vel_z_s: float = 0.08
   action_lowpass_tau_yaw_rate_s: float = 0.08
   action_lowpass_tau_gimbal_yaw_rate_s: float = 0.04
   action_lowpass_tau_gimbal_pitch_rate_s: float = 0.04
   action_lowpass_tau_zoom_rate_s: float = 0.10
   ```
   And bump:
   ```python
   observation_spaces: dict = {"drone_0": 62 + 7, "drone_1": 62 + 7, "drone_2": 62 + 7}
   ```
   conditional on `enable_prev_action_obs` (use `__post_init__` if the cfg framework supports it; otherwise document the dim formula in the docstring and compute it in `__init__` of the env).

2. **`iris_ma_env6_test.py`** —
   - Add state buffers in `__init__`: `self._cmd_vel_filt: torch.Tensor` shape `(N, A, 7)` initialized to zero. Stored alongside `self.cmd_vel`.
   - Precompute `self._lp_alpha: torch.Tensor` shape `(7,)` from cfg τ values and `self.cfg.sim.dt * self.cfg.decimation` at env-init time.
   - In `_pre_physics_step` ([line 815](../../../iris_ma_env6_test.py#L815)): after computing the raw scaled `cmd_vel[:, idx, :]`, branch on `enable_action_lowpass`:
     ```python
     if self.cfg.enable_action_lowpass:
         self._cmd_vel_filt[:, idx, :] = (
             self._lp_alpha * self.cmd_vel[:, idx, :]
             + (1.0 - self._lp_alpha) * self._cmd_vel_filt[:, idx, :]
         )
     else:
         self._cmd_vel_filt[:, idx, :] = self.cmd_vel[:, idx, :]
     ```
     Then update `self.cmd_vel[:, idx, :] = self._cmd_vel_filt[:, idx, :]` so the existing `_apply_action` consumes the filtered command unchanged. **Note**: `_actions` (raw normalized policy output, [-1,1]) is left as-is so the `action_delta` reward operates on the raw policy decision, not on the filter output.
   - In `_get_observations` ([line 2028](../../../iris_ma_env6_test.py#L2028)): if `enable_prev_action_obs`, append `self._cmd_vel_filt[:, idx, :]` (7D) to `ego_obs` *before* the inter-agent concat. Document the field layout in the function docstring.
   - In the reset path ([line 3115](../../../iris_ma_env6_test.py#L3115) neighborhood): zero `self._cmd_vel_filt[env_ids]` alongside `self._last_actions[env_ids]`.

3. **No changes to the shared/critic observation in this slice**. The centralized critic already consumes the per-agent ego obs (via the shared_observation assembly) — the +7 dims will appear there automatically. If `shared_observation_spaces` is computed downstream, verify it reflects the new dim or update the cfg accordingly (Slice 1's regression test catches mismatches).

### Slices

**Slice 1 — bit-exact regression test + smoke test**.
- Unit test under `tests/test_action_smoothness_v1.py`:
  - With `enable_prev_action_obs=False, enable_action_lowpass=False`: env construction + one rollout step produces an observation tensor with the *same dim* and same numerical content as pre-patch. Use a checked-in golden tensor hash on `_get_observations()` output for a fixed seed.
  - With both flags enabled: confirm obs dim is `62 + 7 = 69` per agent; `cmd_vel` differs from raw `_actions * scale` after the second policy step (filter has memory).
  - Edge case: very large τ (e.g., 1.0 s) attenuates `cmd_vel` to near-zero on the first step, then converges to the raw command after ≈4τ.
- AppLauncher template per [iris_ma6/CLAUDE.md](../../../CLAUDE.md) "Generating Tests for Functional Modules".

**Slice 2 — short A/B training run**.
- Three configs, same RNG seed, same curriculum, 200k steps each (~24 wall-h total at current sim rate):
  - `043_baseline`: both flags off (pre-patch behavior).
  - `043_prev_action_only`: `enable_prev_action_obs=True`, `enable_action_lowpass=False`. Isolates the obs-channel contribution.
  - `043_full`: both flags on. The shipping candidate.
- Add to `experiments/registry.py` as `validation_action_smoothness_short`.
- Metrics tracked (via existing metric_tracker):
  - **Smoothness (load-bearing)**: `action_sum`, `action_delta`, plus a new tracked metric `cmd_vel_delta` = mean(|Δ cmd_vel_filt|) per policy step.
  - **Task quality**: `bbox_center`, `bbox_size`, `pair_valid_rate`, `all_invalid_rate`, `tracking_lost_fraction`, `triangulation_rmse`.
  - **Safety**: `collision_per_env`, `cbf_penalty`.
  - **Training health**: mean reward (last-10 avg), KL band stability, σ trace.
- Acceptance bar:
  - `043_full` shows ≥30% reduction in `cmd_vel_delta` vs `043_baseline` at 200k steps (the architectural change must visibly reduce jerk in the *applied* command).
  - Task-quality regressions on `043_full` within ±5% of baseline on all metrics — OR a clear interpretable improvement.
  - `043_prev_action_only` is the diagnostic: if it alone closes most of the smoothness gap, the LP may be optional and τ can be more aggressive (tune up in a follow-up).
  - If smoothness improves but task quality regresses >5%, that is the signal to follow up with reward retuning (ticket 002), not to roll back 043.

**Slice 3 — τ sensitivity sweep (optional, sequential)**.
- Only execute if Slice 2's `043_full` shows directionally correct but insufficient smoothness gain.
- 100k-step runs at τ ∈ {0.04, 0.08, 0.16} s on the translational-vel channels (gimbal/zoom fixed at default).
- Acceptance: pick the smallest τ that hits the 30% `cmd_vel_delta` reduction without >5% task-quality regression. Defer if Slice 2 passes cleanly at default τ.

### Method (training-time validation)

1. **Pre-flight**: Slice 1 unit test passes; smoke-test env construction with both flags on/off; confirm shared_observation dim propagates correctly.
2. **Slice 2 launch**: branch from `main` (current `02c34fe802_ticket040_042_pegasus_matched_params` baseline), apply patch, register experiment, launch three runs. Tensorboard pull at 40k, 80k, 120k, 200k for early-stop on obvious regression.
3. **Decision rule**: `043_full` ships if it hits the smoothness bar without regressing task quality. `043_prev_action_only` is documented as an intermediate option for future tuning. Pre-patch behavior remains opt-in via both cfg flags.

### Acceptance criteria

| Criterion | Result |
|---|---|
| Patch landed on `iris_ma_env6_test_cfg.py` and `iris_ma_env6_test.py` per §"Patch summary" | landed |
| `enable_prev_action_obs=False, enable_action_lowpass=False` produces bit-exact pre-patch obs + cmd_vel for a fixed seed | unit test green (Slice 1) |
| Per-channel τ values are sampling-rate-invariant: changing `decimation` does not change the continuous-time filter response | unit test green (Slice 1) |
| Per-agent observation dim becomes `62 + 7 = 69` when `enable_prev_action_obs=True`; shared_observation dim updates correspondingly | confirmed by smoke test |
| `043_full` Slice-2 run reduces `cmd_vel_delta` ≥ 30% vs `043_baseline` at 200k | tensorboard |
| `043_full` task-quality metrics within ±5% of baseline (or clearly interpretable improvement) | tensorboard + writeup |
| `043_prev_action_only` documented as a diagnostic intermediate | writeup |
| `doc/experiments/<date>_ticket043_prev_action_lowpass.md` writeup landed | written |
| No regression in any existing experiment in the registry (spot-check on a representative one) | confirmed |

### Scope boundary

- **DO**: land the patch behind two feature flags; validate via 3-way A/B; document.
- **DO**: keep both flags `True` as the new default once Slice 2 passes; legacy behavior opt-in via flags-off.
- **DO**: feed *applied filtered* command (`cmd_vel_filt`) into the prev-action obs channel, in physical units (m/s, rad/s) for vel/yaw_rate channels; normalized passthrough for gimbal/zoom.
- **DO**: leave `_actions` (raw normalized policy output) untouched so the `action_delta` reward continues to penalize raw policy decisions, not filter output.
- **DO NOT**: retune reward weights (`action_delta_penalty_scale`, bbox weights, etc.). That is ticket 002's scope and should follow 043, not precede it.
- **DO NOT**: add observation-side low-pass. Rejected at design time (see §"On observation-side low-pass").
- **DO NOT**: change the action space (no Δv action, no slew limit hard-clipping). The first-order LP is the smoothness mechanism.
- **DO NOT**: add CAPS-style gradient smoothness regularizer. Tracked as a future ticket if 043 + reward retune (002) is insufficient.
- **DO NOT**: change controller gains, EKF lag values, or curriculum.
- **DO NOT**: retrain ticket 040/042 baselines from scratch — Slice 2 launches from `main` and treats `043_baseline` as the comparison point.

### Risk

Low to medium.

1. **Obs-dim change invalidates checkpoints.** The +7 dim per agent (and the corresponding +21 in shared/critic) means any pre-043 checkpoint is incompatible with post-043 obs space at load time. Mitigation: 043 is a forward-only feature; pre-patch checkpoints continue to work with flags off. New runs start from scratch — standard practice for an architectural change.

2. **τ defaults too aggressive (over-smoothing).** If τ_vel_xy=0.08 attenuates the policy's intended bandwidth, task quality regresses while smoothness improves. Mitigation: Slice 2's task-quality bar catches this; Slice 3 sweeps τ down. The `cmd_vel_delta` reduction target (≥30%) is intentionally moderate to avoid over-filtering.

3. **τ defaults too lax (insufficient smoothing).** If τ values are too small, the LP does nothing and only the prev-action-in-obs channel contributes. The `043_prev_action_only` diagnostic isolates this; Slice 3 sweeps τ up.

4. **Prev-action channel scaling vs RunningStandardScaler.** The skrl `RunningStandardScaler` will re-fit to the new 7 dims. Statistics for vel channels are in m/s (O(1–10)), yaw_rate in rad/s (O(0.1–1)), gimbal/zoom normalized (O(1)). All within usual range; no special handling. Brief settling period at the start of training is expected.

5. **First-policy-step transient at reset.** With `_cmd_vel_filt` zeroed at reset, the first commanded step is attenuated by `α` (≈ 0.4 at τ=0.08). For all but the very first step this is correct behavior; the transient is consistent with "no prior command exists." Documented; not a bug.

6. **Couples cleanly with delay system v3.** Both the LP filter and the prev-action obs operate on `cmd_vel` (the env's outgoing command to the controller), entirely separate from the ingoing observation path that the delay system filters. No interaction.

### Coupling

- **Ticket 040** (Pegasus physics parity, in-flight) — sister sim2sim ticket; the LP τ defaults are tuned for 040's gain set. Coupling is one-way: 040 lands first, 043 builds on the matched-controller setup.
- **Ticket 042** (per-channel EKF latency, proposed) — orthogonal; prev-action-in-obs operates on a non-EKF channel (the env's own command), so the two stack additively.
- **Ticket 002** (reward weight retuning) — follow-up. After 043 lands, `action_delta_penalty_scale` and bbox weights should be revisited on top of the new architecture.
- **Ticket 037** (critic privileged obs) — the V-loss / σ-collapse failure mode in 037 is partially explained by the architectural gap 043 closes. After 043, re-evaluating critic-priv-obs's V-loss prediction is more meaningful (the actor's mean is no longer wandering).
- **Experiments framework** — adds one experiment (`validation_action_smoothness_short`) to the registry; existing pipeline unchanged.

### Affected files

**Edits**:
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — new cfg block (flags + per-channel τ defaults) and conditional obs-dim bump.
- [iris_ma_env6_test.py](../../../iris_ma_env6_test.py) — `__init__` adds `_cmd_vel_filt` buffer + `_lp_alpha` constant; `_pre_physics_step` adds the LP branch and the `_cmd_vel_filt` update; `_get_observations` appends prev-action to `ego_obs`; reset zeros `_cmd_vel_filt`.

**New**:
- `tests/test_action_smoothness_v1.py` — Slice 1.
- `experiments/registry.py` — add `validation_action_smoothness_short` (Slice 2).
- `doc/experiments/<date>_ticket043_prev_action_lowpass.md` — Slice 2 writeup.

### References

- [iris_ma_env6_test.py:815](../../../iris_ma_env6_test.py#L815) — `_pre_physics_step` (action → cmd_vel mapping; LP insertion point).
- [iris_ma_env6_test.py:853](../../../iris_ma_env6_test.py#L853) — `_apply_action` (consumes `cmd_vel`, unchanged by 043).
- [iris_ma_env6_test.py:2028](../../../iris_ma_env6_test.py#L2028) — `_get_observations` (prev-action concat point).
- [iris_ma_env6_test.py:338](../../../iris_ma_env6_test.py#L338) — existing `_last_actions` buffer (used by reward only).
- [skrl_mappo_rnn_cfg.yaml](../../../agents/skrl_mappo_rnn_cfg.yaml) — agent config (no changes needed for 043; mentioned for context on σ floor).
- [exp 037 doc](../../../experiments/2026-05-22_037_critic_priv_obs.md) — `action_delta` evidence.
- [exp 040/042 in-flight doc](../../../experiments/2026-05-26_01-38-37_mappo_rnn_torch_02c34fe802_ticket040_042_pegasus_matched_params.md) — sibling sim2sim run.
- [iris_ma6/CLAUDE.md](../../../CLAUDE.md) — test-writing conventions for Slice 1.
- Literature on prev-action-in-obs: Hwangbo et al. ("Control of a Quadrotor with RL," RAL 2017); Kaufmann et al. (Swift, Nature 2023); Molchanov et al. ("Sim-to-(Multi)-Real," IROS 2019); Rudin et al. ("Learn to Walk in Minutes," CoRL 2021); Lee et al. (Science Robotics 2020); Margolis & Agrawal ("Walk These Ways").
- Literature on action low-pass / smoothness regularization: Mysore et al. ("CAPS," ICRA 2021) — referenced as a future complement, not in 043's scope.

**Flow**: Low to medium. Code change is mechanically straightforward (a buffer, an EMA update, a concat). Risk is in the validation: a 200k × 3 run is the gate, and feature flags are the rollback. Estimated 2–3 days: (a) patch + Slice 1 test; (b) launch Slice 2 (~24 wall-h); (c) analyze; (d) writeup; (e) optionally Slice 3.
