# Experiment: 037_ticket037_critic_priv_obs_no_dr

**Run dir**: `logs/skrl/iris_ma6/2026-05-20_21-39-19_mappo_rnn_torch_a3f94fdc8b_ticket037_critic_priv_obs_no_dr`
**Commit**: `a3f94fdc8b` (Phase 1 implementation) + `e09e43c11b` (yaml tag)
**Trained**: 2026-05-20 21:39 → 2026-05-22 08:48, 35 h wall-clock, 400 k / 400 k steps
**Base**: `2026-05-18 t034` (per-env jitter; agent_400000.pt)
**Status**: Completed. **Conclusion: Phase 1 prediction FAILED on V-loss; concerning catastrophic forgetting at early-curriculum eval. Recommend ablation before deciding next step.**

---

## 1. Hypothesis

Ticket 037 Phase 1 prediction (from `doc/critic_obs_design.md`):

1. **V-loss should drop** — critic conditioned on env params attributes more return variance, so the asymptotic floor `Var[return | obs, env_params]` is below t034's `Var[return | obs]`.
2. **σ should collapse modestly** — sharper advantages → less exploration noise needed.
3. **action_sum / action_delta should improve** — cleaner gradient → less pushed-around policy.
4. **Task-quality non-regression** vs t034 within ±2 %.
5. **Visibility ratios at {39 k, 99 k, 199 k, 399 k} ≥ 0.80** vs new-run matched-step references.
6. **Re-running probe 036 should still return robust verdict** — critic-only privileged obs doesn't backprop into the actor.

## 2. Configuration delta vs t034

```python
# iris_ma_env6_test_cfg.py — flipped from False to True
enable_full_critic_priv_obs = True   # → 17 non-DR fields populated (7 DR-required dropped, DR is off)
enable_axis_independence    = True   # → 8 per-axis _eff_progress_* tensors sampled independently
domain_randomization.enabled = False # (KEPT — never validated for iris_ma6; A/B stays clean)
```

Per-axis curriculum `(start, end)` all left at default `None` (= inherit `dynamics_*`). The intent was: behavior is *bit-exact-equivalent* to t034 except for the new critic-obs tail. As the data below show, this assumption was wrong (see §6.1).

Network: same MAPPO-RNN arch as t034 except value-network input dim 98 → 130 (+32 = 15 per-agent × 2 + 2 shared = 32 new privileged input dims). Total params 68,175 → 70,223 (+2,048 = first-layer weights).

## 3. Results

### 3.1 Training-time scalars (vs t034 at matched train steps)

Format per cell: `t034 / t037 (Δ%)`.

| Metric                        | 40 k                    | 80 k                    | 120 k                    | 200 k                  | 280 k                  | 360 k                  | 400 k                  |
| ----------------------------- | ----------------------- | ----------------------- | ------------------------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| **V-loss**                    | 0.007 / 0.007 (−6 %)    | 0.010 / 0.012 (+17 %)   | 0.009 / 0.012 (+25 %)    | 0.011 / 0.011 (0 %)    | 0.011 / 0.011 (−0 %)   | 0.012 / 0.012 (−3 %)   | **0.012 / 0.012 (+2 %)** |
| **σ**                         | 0.216 / 0.204 (−6 %)    | 0.228 / 0.244 (+7 %)    | 0.213 / 0.261 (+23 %)    | 0.281 / 0.290 (+3 %)   | 0.277 / 0.285 (+3 %)   | 0.286 / 0.276 (−3 %)   | **0.277 / 0.267 (−4 %)** |
| **action_sum**                | −6.85 / −6.36 (+7 %)    | −10.6 / −11.2 (−6 %)    | −9.94 / −11.96 (−20 %)   | −11.6 / −12.3 (−6 %)   | −12.3 / −11.9 (+3 %)   | −12.8 / −11.4 (+11 %)  | **−12.5 / −10.9 (+13 %)** |
| **action_delta**              | −10.05 / −8.86 (+12 %)  | −11.48 / −12.40 (−8 %)  | −10.17 / −13.74 (−35 %)  | −14.69 / −15.29 (−4 %) | −14.31 / −14.56 (−2 %) | −14.95 / −14.02 (+6 %) | **−14.25 / −13.12 (+8 %)** |
| **KL**                        | 0.022 / 0.025 (+13 %)   | 0.028 / 0.027 (−1 %)    | 0.025 / 0.028 (+14 %)    | 0.022 / 0.023 (+5 %)   | 0.024 / 0.023 (−2 %)   | 0.025 / 0.023 (−7 %)   | 0.026 / 0.023 (−10 %)  |
| **pair_valid_rate**           | 0.877 / 0.898 (+2 %)    | 0.848 / 0.845 (−0 %)    | 0.880 / 0.830 (−6 %)     | 0.795 / 0.812 (+2 %)   | 0.830 / 0.814 (−2 %)   | 0.833 / 0.823 (−1 %)   | **0.831 / 0.811 (−2 %)** |
| **bbox_center** (lower better) | 47.5 / 48.4 (+2 %)     | 42.5 / 41.0 (−3 %)      | 44.4 / 40.0 (−10 %)      | 25.6 / 24.8 (−3 %)     | 25.4 / 24.4 (−4 %)     | 25.4 / 25.3 (−0 %)     | 25.8 / 24.9 (−3 %)     |
| **collision_per_env**         | 0.70 / 0.42 (−40 %)     | 0.24 / 0.27 (+12 %)     | 0.18 / 0.11 (−36 %)      | 0.25 / 0.69 (+178 %)   | 0.14 / 0.56 (+291 %)   | 0.29 / 0.39 (+35 %)    | **0.22 / 0.42 (+95 %)** |
| **tracking_lost_fraction**    | 0.014 / 0.020 (+38 %)   | 0.053 / 0.068 (+29 %)   | 0.028 / 0.070 (+148 %)   | 0.047 / 0.058 (+25 %)  | 0.031 / 0.061 (+93 %)  | 0.029 / 0.048 (+66 %)  | **0.031 / 0.052 (+68 %)** |
| **reward (mean)**             | 4803 / 5012 (+4 %)      | 3641 / 3520 (−3 %)      | 3845 / 3269 (−15 %)      | 2548 / 2437 (−4 %)     | 2642 / 2552 (−3 %)     | 2630 / 2635 (+0 %)     | 2694 / 2668 (−1 %)     |

**Phase 1 prediction check:**

- **V-loss reduction: FAILED.** V-loss is essentially identical (±3 % at every checkpoint after step 200 k). The headline Phase 1 mechanism did not materialize.
- **σ collapse: marginal.** At 400 k, σ is −4 % vs t034 (0.267 vs 0.277). Within noise.
- **Action smoothness: PASS partial.** action_sum −13 %, action_delta −8 % at 400 k (i.e., the magnitude of the negative penalty terms shrank — meaning *less* aggressive commands).
- **Reward mean: flat** at −1 %.

**Concerns:**

- **collision_per_env +95 % at 400 k**, with intermediate-step values up to +291 %.
- **tracking_lost_fraction +68 % at 400 k**, +148 % at step 120 k.
- These breach the ticket's ±2 % task-quality non-regression bar significantly.

### 3.2 Matched-step visibility eval (ticket 037 §6 acceptance)

Eval procedure (per ticket 034 pattern): for each `step S`, evaluate two checkpoints with curriculum pinned to step S via `debug_initial_step`:
- **reference**: `agent_<S>.pt` from this run
- **post-fix**: `agent_400000.pt`

Ratio = `postfix_visibility / reference_visibility`. Pass bar: ≥ 0.80.

| Eval step | Ref ckpt          | Ref vis | Post-fix vis | **Ratio** | Pass? |
|---:|---|---:|---:|---:|:---:|
| 39 000 | agent_40000.pt | 0.946 | 0.877 | **0.927** | ✓ |
| 99 000 | agent_80000.pt | 0.911 | 0.776 | **0.851** | ✓ |
| 199 000 | agent_200000.pt | 0.862 | 0.870 | **1.009** | ✓ |
| 399 000 | agent_400000.pt (self) | 0.870 | 0.870 | **1.000** | ✓ (self) |

**All four ratios PASS the ≥ 0.80 bar.** Visibility retention is preserved.

### 3.3 Matched-step collision rate (concerning)

The visibility-pass conceals a substantial regression in **collision_rate** at the early-curriculum eval:

| Eval step | Ref collision_rate | Post-fix collision_rate | Δ |
|---:|---:|---:|---:|
| 39 000 | 0.307 | **1.570** | **+412 %** |
| 99 000 | 0.215 | 0.128 | −41 % |
| 199 000 | 0.645 | 0.253 | −61 % |
| 399 000 | 0.176 | 0.176 | 0 (self) |

The 400 k policy crashes **5× more often** on the easy-regime distribution (step 39 k curriculum settings) than a freshly-trained-at-step-40 k checkpoint. At mid/late curriculum settings (99 k, 199 k), the 400 k policy is *better* than its matched-step references.

This is **catastrophic forgetting of the easy regime** — the very pattern ticket 034 was designed to prevent. See §6.1 for likely cause.

### 3.4 Re-running ticket 036's probe against the 037 400 k checkpoint

`adaptation_probe.py --probe all --num_envs 1024` against `agent_400000.pt` from this run.

| Probe | t034 (prior) | 037 Phase 1 | Verdict on 037 |
|---|---|---|---|
| **1** — action energy ratio early/steady | 1.496 | 1.354 | "adaptive" (reset-settling artifact, same as t034) |
| **2** — linear/MLP decoding of latents | all R² < 0 (ROBUST) | all R² < 0 (ROBUST — post max −0.99 to −1.20, pre −0.40 to −0.87) | **ROBUST — unchanged** |
| **3** — hidden-state swap detrended | −0.21 nat (robust) | **+0.32 nat (robust)** | **ROBUST** |

**Probe verdict: 037 Phase 1 policy is STILL ROBUST.** As predicted — critic-only privileged obs doesn't backprop into the actor's GRU. The 17-dim post-GRU representation is no more env-id-aware after Phase 1 than after t034.

Probe 2 R² values are slightly more negative on the post-GRU side under 037 (−1.0 to −1.2 vs t034's −0.76 to −1.18), suggesting the GRU representation is slightly *less* linearly-decodable-into-env-params after Phase 1 — possibly because the actor's policy gradient is now driven by a slightly different advantage signal (critic-priv-obs), but the gradient still doesn't pull toward env-id encoding.

[Outputs](../../experiments/outputs/2026-05-22_slice10_probe036/) (in `/tmp/slice10_probe036/`).

## 4. Acceptance criteria check (ticket 037 §6)

| Criterion | Result | Pass? |
|---|---|---|
| Cfg-flagged: `enable_full_critic_priv_obs=False` and `enable_axis_independence=False` reproduce t034 bit-exact | Verified in unit tests (object-identity dispatch) | ✓ |
| 400 k training run completes | 35 h wall-clock | ✓ |
| V-loss at 400 k ≥ 15 % below t034 | V-loss +2 % vs t034 | **✗ FAIL** |
| Visibility ratios at {39 k, 99 k, 199 k, 399 k} ≥ 0.80 | All four pass (0.927, 0.851, 1.009, 1.000) | ✓ |
| Task-quality at 400 k within ±2 % of t034 | pair_valid_rate −2 %, bbox_center −3 % (better), bbox_size −2 % (better), **collision +95 %**, **tracking_lost +68 %** | **✗ FAIL on collisions + tracking_lost** |
| Ticket 036 probe re-run: robust verdict expected | All three probes robust | ✓ |

**Two of six criteria FAIL.** The two failures are the load-bearing ones (V-loss is Phase 1's main mechanism; task-quality non-regression is the ship gate).

## 5. Verdict

Phase 1 as deployed (`enable_full_critic_priv_obs=True` + `enable_axis_independence=True` + `domain_randomization=False`) **does not deliver its predicted benefit and introduces a real task-quality regression**. The probe-036 robust verdict was correctly predicted by the lit review (`doc/critic_obs_design.md` §1.7), so the actor-side outcome is no surprise. The unexpected piece is:

1. V-loss didn't drop — the very mechanism Phase 1 was built around (sharper advantage attribution from privileged env params) didn't move the needle.
2. Catastrophic forgetting of the easy regime (collision +412 % at step-39 k eval) — this is the ticket-034 pathology re-emerging.

## 6. Analysis — what likely went wrong

### 6.1 `enable_axis_independence=True` changes the *distribution*, not just the *RNG stream*

The Slice 7 design assumed that setting `enable_axis_independence=True` with all per-axis `(start, end)` at default `None` would be bit-exact-equivalent to `=False` (because each axis's progress falls back to `dynamics_progress`).

**This was wrong at the distribution level.**

Under `=False` (bundled), one `sample_per_env_progress(global_progress=progress_dynamics, ...)` call gives a single per-(env, agent) effective progress. All 8 sub-randomizations (gimbal τ, drone gains, max_lin_vel scale, FOV, mech offsets, mass/inertia, stiff/damp, target scale) receive the **same** scale per (env, agent). Easy envs are jointly easy; hard envs jointly hard.

Under `=True` with per-axis cfgs at `None`, *8 independent* `sample_per_env_progress` calls happen per reset. Each draws an independent `Uniform(0, dynamics_progress)`. The 8 sub-randomizations are now **independent** per (env, agent). "All easy" environments (all 8 axes near zero) become rare (probability shrinks geometrically with 8 axes). "All hard" also rare. The dominant population is mixed.

This is the *intended* sim-to-real-fidelity improvement of ticket 037's decoupling sub-scope. But it's not bit-exact-equivalent to t034 — it's a different env distribution.

**Consequence for the 037 run**: ticket-034's anti-forgetting jitter was designed under the bundled distribution. Under independent jitter, the volume of "easy" envs the policy sees during full-curriculum training collapses. The 400 k policy never sees genuinely easy training conditions in sufficient quantity, so its early-curriculum competence degrades.

Evidence: collision_per_env at step-39 k eval +412 % vs reference, while at step-199 k it's −61 %. The early regime is what regressed.

### 6.2 V-loss didn't drop — why?

Two compatible explanations:

- **Critic input under-informativeness.** The 17 non-DR fields include 6 controller-gain scales centered at 1.0 with ±5 % range (cfg.gain_randomization.scale_range = (0.95, 1.05)). Within ±5 %, the *return variance attributable to these axes is small*. The critic doesn't need to learn fine attribution because the signal-to-noise is already small. Phase 1's V-loss-reduction prediction implicitly assumed wide variation across the privileged channel; the actual cfg has narrow variation on many of the 17 axes.
- **Per-(env, agent) info missing on max_lin_vel.** Per Slice 1 finding (ticket 038 deferred), `max_lin_vel` is per-env (not per-agent). The critic gets an under-informative read on the velocity axis, which is one of the higher-variance axes.

### 6.3 What this means for Phase 1

Phase 1 with this configuration does not work. Two distinct fixes worth considering:

- **F1: Disable axis independence**, keep critic-priv-obs. Re-run with `enable_axis_independence=False`, `enable_full_critic_priv_obs=True`. This restores the t034 distribution; the critic-priv-obs effect is then isolated. If V-loss still doesn't drop, Phase 1 is fundamentally underwhelming on this env.
- **F2: Widen the high-variance axes** before re-trying. Increase `gain_randomization.scale_range` to ±20 %; increase `max_lin_vel_scale_range`; etc. Larger privileged-channel variance gives the critic something to attribute.

F1 is the cheaper diagnostic. Recommend F1 first.

## 7. Next step recommendation

**Run F1 (ablation): `enable_axis_independence=False`, `enable_full_critic_priv_obs=True`, DR off.** Same 400 k training, 24 h wall-clock. Three possible outcomes:

- **(a) F1 fixes the early-curriculum regression AND V-loss drops** → Phase 1 was correct in mechanism but broken by the axis-independence interaction. The ticket can ship F1's config as Phase 1.
- **(b) F1 fixes the early-curriculum regression but V-loss still flat** → Phase 1's V-loss-reduction prediction is wrong for this env at the current randomization width. Move directly to Phase 2 (aux sysid head on actor) per the ticket's deferred plan.
- **(c) F1 doesn't fix the regression** → Critic-priv-obs itself is destabilizing the actor's training somehow. Investigate further or abandon Phase 1.

Outcome (b) is the most likely a priori, based on the lit review's "the conjunction (critic-only + env params + no actor encoder) doesn't have a direct precedent" caveat.

## 8. Limitations / open

- **One seed.** A second seed under F1's config would let us bound noise vs. signal.
- **`max_lin_vel` per-env (ticket 038)** still in deferred state. If F1 still shows weakness, ticket 038's per-(env, agent) extension might be the right next intervention before declaring critic-priv-obs ineffective.
- **The "narrow randomization" question** the user raised earlier is now load-bearing again. If gain_randomization.scale_range is too narrow, the critic's privileged channel has too little to attribute. A "wider randomization" sibling ticket may need to land before Phase 1's V-loss prediction can be fairly tested.
