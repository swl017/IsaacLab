# Ticket 050 / Slice B — Team / difference (information) reward: A/B result & review

**Commit**: `b4edcfa5af` (Slice A merged). **Slice B implementation is UNCOMMITTED** in the working
tree (`information_reward/`, env wiring in `iris_ma_env6_test.py`, `curriculum_cfg.py`,
`experiment_registry.py`).
**Date**: 2026-06-16
**Experiment IDs**: `t050b_team_reward` (info ON) vs `t050b_baseline_no_info` (info OFF) — both in
`experiment_registry.py`.
**Base**: warm-start from t048 wide lead `2026-06-11_05-17-30_..._net_width_wide/agent_drone_0_final.pt`.

---

## TL;DR — gate FAILS (null / slightly-negative result)

The per-agent difference (counterfactual) information reward produced **no improvement in
re-acquisition** over the no-info control, and **both arms fell below the C2 baseline (0.385)** they
were warm-started from. `time_to_reacq` got **worse**. The implementation is clean and the info
reward was genuinely active (not a wiring bug), so this is a **real null**, with a specific identified
design flaw (see §4). Single seed each — gate required ≥3. **Do not report Slice B as the
contribution yet; revise the reward and re-run multi-seed.**

## 1. Hypothesis
With the cooperation trigger in place (Slice A), the C2 reward (bbox-90 dominance + the LS
triangulation signal going NaN→0 at single-agent loss) is the binding constraint. A team/difference
information reward — pay each agent its *marginal information contribution* `r_i = J(Σ) − J(Σ_{−i})`
from an FIM-with-prior, plus a bbox→team rebalance — was expected to raise `reacq_success_rate` and
lower `time_to_reacq` vs the C2 baseline (0.385 / 0.46 s), without collapsing base tracking.

## 2. Configuration delta (team arm vs control)
Both arms: `enable_track_loss_scenario=True`, `cooperation_metrics.enable=True`, same warm-start ckpt,
same scenario, 200k fine-tune steps. The **only registry difference** is `information_reward.enabled`.
Because the bbox-rebalance fires only when the info reward is on, the team arm differs in **two** ways:

```python
# team_reward:        information_reward.enabled = True
#   -> triangulation slot = r_diff[:,i] * info_scale_max(8)   (per-agent difference reward)
#   -> bbox_center 90 -> 30, bbox_size 30 -> 20               (rebalance, p_rebal=1 at warm-start)
# baseline_no_info:   information_reward.enabled = False
#   -> triangulation slot = shared trace_quality * scale      (C2 reward form)
#   -> bbox_center = 90, bbox_size = 30                        (unchanged)
```

Notes / deviations from the Slice-B design (i_design.md):
- **`enable_critic_gt_target` (privileged critic) was NOT enabled** — it sizes `state_space` in
  `__post_init__`, so a Hydra/registry override lands too late and would mismatch the critic
  (cf. R gap #5 / `project_hydra_post_init_bypass`). The run used the existing
  `enable_full_critic_priv_obs` critic instead. The "stabilize value through the deficit" mechanism
  the credit-assignment story relied on was therefore absent.
- **Rebalance curriculum never exercised**: with `current_step = common_step_counter + debug_initial_step`
  (=200k at warm-start) the 80k–200k rebalance window is already complete at step 0, so the run begins
  at **full team strength** (bbox=30, info=full) — effectively a step function, not the designed ramp.

## 3. Results

### 3.1 Deterministic eval (the gate)
`evaluate.py --track_loss_scenario --step 400000 --num_envs 256`, seed 42
(`/tmp/t050b_eval_{team,base}.json`, log `/tmp/t050b_eval.log`):

| metric | C2 baseline (Slice A) | **team_reward** | **baseline_no_info** | gate |
|---|---|---|---|---|
| `reacq_success_rate` | 0.385 | **0.359** | 0.349 | beat 0.385 → **FAIL** (both below) |
| `time_to_reacq_s` | 0.46 | **0.735** | 0.573 | beat 0.46 → **FAIL** (team worst) |
| `team_track_maintenance` | 0.53 | 0.566 | 0.445 | team > control (+0.12) ✓ |
| `reacq_bearing_align_mean` | 0.798 | 0.816 | 0.651 | team > control |
| `reacq_dist_delta_mean` | −0.086 | **−0.578** | −0.518 | design wanted ↑; went wrong way |
| `track_loss_event_rate` | 2.05 | 1.43 | 1.53 | event mix differs (caveat) |
| `num_episodes` | 601 | 544 | 557 | |

`reacq_success_rate` gap team-vs-control = 0.010 ≪ 1 binomial SE (~0.020 on ~550 episodes) →
**not significant**. The info reward had no detectable effect on the gate metrics.

### 3.2 Training-time `Coop/*` curves (steps 2k–200k, last-20% mean)
Confirm the eval: `reacq_success_rate` flat ~0.37 for **both** arms throughout.

| metric (last 20%) | team | baseline |
|---|---|---|
| `Coop/reacq_success_rate` | 0.367 | 0.369 |
| `Coop/time_to_reacq_mean` | 0.728 | 0.410 |
| `Coop/team_track_maintenance` | 0.564 | 0.516 |
| `Coop/reacq_dist_delta_mean` | −0.423 | −0.259 |

### 3.3 Intervention was active + training healthy (not a no-op bug)
| metric (last 20%) | team | baseline |
|---|---|---|
| `Episode_Reward/drone_0_triangulation` | 65.4 (= `r_diff`·8) | 17.3 (shared trace) |
| `Episode_Reward/drone_0_bbox_center` | 8.8 (rebalanced) | 29.0 (selfish) |
| `Policy / Standard deviation (drone_0)` | 1.23 (stable) | 1.24 (stable) |
| `Loss / Value loss (drone_0)` | 0.008 | 0.001 |
| `Detection/pair_valid_rate` | 0.566 | 0.519 |

σ not collapsed/stuck, value loss low, total reward stable → the null is real, not a divergence or a
zero-magnitude reward.

## 4. Analysis — why it didn't work

**Root cause (likely causal): the FIM-with-prior's key property is computed and discarded.**
The reward only consumes `compute(...)["r_diff"]`; `team_quality` (the shared, dense, *defined-through-
the-deficit* readout — the entire justification for FIM-over-LS) is thrown away. And `r_diff` is forced
to **0 for any agent without a valid detection** (`information_reward.py:116`). So during the single-
agent deficit the whole ticket is about, the **lost agent gets exactly zero info-reward gradient** —
the same vanishing-signal pathology as C2, relocated from the shared-trace term to the difference term.
The reward shapes the *outcome geometry* (good while already tracking) but adds no denser gradient for
the *recovery action* (re-pointing while lost) than the old reward did.

This matches the metrics exactly: `team_track_maintenance` and `bearing_align` improve (better team
geometry while tracking), but `reacq_success_rate` is flat and `time_to_reacq` worsens (recovery itself
not helped). `reacq_dist_delta` going *more* negative falsifies the design's "covariance reward → agents
close in to re-acquire" hypothesis.

**Secondary issues:**
1. **Confounded A/B** — team arm changes reward *form* AND bbox weight (30 vs 90). No "rebalance-only"
   (bbox-30 + shared-trace) arm, so no effect is attributable to the difference reward specifically.
2. **Privileged critic dropped** (see §2) — the value-conditioning the credit story assumed was absent.
3. **Single seed** each; gate required ≥3. Even the +0.12 maintenance edge is unconfirmed.
4. Both fine-tuned arms ended **below** the t048 starting checkpoint (0.385). Caveat: event rates
   differ (2.05 vs ~1.45), so C2-vs-new isn't perfectly apples-to-apples — but team-vs-control is, and
   there the info reward shows nothing on the gate metrics.

**What IS sound:** the ticket framing (C1/C2/C3, estimation-theoretic team reward) is well-grounded;
Slice A is solid (trigger gate 2.05, first-class metrics, default-off, 14/14 tests); the
`information_reward` module is mathematically correct and numerically careful (float64 inversion for
near-target ill-conditioning, masked-sum leave-one-out to avoid prior cancellation, unit tests). The
A/B had a proper control, common warm-start, and a deterministic eval.

## 5. Decision / Todo
Report Slice B honestly as a **null with identified cause**. Before spending multi-seed budget:
1. **Fix the recovery gradient (highest leverage):** give the lost agent a non-zero signal — blend the
   dense `team_quality` into the reward, and/or add **potential-based** shaping on peer-bearing
   alignment so re-pointing-while-lost is rewarded *before* re-acquisition.
2. **De-confound:** add the bbox-30 + shared-trace arm (rebalance only).
3. **Restore the privileged critic** properly (re-finalize `state_space` at env `__init__`).
4. **Multi-seed (≥3)** before any claim.
5. Strategic read (consistent with the parent ticket): the reward axis alone was insufficient → this
   elevates **Slice C (explicit peer-bearing channel)** as the likely load-bearing piece, alongside the
   recovery-gradient fix above. Capacity remains not-the-bottleneck (t048).
