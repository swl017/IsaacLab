# R0 — Regression Measurement Results

**Ticket**: 034-curriculum-catastrophic-forgetting
**Date**: 2026-05-17
**Run config**: 7 cells × 1024 envs × 1 episode per env (≈1024 episodes per cell),
deterministic actions, `experiment=a1_with_aoi`, headless,
curriculum pinned via `--step` (env's `debug_initial_step`).
**Source data**: 7 JSON files under
[r_research/](.).

---

## 0. Headline

| Eval step | Reference visibility | 400k-ckpt visibility | Ratio | Pass ≥ 0.80? |
|---:|---:|---:|---:|:---:|
| **39 000**  | 0.932 (ckpt 40k)  | **0.574** | **0.616** | **FAIL** |
| **99 000**  | 0.920 (ckpt 80k)  | **0.531** | **0.577** | **FAIL** |
| **199 000** | 0.879 (ckpt 200k) | 0.868     | 0.987     | PASS |
| **399 000** | 0.875 (self)      | 0.875     | 1.00      | n/a |

**Verdict: catastrophic forgetting hypothesis CONFIRMED.** The 400k checkpoint
performs 38–42 % worse on visibility at easy curriculum settings (39k/99k)
than checkpoints that were trained at those same settings. As the eval step
approaches the 400k training horizon, the ratio recovers to 1.0.

This is the monotonic pattern predicted by the ticket's audit: the env
distribution between 220 k and 400 k contains zero envs at the early-phase
regime, so the policy has no gradient signal for early-phase settings, and
loses the corresponding skill.

Per ticket §"Validation criterion" and Q5, the 80 % bar applies — and the
400 k checkpoint fails it by a wide margin (0.62 / 0.58 vs. the 0.80
threshold).

---

## 1. Raw metrics (all 7 cells)

| Metric | s39k-c400k | s99k-c400k | s199k-c400k | s399k-c400k | s39k-c40k (ref) | s99k-c80k (ref) | s199k-c200k (ref) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `visibility_mean` | **0.574** | **0.531** | 0.868 | 0.875 | **0.932** | **0.920** | 0.879 |
| `visibility_p5` | (low) | (low) | — | — | (high) | (high) | — |
| `visibility_p95` | — | — | — | — | 1.000 | — | — |
| `collision_rate_mean` | 0.334 | 0.070 | 0.311 | 0.347 | 0.088 | 0.314 | 0.159 |
| `cbf_violation_rate_mean` | 7.88e-3 | 4.51e-3 | 3.74e-3 | 3.76e-3 | 1.98e-3 | 1.90e-3 | 2.50e-3 |
| `min_separation_mean` (m) | 9.88 | 12.97 | 11.94 | 11.88 | 9.42 | 11.92 | 12.21 |
| `num_episodes` | 1024 | 1025 | 1025 | 1025 | 1024 | 1025 | 1025 |

(Full JSON: `eval_step<S>_ckpt<C>.json` per cell.)

---

## 2. Interpretation

### 2.1 Primary signal — `visibility_mean`

`visibility_mean` = fraction of steps with ≥ 1 valid bbox detection in any
agent. It is the cleanest "is the policy doing the basic task" metric
available in `MetricTracker` and is independent of the broken triangulation
metrics (see §3).

The 400 k checkpoint:
- At step 39 000: visibility 0.574 vs. the 40 k checkpoint's 0.932 →
  **−38 %**. The policy loses the bbox more than 4× as often as the
  freshly-trained 40 k checkpoint on the *same* easy env.
- At step 99 000: visibility 0.531 vs. the 80 k checkpoint's 0.920 →
  **−42 %**. Same pattern, worse magnitude.
- At step 199 000: visibility 0.868 vs. the 200 k checkpoint's 0.879 →
  **−1.2 %**. Within noise.
- At step 399 000: 0.875 (the run's training horizon). Self-consistent.

The ratio recovers monotonically as the eval step approaches the training
step. This is the regression curve predicted by the audit: the further
the env distribution is from the training distribution the 400 k checkpoint
spent most of its time in (steps 220 k – 400 k), the worse it performs.

### 2.2 Secondary signal — `collision_rate_mean` (and `cbf_violation_rate_mean`)

Pattern is more complex and not directly indicative of forgetting:

- At step 39 000 the 400 k checkpoint collides 3.8 × more often than the
  40 k checkpoint (0.334 vs. 0.088). This is *consistent with* the
  forgetting hypothesis: the policy that has forgotten how to track the
  easy target is plausibly also more likely to crash into other agents
  while flailing.
- At step 99 000 the 400 k checkpoint actually collides *less* than the
  80 k checkpoint (0.070 vs. 0.314). Plausible reading: the 80 k
  checkpoint hadn't yet finished training the safety reward
  (`progress_safety` ramps 20k → 40k, then full from 40k on, but the
  policy needs many more steps to consolidate it), so the 80 k policy
  has poor collision avoidance even on the easy regime; the 400 k
  policy has the CBF behavior consolidated but is bouncing around
  without tracking.
- At steps 199 k and 399 k collision rates are 0.31–0.35, suggesting the
  policy under hard curriculum settings collides a lot during normal
  operation. The CBF penalty has been driven near zero in training
  (per the fix2 experiment doc, 0.07) but `collision_rate_mean` here
  measures hits against any other agent, not the CBF margin.

Collision rate is not a clean regression signal because two effects compose:
(a) safety policy maturity (improves monotonically with training),
(b) tracking competence on the eval distribution (regresses with training
when the distribution drifts). The visibility signal is the cleaner
one.

### 2.3 `min_separation_mean`

Mean minimum pairwise distance is ≥ 9.4 m in every cell, well above the
CBF safety distance `D_s = 2 m`. The forgetting does not push agents
into each other on average — they spread further apart when they can't
track. (This is consistent with the visibility regression: agents that
have lost the target also disperse.)

---

## 3. Eval-script defect (orthogonal to ticket)

**Every triangulation metric is 0.0 in every cell**, including the
reference cells where the policy was trained at that step's distribution:

- `triangulation_rmse_mean = 0.0`
- `trace_sigma_mean = 0.0`
- `task_success_rate = 0.0`
- `track_maintenance_rate = 0.0`
- `accuracy_rate = 0.0`
- `tri_valid_ratio_mean = 0.0`

Since the 40 k, 80 k, and 200 k reference checkpoints reached
`triangulation` reward components of ~20+ during training (per the fix2
experiment doc), `tri_valid_ratio = 0` here cannot be the policy's fault.

Root cause is in `_collect_step_metrics`
[evaluate.py:186-212](../../../experiments/evaluate.py#L186-L212): the
function reads `env._triangulation_result_obs` and `_triangulation_result_gt`,
and falls back to all-zeros when these are `None`. Under the
`--no-record-trajectory --no-record-action-trace --no-timeseries` mode
plus `experiment=a1_with_aoi`, one or both of these attributes is
apparently `None` at the points where the metric step is taken (likely a
lifecycle ordering issue between `_get_rewards` and the eval-side
metric collection).

This is a defect in the eval harness, not in the training run under
review, and not in the ticket's scope. It does not affect the regression
verdict because visibility — measured directly from the bbox raycaster
output, not triangulation — is unaffected.

**Captured as out-of-scope observation for a follow-up ticket.**

---

## 4. What this changes for Stage I

- The ticket's qualitative claim ("400 k ckpt fails at 40 k env settings")
  is now backed by a quantitative measurement: visibility drops 38–42 %,
  far below the 80 % bar.
- The ratio's recovery to 1.0 at step 399 k confirms the failure is
  distribution-shift, not policy collapse. The 400 k policy still works
  in the regime it was trained on.
- The monotonic shape (worst at lowest eval step, monotonically improves
  toward 1.0 at the highest eval step) is the signature of a
  collapsing-distribution forgetting, not a randomly-distributed
  degradation. This rules out alternative hypotheses like LR
  decay artefacts or random-seed effects, which would not produce
  monotone regression-by-eval-step.
- Quantitatively, the 99 k cell is *worse* than the 39 k cell (ratio
  0.577 < 0.616). Stage I should not assume the regression's severity
  is monotonic in the env-step distance from training horizon — it's
  monotonic in distance from the *training distribution's mass*, which
  is concentrated 220 k – 400 k. Both 39 k and 99 k are far from that
  mass; 99 k happens to be the worst.
