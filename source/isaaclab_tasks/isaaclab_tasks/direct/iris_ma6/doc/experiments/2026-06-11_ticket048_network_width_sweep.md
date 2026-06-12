# Ticket 048 — Network width sweep (MAPPO-RNN): results & Pareto

**Date**: 2026-06-11
**Runs**: 3 × 200k @ seed=42, num_envs=1024, sequential on the 64 GB GPU.
**Env**: bare cfg defaults (post-045/046/047). Width is the only variable.
`gru_num_layers=1` held; `hidden_size = gru_hidden_size` mirrored policy↔value.
**enable_z_motion**: True (3D target motion — t046's 2D treatment was never
adopted as the cfg default; all three runs share it, so the comparison is clean).

| config | hidden/gru | combined p+v params | run dir |
|---|---|---|---|
| baseline | 64/64 | 71,567 (1.0×) | `2026-06-09_23-48-47_..._net_width_baseline` |
| mid | 128/128 | 257,807 (3.6×) | `2026-06-10_14-41-45_..._net_width_mid` |
| wide | 256/256 | 974,351 (13.6×) | `2026-06-11_05-17-30_..._net_width_wide` |

## Headline result — the capacity hypothesis is REJECTED

The motivating hypothesis (t045 100% slew saturation is partly a *capacity*
bottleneck → a wider net can encode a strategy that allocates bandwidth across
single-agent tracking AND cooperative geometry) is **not supported**:

- **Triangulation reward is flat across width**: 51.9 (baseline) / 51.3 (mid) /
  51.0 (wide). More capacity bought **zero** triangulation. The ~51 level is
  delivered by the post-047 reward retune, not by width — baseline already sits
  there.
- **Slew saturation is unchanged**: velocity channels stay 0.90–0.98 saturated
  for all three widths. Width does not redistribute or relieve it (the only
  exception — mid's `gim_pitch` 0.61 vs 0.95 — is a one-off; wide is back at 0.95).

→ This is the negative result the ticket anticipated (Risk #4). The t045
saturation ceiling is **demand-side, not capacity-side**. It strengthens the
case for **Ticket 049 (loosen the slew clip)**: the slew clip itself is the
structural constraint, and no amount of network capacity works around it.

## Secondary effects — small, consistent, single-seed

Wider nets give modest gains on *other* metrics, strongest for wide:

| metric @200k (drone-avg) | baseline | mid | wide | wide vs base |
|---|---|---|---|---|
| Total-reward AUC (40–200k) | 757.8 M | 771.6 M | 794.0 M | **+4.8%** |
| bbox_center (rew) | 48.40 | 52.45 | 53.62 | **+10.8%** |
| triangulation (rew) | 51.91 | 51.31 | 51.05 | −1.6% |
| pair_valid_rate | 0.850 | 0.869 | 0.863 | +1.5% |
| tracking_lost_fraction | 0.0514 | 0.0455 | **0.0433** | −16% (better) |
| collision_fraction | 0.0013 | 0.0024 | 0.0028 | tiny abs |
| total_rms (smoothness) | 0.660 | 0.633 | 0.663 | +0.3% |

The clearest width benefit is **curriculum-dip resilience**: at 160k (entropy
ramp + delay/dropout perception phases) baseline reward dips to 3273 while wide
holds 3996 and mid 3837 — more capacity absorbs the curriculum shift better.

## Wall-time — width is essentially FREE on this hardware

The ticket feared wide ≈ 1.5–2.0× baseline wall-time (GRU quadratic in hidden).
That did **not** materialize:

| config | wall-h | sec/1k steps | ratio |
|---|---|---|---|
| baseline | 14.75 | 268.3 | 1.00× |
| mid | 14.47 | 263.1 | 0.98× |
| wide | 14.16 | 257.5 | **0.96×** |

At num_envs=1024 on the 64 GB GPU the loop is **env-sim-bound** (Isaac Sim
physics + gimbal/camera), not GRU-bound. Width is free here. The Pareto "cost"
axis is therefore flat — the ticket's "ship mid because wide is 1.5× slower"
trade does not apply.

## Training stability — a wart on mid

| config | KL pts >0.04 | KL max | note |
|---|---|---|---|
| baseline | 2 / 100 | 0.059 | clean |
| mid | 47 / 100 (sustained 76k→200k) | **0.811 @168k** | would trip the "KL>0.04 sustained >20k" kill rule |
| wide | 76 / 100 (all mild) | 0.054 | chronically warm but bounded |

mid ran hot on KL across the entire back half and threw a single 0.811 blowup at
168k (entropy-ramp / curriculum-shift region); the KLAdaptiveLR pinned LR at
`min_lr=3e-4` and couldn't tame it. It recovered on the metrics, but this is a
real stability concern. wide, despite more params, stayed bounded (max 0.054).
σ ended at the `max_log_std=0.4` cap (1.49) for baseline & wide; mid 1.32.

## Acceptance-bar check

| bar | result |
|---|---|
| ≥1 widened config `triangulation ≥ 32` @200k | numerically met (51 ≥ 32) but **NOT attributable to width** — baseline is also 51 |
| smoothness `total_rms` within ±10% of baseline | ✅ mid −4.1%, wide +0.3% |
| `collision_fraction ≤ 2× t043 baseline` | ✅ all negligible (0.001–0.003) |
| wall-time/step on Pareto winner ≤ 2.5× baseline | ✅ ≈1.0× (width is free) |

## Recommendation (Slice 3 — gated cfg-default flip)

**Do not flip the default on this single-seed evidence.** Rationale:

1. On the **load-bearing metric (triangulation)** baseline ties the widened
   configs — width did not unlock the cooperative-geometry capacity the ticket
   was testing for. The ticket's own decision rule: "If baseline is the Pareto
   winner (width didn't help): no cfg flip — the value is in the documented
   negative result."
2. The secondary gains (bbox_center +11%, AUC +5%, tracking_lost −16%,
   curriculum resilience) are real and **free** (≈1.0× wall-time), but they are
   **single-seed** and within plausible seed noise. The ticket flags a
   multi-seed re-run as the follow-up "if the headline result is marginal" — it
   is marginal.
3. **mid is disqualified** by the KL instability regardless.

**If** the secondary gains are wanted, **wide** is the candidate (free,
bounded KL, best on every secondary metric) — but **validate with 2–3 seeds
before flipping**, since the deltas are within single-seed noise.

The durable conclusion is the negative one: **capacity is not the bottleneck;
demand (the slew clip) is.** → prioritize **Ticket 049**.

## Deploy-time comparison (Pegasus + PX4 SITL + mas, 2026-06-12)

Rosbags recorded on the full ROS2 deploy stack for all three checkpoints
(`bag_20260611_*_t048_net_width_{baseline64,mid128,wide256}`). px4_1/px4_2 are
the observer agents, px4_3 is the target (ground truth). Steady window 10–24 s,
observers averaged. Extractor: `experiments/outputs/t048_netwidth/deploy_tracking_compare.py`
→ `deploy_tracking_compare.json`.

| deploy metric (obs-avg) | baseline | mid | wide |
|---|---|---|---|
| obs→target horiz dist (m) | 17.50 | **7.28** | 16.57 |
| dist-hold std (m) | 2.36 | 1.30 | **1.17** |
| loiter wander RMS (m) | **4.58** | 2.32 | **1.69** |
| obs horiz speed (m/s) | **1.291** | 0.890 | **0.616** |
| obs vert speed (m/s) | 0.309 | 0.286 | **0.186** |
| target **EST error** mean (m) | **0.73** | **1.61** | 0.84 |
| target **cov trace xy** (m²) | **0.17** | **3.42** | 0.23 |
| gimbal yaw std (deg) | 10.66 | 11.81 | **5.74** |
| gimbal pitch std (deg) | 6.99 | 6.79 | 6.60 |
| gimbal >2 Hz osc (deg) | 1.108 | 1.163 | 1.135 |

**The three policies adopt visibly different deploy strategies** (matches the
operator's eyeball read):

- **baseline — "roams/loiters around the target."** Holds a ~17.5 m standoff but
  wanders 4.58 m RMS at 1.29 m/s (highest of the three) and slews the gimbal yaw
  the most (10.7°). The active coverage gives it the *best raw estimation*
  (0.73 m, cov 0.17) — but at the highest control effort.
- **mid — "approaches very closely" (7.3 m), and it backfires.** The close-in
  single-agent framing **wrecks the cooperative triangulation geometry**:
  estimation error 1.61 m (2× the others) and covariance trace **3.42 m²
  (~15× baseline/wide)**, plus the worst gimbal yaw. The visual "gets close"
  is the *symptom of a degenerate strategy*, not better tracking. Consistent
  with mid's training KL blowup (0.81@168k) and its worst sim track-loss /
  convergence / collision.
- **wide — "most stationary" (the anticipated behaviour).** Lowest speed
  (0.62 m/s), lowest wander (1.69 m), steadiest standoff (dist std 1.17), and
  **gimbal yaw std halved (5.74° vs ~11°)** — a calm body means the gimbal
  barely has to slew. Estimation essentially ties baseline (0.84 m, cov 0.23)
  while flying far calmer.

**Cross-validation with sim-eval** (`{baseline,mid,wide}.json`): aggregate sim
metrics are flat (task_success 0.56/0.54/0.57; tri_rmse_med 1.23/1.34/1.19) —
same "width barely moves the average" as the training reward. But the **ranking
agrees**: sim tri_rmse and deploy estimation both put **mid worst, wide & baseline
better**, and mid is also worst on sim track_loss / convergence / collision. Two
independent datasets converge on the same behavioural picture.

**Gimbal >2 Hz oscillation is identical (~1.1°) across widths** — the divergent
limit-cycle that motivated Ticket 049 does not reproduce in any config
(consistent with t049's "phenomenon does not currently reproduce"). Wide's gimbal
advantage is *low-frequency* (less body motion → less slewing), not damping of
the high-freq mode.

### Revised recommendation — the deploy data tips toward flipping to **wide**

The sim-only conclusion ("no flip; baseline ties on triangulation reward") was
based on the aggregate reward. The deploy evidence is a stronger, physically
grounded case for **wide**:
- Calmest body (½ the speed, ⅓ the wander of baseline) → least excitation of the
  body-rejection loop — directly the sim2real robustness Ticket 049 cares about.
- Best gimbal pointing stability (yaw std halved).
- Estimation ties baseline (0.84 vs 0.73 m — an 11 cm gap baseline buys with 2×
  the speed, 2.7× the wander, 2× the gimbal motion).
- Best sim task_success and tri_rmse; width is free on this GPU.
- **mid is disqualified** (degenerate close-in strategy + KL instability).

Caveat unchanged: single-seed. But the deploy behaviour is consistent with sim
and physically sensible, so the risk of flipping to **wide** is low. If a
confirmation is wanted, a 2–3-seed re-run of baseline-vs-wide closes it.

## Seed-confirm (2026-06-12) — width is within seed noise; wide kept as a behavioural lead

Sim-eval (1024 envs, seed-42 scenarios, step 200k) of a second wide seed (123),
with the checkpoint-step confound isolated by also evaluating wide@42 at 180k:

| task_success | wide@42 200k | wide@42 180k | wide@123 180k |
|---|---|---|---|
|  | 0.571 | 0.537 | **0.372** |
| tri_rmse_med | 1.191 | 1.335 | **1.758** |

- Checkpoint effect (200k→180k, same seed): small (task_success −6%).
- **Seed effect (180k, same scenarios): large — task_success −31%, tri_rmse +32%.**
- At fixed width=256, two seeds span 0.37–0.57 task_success (≈0.17), **~5× the
  original between-width spread (0.033)**. So the v1 "wide marginally best"
  ranking — and any single-seed width claim — is within seed noise. This is the
  stronger negative result: **width has no task-quality effect distinguishable
  from seed noise**, vindicating the original "no flip on single-seed evidence"
  caution.
- **Decision: cfg default kept at wide (256/256) NOT as a proven task win, but
  because wide@42 exhibited a distinctive calm/stationary tracking behaviour no
  other model produced — a behavioural *lead* worth carrying into the
  cooperativeness work (t050).** Revisit only with a ≥3-seed sweep if width is
  ever to be claimed; given the prior is "no effect," not worth the compute now.
- Methodology carry-forward: **single-seed claims in this env are unreliable;
  any t050 headline needs ≥3 seeds.**

Lead checkpoint for t050: `logs/skrl/iris_ma6/2026-06-11_05-17-30_..._net_width_wide`
(wide@42, agent_200000.pt / best_agent.pt).

## Follow-ups

- **Ticket 049 (loosen slew)** — now the highest-value lever; this sweep rules
  out the capacity explanation for the saturation ceiling.
- Multi-seed (≥3) re-run of baseline vs wide if the free secondary gains are
  judged worth chasing before 049.
- Depth axis (`gru_num_layers=2`) remains deferred; width-alone plateaued on the
  headline metric, so depth is unlikely to help the triangulation ceiling either.
