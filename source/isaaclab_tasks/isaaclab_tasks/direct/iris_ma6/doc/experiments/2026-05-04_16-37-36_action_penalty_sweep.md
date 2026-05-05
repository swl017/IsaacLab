# Experiment: Action-Penalty Sweep (4 cases × 4 GPUs)

**Sweep root**: `logs/skrl/iris_ma6_action_penalty_20260504_163714/`
**Date**: 2026-05-04
**Commit base**: `0b8278f00c` (most-recent solo run) with action_weight rebalance applied at launch
**Runs**:

| ID | action_sum | action_delta | GPU | Max step |
|---|---:|---:|:---:|---:|
| sum8_d8   | -8 | -8  | 0 | ~100k |
| sum4_d16  | -4 | -16 | 1 | ~100k |
| sum4_d8   | -4 | -8  | 2 | ~100k |
| sum2_d16  | -2 | -16 | 3 | ~100k |

**Status**: All four runs reached ~100k. **First runs in the SIYI realism series where sigma actually collapses** — driven jointly by the per-axis weight rebalance and the scale increase. Headline finding: `sum8_d8` is the clearest winner; `sum2_d16` is broken; the others sit in between with a clear "magnitude > delta" preference.

---

## 1. Hypothesis (going in)

The 0b8278f00c run showed that doubling penalty scale (`-1/-1` → `-2/-2`) reduced action thrash but did not collapse sigma (stuck at 1.75 throughout). The §6 open item flagged that **the `action_weight` vector itself might need rebalancing** — gimbal/zoom rate axes were at 0.5× weight, possibly under-penalizing the axes most coupled to triangulation quality. This sweep tests two changes simultaneously:

1. **Per-axis weight rebalance** (applied to all 4 cases):
   - `action_weight: [1, 1, 5, 1, 0.5, 0.5, 0.5] → [1, 1, 5, 1, 1, 1, 1]`
   - `action_delta_weight: [1, 1, 1, 1, 0.5, 0.5, 0.5] → [1, 1, 1, 1, 1, 1, 1]`
   - Net effect: gimbal/zoom rate axes now get **2× the weighted penalty** for the same scale.
2. **Scale sweep**: `(action_sum, action_delta) ∈ {(-8,-8), (-4,-16), (-4,-8), (-2,-16)}` — explores both magnitude/delta absolute scale and their ratio.

Combined effective gimbal-axis penalty vs prior 0b8278f00c (-2 scale × 0.5 weight = -1 effective):
- sum8_d8: 8× scale × 1.0 weight = **-8** (8× heavier than 0b8278)
- sum4_d16 delta: 16× × 1.0 = **-16** (16×)
- sum2_d16 sum: 2× × 1.0 = **-2** (2×)

## 2. Configuration delta (vs 0b8278f00c)

```python
# Per-axis weight rebalance (applied to all 4 sweep cases)
action_weight:        [1, 1, 5, 1, 0.5, 0.5, 0.5] → [1, 1, 5, 1, 1, 1, 1]
action_delta_weight:  [1, 1, 1, 1, 0.5, 0.5, 0.5] → [1, 1, 1, 1, 1, 1, 1]

# Scale sweep
action_sum_penalty_scale:   -2.0 → {-8, -4, -4, -2}
action_delta_penalty_scale: -2.0 → {-8, -16, -8, -16}
# all other knobs unchanged from 0b8278f00c (drift off, λ_cbf=1.0,
# envelope 60/25, dynamics 220k–240k, etc.)
```

## 3. Results

### 3.1 Sigma collapse — the headline change

| Step | sum8_d8 | sum4_d16 | sum4_d8 | sum2_d16 | 0b8278 | 0d4fa |
|----:|---:|---:|---:|---:|---:|---:|
| 4k   | 0.435 | 0.392 | 0.459 | 0.403 | 0.579 | 0.452 |
| 20k  | 0.209 | 0.178 | 0.302 | 0.233 | 0.752 | **0.137** |
| 40k  | 0.203 | **0.157** | 0.195 | 0.213 | 1.052 | 0.135 |
| 80k  | 0.246 | 0.190 | 0.243 | 0.214 | 1.586 | 0.139 |
| 100k | 0.254 | 0.197 | 0.317 | 0.218 | 1.750 | 0.155 |

**All four sweep runs collapse sigma into the 0.16–0.32 range — 5–10× tighter than 0b8278's 1.75 plateau** and within 50% of 0d4fa's 0.14. This is the first SIYI-stack run series where the policy actually commits to a basin instead of riding the entropy bonus. `sum4_d16` reaches the tightest sigma at 40k (0.16, within 20% of 0d4fa).

The collapse driver is the **per-axis weight rebalance**, not just the scale increase. The 0b8278f00c run had `action_sum=-2` but with `gimbal/zoom weights = 0.5`, the effective gimbal penalty was `-2 × 0.5 = -1`. Even `sum2_d16` (`-2 × 1.0 = -2`) — only 2× heavier than 0b8278 on the gimbal axes — collapses sigma. The 0.5× gimbal weighting in prior runs was the load-bearing knob that prevented sigma collapse.

### 3.2 Tracking quality at 100k — magnitude beats delta

| Variant | sigma | pair_valid | ep_len | bbox_center | bbox_size | tri_d0 | coll_per_env |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sum8_d8**  | 0.25 | **0.78** | **488** | **35.9** | **42.3** | **13.4** | 1.80 |
| sum4_d16 | 0.20 | 0.61 | 429 | 29.8 | 33.0 | 9.63 | 0.77 |
| sum4_d8  | 0.32 | 0.58 | 432 | 26.8 | 33.1 | 8.50 | 0.85 |
| sum2_d16 | 0.22 | **0.43** | **347** | 8.16 | 21.7 | 4.67 | 0.21 |
| 0b8278   | 1.75 | 0.84 | 492 | 39.1 | 44.5 | 19.5 | 1.74 |
| 0d4fa    | 0.16 | 0.89 | 496 | 37.8 | 45.7 | 27.6 | 0.20 |

`sum8_d8` is unambiguously the best of the four:
- Highest pair_valid (0.78), within 7% of 0b8278's 0.84 and 13% of 0d4fa's 0.89.
- Episode length 488, essentially equal to 0b8278 (492) and 0d4fa (496).
- Best `bbox_center` and `bbox_size` of the sweep — within 8–12% of 0b8278.
- Tri reward 13.4 vs 0b8278's 19.5 — a 31% regression from the prior solo run, but the lowest gap of any sweep variant.

`sum2_d16` is broken: light magnitude penalty (`-2`) with heavy delta penalty (`-16`) prevents the policy from making the sharp corrections needed for tracking. ep_len drops to 347, pair_valid to 0.43, bbox_center to 8 — barely tracking. Heavy delta + light magnitude is the wrong combination on the SIYI gimbal where slewing is already saturation-limited at 73°/s.

`sum4_d16` and `sum4_d8` sit in between, both with pair_valid 0.58–0.61 and tri 8.5–9.6. The (4,16) variant has tighter sigma (0.20 vs 0.32) but the heavier delta penalty doesn't pay off in tracking quality.

### 3.3 The triangulation regression vs 0b8278

Every sweep variant has worse `tri_d0` at 100k than 0b8278f00c (sum8_d8: 13.4 vs 19.5). This was unexpected — heavier penalties were supposed to *help* the policy commit to better-quality formations. The mechanism is more subtle:

- 0b8278 had sigma=1.75 and pair_valid=0.84. With the wide policy distribution, the **average** behaviour stumbles into wide-baseline triangulation geometries even though no single committed strategy is being pursued. tri_d0 = 19.5 is the integral over a high-variance policy.
- sum8_d8 has sigma=0.25 and pair_valid=0.78. The committed strategy keeps both targets in frame consistently but in a slightly tighter formation — better bbox quality per episode but smaller triangulation baselines on average.

So 0b8278's high tri reward was **partially an artifact of policy spread** rather than principled triangulation behaviour. The trade-off the sweep reveals: collapsing sigma ⇒ tighter, more consistent formation ⇒ more reliable bbox_center/bbox_size but smaller triangulation baselines per episode. The right path forward is probably a **wider initial-state distribution** for the formation when sigma is committed, not less penalty.

### 3.4 Collisions

| Variant | coll_per_env @ 100k |
|---|---:|
| sum8_d8 | 1.80 |
| sum4_d16 | 0.77 |
| sum4_d8 | 0.85 |
| sum2_d16 | 0.21 |
| 0b8278 | 1.74 |
| 0d4fa | 0.20 |

Heavier delta penalty (sum*_d16) drops collision exposure substantially — the policy keeps the gimbal smoother and stays pointed at the target, which incidentally prevents the close-quarters thrash that produced collisions in 0b8278. sum8_d8 maintains 0b8278's collision rate (~1.8) — its delta scale (`-8`) isn't heavy enough to suppress close maneuvers.

This suggests **a hybrid `(sum=8, delta=16)` configuration would be worth testing** — keeping sum8's tracking quality while picking up sum4_d16's collision suppression.

## 4. Analysis — what the sweep tells us

### 4.1 The `action_weight` vector was the prior runs' bottleneck, not penalty scale

The 0b8278f00c journal recommended a scale-only sweep (`-4, -8, -15, -30`). This sweep effectively bumped the *effective* gimbal-axis penalty by 2× for free (via the weight rebalance from 0.5 → 1.0) before changing scale. Even `sum2_d16` — only 2× heavier than 0b8278 on the gimbal axes — collapses sigma. The scale sweep above 2× then produces the magnitude/delta trade-offs in §3.2.

This is a useful re-framing: every prior iris_ma6 run since the action penalty was first introduced had `gimbal/zoom weights = 0.5`. That 0.5× factor was the silent load-bearing knob preventing sigma collapse. The 0d4fa6906b baseline used the same 0.5× weights but with `-30/-15` scales — i.e., it compensated by raising scale 6× higher. The sweep shows the right way to get 0d4fa-like sigma collapse without the giant scale magnitudes is to fix the per-axis weights first.

### 4.2 Magnitude > delta for SIYI gimbal regime

`sum8_d8` (heavy magnitude, equal delta) cleanly beats `sum4_d16` (light magnitude, heavy delta) on every tracking metric (pair_valid 0.78 vs 0.61, bbox_center 36 vs 30, ep_len 488 vs 429). The SIYI gimbal saturates at 73°/s — sharp corrections (high `|action_delta|`) are physically required to track an agile target. Heavy delta penalty fights the physical regime; heavy magnitude penalty just discourages sustained high commands, which is fine.

`sum2_d16` makes this concrete: `-16` delta with `-2` magnitude is the worst configuration — the policy can use big actions (light magnitude penalty) but is heavily punished for changing them (heavy delta penalty). Result: it picks a single sustained action and rides it, breaking tracking.

### 4.3 Collapsed sigma traded tri reward for tracking consistency

The unexpected `tri_d0` regression vs 0b8278 (19.5 → 13.4 at 100k) is real but should be interpreted carefully — 0b8278's tri reward came from a policy that was *not committed* and stumbled into wide formations on average. The sweep produces a *committed* policy with smaller average baselines. The right way to recover absolute tri reward post-sigma-collapse is to widen the initial-state distribution for inter-agent geometry, not to weaken the penalty.

## 5. Decision

**Sweep winner: `sum8_d8` (action_sum_penalty_scale=-8, action_delta_penalty_scale=-8)**, with the per-axis weight rebalance baked in.

It collapses sigma (0.25 vs 0d4fa's 0.16), maintains 0b8278-level tracking (pair_valid 0.78 vs 0.84, ep_len 488 vs 492), and trades only ~30% tri reward for the qualitative win of policy commitment. The remaining gaps are addressable downstream — collision rate via a delta bump (probably `(8, 16)`), tri reward via wider geometry sampling.

**Recommended next runs**:

1. **Continue `sum8_d8` to 400k** to see whether the late-phase tail recovers vs 0b8278. The sigma-collapsed regime should hold formation through burst dropout where the loose-sigma regime broke.
2. **Test `(sum=8, delta=16)`** as a single-knob ablation on top of `sum8_d8`. Tests whether picking up `sum4_d16`'s collision suppression while keeping `sum8_d8`'s tracking quality is feasible.
3. **Defer the geometry-widen change** until after the late-phase data is in. If sum8_d8 holds tri reward through 200k–400k, the regression at 100k may resolve on its own.

## 6. Open items

- **Late phase**: all four runs stopped at 100k. Need at least one of them (probably sum8_d8) to run to 400k to compare against 0b8278's late-phase trajectory.
- **(8, 16) hybrid**: untested but predicted from the §3.4 analysis.
- **The pre-delay tri gap (sweep vs 0b8278) might invert post-delay** — the committed policy should handle observation corruption better than the uncommitted one, recovering tri reward in absolute terms once delay/dropout layers on. Falsifiable with the late-phase data.
- **Reverting `action_weight[2] = 5` (vz)** — this was preserved at 5 across the sweep. Worth a one-knob test once the magnitude/delta regime is locked in. If the heavy vz weight was holding back vertical maneuvers needed to descend toward agile targets, easing it could pay off.

## 7. Side notes

- The bash script's comment says "case 1: (-8, -16)" but the `PENALTY_PAIRS` array actually contains `"8 8"` first. The script comments are stale; the env.yaml of each run is authoritative.
- All four runs use the dynamics curriculum at 220k–240k (inherited from 0b8278f00c). At 100k the dynamics ramp hasn't fired yet — these data are pre-dynamics, pre-delay.
- The sigma trajectories diverge between 4k and 20k: every variant starts at sigma ≈ 0.4 and rapidly drops over the first 16k steps. This is the heavy-penalty regime forcing concentration early. Compare to 0b8278 which started at sigma 0.58 at 4k and *climbed* to 0.75 / 1.05 / 1.59 at 20k/40k/80k — the policy was being pulled apart by the entropy bonus when penalty was too light.
- The `sum2_d16` failure provides a useful negative datapoint: the magnitude/delta ratio matters for the SIYI gimbal regime, not just the absolute scale. Future per-axis weight changes should keep this in mind.
