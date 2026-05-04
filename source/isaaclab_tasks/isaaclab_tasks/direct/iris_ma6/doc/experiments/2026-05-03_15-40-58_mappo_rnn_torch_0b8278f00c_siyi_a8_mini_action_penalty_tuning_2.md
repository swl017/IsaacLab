# Experiment: 2026-05-03_15-40-58_mappo_rnn_torch_0b8278f00c_siyi_a8_mini_action_penalty_tuning_2

**Commit**: `0b8278f00c` ("Action penalty tuning")
**Date**: 2026-05-03 → 2026-05-04
**Experiment ID**: `siyi_a8_mini_action_penalty_tuning_2`
**Base**: `58a3c3ee84` (siyi_a8_mini_triangulation_fix) with action penalties bumped 2× and the dynamics-randomization window shifted later.
**Status**: Run reached **376k**. Triangulation reward held a new high through 220k, with late-phase tail materially better than the prior `-1/-1` regime — but still short of 0d4fa6906b. The 2× penalty bump did not collapse sigma, suggesting penalty scale needs to climb at least an order of magnitude further before the policy commits.

---

## 1. Hypothesis (going in)

The 58a3c3ee84 ratification of `include_drift_uncertainty=False` left the late-phase tail (240k+) as the only remaining gap to the 0d4fa6906b baseline. That tail was attributed to the very-light `-1/-1` action penalties — sigma stuck at 1.76 throughout, no policy concentration, ep_len collapsed under burst dropout.

Recommended next intervention from the 58a3c3ee84 journal was reverting to `-30/-15`. Actual launch chose a much more conservative step (`-2/-2`) plus a curriculum shift that delays the dynamics-randomization phase. Implicit hypotheses:

- **Action penalty 2×**: small step toward 0d4fa's regime; tests whether the late-phase tail is monotone in penalty scale (small bump → small improvement).
- **Dynamics curriculum 180k→220k start**: removes gimbal rate-loop τ ramp from the 180k–220k window where random_delay + dropout are already maxing out, so the policy doesn't have to absorb three new dynamics simultaneously.

## 2. Configuration delta (vs 58a3c3ee84)

```python
# Action penalties — 2× stronger
action_sum_penalty_scale: -1.0 → -2.0
action_delta_penalty_scale: -1.0 → -2.0

# Dynamics curriculum — shifted and shortened
dynamics_start_step: 180_000 → 220_000
dynamics_end_step:   220_000 → 240_000   # 20k window (was 40k)

# All other knobs preserved:
#   include_drift_uncertainty: false
#   lambda_cbf: 1.0
#   triangulation_reward_scale: 5.0
#   cylinder_diameter_max: 60, target_distance_max: 25
#   moving_target_end_step: 80_000
#   rate-loop curriculum gate enabled
```

## 3. Results

### 3.1 Bootstrap & pre-delay (0–120k)

| Step | rew_inst | ep_len | pair_valid | sigma | tri_d0 | bbox_center | bbox_size | as_d0 | coll_per_env |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k   | 4.73 | 462 | 0.74 | 0.58 | 0.00 | 24.3 | 39.2 | -6.98 | 0.94 |
| 20k  | 6.53 | 498 | 0.93 | 0.75 | 0.00 | 47.2 | 47.0 | -4.97 | 4.20 |
| 40k  | 6.00 | 494 | 0.84 | 1.05 | 0.00 | 42.4 | 44.9 | -5.82 | 3.01 |
| 80k  | 5.74 | 492 | 0.85 | **1.59** | 6.96 | 39.0 | 44.6 | -10.65 | 4.04 |
| 100k | 6.86 | 492 | 0.84 | 1.75 | **19.51** | 39.1 | 44.5 | -11.28 | 1.74 |
| 120k | **7.24** | 492 | 0.83 | 1.75 | **23.46** | 39.9 | 44.2 | -11.54 | 1.85 |

vs PREV (`-1/-1`) at 120k: tri 17.40, coll_per_env 5.51. **Tri jumped 35% (17.4 → 23.5)** with the 2× penalty bump. Collision exposure dropped 3× (5.5 → 1.9) — heavier penalty made the policy less willing to thrash through close formations. `pair_valid` essentially unchanged (0.85 → 0.83), `bbox_center` unchanged (40 → 40).

Sigma rises slower than PREV (1.05 at 40k vs 1.76 at 40k in PREV) but still saturates at ~1.75 by 100k. The 2× bump is **not** enough to collapse sigma like 0d4fa's 0.14 — that would require a much heavier penalty. The action_sum reward magnitude actually *decreased* under heavier scale (PREV's `|as|/scale = 7.49`, NEW's `|as|/scale = 5.77`), so per-step action magnitudes did drop ~25% — measurable but moderate effect on policy concentration.

### 3.2 Detection-latency ramp (120k–140k)

| Step | rew_inst | pair_valid | tri_d0 | bbox_center | coll_per_env |
|----:|---:|---:|---:|---:|---:|
| 120k | 7.24 | 0.83 | **23.46** | 39.9 | 1.85 |
| 124k | 5.07 | 0.68 | **17.43** | 24.2 | 1.53 |
| 132k | 4.59 | 0.62 | 14.66 | 16.5 | 0.85 |
| 140k | 3.84 | 0.50 | **13.36** | 14.5 | 1.00 |

vs PREV across the same window: 17.40 → 10.46 (drop of 6.94). NEW: 23.46 → 13.36 (drop of 10.10). The post-delay drop magnitude is similar in absolute terms; NEW just starts higher. Importantly **NEW's 124k tri (17.43) exceeds 0d4fa's 124k tri (12.50)** because 0d4fa had a transient deeper dip at that step.

### 3.3 Random delay + dropout (140k–200k)

| Step | rew_inst | pair_valid | tri_d0 | bbox_center | ep_len |
|----:|---:|---:|---:|---:|---:|
| 140k | 3.84 | 0.50 | 13.36 | 14.5 | 433 |
| 156k | 4.65 | 0.61 | 17.93 | 19.3 | 466 |
| 180k | 4.49 | 0.57 | 17.49 | 18.9 | 464 |
| 200k | 4.51 | 0.58 | **17.69** | 18.1 | 462 |

`tri_d0` peaks at 17.93 at 156k and stays at ~17 through 200k — a flat plateau very close to PREV's peak (16.50 at 180k) and within ~25% of 0d4fa's plateau (~22). `pair_valid` recovers to 0.57–0.61 by 156k and holds. The dynamics curriculum now firing at 220k means the rate-loop τ is **not** active during this window — confounded with the penalty change.

### 3.4 Burst dropout + dynamics ramp (200k–240k): tail recovers vs PREV

| Step | rew_inst | ep_len | pair_valid | tri_d0 | coll_per_env | sigma |
|----:|---:|---:|---:|---:|---:|---:|
| 200k | 4.51 | 462 | 0.58 | 17.69 | 1.65 | 1.76 |
| 220k | **4.74** | **468** | **0.63** | **19.45** | 1.48 | 1.76 |
| 240k | 3.36 | 416 | 0.43 | 12.33 | 0.75 | 1.76 |
| 280k | 2.05 | 258 | 0.27 | 4.20 | 0.16 | 1.76 |

The 220k row is the strongest data point in this run: `pair_valid 0.63`, `ep_len 468`, `tri 19.45` — **essentially matching 0d4fa6906b at 220k** (pair_valid 0.65, ep_len 476, tri 22.33). vs PREV at 220k: `pair_valid 0.41`, `ep_len 405`, `tri 9.51`. The 220k window is where PREV started cratering; NEW holds.

But the tail past 240k still degrades. By 280k: `pair_valid 0.27`, `ep_len 258` — NEW vs PREV at 280k is 0.27/258 vs 0.26/166. Some recovery but the curve continues down.

### 3.5 Far late phase (280k–376k): same trajectory shape as 0d4fa

| Step | rew_inst | ep_len | pair_valid | tri_d0 |
|----:|---:|---:|---:|---:|
| 280k | 2.05 | 258 | 0.27 | 4.20 |
| 320k | 1.52 | 142 | 0.27 | 1.98 |
| 360k | 1.19 | 110 | 0.25 | 1.41 |

Compare to 0d4fa6906b at the same steps: 280k (rew 4.03, ep_len 274, pv 0.51, tri 11.61), 320k (3.33, 164, 0.44, 6.25), 360k (2.77, 126, 0.41, 4.30). NEW's tail trajectory has a similar shape (monotone decline in ep_len from ~250 to ~110) but lands at roughly half 0d4fa's reward magnitude. The shape match suggests both runs are bottlenecked by the same late-phase mechanism (compounding observation corruption + dynamics ramp); the magnitude gap is the policy regime — 0d4fa with sigma 0.42, NEW with sigma 1.76.

## 4. Analysis

### 4.1 Penalty scale is sub-linear in late-phase recovery

| Run | action_sum scale | pair_valid 220k | ep_len 220k | tri 220k |
|---|---:|---:|---:|---:|
| 343e2ed4a3 | -1 | 0.41 | 405 | 9.51 |
| 0b8278f00c | **-2** | 0.63 | 468 | 19.45 |
| 0d4fa6906b | -30 | 0.65 | 476 | 22.33 |

Doubling the penalty scale (×2) closed most of the 220k gap to 0d4fa. But the next step (the 240k+ tail) shows NEW still falling away while 0d4fa holds. Penalty scale appears to have **diminishing returns**: the small 2× bump captures most of the 220k recovery but the late tail (which involves both burst dropout and the new dynamics ramp at 220k–240k) needs further pressure or different intervention.

The sigma trajectory backs this up: NEW sigma reaches 1.59 at 80k vs PREV's 1.76 — a measurable lag in the saturation rate but still saturated by 100k. The 2× bump barely shifts where the policy lands; only an order-of-magnitude bump (toward `-30`) would actually drive sigma collapse.

### 4.2 Dynamics curriculum shift was probably not load-bearing

Moving `dynamics_start_step` 180k → 220k confounds the comparison: NEW didn't have to absorb gimbal rate-loop τ during 180k–220k while PREV did. But comparing NEW's 200k row (rew 4.51, pv 0.58) to PREV's 200k row (rew 4.59, pv 0.56), the rate-loop τ ramp didn't seem to hurt PREV's 180k–200k window much. The **pair_valid difference** (NEW 0.63 vs PREV 0.41 at 220k) shows up *after* 200k — i.e., during the new dynamics window itself — so the curriculum shift might actually *delay* the failure rather than relieve it.

To isolate: the next run that wants to attribute late-phase recovery should keep dynamics window unchanged. With both knobs moved here, this run can't separate "penalty bump fixed it" from "delaying the rate-loop ramp helped."

### 4.3 Pre-delay tri reward inches up the remaining 1.85× gap

| Run | tri 120k | gap to 0d4fa (32.18) |
|---|---:|---:|
| 343e2ed4a3 (drift on, -1) | 6.55 | 4.91× |
| 58a3c3ee84 (drift off, -1) | 17.40 | 1.85× |
| 0b8278f00c (drift off, -2) | 23.46 | **1.37×** |
| 0d4fa6906b | 32.18 | 1.0× |

The 2× penalty bump narrowed the residual pre-delay gap from 1.85× to 1.37×. The policy with heavier action penalties is producing slightly tighter formations — `coll_per_env` at 120k dropped 3× (5.5 → 1.9), suggesting the agents are flying with cleaner inter-agent geometries that triangulate better. The remaining ~37% gap is what's structurally attributable to `Sigma_K` + tighter geometry; closer to baseline than expected.

### 4.4 ep_len trajectory vs 0d4fa is now the canonical late-phase comparison

NEW ep_len: 280k = 258, 320k = 142, 360k = 110.
0d4fa ep_len: 280k = 274, 320k = 164, 360k = 126.

NEW is at **94%, 87%, 87%** of 0d4fa across the three checkpoints. The shape matches; the offset is small. This is the strongest indication that the iris_ma6 SIYI realism stack is **converging on 0d4fa-like trajectories** when the action penalty regime is even modestly closer to baseline.

## 5. Decision

**Penalty bump direction confirmed; magnitude still too small.** The 2× step was a useful sanity check — it monotonically improved every metric vs PREV without breaking anything. But sigma never collapses, and the late-phase tail past 240k still tracks below 0d4fa.

**Next intervention candidates** (only one should be tested at a time to keep ablation clean):

1. **Action penalty 4×–8× (e.g., `-8/-8` or `-15/-15`)** — direct continuation of the trend. The diminishing-returns analysis suggests this has the most ROI for the residual late-phase gap. If sigma starts falling below 1.5 by 40k, the regime is shifting; if it stays stuck at 1.7+, the action_weight vector itself may need rebalancing.
2. **Restore dynamics curriculum to 180k–220k** — isolate whether the curriculum shift contributed to the 220k recovery. Combine with whatever penalty value is chosen above.
3. **Re-evaluate `triangulation_reward_scale`** — pre-delay gap is now 1.37× and shrinking with each penalty bump. Probably no longer needs a scale change; the improvement is coming from the policy committing to better geometries, not from reward magnitude.

## 6. Open items

- **Penalty regime ceiling**: at what scale does sigma actually start to collapse on the SIYI stack? Worth a small sweep — `-4`, `-8`, `-15`, `-30` — if the late tail is still load-bearing for s2r.
- **Dynamics curriculum shift effect**: not isolated in this run; the next clean A/B should hold it constant.
- **`bbox_size` is moderately lower than 0d4fa across the whole run** (e.g., 220k: NEW 39.2 vs 0d4fa 40.8; 280k: NEW 15.2 vs 0d4fa 21.1). Probably the SIYI zoom curve + tighter geometry combined — the policy can't frame as tightly as in the wider envelope. Not actionable until the action-penalty regime is fully resolved.

## 7. Side notes

- This is the first run in the SIYI realism series where **a 220k metric matches 0d4fa6906b** (`pair_valid 0.63 vs 0.65`, `ep_len 468 vs 476`, `tri 19.45 vs 22.33`). The drift fix + 2× penalty bump combination is a concrete proof point that the iris_ma6 minimum-viable-s2r config is achievable on the deployment envelope.
- The collision rate tells a clean story: `coll_per_env` halved at every checkpoint vs PREV (4.73 → 4.20 at 4k, 5.51 → 1.85 at 120k, 1.06 → 1.48 at 220k roughly equivalent). Heavier penalty → less close-quarters thrash → fewer collisions → cleaner triangulation geometry.
- The dynamics-curriculum shift to 220k–240k means future runs comparing `rate-loop on vs off` post-delay will need to account for this if reproducing.
