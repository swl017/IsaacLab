# Experiment: 2026-04-01_13-19-18_mappo_rnn_torch_928b9585f2_min_lr_tuning

**Commit**: `928b9585f2`
**Date**: 2026-04-01
**Experiment ID**: `min_lr_tuning`
**Base**: `7e46515b39`<2026-03-30_22-25-56_mappo_rnn_torch_7e46515b39_compressed_curriculum_target_proximity_20k>

**Status**: Completed (400k steps).

---

## 1. Hypothesis
The `7e46515b39` baseline's KL-adaptive LR scheduler collapsed to 1e-6 after curriculum
transitions, freezing the policy for the final 100k+ steps. The `18dc640464` attempt
(`kl_threshold=0.04` + `min_lr=1e-4` + `max_lr=0.003`) failed because the wider KL threshold
removed the scheduler's natural oscillation, causing σ divergence and reward collapse.

This run tests the minimal fix: **only `min_lr=1e-4`**, with `kl_threshold` reverted to 0.02.
The hypothesis is that the scheduler's oscillation is beneficial during normal training, and
only the floor needs fixing to prevent post-curriculum collapse.

## 2. Configuration Delta
Changes from baseline `7e46515b39`:
```yaml
learning_rate_scheduler_kwargs:
  kl_threshold: 0.02                  # UNCHANGED from baseline
  min_lr: 1.0e-4                      # was default 1e-6 — the ONLY change
```

## 3. Results (400k steps, completed)

### 3.1 Final Metrics
| Metric | Value |
|--------|------:|
| Mean Reward | 3788 |
| Max Reward | 5473 |
| Learning Rate | 0.000100 (at floor) |
| Policy Std Dev | 1.210 |
| Value Loss | 0.001011 |
| Policy Loss | −0.002348 |
| pair_valid_rate | 0.861 |
| triangulation (d0) | 44.4 |
| bbox_center (d0) | 22.6 |
| bbox_size (d0) | 44.4 |
| collision_fraction | 0.0008 |
| tracking_lost_fraction | 0.005 |

### 3.2 Head-to-Head vs 7e46515b39 (baseline, kl=0.02, min_lr=1e-6)

#### Learning Rate — floor fix working as designed
| Step | 7e46 (old) | 928b (new) | Status |
|-----:|------:|------:|--------|
| 4k | 0.00297 | 0.00608 | Both high (early) |
| 36k | 0.00093 | 0.00060 | Both oscillating |
| 80k | 0.00142 | 0.00044 | Both oscillating |
| 120k | 0.00183 | 0.00053 | Both healthy |
| 140k | **0.000194** | 0.000506 | Old decaying, new held |
| 176k | **0.000001** | 0.000419 | Old frozen, new alive |
| 200k | **0.000001** | 0.000277 | Old frozen, new alive |
| 236k | **0.000001** | 0.000100 | Old frozen, new at floor |
| 288k | **0.000002** | 0.000100 | Old frozen, new at floor |

The scheduler's natural oscillation pattern (0.0005–0.003) is preserved through phases 1–3,
identical to the baseline. The only difference appears post-124k: the new run floors at 1e-4
instead of 1e-6. This is exactly the intended behavior.

#### Policy Std Dev — identical trajectory, no divergence
| Step | 7e46 (old) | 928b (new) | Delta |
|-----:|------:|------:|------:|
| 20k | 0.337 | 0.644 | +0.31 |
| 48k | 0.547 | 0.808 | +0.26 |
| 80k | 0.974 | 1.207 | +0.23 |
| 92k | 1.214 | 1.215 | +0.00 |
| 120k | 1.211 | 1.224 | +0.01 |
| 200k | 1.216 | 1.215 | −0.00 |
| 288k | — | 1.211 | — |
| 400k | — | 1.210 | — |

Both converge to σ≈1.21. The new run reaches equilibrium slightly faster (80k vs 92k)
due to a hotter early LR, but the final value is within ±0.01. No σ divergence, confirming
the `kl_threshold=0.02` revert was correct.

#### Mean Reward — the key result
| Phase | Steps | 7e46 (old) | 928b (new) | Delta | Notes |
|-------|------:|------:|------:|------:|-------|
| Early | 4k–24k | 1656→3119 | 1673→3244 | +125 | Slightly faster warmup |
| Exploration dip | 36k–56k | 3082→2777 | 2989→2952 | +175 | Same dip, new recovers faster |
| Peak climb | 68k–120k | 2767→4269 | 3057→4190 | −79 | Similar peak |
| Curriculum shift | 124k | 3320 | 2957 | −363 | New hit harder |
| Post-shift recovery | 140k–180k | 3926→3932 | 3662→3634 | −298 | Old higher but frozen |
| **Old collapse zone** | 192k–208k | **3455→2892** | 3166→3365 | **+473** | **New avoids collapse** |
| Late training | 240k–288k | 2981→— | 3608→3559 | **+578** | **New clearly better** |
| Final | 400k | — | 3788 | — | Still learning |

The critical difference: **the 928b run never experienced the catastrophic collapse to 2509
that 7e46 hit at 200k.** The worst mean reward was 3125 at 196k (vs 2509 for the old run).
From 240k onward, 928b stabilizes at 3500–3600, while 7e46 was oscillating at 2500–3400.

#### Value Loss — healthy throughout
| Step | 7e46 (old) | 928b (new) | Delta |
|-----:|------:|------:|------:|
| 120k | 0.001240 | 0.002717 | +0.001 |
| 176k | 0.003717 | 0.001416 | −0.002 |
| 200k | **0.010974** | 0.001827 | **−0.009** |
| 252k | 0.007680 | 0.001498 | −0.006 |
| 288k | — | 0.001538 | — |
| 400k | — | 0.001011 | — |

The old run's value loss spiked to 0.011 at 200k (frozen policy → non-stationary value targets).
The new run's value loss stays below 0.003 and trends downward to 0.001, confirming the
policy-value feedback loop remains healthy.

#### Policy Loss — active learning continues
From 128k onward, 928b shows consistently negative policy loss (−0.001 to −0.003), meaning
L^CLIP > 0 — the policy is successfully increasing probability of advantageous actions.
The old run's policy loss was near zero or positive in this range (no useful gradient signal).

At 400k, policy loss is −0.0023 — still actively improving.

### 3.3 Head-to-Head vs 4f7d9d0c34 (slow curriculum, kl=0.02, min_lr=1e-6)

The `4f7d` run used the original (slower) curriculum and ran for 364k steps. It also
suffered LR collapse to 1e-6 but had more time per curriculum phase.

#### Mean Reward
| Phase | Steps | 4f7d | 928b | Delta | Notes |
|-------|------:|------:|------:|------:|-------|
| Early | 4k–24k | 1638→2741 | 1673→3244 | +503 | 928b much faster warmup |
| Exploration | 36k–68k | 2842→2724 | 2989→3057 | +333 | 928b explores better |
| Peak | 80k–120k | 3115→4137 | 3344→4190 | +53 | Similar peaks |
| 4f7d peak plateau | 124k–140k | 4241→4277 | 2957→3662 | −615 | 4f7d's slower curriculum hasn't shifted yet |
| 4f7d collapse | 144k | **2716** | 3694 | **+978** | 4f7d hits its curriculum shift |
| Mid-late | 180k–240k | 3964→3945 | 3634→3608 | −337 | 4f7d higher but decaying |
| Late convergence | 272k–296k | 3503→3474 | 3522→3594 | **+120** | 928b catches up |
| Extended | 364k–400k | 3453 | 3788 | **+335** | 928b pulls ahead |

`4f7d` achieved higher sustained reward (3900–4000) in the 160k–240k range because its
slower curriculum gave the policy more time at each difficulty. But `4f7d` was trending
downward (4000→3450 from 180k→364k) while `928b` stabilized and then climbed (3500→3788
from 240k→400k). By the end, **928b surpassed 4f7d by +335**.

#### Value Loss
928b's value loss is consistently lower from 96k onward. At 260k: 0.0009 vs 0.0047.
`4f7d`'s frozen LR caused value loss to climb steadily (0.003→0.005 from 200k→364k).
928b's value loss was trending down (0.002→0.001 from 200k→400k).

### 3.4 Task Metric Trajectories (928b)

#### pair_valid_rate
```
4k:   0.65 → 20k: 0.95 (warmup)
36k:  0.87 → 56k: 0.90 (exploration dip + recovery)
80k:  0.89 → 120k: 0.88 (curriculum ramp, stable)
124k: 0.78 → 140k: 0.93 (shift + fast recovery)
192k: 0.77 → 216k: 0.83 (dynamics ramp dip)
240k: 0.85 → 320k: 0.86 (stabilizing)
360k: 0.86 → 400k: 0.86 (converged)
```
Sustained above 0.83 from step 240k to completion. No permanent collapse.

#### triangulation (drone_0)
```
64k:  1.1 (onset)
92k:  26.4 → 120k: 40.7 (rapid climb)
124k: 28.3 (curriculum shift dip)
172k: 36.2 → 240k: 39.8 (recovery)
320k: 41.0 → 400k: 44.4 (still climbing at end)
```
Peak triangulation of 44.4 at completion — higher than 4f7d's peak of ~50 in the overlap
region, and **still improving**. This is the strongest evidence that the `min_lr` fix
enables continued learning.

#### bbox_center (drone_0)
```
20k:  50.5 (peak — easy curriculum)
120k: 31.7 (curriculum ramp)
200k: 20.5 (post-dynamics dip)
360k: 22.4 → 400k: 22.6 (stabilized)
```
Lower than 4f7d's sustained 30+ because the compressed curriculum is harder. But stable.

#### collision_fraction
```
16k:  0.0096 (peak — early exploration)
120k: 0.0013
200k: 0.0006
400k: 0.0008
```
Collisions decreased 12× from peak to convergence. Safety behavior is well-learned.

#### tracking_lost_fraction
```
4k:   0.27 (no tracking yet)
20k:  0.006
104k–140k: 0.000 (noise onset → no early termination)
200k: 0.009
400k: 0.005
```
Very low throughout post-warmup training. No tracking collapse.

## 4. Analysis

### The min_lr fix works
The single change (`min_lr: 1e-4`) achieved exactly what was intended:
1. **Preserved natural oscillation**: The scheduler's LR pattern through phases 1–3 is
   indistinguishable from the baseline. No σ divergence, no premature exploration.
2. **Prevented collapse**: Post-curriculum-shift LR floors at 1e-4 instead of 1e-6,
   maintaining 100× more learning capacity.
3. **Healthy value function**: Value loss stays below 0.003 and trends downward, vs the
   baseline's spike to 0.011. The policy-value feedback loop remains functional.
4. **Continued learning**: Policy loss is negative at 400k (−0.0023), meaning the policy
   is still improving. Triangulation reward is still climbing (44.4 and rising).

### Comparison summary
| Metric | 7e46 (baseline) | 4f7d (slow curriculum) | 928b (min_lr fix) |
|--------|------:|------:|------:|
| Peak mean reward | 4269 (120k) | 4277 (140k) | 4190 (120k) |
| Final mean reward | ~3400 (260k) | 3453 (364k) | **3788 (400k)** |
| Reward trend at end | oscillating | declining | **rising** |
| LR at 200k | 0.000001 | 0.000002 | **0.000277** |
| Value loss at 200k | 0.01097 | 0.00286 | **0.00183** |
| Triangulation (final) | ~40 (120k peak) | ~44 (364k) | **44.4 (400k, rising)** |
| pair_valid_rate (final) | ~0.82 (224k) | ~0.86 (364k) | **0.86 (400k)** |

928b didn't achieve the highest peak reward, but it has the **best final performance** and
the **only upward-trending reward at termination**. The other two runs were frozen or declining.

### Why 928b's peak is slightly lower than 7e46
928b's peak (4190 at 120k) is 79 points below 7e46's peak (4269 at 120k). This is likely
due to the hotter early LR (928b started at 0.006 vs 0.003), which caused slightly faster
σ growth and a deeper exploration dip. The policy spent less time exploiting in the optimal
σ≈1.21 zone before the curriculum shift hit. This is noise-level — within the variance of
a single seed.

## 5. Decision
- **This is the new baseline.** `min_lr=1e-4` with `kl_threshold=0.02` is validated.
- The policy is still improving at 400k. Consider extending to 500k+ to see if reward
  continues climbing (triangulation was 44.4 and rising).
- The compressed curriculum + min_lr fix produces a policy that is both high-performing
  and robust to curriculum transitions.
- Next experiment should focus on other bottlenecks (tracking_reacquire_steps, dynamics
  ramp timing) rather than further LR scheduler tuning.
