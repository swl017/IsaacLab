# Experiment: 2026-04-01_04-33-26_mappo_rnn_torch_18dc640464_min_lr_and_kl_threshold_tuning

**Commit**: `18dc640464`
**Date**: 2026-04-01
**Experiment ID**: `min_lr_and_kl_threshold_tuning`
**Base**: `7e46515b39`<2026-03-30_22-25-56_mappo_rnn_torch_7e46515b39_compressed_curriculum_target_proximity_20k>

**Status**: Failed — policy diverged. Terminated at 128k steps.

---

## 1. Hypothesis
The KL-adaptive LR scheduler collapsed the learning rate to 1e-6 in the baseline run (`7e46515b39`)
after curriculum transitions, freezing the policy. Two fixes applied:
1. `min_lr: 1e-4` — prevent LR floor collapse
2. `kl_threshold: 0.04` (was 0.02) — widen the scheduler's dead zone so curriculum-induced KL
   doesn't trigger LR reductions
3. `max_lr: 0.003` — cap upward LR growth (added after `44274eee4c` run exploded with default max_lr=0.01)

## 2. Configuration Delta
Changes from baseline `7e46515b39`:
```yaml
kl_threshold: 0.04                    # was 0.02
learning_rate_scheduler_kwargs:
  kl_threshold: 0.04                  # was 0.02
  max_lr: 0.003                       # was default 0.01
  min_lr: 0.0001                      # was default 1e-6
```
All other parameters identical.

## 3. Results (128k steps)

### 3.1 Head-to-Head vs Baseline (7e46515b39)

#### Learning Rate
| Step | 7e46 (old) | 18dc (new) | Delta |
|-----:|------:|------:|------:|
| 4k | 0.00297 | 0.00298 | +0.000 |
| 20k | 0.00095 | 0.00254 | +0.002 |
| 48k | 0.00114 | 0.00300 | +0.002 |
| 80k | 0.00142 | 0.00300 | +0.002 |
| 100k | 0.00131 | 0.00135 | +0.000 |
| 120k | 0.00183 | 0.00277 | +0.001 |
| 124k | 0.00070 | 0.00139 | +0.001 |

LR stayed at or near `max_lr=0.003` for almost the entire run (4k–96k). The wider
`kl_threshold=0.04` meant the "increase" trigger (KL < 0.02) fired freely, pushing LR to
the ceiling. The scheduler only braked briefly at 100k (0.00135) and 124k (0.00139).

**Comparison**: The old run's LR oscillated between 0.0008–0.0018, creating natural
consolidation periods. The new run was pinned at the ceiling — no oscillation, no consolidation.

#### Policy Standard Deviation
| Step | 7e46 (old) | 18dc (new) | Delta |
|-----:|------:|------:|------:|
| 4k | 0.559 | 0.616 | +0.06 |
| 20k | 0.337 | 0.801 | +0.46 |
| 48k | 0.547 | 1.234 | +0.69 |
| 80k | 0.974 | 1.340 | +0.37 |
| 92k | 1.214 | 1.499 | +0.28 |
| 120k | 1.211 | 1.617 | +0.41 |
| 124k | 1.212 | 1.659 | +0.45 |

**Critical failure**: σ never stabilized. The old run plateaued at σ≈1.21 by step 92k.
The new run kept climbing to 1.66 at 124k. The sustained high LR (0.003 vs 0.001) kept
the entropy gradient strong enough to push σ past the old equilibrium. The policy became
too stochastic to exploit.

Why: The entropy gradient ∂(-c_e·H)/∂ln(σ) = -0.01 is constant. At LR=0.003, the effective
push on ln(σ) is -0.01 × 0.003 = 3e-5 per step. At LR=0.001 (old run), it's 1e-5 per step.
The 3× stronger entropy push overwhelmed the policy gradient's pull toward lower σ.

#### Mean Reward
| Step | 7e46 (old) | 18dc (new) | Delta |
|-----:|------:|------:|------:|
| 4k | 1656 | 1951 | +295 |
| 16k | 3086 | 3411 | +324 |
| 36k | 3082 | 3064 | −18 |
| 60k | 2781 | 2168 | −613 |
| 80k | 2741 | 1519 | −1222 |
| 100k | 3708 | 1240 | −2468 |
| 120k | 4269 | 1677 | −2593 |
| 124k | 3320 | 1084 | −2236 |

The new run started stronger (+324 at 16k) due to faster early learning, then diverged
catastrophically. By 100k the old run was at its peak (3708) while the new run was at 1240.
**The exploration phase never ended** — the policy kept getting more random instead of
converging to exploit discovered strategies.

#### Key Task Metrics
| Metric | 7e46 @ 120k | 18dc @ 120k | Verdict |
|--------|------:|------:|---------|
| pair_valid_rate | 0.93 | ~0.30 | 3× worse |
| bbox_center (d0) | 38 | 10.7 | 4× worse |
| bbox_size (d0) | ~45 | 21.5 | 2× worse |
| triangulation (d0) | 47 | 8.1 | 6× worse |
| tracking_lost_frac | 0.0% | 60.6% | Broken |

The policy essentially lost the ability to track targets. `pair_valid_rate` dropped to 0.30
at 100k (both drones seeing the target only 30% of the time). `tracking_lost_fraction` spiked
to 60.6% — episodes were terminating early because drones couldn't maintain visual contact.

#### Value Loss
| Step | 7e46 (old) | 18dc (new) | Delta |
|-----:|------:|------:|------:|
| 20k | 0.000247 | 0.001015 | +0.001 |
| 48k | 0.000809 | 0.004042 | +0.003 |
| 80k | 0.002437 | 0.005839 | +0.003 |
| 100k | 0.004219 | 0.009189 | +0.005 |
| 120k | 0.001240 | 0.003290 | +0.002 |

Consistently 3–5× higher. The non-stationary policy (ever-increasing σ) made the value
function's prediction task much harder. Noisy advantages ⟹ noisy policy gradients ⟹
more σ growth ⟹ noisier advantages. A vicious cycle.

### 3.2 Intermediate Run: 44274eee4c (same commit, no max_lr cap)
Also failed. LR exploded to 0.01 (KLAdaptiveLR default max), σ hit 1.75 by step 8k,
reward collapsed to 186 by step 64k. Adding `max_lr=0.003` prevented the LR explosion
but didn't prevent the σ divergence — it just slowed it.

## 4. Analysis

### Root Cause: kl_threshold=0.04 is too permissive
The `kl_threshold` change from 0.02 → 0.04 was the primary failure. It had two effects:
1. **LR stayed at ceiling**: The "increase" trigger (KL < 0.02) fired almost every update,
   pushing LR to `max_lr=0.003` and keeping it there. The old run's tighter threshold
   (KL < 0.01 to increase) created natural LR oscillation that balanced exploration/exploitation.
2. **Early stopping was less aggressive**: The agent-level `kl_threshold=0.04` meant learning
   epochs were rarely cut short, allowing larger per-update policy changes that further
   destabilized the value function.

### Why the old run's LR oscillation was beneficial
The old run's LR pattern (0.0008–0.0018) was not a bug — it was an emergent annealing schedule:
- When the policy made good progress → KL spiked → LR dropped → consolidation period
- When the policy stabilized → KL dropped → LR rose → learning resumed

This oscillation naturally balanced exploration and exploitation. The new run eliminated
this mechanism by making the scheduler too permissive.

### The σ divergence mechanism
With sustained LR=0.003 and entropy_loss_scale=0.01:
- Entropy gradient push on ln(σ): constant −0.01 per update step
- Policy gradient pull on ln(σ): proportional to advantage magnitude × LR
- At high σ, advantages become noisy (wide action distribution → high variance in returns)
- Noisy advantages → weak policy gradient pull → entropy push wins → σ grows further

The old run found equilibrium at σ≈1.21 because its lower LR (~0.001) gave a weaker entropy
push that balanced the policy gradient pull. At LR=0.003, the equilibrium shifts to σ≈1.6+,
which is too stochastic for the task.

## 5. Decision

### Conclusion: Revert kl_threshold to 0.02, keep only min_lr fix
The original diagnosis was correct — the LR collapse after curriculum shifts was the problem.
But `kl_threshold=0.04` was too aggressive a fix. It eliminated the scheduler's beneficial
oscillation behavior and caused σ divergence.

The correct minimal fix is:
```yaml
kl_threshold: 0.02                    # REVERTED to original
learning_rate_scheduler_kwargs:
  kl_threshold: 0.02                  # REVERTED to original
  min_lr: 1.0e-4                      # KEEP — prevents floor collapse
```

`min_lr=1e-4` alone addresses the original problem: after a curriculum shift causes repeated
LR reductions, the floor at 1e-4 (vs 1e-6) keeps the policy learning at ~10× below the
productive range rather than effectively frozen. This is 100× higher than the default floor
but doesn't affect the scheduler's normal oscillation behavior during phases 1–3.

### Lessons Learned
1. **The KL-adaptive scheduler's oscillation is a feature, not a bug.** It creates natural
   annealing that balances exploration and exploitation. Widening the dead zone eliminated
   this beneficial mechanism.
2. **LR ceiling matters as much as LR floor.** The `max_lr` default of 0.01 is dangerous
   with permissive thresholds. Always set `max_lr` explicitly.
3. **σ equilibrium depends on LR.** Higher sustained LR → entropy gradient dominates →
   σ equilibrium shifts upward → policy too stochastic. The relationship is not linear —
   going from LR=0.001 to LR=0.003 pushed σ from 1.21 to 1.66, a qualitative regime change.
4. **Fix the floor, not the thermostat.** The original LR collapse was a floor problem
   (1e-6 too low). Changing the thermostat (kl_threshold) had cascading side effects.
