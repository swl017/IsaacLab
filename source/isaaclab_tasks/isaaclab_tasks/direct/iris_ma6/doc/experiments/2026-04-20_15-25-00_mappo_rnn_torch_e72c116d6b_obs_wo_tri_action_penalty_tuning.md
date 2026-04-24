# Experiment: 2026-04-20_15-25-00_a9_bbox_size_baseline_e72c116d6b_seed42

**Commit**: `e72c116d6b` ("Action penalty tuning")
**Date**: 2026-04-20
**Experiment ID**: `e72c116d6b_obs_wo_tri_action_penalty_tuning`
**Base**: `0d4fa6906b` (no-triangulation-in-obs + DR deferred) with swapped action penalty emphasis

**Status**: Completed 400k. Very similar trajectory to 0d4fa through 140k, slightly worse at 160k onward, similar terminal state.

---

## 1. Hypothesis

`0d4fa6906b` used action penalties (-30, -15) — heavier magnitude penalty than smoothness.
This run swaps the emphasis: (-15, -30) — lighter magnitude penalty, heavier smoothness
penalty. Hypothesis: the policy should still be efficient (heavy smoothness keeps actions
consistent) but have more freedom on individual action magnitudes.

## 2. Configuration Delta

### vs 0d4fa6906b (the ONE change)

```python
action_sum_penalty_scale: -30.0 → -15.0     # 2x lighter (magnitude penalty reduced)
action_delta_penalty_scale: -15.0 → -30.0   # 2x heavier (smoothness penalty doubled)
```

Total penalty budget (−45) is unchanged; only the split is reweighted. All other
configuration identical to 0d4fa6906b (no triangulation in obs, DR deferred).

## 3. Results

### 3.1 Bootstrap & Phase 1 (0-80k): Slightly Better Start

| Step | Reward | pair_valid | track_lost | sigma | LR | ent_loss | ep_len |
|-----:|-------:|-----------:|-----------:|------:|----:|---------:|-------:|
| 4k | -1.51 | 0.482 | 0.441 | 0.377 | 0.008 | -0.006 | 383 |
| 20k | 6.75 | 0.906 | 0.006 | **0.118** | 0.001 | +0.015 | 498 |
| 40k | 6.41 | 0.895 | 0.028 | 0.122 | 0.001 | +0.013 | 493 |
| 60k | 6.23 | 0.847 | 0.014 | 0.119 | 0.0004 | +0.018 | 495 |
| 80k | 6.91 | 0.844 | 0.020 | 0.127 | 0.0001 | +0.015 | 493 |

**Sigma at 0.118 — the tightest yet.** Heavy smoothness penalty forces the policy into
an even more consistent action profile. Bootstrap slightly cleaner than 0d4fa (reward
-1.5 @ 4k vs -3.8; 6.75 @ 20k vs 6.77).

### 3.2 Noise & 124k Transition: Similar Recovery

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 100k | 8.05 | 0.871 | 0.015 | 0.133 | 495 |
| 108k | 8.27 | 0.888 | 0.010 | 0.142 | 496 |
| 120k | **8.35** | **0.891** | 0.012 | 0.142 | 496 |
| 124k | 4.22 | 0.404 | 0.333 | 0.144 | 437 |
| 128k | 5.44 | 0.587 | 0.119 | 0.148 | 480 |
| 140k | **6.11** | **0.734** | 0.056 | 0.158 | 488 |

Peak 8.35 at 120k — matches 0d4fa exactly. The 124k dip and recovery is nearly
identical (0.40 → 0.59 → 0.73 over 16k steps). At 140k, this run is slightly **better**
than 0d4fa (0.73 vs 0.68).

### 3.3 Post-Transition (160-200k): Worse Than 0d4fa

| Step | Reward 0d4fa | Reward e72c | pair_valid 0d4fa | pair_valid e72c |
|-----:|-------------:|------------:|----------------:|----------------:|
| 160k | 5.91 | **4.11** | 0.742 | **0.502** |
| 180k | 5.68 | 3.82 | 0.739 | 0.447 |
| 200k | 5.24 | 3.82 | 0.677 | 0.458 |

**This run degrades sooner.** At 160k (dropout at max, random delay complete),
pair_valid drops to 0.50 vs 0d4fa's 0.74. The swapped penalty emphasis hurt the
delay+dropout ramp, possibly because:
- Heavy smoothness penalty (-30) discourages the quick corrective actions needed to
  recover from dropped frames
- Lighter magnitude penalty (-15) allows larger baseline action magnitudes, which
  become harder to smooth when observations are intermittent

### 3.4 Late Phase (220-400k): Similar Terminal State

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 240k | 3.57 | 0.454 | 0.272 | 0.308 | 421 |
| 280k | 3.13 | 0.397 | 0.208 | 0.423 | 255 |
| 320k | 2.65 | 0.372 | 0.124 | 0.424 | 156 |
| 400k | **1.80** | **0.345** | 0.077 | 0.424 | **108** |

Sigma stabilizes at 0.42 (same as 0d4fa). Episode length terminally drops to 108 (vs
0d4fa's 112). Reward 1.80 at 400k (vs 0d4fa's 2.43). Terminal state is slightly worse
than 0d4fa but in the same degradation regime.

## 4. Analysis

### The (-15, -30) split is strictly worse than (-30, -15)

Head-to-head, matched-step comparison vs 0d4fa:

| Metric | 0d4fa (-30,-15) | e72c (-15,-30) | Winner |
|--------|----------------:|---------------:|:---|
| Sigma at 20k | 0.137 | 0.118 | e72c (tighter) |
| Peak reward (120k) | 8.35 | 8.35 | tie |
| pair_valid 140k | 0.676 | 0.734 | e72c |
| pair_valid 160k | **0.742** | 0.502 | **0d4fa** |
| pair_valid 200k | **0.677** | 0.458 | **0d4fa** |
| pair_valid 400k | 0.400 | 0.345 | 0d4fa |
| ep_len 400k | 112 | 108 | tie |

Through 140k, e72c looks slightly better (tighter sigma, quicker 124k recovery). But
from 160k onward, 0d4fa is consistently +0.17 pair_valid ahead. The swap hurt where it
mattered — handling the delay + dropout ramps.

### Why heavy smoothness hurts dropout recovery

Dropout means intermittent observation blackouts. The optimal response is:
- Hold last action briefly (smoothness helps)
- Then transition to a new action when observation returns (smoothness hurts)

A -30 smoothness penalty over-weights the "hold" half of this tradeoff. The policy
smooths through observation gaps but takes too long to adjust when observations return.
This is visible in the 160k metrics — delay+dropout active simultaneously at that point.

### The real decision: which error matters more at the current task level

0d4fa's config (-30, -15) prioritizes keeping actions small in absolute magnitude. In a
noisy-observation setting, this is safer — small wrong actions cost less than smoothed
wrong actions. e72c's config prioritizes keeping actions consistent step-to-step, which
assumes the underlying signal is smooth. In a dropout-corrupted setting, the signal is
not smooth, so the constraint works against the task.

## 5. Decision

**Swapped penalty emphasis (-15, -30) is worse than the (-30, -15) baseline.** Revert.

The degradation timing (~160k, when dropout completes) points to a specific interaction
between smoothness penalty and observation dropout. This suggests an interesting
experiment: **adaptive action penalties** that reduce during observation-corruption
phases. But the first-order priority is to identify and fix the burst dropout
degradation seen in both 0d4fa and this run (200-400k terminal state degradation is
identical between them).

**Next experiments**:
1. Revert to (-30, -15) split — use 0d4fa as current best baseline
2. Disable burst dropout entirely to confirm it's the terminal degradation driver
3. Tune burst dropout parameters (p_onset lower, p_recovery higher) so bursts are rarer
   and shorter
