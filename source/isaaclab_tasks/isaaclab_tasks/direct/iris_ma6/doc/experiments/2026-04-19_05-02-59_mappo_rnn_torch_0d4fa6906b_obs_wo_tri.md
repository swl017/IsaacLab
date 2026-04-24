# Experiment: 2026-04-19_05-02-59_a9_bbox_size_baseline_0d4fa6906b_seed42

**Commit**: `0d4fa6906b` ("Defer domain randomization")
**Date**: 2026-04-19
**Experiment ID**: `0d4fa6906b_obs_wo_tri`
**Base**: `3556bf75a4` with penalties reverted, triangulation dropped from observation, DR deferred

**Status**: Completed 400k. Peak 8.35 at 120k, survives 124k transition (dip to 0.37 → recovers to 0.74), slow late-phase degradation.

---

## 1. Hypothesis

The `3556bf75a4` run showed that light action penalties (-1/-2) caused a sigma-0.80
policy that collapsed catastrophically at the 124k delay transition. This run tests
three hypotheses simultaneously:

1. **Restore heavy penalties** (-30/-15) — confirmed in 3556 report as necessary
2. **Drop triangulation from observation** — the actor no longer sees the 6D triangulation
   tail (tri_pos + tri_std). Reward-side triangulation still computed for task rewards.
   Hypothesis: removing redundant info simplifies the learned state representation and
   may improve training stability.
3. **Defer domain randomization** — `DomainRandomizationCfg.enabled=False`,
   `MountOffsetRandomizationCfg.enabled=False`. Removes physics/camera/gimbal DR that
   ramps at 180k-200k. Hypothesis: DR may have been contributing to late-phase
   instability seen in bda2639a71.

## 2. Configuration Delta

### vs 3556bf75a4

```python
# Actions — reverted
action_sum_penalty_scale: -1.0 → -30.0
action_delta_penalty_scale: -2.0 → -15.0

# Triangulation — dropped from OBS only
enable_triangulation: True → False   # observation tail OFF
# (reward-side triangulation still computed in _compute_rewards())

# Domain randomization — disabled
domain_randomization.enabled: True → False
mount_offset.enabled: True → False
```

Observation dim drops: 31D ego + 16D inter-agent + **0D** triangulation (was 6D).

Env code `_compute_triangulation()` calls were decoupled from the `enable_triangulation`
flag — the flag now only gates the observation tail and the viz ellipsoid. Task rewards
at levels 1/2/3 still use the full triangulation machinery.

## 3. Results

### 3.1 Bootstrap & Phase 1 (0-80k) — Tightest Sigma Yet

| Step | Reward | pair_valid | track_lost | sigma | LR | ent_loss | ep_len |
|-----:|-------:|-----------:|-----------:|------:|----:|---------:|-------:|
| 4k | -3.81 | 0.449 | 0.526 | 0.452 | 0.008 | -0.008 | 361 |
| 20k | 6.77 | **0.919** | 0.005 | **0.137** | 0.0006 | +0.012 | 498 |
| 40k | 6.50 | 0.935 | 0.005 | 0.135 | 0.001 | +0.012 | 498 |
| 60k | 5.73 | 0.835 | 0.030 | 0.146 | 0.0004 | +0.014 | 494 |
| 80k | 7.15 | 0.891 | 0.012 | 0.139 | 0.001 | +0.011 | 496 |

**Sigma stabilized at 0.14 — tightest of any post-2be3 run with heavy penalties.** The
policy found an even more efficient strategy than bda2639a71 (0.15). Dropping
triangulation from obs didn't hurt; it may have helped (simpler state → cleaner gradient).

Entropy loss is **positive** through 80k (+0.011 to +0.014), meaning entropy is
decreasing — the policy is exploiting. This is the healthy signature of a policy
concentrating on a good strategy, not suffering from sigma stagnation.

### 3.2 Noise Ramp & 124k Transition (100-140k): Survived Cleanly

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 100k | 8.03 | 0.885 | 0.021 | 0.155 | 496 |
| 108k | 8.30 | 0.906 | 0.009 | 0.159 | 497 |
| 116k | 8.20 | 0.885 | 0.014 | 0.160 | 496 |
| 120k | **8.35** | **0.898** | 0.011 | 0.158 | 497 |
| 124k | 3.67 | 0.367 | 0.393 | 0.159 | 419 |
| 128k | 4.91 | 0.521 | 0.233 | 0.160 | 459 |
| 140k | 5.90 | **0.676** | 0.090 | 0.163 | 485 |

**Peak reward 8.35 at 120k — near 3556's peak (8.60) but without the catastrophe.**
The 124k dip was similar to bda2639a71 (pair_valid 0.90 → 0.37 → recovered to 0.68
at 140k). Delay ramp handled cleanly.

### 3.3 Delay System Ramp (140-200k): Stable

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 140k | 5.90 | 0.676 | 0.090 | 0.163 | 485 |
| 160k | 5.91 | 0.742 | 0.037 | 0.182 | 491 |
| 180k | 5.68 | 0.739 | 0.051 | 0.192 | 489 |
| 200k | 5.24 | 0.677 | 0.109 | 0.211 | 480 |

pair_valid climbed back to 0.74 at 160k-180k (better than 140k). Sigma gently rising
(0.16 → 0.21) from entropy bonus. Reward holds ~5.2-5.9.

### 3.4 Late-Phase Degradation (200-400k)

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 220k | 4.84 | 0.651 | 0.144 | 0.264 | 476 |
| 240k | 4.43 | 0.605 | 0.175 | 0.392 | 454 |
| 280k | 4.03 | 0.509 | 0.145 | 0.421 | 274 |
| 320k | 3.33 | 0.440 | 0.097 | 0.422 | 164 |
| 400k | 2.43 | **0.400** | 0.061 | 0.422 | **112** |

Gradual degradation through 200k-400k. Sigma climbs from 0.21 → 0.42 (stabilized by
280k). Episode length drops from 480 → 112 by 400k. Unlike 3556's terminal freeze at
sigma=1.26, this run's sigma caps at 0.42.

**Note**: burst dropout ramps 200k-220k (still active even with DR deferred). Dropout
full at 180k. The late-phase stress comes from these, not DR.

## 4. Analysis

### Triangulation-free observation is not worse (and may be better)

bda2639a71 (with triangulation in obs) had sigma 0.15 at 100k.
This run (no triangulation) has sigma 0.14 at 100k.

Very similar trajectories. The critic still has access to triangulation (it's in the
reward), and the actor has enough information from ego + inter-agent observations to
learn the task. Removing the 6D triangulation tail **simplified the actor's input
without losing performance**.

### DR deferral helped late phase modestly

Comparing bda2639a71 vs 0d4fa at matched steps past 200k:

| Step | bda2639a71 pair_valid | 0d4fa pair_valid | DR active? |
|-----:|----------------------:|-----------------:|:---|
| 200k | 0.603 | 0.677 | bda: starting ramp; 0d4f: off |
| 220k | 0.395 | 0.651 | bda: ramping; 0d4f: off |
| 240k | 0.369 | 0.605 | bda: full DR; 0d4f: off |

Deferring DR gives +0.25 pair_valid at 240k. The DR ramp (180-200k) was actively
degrading bda2639a71's policy. Without it, the policy holds better late-phase
performance.

But the ep_len still collapses by 400k (112 vs bda2639a71's ~344 at 240k was already
declining). So **DR deferral helps but doesn't fix the terminal degradation** — the
burst dropout ramp (200-220k) still causes problems.

### The 124k transition story so far

| Run | 120k pair_valid | 124k pair_valid | 128k pair_valid | 140k pair_valid |
|-----|---------------:|---------------:|---------------:|---------------:|
| 2695ffe1e3 (broken) | 0.752 | **0.157** | 0.146 | 0.144 |
| bda2639a71 (fixed) | 0.777 | **0.409** | 0.627 | 0.667 |
| 3556 (-1/-2, σ=0.80) | 0.854 | **0.088** | 0.069 | 0.082 |
| **0d4fa (this)** | **0.898** | **0.367** | **0.521** | **0.676** |

0d4fa's 124k dip is the deepest of the surviving runs (0.37), but it fully recovers to
0.68 by 140k — matching bda2639a71's trajectory.

## 5. Decision

**Three changes, three wins** (qualified):
1. **Action penalty revert to -30/-15**: confirmed essential. Sigma stays tight.
2. **Triangulation out of obs**: neutral-to-positive. Simplifies training without loss.
3. **DR deferral**: modest improvement in late phase. Doesn't solve terminal degradation.

**New best baseline**. Peak reward 8.35, survives 124k transition, holds through most
of the curriculum. The terminal ep_len collapse (485 → 112 by 400k) is the remaining
concern, likely driven by the burst dropout curriculum.

**Next experiments**:
1. Disable or defer burst dropout — test whether it's the terminal degradation cause
2. Tune action penalty split (current run used -30/-15; try -15/-30 to prioritize
   smoothness over magnitude)
3. Re-enable DR at a later step (e.g., 300k) to verify the deferral helps or if DR
   was just coincident with another issue
