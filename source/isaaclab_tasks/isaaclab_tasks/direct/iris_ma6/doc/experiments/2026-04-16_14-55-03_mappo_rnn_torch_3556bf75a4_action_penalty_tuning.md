# Experiment: 2026-04-16_14-55-03_a9_bbox_size_baseline_3556bf75a4_seed42

**Commit**: `3556bf75a4` ("Collaps fix is now the new baseline; Action penalty tuning")
**Date**: 2026-04-16
**Experiment ID**: `3556bf75a4_action_penalty_tuning`
**Base**: `bda2639a71` (min_steps=2 baseline) with drastically lightened action penalties

**Status**: Completed 400k. Peak performance at 120k (reward 8.60), catastrophic collapse at 124k, partial recovery fails, frozen at terminal state by 280k.

---

## 1. Hypothesis

`bda2639a71` showed that the ticket-029 dual-cache + `min_steps=2` combination produces a
healthy training curve (sigma=0.14 plateau, pair_valid ~0.80 through 120k). Given this
stability, action penalties (which were heavy at -30/-15 to match 2be3) could be relaxed
to allow more aggressive action usage — particularly relevant for future sim-to-real
transfer where the wider PX4-matched action space benefits from less penalization.

This run drops action penalties dramatically:
- `action_sum_penalty_scale: -30 → -1` (30x lighter)
- `action_delta_penalty_scale: -15 → -2` (7.5x lighter)

Expected: better use of action range, possibly higher reward, but larger sigma.

## 2. Configuration Delta

### vs bda2639a71 (the ONE change)

```python
action_sum_penalty_scale: -30.0 → -1.0     # 30x lighter
action_delta_penalty_scale: -15.0 → -2.0   # 7.5x lighter
```

Plus a minor comment change (`delayed/observed state` → `delayed state without noise`)
in the env code. No other functional changes to the running env.

(`iris_ma_env6_v1.py` observation redesign adds new files but they're not the running
env — the run used `iris_ma_env6_test.py`.)

## 3. Results

### 3.1 Bootstrap Through Curriculum (0-120k): Exceptional

| Step | Reward | pair_valid | track_lost | sigma | LR | ent_loss | ep_len |
|-----:|-------:|-----------:|-----------:|------:|----:|---------:|-------:|
| 4k | 4.15 | 0.719 | 0.177 | 0.555 | 0.003 | -0.010 | 461 |
| 20k | **6.49** | **0.969** | 0.001 | 0.341 | 0.0003 | 0.000 | 499 |
| 40k | 6.24 | 0.947 | 0.009 | 0.384 | 0.0001 | 0.000 | 498 |
| 60k | 5.76 | 0.853 | 0.039 | 0.453 | 0.0001 | -0.001 | 487 |
| 80k | 6.79 | 0.908 | 0.022 | 0.564 | 0.0006 | -0.005 | 495 |
| 100k | 8.12 | 0.799 | 0.051 | 0.700 | 0.0001 | -0.005 | 488 |
| 108k | **8.65** | 0.843 | 0.017 | 0.729 | 0.0001 | -0.006 | 494 |
| 120k | 8.60 | 0.854 | 0.029 | 0.802 | 0.0001 | -0.007 | 494 |

**Peak reward 8.65 at 108k — the highest of any run to date.** With light penalties, the
policy exploited the full action range. Sigma grew from 0.34 → 0.80 by 120k (vs
bda2639a71's 0.15 plateau) — the policy used more exploration because actions were
cheaper.

Noise ramp (100-120k) handled fine: pair_valid stayed 0.80-0.85. The fat sigma (0.70+)
seemed to absorb the noise onset.

### 3.2 The 124k Catastrophic Collapse

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 120k | 8.60 | 0.854 | 0.029 | 0.802 | 494 |
| 124k | **0.33** | **0.088** | **0.982** | 0.812 | **122** |
| 128k | 0.46 | 0.069 | 0.997 | 0.812 | 117 |

Sudden total collapse at the delay mode transition. Unlike bda2639a71 (which dipped to
pair_valid 0.41 at 124k and recovered to 0.63 by 128k), this run's pair_valid crashed
to **0.09** and got worse before any recovery.

The collapse is even more severe than the pre-fix 2695ffe1e3 run (which bottomed at
pair_valid 0.15). The difference: sigma was 0.80 here vs 0.40 in 2695. At high sigma,
the policy's actions are much noisier — so when the delay transition perturbs the
observation distribution, the combination of noisy actions + corrupted observations
produces an unrecoverable state.

### 3.3 Partial Recovery (140k-200k) Then Slow Bleed

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 140k | 2.07 | 0.082 | 0.664 | 0.823 | 298 |
| 160k | 4.95 | 0.486 | 0.307 | 0.870 | 439 |
| 180k | 4.58 | 0.446 | 0.312 | 0.943 | 438 |
| 200k | 4.09 | 0.370 | 0.336 | 1.042 | 413 |

pair_valid partially recovered (0.09 → 0.49) through the delay ramp phase, but never
matched the pre-collapse level. Sigma continued to climb (0.80 → 1.04) as the entropy
bonus dominated the weakened exploitation gradient.

### 3.4 Terminal Freeze (200k-400k)

| Step | Reward | pair_valid | sigma | ep_len |
|-----:|-------:|-----------:|------:|-------:|
| 240k | 2.73 | 0.207 | 1.256 | 283 |
| 280k | 2.21 | 0.217 | 1.257 | 145 |
| 320k | 1.62 | 0.200 | 1.257 | 106 |
| 400k | 1.20 | 0.186 | 1.257 | **93** |

Sigma maxed out at 1.257 (near the 2.01 ceiling from `max_log_std=0.7`). Episode length
dropped steadily to 93 — every episode truncates within 3.7 seconds. Policy permanently
frozen at the terminal state. Reward bleeds from 4.9 → 1.2 as the curriculum continues
ramping features (dropout, dynamics DR, burst dropout) the frozen policy cannot adapt to.

## 4. Analysis

### Light penalties create a fragile high-sigma policy

The policy achieved peak reward 8.65 — the best of any run — but at the cost of using a
sigma of 0.80 at 120k. Compare across runs at step 120k:

| Run | action_sum | action_delta | sigma | pair_valid | peak reward | survives 124k? |
|-----|-----------:|-------------:|------:|-----------:|------------:|:---|
| 2be3 | -30 | -15 | 0.12 | 0.89 | 7.72 | Yes (pair_valid stayed 0.90+) |
| bda2639a71 | -30 | -15 | 0.15 | 0.78 | 7.49 | Yes (dip to 0.41, recovered to 0.63) |
| **3556 (this)** | **-1** | **-2** | **0.80** | **0.85** | **8.60** | **No (crashed to 0.09)** |

Light penalties → more exploration → wider sigma → more performant mean policy but
more brittle distribution. When the 124k delay transition hits, the wide sigma means
many action samples are far from the deterministic mean — and those far samples
collide with the corrupted observations to produce catastrophic outcomes.

### Sigma is not the direct robustness predictor — its interaction with action cost is

Earlier experiments (949d, 492f, 025af4d) suggested higher sigma is more robust. This
run inverts that: sigma=0.80 was catastrophically fragile. The resolution: sigma alone
isn't the predictor — it's **whether the policy explores efficiently**. Heavy penalties
force the policy to concentrate mass on efficient actions; light penalties let it
spread mass over inefficient actions too. When observations corrupt, the "spread mass"
strategies fail hard because they lack consistent commitment to any strategy.

### Comparison to bda2639a71 late-phase degradation

bda2639a71 was degrading at 240k (pair_valid 0.37, sigma 0.44). This run's degradation
is **much worse** at the same step (pair_valid 0.21, sigma 1.26, ep_len 283 vs 344).
The light penalties don't just cause the 124k cliff — they also make the policy less
resilient to the late-phase disturbances (burst dropout 200-220k, dynamics DR ramping).

## 5. Decision

**Light action penalties are unsafe.** The 30x/7.5x reduction gave a 1% reward peak
boost (8.65 vs bda2639a71's 7.49) at the cost of total training failure at the delay
transition.

**The -30/-15 penalties from 2be3 are load-bearing.** They don't just penalize action
magnitude — they regulate the policy's action distribution width, which determines
robustness to observation corruption. Any relaxation of action penalties must be
tested carefully, likely with a simultaneous change to how the policy handles delay
transitions.

**Next experiment directions**:
1. Restore action penalties to -30/-15 (revert this run's only change)
2. Alternative: keep -1/-2 but disable/remove the delay mode transition entirely —
   use "random" mode with progress from step 0 (skip the "none" → "fixed" cliff)
3. Alternative: keep -1/-2 but add a gradual warmup phase that initializes the delay
   buffer before the curriculum starts
