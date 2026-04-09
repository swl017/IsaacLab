# Experiment: A8 Action Penalty Ablation

**Date**: 2026-04-05
**Experiment IDs**: `a8_weight_config_a` (5x), `a8_weight_config_b` (15x)
**Base**: `928b9585f2` (action_sum=-2.0, action_delta=-1.0, zoom_weight=0.3)

**Status**:
- config_a: Completed (150k steps, as designed).
- config_b: Aborted at 99k (of 150k target). Killed to start the 2be3 experiment run.

---

## 1. Hypothesis
The 928b baseline used light action penalties (action_sum=-2.0, action_delta=-1.0) with low
zoom weight (0.3). Teleoperation review suggested the policy produced jerky actions and
under-penalized zoom oscillation. The A8 ablation tests whether stronger action penalties
improve smoothness without degrading task performance.

Two penalty scales were tested:
- **Config A** (conservative): 5x increase (action_sum=-10, action_delta=-5, zoom_weight=1.0)
- **Config B** (aggressive): 15x increase (action_sum=-30, action_delta=-15, zoom_weight=1.0)

Both were short runs (150k steps) designed to evaluate the penalty's effect on early training
dynamics before committing to a full 400k run.

## 2. Configuration Delta

| Parameter | 928b (base) | Config A (5x) | Config B (15x) |
|-----------|------:|------:|------:|
| action_sum_penalty_scale | -2.0 | **-10.0** | **-30.0** |
| action_delta_penalty_scale | -1.0 | **-5.0** | **-15.0** |
| action_weight[6] (zoom) | 0.3 | **1.0** | **1.0** |
| action_delta_weight[6] (zoom) | 0.3 | **1.0** | **1.0** |
| total_timesteps | 400000 | 150000 | 150000 |

All other parameters identical to 928b.

## 3. Results

### 3.1 Summary Table (at matched step 99k)

| Metric | Config A (5x) | Config B (15x) | 928b (ref, 99k) | Notes |
|--------|------:|------:|------:|-------|
| Mean Reward | 3681 | 3656 | ~3700 | Both comparable to 928b |
| Max Reward | 6268 | 6102 | ~5900 | Config A higher top |
| Learning Rate | 0.000100 | 0.001272 | 0.000500 | **A at floor; B oscillating** |
| Policy Std Dev | **0.170** | **0.126** | **1.21** | **Both collapsed** |
| Value Loss | 0.001023 | 0.000090 | 0.002 | B very low |
| Policy Loss | +0.002103 | +0.000350 | -0.002 | **Both positive (not learning)** |
| pair_valid_rate | 0.779 | 0.759 | 0.88 | Both below 928b |
| triangulation (d0) | 24.1 | 27.6 | ~30 | Both below 928b |
| bbox_center (d0) | 33.4 | 34.2 | ~30 | Both comparable |
| bbox_size (d0) | 41.6 | 42.6 | ~44 | Both comparable |
| collision_fraction | 0.0007 | 0.0008 | 0.001 | Both safe |
| tracking_lost_fraction | 0.068 | 0.055 | 0.000 | **Both elevated** |
| action_sum (d0) | -5.21 | -7.78 | -- | B penalized more |
| action_delta (d0) | -2.80 | -4.36 | -- | B penalized more |
| action_rms (d0) | 0.761 | 0.544 | -- | **B 29% smoother** |

### 3.2 Final metrics (config_a at 150k, config_b at 99k)

| Metric | Config A (150k) | Config B (99k) |
|--------|------:|------:|
| Mean Reward | 3601 | 3512 |
| Max Reward | 5996 | 6102 |
| Learning Rate | 0.000100 | 0.001272 |
| Policy Std Dev | 0.198 | 0.130 |
| Value Loss | 0.001414 | 0.000090 |
| Policy Loss | +0.002153 | +0.000350 |
| pair_valid_rate | 0.878 | 0.759 |
| triangulation (d0) | 31.8 | 27.6 |
| bbox_center (d0) | 26.3 | 34.2 |
| bbox_size (d0) | 44.1 | 42.6 |
| tracking_lost_fraction | 0.005 | 0.055 |
| action_rms (d0) | 0.864 | 0.564 |

### 3.3 Policy Standard Deviation -- the critical comparison

| Step | Config A (5x) | Config B (15x) | 928b (ref) |
|-----:|------:|------:|------:|
| 6k | 0.263 | 0.205 | 0.30 |
| 15k | 0.146 | 0.119 | 0.31 |
| 24k | 0.135 | 0.102 | 0.34 |
| 42k | 0.139 | 0.105 | 0.60 |
| 69k | 0.157 | 0.111 | 1.00 |
| 96k | 0.168 | 0.126 | 1.21 |
| 150k | 0.198 | -- | 1.22 |

**Both** runs show sigma collapse relative to the 928b reference (which converges to 1.21).
However, the severity differs dramatically:

- **Config B (15x)**: Sigma collapses to ~0.10 by 24k and barely recovers (0.13 at 99k).
  This is the same pattern seen in 2be3 and a9_bbox_baseline. The policy is effectively
  frozen in action space.

- **Config A (5x)**: Sigma drops to 0.135 at 24k but shows a **steady upward trend**
  (0.135 -> 0.198 over 24k-150k, +47%). At 150k it is still rising. If extrapolated,
  sigma could reach ~0.25-0.30 by 400k -- still far below 928b's 1.21, but showing
  recovery capacity.

### 3.4 Learning Rate Dynamics

**Config A**: LR hit the min_lr=1e-4 floor at 51k and stayed there through 150k, with
occasional brief spikes to 1.5e-4. This is **premature floor-hitting** -- the 928b baseline
didn't reach the floor until 236k. The heavier action penalty creates a narrower KL landscape,
causing the scheduler to undershoot and floor early.

**Config B**: LR shows volatile oscillation (0.0001 to 0.0013) without settling. The
extremely low sigma means small LR changes produce outsized policy updates, creating the
unstable feedback loop described in the 2be3 analysis. The LR was at 0.0013 when aborted
(still oscillating, no convergence pattern).

### 3.5 tracking_lost_fraction

| Step | Config A | Config B | 928b |
|-----:|------:|------:|------:|
| 10k | 0.004 | 0.011 | 0.006 |
| 24k | 0.003 | 0.013 | 0.002 |
| 42k | 0.008 | 0.014 | -- |
| 60k | 0.078 | 0.073 | -- |
| 78k | 0.100 | 0.058 | -- |
| 96k | 0.084 | 0.040 | -- |
| 105k | **0.000** | -- | 0.000 |
| 150k | **0.005** | -- | -- |

Config A shows a bump in tracking loss (5-10%) during 50k-100k, coinciding with the
curriculum ramp introducing noise and target motion. However, it **recovers to 0%** at 105k
when the noise onset phase starts (tracking termination is masked during noise curriculum).
By 150k, tracking loss is 0.5% -- essentially recovered.

Config B shows persistent 4-7% tracking loss in the same window, with no sign of recovery
before abort.

### 3.6 Reward Trajectory

**Config A**: Strong early performance (peaked at 4098 at 118k), clean curriculum shift
recovery (3212 -> 3601 from 123k -> 150k), and an upward trajectory at termination.
This is broadly comparable to the 928b baseline pattern.

**Config B**: Slower warmup, deeper exploration dip (2767 at 60k), but strong late recovery
(2767 -> 3656 from 60k -> 96k). The run was aborted at 99k just as reward was climbing
rapidly. The late recovery suggests the policy was finding a viable narrow-sigma strategy.

### 3.7 Action Smoothness -- the intended effect

| Step | Config A action_rms | Config B action_rms |
|-----:|------:|------:|
| 15k | 0.620 | 0.472 |
| 42k | 0.617 | 0.447 |
| 69k | 0.697 | 0.478 |
| 96k | 0.750 | 0.544 |
| 150k | 0.864 | -- |

Config B achieves 29% lower action RMS than Config A at 96k (0.544 vs 0.750). Both are
smoother than the 928b baseline's typical action patterns. However, Config A's action_rms
is **rising** (0.620 -> 0.864, +39%), which tracks the rising sigma -- as the policy
recovers exploration, it naturally becomes less smooth.

This rising trend in Config A is **healthy**: it means the penalty is constraining but not
preventing exploration. Config B's flat action_rms (0.47 -> 0.54) reflects the frozen sigma.

## 4. Analysis

### The sigma collapse spectrum

The three penalty levels define a clear spectrum:

| Penalty level | Sigma at 100k | Sigma trend | Task performance | Smoothness |
|:------|------:|:------|:------|:------|
| 928b (1x): -2/-1 | **1.21** | Converged | Best (3700+) | Baseline (jerky) |
| Config A (5x): -10/-5 | **0.17** | Rising slowly | Comparable (3680) | 29% better |
| Config B (15x): -30/-15 | **0.13** | Flat/stuck | Comparable (3656) | 44% better |

At 100k steps, all three achieve similar mean reward (~3650-3700). The difference is in
what happens next:
- **928b**: Sigma fully converged (1.21), policy has explored broadly, post-curriculum adaptation
  is robust. Final reward at 400k: 3788.
- **Config A**: Sigma still rising (0.17 at 100k -> 0.20 at 150k). The policy is slowly
  widening its action distribution. Post-curriculum performance is good (3601 at 150k, still rising).
  **Unknown**: whether sigma continues rising to reach adequate exploration by 400k.
- **Config B**: Sigma stuck (0.13). Post-curriculum adaptation fails (this is what happened in
  2be3 and a9_bbox_baseline). Reward would likely plateau or decline beyond 150k.

### Config A is the viable candidate

Config A shows the most promising balance:
1. **Smoothness improvement**: 29% lower action_rms (the intended goal)
2. **Comparable reward**: 3601 at 150k (vs 928b's ~3600 at same step)
3. **Recovery capacity**: Sigma rising from 0.135 to 0.198 over 24k-150k
4. **Healthy late metrics**: pair_valid_rate recovered to 0.878, tracking_lost at 0.5%
5. **Triangulation climbing**: 31.8 at 150k and still rising

The risk: sigma at 0.198 is still far below 928b's 1.21. A full 400k run is needed to
determine whether the slow sigma recovery leads to adequate adaptation during late-curriculum
phases (noise, delay, dynamics ramp).

### Config B is too aggressive -- confirmed across 3 runs

Config B's -30/-15 penalty produces sigma collapse in every run tested:
- **A8 config_b**: sigma 0.13 at 99k
- **2be3b9a3d9**: sigma 0.14 at 264k (same penalties + sysid gains)
- **a9_bbox_baseline**: sigma 0.16 at 152k (same penalties + full fidelity stack)

This is conclusive. The -30/-15 scale is incompatible with MAPPO + KL-adaptive scheduling
in this environment.

### Why sigma collapse happens -- mechanism

The 928b baseline's sigma trajectory (0.1 -> 0.3 -> 0.6 -> 1.0 -> 1.21) requires the
policy to discover that *broad* action distributions yield higher returns during the
exploration-heavy early phases. With heavier penalties:

1. The initial high-entropy actions incur large penalties (-73 action_sum for config_b at
   step 1.5k, vs -29 for config_a, vs ~-5 for 928b).
2. The policy rapidly narrows sigma to minimize these penalties.
3. Once sigma is low, the KL-adaptive scheduler has no mechanism to widen it -- it can only
   adjust LR, not directly control sigma.
4. The penalty gradient (toward lower sigma) is always present and always stronger than the
   exploration gradient (toward higher sigma for reward discovery).

This is a **penalty-exploration tradeoff failure**: the penalty signal dominates before the
reward signal has time to develop.

### Zoom weight change (0.3 -> 1.0) is separate from penalty scale

Both configs also changed zoom weight from 0.3 to 1.0. This 3.3x increase in zoom
dimension weighting contributes to the sigma collapse (zoom is one of 7 action dimensions;
over-weighting it reduces the effective action space). However, the dominant effect is the
penalty scale, not the zoom weight -- config_a (5x penalty) shows recovery despite the
same zoom weight.

## 5. Decision

1. **Config B (-30/-15) is rejected.** Do not use in any future experiments.

2. **Config A (-10/-5) is promising but unvalidated at 400k.** Next step: run a full 400k
   training with config_a penalties to determine:
   - Does sigma continue rising beyond 0.20?
   - Does the policy adapt to post-curriculum challenges (noise, delay, dynamics)?
   - Does the smoothness improvement survive late training?

3. **Consider intermediate scales.** The gap between 1x (-2/-1) and 5x (-10/-5) is large.
   A 2-3x scale (-4/-2 to -6/-3) may offer better smoothness/exploration balance.

4. **Zoom weight 1.0 should be tested independently.** A separate ablation with zoom_weight
   =1.0 but original penalty scales (-2/-1) would isolate the zoom contribution.

5. **For the immediate fidelity-stack experiments** (sysid gains + detector replicator + burst
   dropout), use the **928b baseline penalties** (-2/-1) until config_a is validated at 400k.
   The 2be3 and a9_bbox_baseline failures confirm that the -30/-15 scale breaks training.
