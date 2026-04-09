# Experiment: 2026-04-09_04-52-12_a9_bbox_size_baseline_40aab3574f_seed42

**Commit**: `40aab3574f`
**Date**: 2026-04-09
**Experiment ID**: `a9_bbox_size_baseline` (from experiment_registry.py)
**Base**: `928b9585f2`<2026-04-01_13-19-18_mappo_rnn_torch_928b9585f2_min_lr_tuning>

**Status**: Ongoing -- 152k of 400k steps as of 2026-04-09 12:43. Latest checkpoint: agent_120000.pt.

---

## 1. Hypothesis
This is the **baseline** arm of the A9 ablation (ticket-014: bbox_size_reward). It tests whether
the default bbox_size_reward (scale=60, target=20% of image area) is necessary, or if the policy
can discover optimal zoom from the miss-rate signal + triangulation reward alone.

This run keeps bbox_size_reward_scale=60.0 (default). The paired experiment `a9_no_bbox_size`
(bbox_size_reward_scale=0.0) will be compared against this baseline.

The broader question: is the bbox_size heuristic helpful or harmful? It provides a strong
shaping signal early in training (zoom to keep targets at ~20% image fill), but may conflict
with the optimal zoom for triangulation (which wants sufficient angular separation, not
necessarily large targets). The A9 ablation will answer whether removing this signal improves
late-training triangulation quality.

This is the first experiment run using the full sim-to-real fidelity stack:
- Detector replicator (bbox noise, FP/FN)
- Burst dropout (Gilbert-Elliott)
- Continuous AoI
- Sysid-matched controller gains (from `2be3b9a3d9`)
- Level-3 aerodynamics

## 2. Configuration Delta
Default `IrisMA6TestEnvCfg` at commit `40aab3574f` -- no env_overrides (baseline).

Key features enabled vs 928b baseline:
```python
# New since 928b (enabled by default in 40aab3574f):
detector_replicator: enabled=True      # bbox noise + FP/FN
burst_dropout: enabled=True            # Gilbert-Elliott comm model
continuous_aoi: True                   # Sub-step AoI precision
aero_fidelity_level: 3                 # Full aerodynamics
sysid_matched_gains: True              # PX4-matched controller

# Unchanged from 928b:
bbox_size_reward_scale: 60.0           # THIS is the variable under test
min_lr: 1e-4                           # Validated in 928b
kl_threshold: 0.02                     # Validated in 928b
```

Note: Action penalty scales are at the `2be3b9a3d9` levels (action_sum=-30, action_delta=-15),
meaning this run inherits the aggressive penalty that caused exploration collapse in 2be3.

## 3. Results (152k steps, ongoing)

### 3.1 Current Metrics
| Metric | Value (152k) | 928b (152k) | 928b (400k) | Notes |
|--------|------:|------:|------:|-------|
| Mean Reward | 2397 | ~3600 | 3788 | **-33% vs 928b** |
| Max Reward | 5584 | ~5400 | 5473 | comparable max |
| Learning Rate | 0.000758 | 0.000506 | 0.000100 | healthy |
| Policy Std Dev | 0.159 | 1.224 | 1.210 | **collapsed (~0.16)** |
| Value Loss | 0.000011 | 0.002717 | 0.001011 | very low |
| Policy Loss | +0.002563 | -0.002348 | -0.002348 | **positive (not learning)** |
| pair_valid_rate | 0.679 | 0.930 | 0.861 | **-27%** |
| triangulation (d0) | 22.2 | ~35 | 44.4 | **-37%** |
| bbox_center (d0) | 19.5 | ~28 | 22.6 | -31% |
| bbox_size (d0) | 40.9 | ~44 | 44.4 | -8% |
| collision_fraction | 0.0007 | 0.0013 | 0.0008 | good |
| tracking_lost_fraction | **0.201** | 0.000 | 0.005 | **40x worse** |

### 3.2 Training Trajectory

#### Mean Reward -- early collapse, partial recovery
```
4k:   -2383 (negative -- worse init than 928b)
20k:   2727 (fast warmup)
36k:   2540
52k:   2224 (declining)
68k:   2019 (still declining)
84k:   1811 (bottom)
100k:  2339 (recovery starts)
116k:  2836 (local peak)
132k:  1991 (curriculum shift dip)
148k:  2573 (recovering)
152k:  2397
```
Pattern: reward declined from 20k to 84k (unusual -- 928b was rising in this range), then
slowly recovered. The 84k trough coincides with the tracking lost fraction peak (0.52 at 84k).

#### pair_valid_rate -- severe mid-training collapse
```
4k:   0.22 (init)
20k:  0.89 (learned tracking)
36k:  0.85
52k:  0.76
68k:  0.64  <- declining rapidly
84k:  0.59  <- bottom
100k: 0.66 (recovery)
116k: 0.71
132k: 0.54 (curriculum shift)
148k: 0.71
152k: 0.68
```
pair_valid_rate declined from 0.89 (20k) to 0.59 (84k) -- a 34% drop that 928b never
experienced (928b stayed above 0.83 throughout). Partial recovery to ~0.70, but still
well below the 928b baseline of 0.86.

#### tracking_lost_fraction -- alarmingly high
```
4k:   0.87 (warmup)
20k:  0.01
36k:  0.03
52k:  0.11 <- rising
68k:  0.34 <- critical
84k:  0.52 <- HALF OF EPISODES TERMINATED by tracking loss
100k: 0.21 (recovering)
116k: 0.23
132k: 0.35 (curriculum shift)
148k: 0.14
152k: 0.20
```
Between 52k-132k, 10-52% of episodes terminate due to tracking loss. The 928b baseline
had tracking_lost_fraction < 0.01 throughout this range (and 0.0 during the noise onset
window). This is the dominant failure mode.

#### Policy Std Dev -- collapsed (same pattern as 2be3)
```
4k:   0.580
20k:  0.177
36k:  0.161
52k:  0.155
68k:  0.153  <- converged low
84k:  0.174
100k: 0.167
116k: 0.158
152k: 0.159
```
Same exploration collapse as the 2be3 run: sigma converges to ~0.16 by 36k and stays there.
928b had sigma ~1.21 at these steps. This confirms the 15x action penalty is the root cause
(same penalty config, same collapse pattern).

#### triangulation -- growing but behind baseline
```
68k:  3.3 (onset)
84k:  10.2
100k: 20.6
116k: 25.0
132k: 18.4 (curriculum dip)
148k: 23.8
152k: 22.2
```
Triangulation reward is following a reasonable trajectory (onset at 68k, climbing).
Still behind 928b's pace (928b had 40.7 at 120k vs 25.0 here at 116k), but the
upward trend suggests the triangulation objective is being learned despite other issues.

#### bbox_size -- maintained well
```
20k:  46.3
84k:  31.1 (dip during tracking loss period)
116k: 39.2 (recovery)
148k: 42.6
152k: 40.9
```
bbox_size reward stayed relatively healthy, recovering to ~41 after the mid-training dip.
This is expected since bbox_size_reward_scale=60 (the test variable) provides strong shaping.
The paired a9_no_bbox_size experiment will show if this signal is load-bearing.

#### Learning Rate -- healthy oscillations
```
4k:   0.00976
20k:  0.00156
52k:  0.00142
84k:  0.00145
100k: 0.00026
116k: 0.00126
148k: 0.00056
152k: 0.00076
```
LR oscillation is more moderate than 2be3's wild swings (no 0.007 spikes). The min_lr=1e-4
floor has not been hit yet at 152k. Scheduler behavior looks healthy.

## 4. Analysis

### Same sigma collapse, different cause attribution
This run exhibits the same sigma collapse (~0.16 vs ~1.21) as the 2be3 run. Both share:
- action_sum_penalty_scale = -30.0
- action_delta_penalty_scale = -15.0

This **confirms the 15x action penalty is the primary cause** of exploration collapse, not the
sysid-matched gains or level-3 aerodynamics. The sigma collapse appears in both runs despite
being on different commits with different feature sets enabled.

### tracking_lost_fraction is the new bottleneck
The 928b baseline never had tracking loss issues (< 1% throughout). This run shows 10-52%
tracking loss between 52k-132k steps. Possible causes:
1. **Low sigma + stiff gains**: With sigma=0.16, the policy lacks exploratory corrections.
   Combined with stiffer gains (actions have larger state effects), small tracking errors
   compound faster than the policy can correct.
2. **Detector replicator noise**: FP/FN injections and bbox noise (new since 928b) degrade
   observation quality. The low-exploration policy cannot develop robust tracking behaviors.
3. **Burst dropout**: Consecutive frame drops (Gilbert-Elliott) create blackout windows where
   the policy receives no target information. Without sufficient exploration, it cannot
   learn recovery strategies.

### bbox_size reward appears load-bearing (preliminary)
Despite all the issues, bbox_size reward maintained at ~41 (vs 928b's ~44). The strong
shaping signal (scale=60) is keeping zoom behavior reasonable. The paired a9_no_bbox_size
will reveal whether removing this signal causes zoom collapse.

### Value loss anomaly
Value loss is extremely low (0.000011 vs 928b's 0.001011) -- 100x smaller. This may indicate
the value function is not being challenged (low-variance returns from a rigid policy), or
the value predictions are trivially accurate for a narrow policy. Either way, the value
function is not providing meaningful learning signal.

## 5. Decision / Outlook

### Prognosis: poor
At 152k steps (38% complete), this run shows the same sigma collapse as the aborted 2be3 run.
While reward has recovered from the 84k trough, it is still 33% below the 928b baseline at
the equivalent step, and tracking_lost_fraction remains 40x worse.

### Should this run continue?
**Yes, let it complete to 400k** -- despite poor metrics. Reasons:
1. This is the baseline arm of a paired ablation (A9). The paired `a9_no_bbox_size` needs
   this exact run as its control, regardless of absolute performance.
2. The triangulation trend is still rising (22.2 at 152k), so there may be late-training
   recovery.
3. The data is valuable for understanding how the full fidelity stack (detector replicator +
   burst dropout + continuous AoI) interacts with the action penalty scale.

### Next steps
1. **Revert action penalties** for the next baseline attempt: use 928b's original values
   (action_sum=-2.0, action_delta=-1.0) with the new fidelity stack.
2. **Run A8 ablation first**: Determine the correct action penalty scale independently before
   combining with the fidelity features.
3. **Compare a9_baseline vs a9_no_bbox_size** when both complete -- even with sigma collapse,
   the relative difference between the two arms will answer the ablation question.
4. **Sigma collapse root cause**: The action penalty scale (-30/-15) is confirmed as the cause
   across two independent runs. Future experiments MUST use calibrated penalty scales.
