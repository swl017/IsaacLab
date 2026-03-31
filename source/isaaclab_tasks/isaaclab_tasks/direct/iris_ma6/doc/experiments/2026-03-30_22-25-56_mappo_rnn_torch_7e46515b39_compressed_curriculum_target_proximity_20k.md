# Experiment: 2026-03-30_22-25-56_mappo_rnn_torch_7e46515b39_compressed_curriculum_target_proximity_20k

**Commit**: `7e46515b39`
**Date**: 2026-03-30
**Experiment ID**: `a1_with_aoi`
**Base**: `a07987a412`<2026-03-30_00-31-25_mappo_rnn_torch_a07987a412_compressed_curriculum_target_proximity> (bug-fixed: `current_step` now uses actual training step)

Training in process (224k/200k — extended).

---

## 1. Hypothesis
Compressed curriculum with 20k offset based on accidental no-curriculum ablation findings:
- Policy converges on step-0 difficulty by 16k (entropy loss turns positive)
- Curriculum ramps should start at 20k to avoid wasting compute
- 20k per phase to prevent value function overfitting to any single regime
- Target proximity penalty added

## 2. Configuration Delta
Compressed curriculum (180k total, 20k offset):
```
Step:    0k   20k   40k   60k   80k  100k  120k  140k  160k  180k
Warmup:  [free]
AgentVel:      [──ramp──]
Safety:        [──ramp──]
Tracking:      [────────ramp────────]
Target:              [──────────ramp──────────]
Coord:                        [────ramp────]
Noise:                                      [──ramp──]
FixDelay:                                         [──ramp──]
RndDelay:                                               [──ramp──]
Dropout:                                                      [ramp]
Dynamics:                                                       [ramp]
```
- Bug fix: `current_step` from actual training step (was frozen at 0 in a07987a412)
- Target proximity penalty added
- `all_end_step`: 180k, `timesteps`: 200k

## 3. Results (224k, training in progress)

### 3.1 Training Curves

#### Phase 0: Warm-up (0-20k)
**Metric: `pair_valid_rate`**
- 0.69 → 0.97 by 20k. Identical to no-curriculum ablation — confirms 20k free phase works.

**Metric: `drone_0_bbox_center`**
- 19 → 53 by 20k. Same saturation as ablation.

**Metric: `Policy std`**
- Drops from 0.56 → 0.34 at 20k, then reverses. The entropy convergence signal at 16-20k.

#### Phase 1: Velocity + Safety + Tracking (20-60k)
**Metric: `pair_valid_rate`**
- Stays above 0.94 through 40k. Gradual decline to 0.87 at 60k.
- **No cliff.** Smooth transition from warm-up to curriculum.

**Metric: `drone_0_bbox_center`**
- 53 → 48 at 60k. Gentle decline as difficulty ramps.

**Metric: `CBF penalty`**
- Ramps from 0 at 20k to -0.07 at 40k. Safety learning is healthy.

#### Phase 2: Target Motion + Coordination (40-100k)
**Metric: `pair_valid_rate`**
- Dips to 0.77 at 80k (target motion at full speed), recovers to 0.93 at 104k
  as coordination reward drives better geometry.

**Metric: `drone_0_triangulation`**
- Ramps from 0 at 60k to **47 at 120k** — comparable to 4f7d9d0c34's best (50).
- Fast ramp: 0 → 44 in 40k steps.

**Metric: `Reward / Total reward`**
- Climbs from 2781 at 60k to **4269 at 120k**. Matches 4f7d9d0c34's peak (4277).

#### Phase 3: Noise + Delay (100-180k)
**Metric: `pair_valid_rate`**
- Noise onset (100k): dips from 0.83 to 0.89 at 124k — wait, it actually stays high.
- Delay onset (120k): dip to 0.89 at 124k, recovers to 0.96 by 140k.
- **pair_valid_rate above 0.90 from 104k through 180k!** Best sustained performance.

**Metric: `drone_0_bbox_center`**
- Dips from 38 to 25 at 124k (delay shock), recovers to 31 by 132k.
- Gradual decline to 28 at 180k.

**Metric: `drone_0_triangulation`**
- Dip to 33 at 124k, recovers to 46 at 168k. Stabilizes at 45.

**Metric: `tracking_lost_fraction`**
- Drops to 0% at 104k (noise starts, defeating truncation — same flaw as before).
- Comes back briefly at 144k-200k (0.1-2%) during delay phases.

#### Phase 4: Post-Curriculum (180k+)
**Metric: `pair_valid_rate`**
- **Collapse at 196k**: 0.91 → 0.63. Recovers to 0.82 at 224k.
- This is ~16k steps after the dynamics ramp finishes at 180k.

**Metric: `drone_0_bbox_center`**
- Collapse from 28 → 7.6 at 200k. Recovering to 21 at 224k.

**Metric: `drone_0_triangulation`**
- Dips from 45 → 28 at 196k. Recovering to 40 at 224k.

**Metric: `Reward / Total reward`**
- Dips from 3932 → 2509 at 200k. Recovering to 3408 at 224k.

**Metric: `Value loss`**
- Spikes to 0.011 at 200k (highest ever). Confirms value function shock.

**Metric: `Policy std`**
- Stable at 1.21 throughout 96k-224k. No runaway growth.

### 3.2 Behavior Observations
Not tested yet.

## 4. Analysis
- **The compressed curriculum works.** The policy navigates noise AND delay introduction
  (100-180k) while maintaining pair_valid_rate above 0.90 — something no previous
  experiment achieved. The 20k-per-phase schedule kept the value function plastic.
- **The 20k warm-up offset is validated.** 0-20k is identical to the no-curriculum ablation
  (pair_valid_rate 0.97, bbox_center 53), confirming the policy converges before the
  curriculum starts. No wasted curriculum progression.
- **Curriculum phases complete faster.** The policy reaches full coordination (triangulation 47)
  by 120k, vs 140k in 4f7d9d0c34. The tighter schedule forces faster adaptation.
- **The 196k collapse is surprising.** It happens AFTER the curriculum ends (180k).
  dynamics_start=180k, dynamics_end=200k, so at 196k the dynamics randomization is at 80%.
  The value loss spike to 0.011 confirms value function shock from mass/inertia variation.
  But the policy **is recovering** (0.63 → 0.82 at 224k), unlike 395280af3d where the
  collapse at 142k was permanent.
- **The recovery is the key difference from previous experiments.** The compressed curriculum
  seems to produce a more resilient policy — it gets shocked by dynamics but adapts within
  ~28k steps. The value function may be more plastic from the rapid curriculum changes.
- **Tracking-lost truncation still drops to 0% under noise** (104k onwards). Same flaw.
  The debounced reacquire (5 steps) is still not enough.

## 5. Decision / Todo
- Continue training past 224k to confirm recovery completes.
- The dynamics phase (180-200k) is the remaining bottleneck. Options:
  - Start dynamics earlier (overlap with delay ramp) so it's not a fresh shock.
  - Extend dynamics ramp to 40k steps (180-220k) instead of 20k.
  - Or accept that a transient 20k dip followed by recovery is tolerable.
- Increase tracking_reacquire_steps to 25+ for noise robustness.
- If recovery stabilizes above 0.85 pair_valid_rate, this becomes the definitive baseline.

## 6. Post-Mortem: LR Collapse Root Cause & Fix

### Diagnosis
The KL-adaptive LR scheduler (`KLAdaptiveLR`) drove the learning rate from 0.0018 (peak at 120k)
to 0.000001 (floor at 176k–200k), effectively freezing the policy for the final 100k steps.
The default `min_lr=1e-6` allowed this.

**Mechanism**: Curriculum shift at ~124k caused a distribution shift → high KL between consecutive
updates → scheduler divided LR by 1.5 repeatedly → LR hit 1e-6 floor → policy frozen →
value function diverged (loss 0.001→0.011) → noisy advantages → any attempted update produced
high KL → LR stayed at floor. A vicious cycle.

**Evidence from logs (drone_0)**:
| Step | LR | Value Loss | Mean Reward |
|-----:|------:|------:|------:|
| 120k | 0.00183 | 0.00124 | 4269 (peak) |
| 144k | 0.00016 | 0.00470 | 3862 |
| 176k | 0.000001 | 0.00372 | 3935 |
| 200k | 0.000001 | 0.01097 | 2509 (trough) |
| 224k | 0.000036 | 0.00543 | 3408 |

At LR=1e-6 with grad_norm_clip=0.3, max parameter change per step is ~3×10⁻⁷ — effectively zero.

### Fixes Applied
Two changes to `agents/skrl_mappo_rnn_cfg.yaml`:

**1. `min_lr: 1.0e-4`** (was 1e-6 default)

1e-4 is ~10× below the productive LR range (0.001–0.002) observed during phases 2–3,
low enough to still dampen high-KL instability but high enough to keep the policy adapting.
This is the safety net — prevents the vicious cycle even if the scheduler overreacts.

**2. `kl_threshold: 0.04`** (was 0.02)

The root cause: `kl_threshold=0.02` made the scheduler too sensitive for curriculum learning.
The dead zone where LR is unchanged was only KL ∈ [0.01, 0.04]. Curriculum phase transitions
naturally produce KL in the 0.04–0.08 range — moderate, not catastrophic — but the old threshold
treated this as "too high" and braked the LR repeatedly.

With `kl_threshold=0.04`, the new behavior:
| KL range | Action | When this fires |
|----------|--------|-----------------|
| KL < 0.02 | LR × 1.5 | Stable learning (large σ absorbs μ changes) |
| 0.02–0.08 | no change | Curriculum transitions, normal exploration |
| KL > 0.08 | LR ÷ 1.5 | Genuinely destabilizing updates |

The "increase" trigger (KL < 0.02) still covers the productive regime observed at 92k–120k.
The "reduce" trigger (KL > 0.08) only fires for truly dangerous updates — the PPO clip at
ε=0.2 is already enforcing trust-region below that. The wider dead zone absorbs curriculum
transitions without reacting.

Note: SKRL only supports `KLAdaptiveLR` via YAML config (no cosine annealing without code
changes). These two parameters are the available levers.

See `doc/research/ppo_mappo_algorithm.md` for the full mathematical analysis.
