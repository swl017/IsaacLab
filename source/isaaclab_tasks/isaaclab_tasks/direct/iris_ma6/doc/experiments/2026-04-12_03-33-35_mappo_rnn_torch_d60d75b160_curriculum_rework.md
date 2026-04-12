# Experiment: 2026-04-12_03-33-35_mappo_rnn_torch_d60d75b160_curriculum_rework

**Commit**: `d60d75b160`
**Date**: 2026-04-12
**Experiment ID**: `curriculum_rework`
**Base**: `91f5da3592` (failed two-track schedule) — reworked with framework-compliant
staging, background FN, entropy bump, σ floor

**Status**: Failed. Catastrophic entropy-driven σ explosion at step 4k. Policy never
learned anything — reward was negative from step 0 through 76k.

---

## 1. Hypothesis

Applied the full set of curriculum and hyperparameter fixes derived from the Chapter 3
framework evaluation:
- Free warmup 0-10k (bootstrap window)
- Framework-compliant stage ordering (tracking+safety → target+velocity → coordination → corruption)
- FP/FN background at constant 0.1 from 15k-80k
- Coordination staggered to 60-90k (shift in mid-late Phase 2, sequential after target)
- Agent velocity coupled with moving target (20-60k)
- Observation corruption pushed to 80-130k
- `entropy_loss_scale = 0.02` (2x baseline, counteract wide-action-space exploitation)
- `min_log_std = -1.0` (σ floor at 0.37, prevent σ collapse)
- `tracking_lost_timeout_s = 3.0` (more recovery time during FP/FN ramp)

## 2. Configuration Delta

From `91f5da3592`:
```yaml
# MAPPO hyperparameters
entropy_loss_scale: 0.02          # was 0.01
min_log_std: -1.0                 # was -5.0

# Env config
tracking_lost_timeout_s: 3.0      # was 2.0

# Curriculum — see schedule below
```

Full curriculum schedule (framework-classified):

| Phase | Start | End | Classification |
|-------|------:|----:|----------------|
| Free warmup | 0k | 10k | — |
| Tracking (expand) | 10k | 40k | Block A |
| Safety (shrink) | 10k | 40k | Block A |
| FP/FN background (flat 0.1) | 15k | 80k | Baseline mod |
| Target + Velocity (coupled complex) | 20k | 60k | Block B |
| Dynamics DR (physics expand) | 30k | 80k | Block B |
| Coordination (shift) | 60k | 90k | Block C |
| Noise, Delay, FP/FN ramp | 80k | 120k | Block D |
| Dropout | 90k | 130k | Block D |
| Burst | 100k | 130k | Block D |
| Post-curriculum | 130k | 320k | — |

## 3. Results (76k, aborted)

### 3.1 The Policy Never Bootstrapped

| Step | Reward | pair_valid | sigma | LR | episode_len | tracking_lost |
|-----:|-------:|-----------:|------:|---:|------------:|--------------:|
| 4k | **-775** | 0.062 | **1.09** | 0.010 | 150 | 0.968 |
| 8k | -805 | 0.043 | **1.62** | 0.010 | 138 | 1.000 |
| 12k | -788 | 0.050 | **1.78** | 0.010 | 140 | 1.000 |
| 16k | -783 | 0.056 | **1.78** | 0.010 | 143 | 1.000 |
| 32k | -752 | 0.011 | 1.78 | 0.010 | 128 | 1.000 |
| 60k | -741 | 0.003 | 1.78 | 0.000300 | 125 | 0.998 |
| 76k | -741 | 0.003 | 1.78 | 0.000300 | 124 | 0.954 |

**Reward was negative from step 0 and never recovered.** pair_valid peaked at 0.06
(vs predicted 0.91). Episodes were truncated to 124-150 steps from the start.
The policy was outputting random noise — never learned basic tracking.

### 3.2 Root Cause: `entropy_loss_scale = 0.02` + `min_log_std = -1.0` → σ Explosion

This is a completely new failure mode, distinct from all previous experiments:

| Step | σ | Entropy loss | What happened |
|-----:|---:|-------------:|---------------|
| 4k | **1.09** | **-0.038** | σ already above 928b's equilibrium value |
| 8k | **1.62** | **-0.060** | σ blowing past max_log_std ceiling check |
| 12k | **1.78** | **-0.079** | Saturated at `exp(max_log_std)` = `exp(0.576)` ≈ 1.78 |
| 16k | 1.78 | -0.131 | Pinned at ceiling, entropy loss still strongly negative |
| 28k | 1.78 | **-0.247** | Maximum entropy pressure ever recorded |

**σ hit the `max_log_std` ceiling by step 12k and stayed there for the entire run.**

The mechanism:
1. `entropy_loss_scale = 0.02` (2x baseline) pushes σ upward at 2x the normal rate
2. `min_log_std = -1.0` (σ_min = 0.37) prevents Phase 1 σ collapse — σ never drops
   below 0.37, so it starts growing immediately instead of first collapsing then rebounding
3. **Combined effect**: at step 0 (σ ≈ 0.6), entropy push is 2x strength AND there's no
   downward exploitation gradient to balance it (no curriculum active). σ rockets from
   0.6 → 1.09 → 1.62 → 1.78 (ceiling) in just 12k steps.
4. At σ = 1.78 in the wide action space: physical yaw noise = 1.78 × 90 deg/s = 160 deg/s,
   physical gimbal noise = 1.78 × 360 deg/s = 640 deg/s. **The actions are pure noise.**
   The drone is spinning randomly — impossible to learn anything.

**Why previous runs didn't have this problem**: they used `entropy_loss_scale = 0.01` and
`min_log_std = -5.0`. The lower entropy coefficient allowed exploitation to compress σ
during Phase 1 (0.6 → 0.17-0.30). The lower floor allowed σ to collapse. Then σ rebounded
slowly during Phase 2 with the curriculum providing exploitable reward signal.

**The two "σ protection" fixes were individually reasonable but together catastrophic.**
The hard floor prevented Phase 1 compression. The doubled entropy coefficient overwhelmed
the (absent, due to no curriculum) exploitation gradient. They created a system where the
only stable σ is the ceiling.

### 3.3 Comparison to Prediction

| Metric | Predicted | Actual | Match |
|--------|----------|--------|:-----:|
| pair_valid @ 10k | 0.91 | 0.05 | ✗ |
| pair_valid @ 20k | > 0.90 | 0.05 | ✗ |
| sigma @ 10k | 0.30 (collapsing) | 1.78 (ceiling) | ✗ |
| reward @ 10k | 2800 | -788 | ✗ |
| episode_len @ 10k | 497 | 140 | ✗ |
| LR trajectory | healthy oscillation | max → floor | ✗ |

**The prediction was completely wrong.** It assumed the entropy bump and σ floor would
provide "a safety margin" while the exploitation gradient from curriculum ramps would
prevent σ runaway. This assumption was incorrect — during the 0-10k free warmup window,
there is **no curriculum**, hence **no exploitation gradient**, hence **nothing to resist
the entropy push**. The prediction missed this critical interaction.

### 3.4 The Action Magnitude Evidence

| Step | action_sum (episode) | Interpretation |
|-----:|---------------------:|----------------|
| 4k | -16.4 | Actions are large (random at σ=1.09) |
| 12k | -18.7 | Even larger (σ=1.78, pinned at ceiling) |
| 76k | -17.5 | Still large, no change — policy never learned |

Compare to successful runs where action_sum at step 4k was -8 to -15 (lower because
σ was 0.3-0.6, not 1.09-1.78).

### 3.5 Late-Phase Recovery Attempt

Interesting: from step 56k onward, tracking_lost slowly improved from 1.000 to 0.954.
Policy loss went very slightly negative (-0.0002). The policy was beginning to learn
*something* — possibly that "output small actions" → "less negative reward." But with
σ pinned at 1.78, even a perfect mean action is drowned in noise.

### 3.6 Curriculum Schedule Comparison: d60d (new) vs 492f (old-style)

| Phase | 949d (928b-style) | 492f (compressed 2x) | d60d (framework) |
|-------|-------------------|---------------------|------------------|
| Free warmup | 0-20k | 0-10k | 0-10k |
| Tracking | 20-60k (40k) | 10-30k (20k) | 10-40k (30k) |
| Safety | 20-40k (20k) | 10-30k (20k) | 10-40k (30k) |
| Agent velocity | 20-40k (20k) | 10-30k (20k) | **20-60k (40k)** |
| Moving target | 40-80k (40k) | 20-40k (20k) | **20-60k (40k)** |
| Coordination | 60-100k (40k) | **30-50k (20k)** | **60-90k (30k)** |
| Dynamics DR | 180-200k (20k) | **90-110k (20k)** | **30-80k (50k)** |
| Task L2 | 40-80k | 20-40k | — (level 1 only) |
| Task L3 | 80-120k | 40-60k | — |
| Noise | 100-120k (20k) | 50-70k (20k) | **80-120k (40k)** |
| Fixed delay | 120-140k (20k) | 60-80k (20k) | removed |
| Random delay | 140-160k (20k) | 70-90k (20k) | **80-120k (40k)** |
| Dropout | 160-180k (20k) | 80-100k (20k) | **90-130k (40k)** |
| Burst | 200-220k (20k) | 100-120k (20k) | **100-130k (30k)** |
| FP/FN | 100-120k (20k) | **100-200k linear** | **15-80k bg 0.1, 80-120k ramp** |
| Curriculum ends | ~220k | ~200k | **130k** |
| Post-curriculum | 220-400k (180k) | 200-320k (120k) | **130-320k (190k)** |
| Total | 400k | 320k | 320k |

Key differences from 949d (928b-style, the original reference schedule):
1. **Much shorter total curriculum** — 130k vs 220k active phases
2. **Coordination timing preserved** — 60-90k matches 949d's 60-100k (the correct Phase 2 placement)
3. **Dynamics DR moved 150k earlier** — from 180-200k to 30-80k (physics-side, safe early per §A.1)
4. **Observation corruption compressed** — 949d spread noise→delay→dropout over 100-180k (80k); d60d packs it into 80-130k (50k) with longer per-ramp durations (40k vs 20k)
5. **FP/FN radically different** — 949d had a short 20k ramp co-located with noise; d60d has background exposure from 15k + 40k ramp from 80k

Key differences from 492f (compressed 2x):
1. **Coordination moved from 30-50k to 60-90k** — §A.4 rule 3 (no overlapping shifts with target)
2. **Velocity coupled with target 20-60k** — physical feasibility (agent can catch target)
3. **Dynamics DR moved from 90-110k to 30-80k** — physics-side expand runs with Phase 2, not during corruption block
4. **All observation corruption pushed 30k later** — enters after coord shift completes
5. **FP/FN uses background-then-ramp** — constant 0.1 from 15k builds bbox-empty handling early
6. **All ramps lengthened from 20k to 30-50k** — within Goldilocks range [2Ts, 5Ts]

### 3.7 Head-to-Head Training Metrics vs 492f

492f used the pre-rework curriculum (original 928b-style timing with compressed 2x phases)
and the same MAPPO hyperparameters as 928b (`entropy_loss_scale=0.01`, `min_log_std=-5.0`,
`kl_threshold=0.03`, `learning_epochs=6`, `min_lr=3e-4`). It is the best "old curriculum
+ new action space" run — it trained well through 100k before collapsing at noise onset.

#### Training trajectory comparison

| Step | d60d reward | 492f reward | d60d σ | 492f σ | d60d pair_valid | 492f pair_valid |
|-----:|------------:|------------:|-------:|-------:|----------------:|----------------:|
| 4k | **-775** | 1009 | **1.09** | 0.43 | **0.06** | 0.63 |
| 8k | -805 | **2793** | **1.62** | 0.27 | 0.04 | **0.91** |
| 12k | -788 | **3326** | **1.78** | 0.22 | 0.05 | **0.95** |
| 20k | -775 | **3529** | 1.78 | 0.18 | 0.05 | **0.95** |
| 40k | -754 | **3372** | 1.78 | 0.18 | 0.00 | **0.93** |
| 60k | -741 | **2912** | 1.78 | 0.35 | 0.00 | **0.89** |
| 76k | -741 | **3110** | 1.78 | 0.42 | 0.00 | **0.89** |

492f outperforms d60d on every metric at every step by 1-2 orders of magnitude. This
is not a marginal difference — d60d fundamentally never learned, while 492f trained
successfully for 100k steps before its noise-onset collapse.

#### The opposite σ failure modes

| | d60d (σ explosion) | 492f (σ deficit) |
|---|---|---|
| σ at step 4k | **1.09** (too high) | 0.43 (healthy) |
| σ at step 12k | **1.78** (ceiling) | **0.22** (too low) |
| σ at step 60k | 1.78 (stuck) | 0.35 (slowly growing) |
| σ at step 76k | 1.78 (stuck) | 0.42 (growing) |
| Physical noise (yaw) | 160 deg/s (pure chaos) | 38 deg/s (usable) |
| Entropy loss | -0.04 to **-0.25** (extreme push) | +0.001 to +0.011 (weak pull) |
| Root cause | Entropy 2x + floor → no resistance | Entropy 1x + no floor → over-exploitation |
| Failure mode | Never bootstrapped | Bootstrapped, then noise-onset collapse |

These two runs define the bookend failure modes of the entropy-σ system:
- **d60d** (entropy=0.02, floor=-1.0): entropy dominates → σ explosion → pure noise → no learning
- **492f** (entropy=0.01, floor=-5.0): exploitation dominates → σ collapse → narrow strategy → noise-fragile

The solution lies between them: **entropy=0.01 (as 492f) + floor=-1.0 (new)**.

#### Why the curriculum comparison is meaningless for this run

The d60d curriculum was never tested. The σ explosion occurred during the 0-10k free
warmup — before *any* curriculum stage activated. The framework-compliant staging
(staggered blocks, coupled velocity+target, coordination at 60k, FP/FN background)
had zero influence on this failure.

The curriculum improvements ARE tested indirectly through 492f comparison: 492f's
curriculum failed at noise onset (100k) because it stacked observation corruption too
early with insufficient σ. The d60d curriculum addresses this with later corruption
(80-130k), staggered shifts, and FP/FN background exposure. These improvements are
valid but require a working hyperparameter configuration to evaluate.

#### Entropy loss comparison — the smoking gun

| Step | d60d entropy loss | 492f entropy loss | Ratio |
|-----:|---------:|---------:|------:|
| 4k | -0.038 | -0.006 | **6.7x** |
| 8k | -0.060 | +0.001 | (opposite sign) |
| 16k | -0.131 | +0.009 | (opposite sign) |
| 28k | **-0.247** | +0.015 | (opposite sign) |
| 60k | -0.007 | +0.003 | 2.3x |
| 76k | -0.002 | +0.003 | (opposite sign) |

In 492f, entropy loss is **positive** from step 8k onward — the policy is actively
*reducing* entropy (shrinking σ). The 0.01 entropy coefficient is too weak to resist.

In d60d, entropy loss is **negative** throughout — the 0.02 coefficient overwhelms any
exploitation gradient. At step 28k, entropy loss reaches -0.247 (the largest entropy
loss magnitude in any iris_ma6 experiment), reflecting the maximum possible gradient
pressure toward σ expansion with no counterbalancing force.

The next run (entropy=0.01, floor=-1.0) should show entropy loss that is:
- Negative in Phase 1 (entropy pushes against floor) — unlike 492f's positive
- Small in magnitude (-0.005 to -0.010) — unlike d60d's -0.247
- Transitioning to near-zero as σ rebounds in Phase 2

## 4. Analysis

### The σ Protection Paradox

We arrived at `entropy_loss_scale=0.02` + `min_log_std=-1.0` by solving the *sigma
deficit problem* (§A.2 caveat: wide action space σ ceiling lower than 928b). The fixes
were:
- Higher entropy coefficient → more upward σ pressure → σ reaches higher equilibrium
- Higher σ floor → σ can't collapse below 0.37 during Phase 1

But these fixes assumed the exploitation gradient would always be present to balance
the entropy push. During the 0-10k free warmup, there is no curriculum → no difficulty
→ trivial task → exploitation gradient is very weak. The entropy coefficient dominates
*completely*, launching σ to ceiling before exploitation has a chance.

This is a **Phase 1 → Phase 2 interaction failure**: the fixes were designed for Phase 2
(where exploitation is present) but applied globally (including Phase 1 where exploitation
is absent).

### Comparison to Chapter 3 Framework

The framework (§A.2) identifies Phase 1 as "σ narrows as policy exploits obvious
advantages." Both our fixes fought this natural Phase 1 dynamic:
- `min_log_std=-1.0` prevents σ from narrowing below 0.37
- `entropy_loss_scale=0.02` doubles the opposing force

The framework didn't anticipate a scenario where both anti-collapse mechanisms
are active simultaneously in Phase 1 with no exploitation gradient to counterbalance.

### Which Fix Caused the Failure?

Both contributed, but **entropy_loss_scale = 0.02 is the primary cause**:

At `entropy_loss_scale = 0.01` (baseline), σ collapsed to 0.17-0.30 in Phase 1 across
all runs (928b, 36700, 949d, 492f, 5ce56, 91f5da). The exploitation gradient always
won during Phase 1 at the baseline coefficient.

At `entropy_loss_scale = 0.02`, the entropy push is 2x stronger. In Phase 1 (no
curriculum, weak exploitation), this tips the balance: entropy dominates and σ explodes.
The `min_log_std=-1.0` floor is secondary — it prevented recovery *if* σ had been
compressed first, but the primary driver was the entropy coefficient.

## 5. Decision

### Immediate fix
Revert `entropy_loss_scale` from 0.02 back to **0.01**. The 2x entropy coefficient
is incompatible with the free-warmup bootstrap window.

### Retain `min_log_std = -1.0` with caution
The σ floor did not cause this failure directly — it just prevented recovery from what
the entropy coefficient caused. At `entropy_loss_scale=0.01`, Phase 1 exploitation is
strong enough to compress σ to 0.17-0.30, which is still above the floor of 0.37?

Actually: **if min_log_std = -1.0 (σ_min = 0.37) and Phase 1 tries to compress σ to
0.17-0.30, the floor prevents compression. σ stays at 0.37 instead of 0.17.** This
means Phase 1 never "bottoms out" — the entropy gradient and exploitation gradient
are balanced at σ=0.37, not at σ=0.17. This changes Phase 2 dynamics:
- At σ=0.37 (floor), exploitation gradient is weaker than at σ=0.17 (natural bottom)
- Entropy coefficient (0.01) may dominate earlier, starting σ rebound sooner
- Net effect: σ might grow faster through Phase 2 than in baseline runs

**This might actually solve the σ deficit problem without the entropy bump.**

### Recommended next config
```yaml
entropy_loss_scale: 0.01    # REVERT to baseline
min_log_std: -1.0           # KEEP — provides σ floor without entropy explosion
```

The σ floor alone should give us:
- Phase 1 σ bottoms at 0.37 (vs 0.17-0.30 in all previous runs)
- Earlier Phase 2 rebound (because entropy coefficient starts winning sooner at σ=0.37)
- Higher Phase 2 σ ceiling (rebound starts from higher floor)
- Without the catastrophic free-warmup explosion (exploitation can still compress to 0.37)

### Curriculum changes: keep as implemented
The curriculum schedule is framework-compliant and was not tested in this run (the
policy never got past the warmup window). The curriculum changes are independent of
the hyperparameter failure and should be retained.

### Monitoring points for next run
1. **Step 4k**: σ should be ~0.37-0.50 (not exploding). pair_valid should be > 0.40.
2. **Step 10k**: σ should be ~0.37 (at floor, exploitation is balancing entropy).
   pair_valid should be > 0.85.
3. **Step 30k**: σ should start rebounding from 0.37 (Phase 2 onset, entropy wins).
4. **Step 60k**: σ should be > 0.50 (before coordination shift).
