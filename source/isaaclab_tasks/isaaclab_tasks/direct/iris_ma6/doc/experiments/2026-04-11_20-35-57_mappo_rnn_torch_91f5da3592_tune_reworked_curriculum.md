# Experiment: 2026-04-11_20-35-57_mappo_rnn_torch_91f5da3592_tune_reworked_curriculum

**Commit**: `91f5da3592`
**Date**: 2026-04-11
**Experiment ID**: `tune_reworked_curriculum`
**Base**: `5ce56e3ae8` (failed all-from-zero) — corrected with two-track curriculum

**Status**: Failed at ~52k. Same failure mode as 5ce56 (never learned base tracking),
but at a slower rate — the 30k delay on observation corruption was insufficient.

---

## 1. Hypothesis

The 5ce56 experiment proved that ramping observation corruption (noise/delay/dropout/
FP/FN) from step 0 destroys bootstrappability. The fix was a **two-track curriculum**:
- **Task difficulty** (velocity, safety, tracking, target, coordination, dynamics DR)
  starts at step 0 — keeps σ from collapsing
- **Observation corruption** (noise, delay, dropout, burst, FP/FN) starts at 30-40k —
  gives the policy a clean window to learn base bbox tracking before observations corrupt

Expected: by step 30k, pair_valid > 0.85 (matching 928b/36700 trajectories), then
gradual degradation as corruption ramps in.

## 2. Configuration Delta

From 5ce56 (curriculum changes only):

| Phase | 5ce56 (failed) | 91f5da (this run) |
|-------|---------------|-------------------|
| Velocity, Safety, Tracking | 0-40k | 0-40k (unchanged) |
| Target, Coordination | 0-60k | 0-60k (unchanged) |
| Dynamics DR | 0-100k | 0-100k (unchanged) |
| **Noise** | 0-100k | **30k-100k** |
| **Random delay** | 0-100k | **30k-100k** |
| **Dropout** | 0-100k | **30k-100k** |
| **Burst** | 0-100k | **40k-100k** |
| **FP/FN** | 0-100k | **40k-100k** |

MAPPO hyperparameters unchanged: `kl_threshold=0.03`, `learning_epochs=6`, `min_lr=3e-4`.

## 3. Results (52k, aborted)

### 3.1 Same Failure Pattern as 5ce56

| Step | Reward | pair_valid | bbox_center | bbox_size | episode_len | tracking_lost |
|-----:|-------:|-----------:|------------:|----------:|------------:|--------------:|
| 4k | 446 | 0.363 | 3.6 | 19.0 | 330 | 0.648 |
| 8k | **1314** | **0.621 (peak)** | 10.2 | 36.7 | 461 | 0.232 |
| 12k | 1143 | 0.506 | 9.5 | 32.8 | 439 | 0.303 |
| 16k | 948 | 0.402 | 7.4 | 27.7 | 402 | 0.418 |
| 20k | 564 | 0.268 | 4.8 | 19.6 | 331 | 0.644 |
| 24k | 376 | 0.203 | 3.7 | 15.5 | 300 | 0.713 |
| 32k | 226 | 0.126 | 2.0 | 9.3 | 234 | 0.837 |
| 40k | 265 | 0.126 | 2.1 | 9.3 | 236 | 0.846 |
| 52k | 112 | 0.090 | 1.0 | 5.3 | 179 | 0.949 |

**Same shape as 5ce56**: peak at step 8k (pair_valid 0.62 vs 5ce56's 0.57), then
monotonic decline. By step 32k, pair_valid is 0.13 — the policy has not learned to
track. The 30k clean window was not enough to establish the base skill.

### 3.2 Comparison: 91f5da vs 5ce56 vs Successful Runs

| Metric @ 8k | 928b | 36700 | 5ce56 | **91f5da** |
|-------------|-----:|------:|------:|-----------:|
| pair_valid | 0.91 | 0.91 | 0.57 | **0.62** |
| reward | 2792 | 2793 | 1002 | **1314** |
| bbox_center | 35.6 | 36.3 | 7.6 | **10.2** |
| episode_len | 497 | 497 | 442 | **461** |
| tracking_lost | 0.011 | 0.011 | 0.349 | **0.232** |

**91f5da is marginally better than 5ce56** but still drastically worse than the
successful runs. The pattern is identical: reward peaks at 8k then declines.
The policy is climbing the early gradient but stalling before reaching solid
tracking (~0.9 pair_valid).

### 3.3 Why the 30k Window Wasn't Enough

The hypothesis was that 30k clean steps would let the policy learn base tracking.
This is consistent with 928b/36700 reaching pair_valid 0.95 by step 12-16k.
**But 91f5da only reached 0.62 by step 8k and then declined.**

The decline starts at step 12k — well before the noise/delay/dropout start at 30k.
This means **observation corruption is not the only thing wrong**. Something else
in the 0-12k window is preventing the bootstrap from completing:

1. **Task difficulty ramps from 0**: At step 12k, velocity is at 30%, safety at 30%,
   tracking at 30%, target at 20%, coordination at 20%. In 928b's schedule, all of
   these were at 0% at step 12k (free warmup until 20k). This is a meaningfully
   different early environment.

2. **Dynamics DR ramps from 0**: At step 12k, dynamics randomization is at 12%.
   In 928b/36700, dynamics DR was 0% throughout the entire bootstrap window.
   Even though dynamics DR is filtered through the controller, it adds noise to
   the action→state mapping that the policy is trying to learn.

3. **Coordination reward ramps from 0**: At step 12k, coordination is at 20%. This
   adds a confusing reward signal early — the policy is being told "good triangulation
   geometry is rewarded" before it has even learned to point the camera at the target.

The 30k delay only addressed observation corruption. Task-side complications (DR,
coordination, target motion) were still present from step 0 and prevented the
bootstrap from completing within 12k steps.

### 3.4 Learning Dynamics — Healthy But Pointing the Wrong Way

| Metric | Behavior | Interpretation |
|--------|----------|----------------|
| LR | Stayed at max (0.01) for 36k steps | KL stayed below threshold throughout |
| Sigma | 0.41 → 0.53 → 0.50 | Healthy growth then plateau |
| Entropy loss | Negative throughout (-0.005 to -0.012) | Entropy bonus winning |
| Value loss | Rising from 0.002 → 0.020 | V_φ can't track changing returns |
| Policy loss | Positive throughout (+0.002 to +0.012) | Policy update doesn't help |

The MAPPO machinery is fine: LR healthy, sigma growing, entropy active. But the
**value function is diverging** (loss climbing 0.002 → 0.020) because the policy
can't establish a stable strategy for V_φ to predict. The reward signal is too
noisy (multiple weak gradients pulling in different directions) for V_φ to learn
a useful baseline.

This is the **bootstrap failure signature**: healthy learning machinery, but no
stable signal to learn from.

---

## 4. Analysis

### 4.1 Assessment of Chapter 3's Curriculum Design Framework

I read chapters 1-4 in detail. The framework is fundamentally sound, with the
shrink/expand/shift taxonomy and phase-matching rules correctly explaining most
of our experiments. Three observations and one addition:

**Strong agreement:**
- The collapse cascade math (§4.4) is correct and matches lr_collapse_analysis
- The 2x value-function-settling-time heuristic for ramp duration (§A.5) explains
  why 928b's 20-40k ramps work while 36700's 10-20k ramps were marginal
- The shrink-vs-expand-vs-shift taxonomy correctly predicts σ pressure direction

**Where the framework needs nuance:**

**Observation 1: Phase 2 σ is bounded by action space scale.**
The framework assumes σ rebounds to ~1.2 in Phase 2 (matching 928b). This is true
for narrow action spaces. With the wide action space (2-4x larger physical effects
per unit normalized action), Phase 2 σ caps at ~0.5-0.6 — sufficient for non-
corruption shifts but inadequate for observation corruption shifts. The framework
should explicitly note that **σ ceiling is action-scale-dependent** and that Phase
2 in a wide action space is qualitatively different from Phase 2 in a narrow one.

**Observation 2: §A.1 conflates physics-side and observation-side stages.**
The taxonomy lumps "domain randomization" (physics) and "observation noise"
(sensor) together as "expand" stages. They are not equivalent in their effect on
learning:
- **Physics DR** is filtered through the controller. Mass/inertia perturbations
  produce graceful step-response variations. The action→state mapping is preserved
  but slightly noisy. A randomly-initialized policy can still discover that "pitch
  forward → move forward."
- **Observation corruption** creates entirely new observation states (bbox=0)
  with no learned response. There's no "graceful degradation" — when the bbox
  zeros, the entire downstream computation breaks. A randomly-initialized policy
  has no way to bootstrap from this.

The 5ce56 failure proved this distinction matters enormously. Physics DR from
step 0 was harmless. Observation corruption from step 0 was fatal.

**Observation 3: Long ramps have a hidden cost.**
§A.5 recommends ramps "at least 2x value function settling time." This prevents
the value function from getting stuck behind. But it ignores a competing concern:
if the ramp is too slow, the policy never receives a strong enough gradient
signal to learn the new regime — it just drifts along. The 36700 run's 100k FP/FN
ramp gave V time to settle but also let the policy forget bbox-empty handling
between low-rate exposures. **Ramp speed is a Goldilocks problem**, not strictly
"longer is better."

**Adding the Bootstrap Principle:**

> **At step 0, at least one signal in the observation→reward chain must be
> perfectly reliable so a randomly-initialized policy can learn cause and effect.
> Curriculum stages that corrupt this bootstrap signal must wait until the
> bootstrap skill is established (typically 12-20k steps).**

This is the principle that 5ce56 violated. The Chapter 3 framework's "shrink/
expand/shift in compatible phases" is necessary but not sufficient — it doesn't
distinguish between perturbing an existing skill (fine in Phase 2) and corrupting
the signal needed to bootstrap a new skill (only viable after bootstrap is done).

### 4.2 Comparison to 36700751f6 Through the Updated Framework

Now apply the framework (with the bootstrap principle) to compare 91f5da against
the most successful prior run, 36700:

| Criterion | 36700 | 91f5da | Winner |
|-----------|-------|--------|--------|
| **Bootstrap window cleanliness** | Steps 0-10k all curriculum off | Steps 0-30k observations clean, but task-side ramps active | **36700** |
| **Phase 1 task difficulty** | Free warmup (no curriculum) → trivial task → σ collapses | Velocity/safety/tracking/target/coord/DynDR all ramping → too many gradients | **36700** wins on bootstrap, **91f5da** wins on σ floor |
| **Phase 2 transition smoothness** | Curriculum starts at 10k, observation corruption at 50-100k → clear separation | All task-side at 0, observation at 30-40k → muddled separation | **36700** |
| **Match to 2x V-settling rule** | 20k ramps (acceptable) | 40-100k ramps (overkill, may lose gradient) | Tied |
| **Stack count at any time** | Max 3 stages overlapping | Max 6 stages overlapping at step 30k | **36700** (cleaner) |
| **σ trajectory** | 0.43 → 0.55 (peaked) | 0.41 → 0.53 → 0.50 (collapsed) | **36700** |
| **pair_valid achieved** | 0.96 (peak at 16k) | 0.62 (peak at 8k) | **36700** |
| **Made it to noise onset?** | Yes (60k, with sigma 0.47) | No (collapsed before 30k) | **36700** |

**36700 is decisively better.** The "all-from-zero task difficulty" idea, which
came from the σ collapse analysis, was wrong. The framework correctly predicted
36700's structure: free warmup → narrow task ramps → observability later. The
attempt to stack everything from step 0 violated the **stack count rule (§A.4
rule 1)**: "no more than 2-3 stages active simultaneously."

### 4.3 What Went Wrong: Too Many Concurrent Stages at Bootstrap

At step 8k of 91f5da, the active stages were:
1. Agent velocity (20% — slight tilt requirement)
2. Safety/CBF (20% — slight collision penalty)
3. Tracking initial-state difficulty (20%)
4. Moving target (13% — slight motion)
5. Coordination reward (13% — slight triangulation pressure)
6. Dynamics DR (8% — slight mass/gain noise)
7. Task L2 (20% — slight GT-anchored reward shift)

**Seven simultaneous stages**, all at low percentages. Individually each is
small. Cumulatively, the gradient signal is fragmented across seven competing
objectives, none of which can establish a clean reward landscape for V_φ to
learn. This violates §A.4 rule 1 (max 2-3 concurrent stages) by 2x.

The 36700 schedule had at most 3 stages active at once during bootstrap (velocity
+ safety + tracking, all from 10k). The free warmup window (0-10k) had ZERO
stages — pure base learning. This satisfied the bootstrap principle.

### 4.4 The Core Lesson

The σ collapse problem (3461/949d/492f) and the bootstrap failure problem
(5ce56/91f5da) are the **opposite ends of a spectrum**:

- **Too little early difficulty** → trivial task → σ collapses → can't handle
  late-phase shifts
- **Too much early difficulty** → no clean bootstrap signal → policy never
  learns anything → permanent failure

The successful runs (928b, 36700) sit in the middle: a brief free warmup
(allowing bootstrap), then a few simple ramps (velocity + safety), then
progressively more stages as the policy gains capacity.

**The framework's §A.3 phase-matching rules combined with the bootstrap principle
give the correct prescription**: Phase 1 (0-15k) should have at most 1-2 simple
expand stages (velocity, safety) and zero observation corruption. This is what
36700 did.

## 5. Decision

**Abort and revert to 36700-style structure.** The "all from zero" experiments
(5ce56, 91f5da) have proven that aggressive curriculum start times destroy
bootstrappability, regardless of which specific stages are involved.

### Recommended next schedule (returning to 36700 structure with refinements)

```
Phase                    Start    End      Duration
─────────────────────────────────────────────────────
[Free warmup: 0-10k — no curriculum, pure bootstrap]
Agent velocity           10k      40k      30k
Safety                   10k      40k      30k
Tracking                 10k      40k      30k
Moving target            20k      60k      40k
Coordination             20k      60k      40k
Dynamics DR              30k      80k      50k    (after base skill established)
Task L2                  20k      40k      20k
Task L3                  40k      60k      20k
─── observation corruption ─────────────────────────
Noise                    50k      100k     50k
Random delay             50k      100k     50k
Dropout                  60k      110k     50k
Burst                    70k      110k     40k
FP/FN                    80k      120k     40k
─── all curriculum ends at 120k ────────────────────
Post-curriculum          120k     320k     200k
─────────────────────────────────────────────────────
```

Key differences from 91f5da:
- **Restore 10k free warmup** (bootstrap window with zero stages)
- **Stagger task difficulty ramps**: not all at step 0, max 3 concurrent
- **Push dynamics DR to 30k** (after base tracking)
- **Stagger observation corruption**: 50k, 50k, 60k, 70k, 80k start times
- **2x value-settling rule**: all ramps 30-50k (was 20k in 36700, 40-100k in 91f5da)

### What we learned (to update the framework)

1. **The Bootstrap Principle**: Add to §A.3. Phase 1 needs at least one perfectly
   clean signal. Stages that corrupt the bootstrap signal must wait.
2. **Stack count must be enforced at bootstrap**: §A.4 rule 1 (max 2-3 stages)
   applies *especially* during 0-15k. Never have more than 2 stages active
   during bootstrap.
3. **Observation corruption is qualitatively different from physics DR**: Update
   §A.1 to split "expand" into "physics-side expand" (safe early) and
   "observation-side expand" (only after bootstrap).
4. **Action space scale determines Phase 2 σ ceiling**: Add to §3.4 — the σ
   trajectory in a wide action space is not a 1:1 transfer from a narrow one.
   Phase 2 σ may cap at 0.5-0.6, requiring the curriculum to be designed so
   late-phase shifts are tolerable at this lower σ.
