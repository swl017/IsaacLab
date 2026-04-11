# Experiment: 2026-04-11_02-37-24_mappo_rnn_torch_5ce56e3ae8_curriculum_rework

**Commit**: `5ce56e3ae8`
**Date**: 2026-04-11
**Experiment ID**: `curriculum_rework`
**Base**: `36700751f6` (compressed 2x curriculum)

**Status**: Failed. Aborted at 180k. Policy never learned basic tracking.

---

## 1. Hypothesis

Previous runs showed two failure modes:
1. **3461/949d/492f**: Sigma collapses to ~0.17 in early training because the wide
   action space lets the policy solve trivial step-0 tasks with tiny normalized actions.
2. **36700**: Compressed curriculum partially helped but noise onset still hit before
   sigma recovered.

The hypothesis: removing the "free warmup" entirely (start ALL curriculum phases from
step 0 with gentle ramps) would prevent sigma collapse since the policy never sees a
trivially easy task. The compressed schedule already moved phases to start at 10k —
moving them to 0 was the logical conclusion.

Additionally:
- **Dynamics DR from step 0** (literature consensus: OpenAI Rubik's, RMA, Rudin et al.)
- **FP/FN from step 0** so the policy never overfits to "bbox always valid"
- **Skip fixed delay**, ramp random delay from 0 (negligible at low progress)

## 2. Configuration Delta

All phase start steps moved to 0:

| Phase | 36700 (old) | 5ce56 (new) |
|-------|-------------|-------------|
| Agent velocity | 10k-30k | **0k-40k** |
| Safety | 10k-30k | **0k-40k** |
| Tracking | 10k-30k | **0k-40k** |
| Moving target | 20k-40k | **0k-60k** |
| Coordination | 30k-50k | **0k-60k** |
| Noise | 50k-70k | **0k-100k** |
| Fixed delay | 60k-80k | **removed** |
| Random delay | 70k-90k | **0k-100k** |
| Dropout | 80k-100k | **0k-100k** |
| Burst dropout | 100k-120k | **0k-100k** |
| FP/FN | 100k-200k | **0k-100k** |
| Dynamics DR | 90k-110k | **0k-100k** |
| Task L2 | 20k-40k | **0k-40k** |
| Task L3 | 40k-60k | 40k-60k (unchanged) |

Also: `get_delay_mode()` simplified to always return `"random"` (removed
none→fixed→random state machine).

MAPPO hyperparameters unchanged: `kl_threshold=0.03`, `learning_epochs=6`, `min_lr=3e-4`.

## 3. Results

### 3.1 The Policy Never Learned Basic Tracking

| Step | Reward | pair_valid | bbox_center | bbox_size | episode_len | tracking_lost |
|-----:|-------:|-----------:|------------:|----------:|------------:|--------------:|
| 4k | 334 | 0.329 | 3.0 | 17.6 | 322 | 0.698 |
| 8k | **1002** | **0.569 (peak)** | 7.6 | 33.1 | 442 | 0.349 |
| 12k | 775 | 0.425 | 6.4 | 27.1 | 398 | 0.501 |
| 16k | 669 | 0.351 | 5.6 | 23.9 | 382 | 0.521 |
| 32k | 280 | 0.160 | 2.1 | 10.2 | 240 | 0.879 |
| 60k | -17 | 0.036 | 0.12 | 0.92 | 110 | **0.999** |
| 100k | -53 | 0.018 | 0.06 | 0.47 | 101 | 0.771 |
| 180k | -72 | 0.015 | 0.04 | 0.32 | 96 | 0.484 |

**Peak performance at step 8k** with pair_valid 0.57 — never beaten. By 60k, pair_valid
was 0.04 (essentially random). Reward went **negative** at 60k and stayed there. Episodes
truncated to ~96 steps and never recovered.

For comparison, every previous run reached pair_valid ~0.95 within 12k steps. This run
peaked at 0.57.

### 3.2 Learning Dynamics Were Healthy

This is the surprising part — all the indicators that something is wrong in the
*learning* sense are absent:

| Metric | Behavior | Interpretation |
|--------|----------|----------------|
| LR | Oscillated 0.0003-0.010 | Healthy, scheduler working |
| Sigma | 0.43 → 0.61 | Growing, no collapse |
| Entropy loss | Negative throughout (-0.005 avg) | Entropy bonus winning |
| Policy loss | Mostly negative from 44k onward | Policy improving |
| Value loss | 0.003-0.005 (stable) | Value function healthy |

**Sigma even reached 0.61** — higher than any previous run at this step. The MAPPO
machinery was doing exactly what we wanted. The problem was elsewhere.

### 3.3 The Failure Mode: Unsolvable Task at Step 0

At step 4k, the curriculum state was:
- Agent velocity: 10% (3.7 m/s)
- Safety: 10%
- Tracking difficulty: 10%
- Moving target: ~7% (target moving)
- Coordination reward: 7% (confusing early signal)
- **Noise: 4% (small but real)**
- **Random delay: 4%**
- **Dropout: 4% (~0.2% per step)**
- **FP/FN: 4% (occasional zeroed bboxes)**
- **Burst: 4%**
- **Dynamics DR: 4%**

Individually each is tiny. Cumulatively, the policy is trying to learn basic tracking
while the target moves, dynamics are noisy, observations drop out, and bboxes
occasionally vanish. **It's like learning to walk on a trampoline in the dark.**

A randomly-initialized policy needs at least one perfectly clean signal to bootstrap
from. In 928b/492f/36700 that signal was clean observations of a slow target — "point
camera at target" was learnable in the first few hundred steps. Here, no signal was
clean enough at step 0 to anchor learning.

### 3.4 Comparison to Successful Runs

| Metric @ 8k | 928b | 492f | 36700 | **5ce56** |
|-------------|-----:|-----:|------:|----------:|
| pair_valid | 0.91 | 0.91 | 0.91 | **0.57** |
| reward | 2792 | 2793 | 2793 | **1002** |
| bbox_center | 35.6 | 36.3 | 36.3 | **7.6** |
| episode_len | 497 | 497 | 497 | **442** |
| tracking_lost | 0.011 | 0.011 | 0.011 | **0.349** |

By step 8k, every previous run had achieved >0.9 pair_valid rate. This run was at 0.57
and declining. The deficit was structural, not transient.

## 4. Analysis

### The "never see a clean world" principle was correctly identified but wrongly applied

The previous failures (3461, 949d, 492f) had sigma collapse caused by **too-easy task
dynamics** — the wide action space let the policy solve step-0 with tiny normalized
actions, contracting sigma. The fix was to ensure the task is never trivially easy.

But "task difficulty" and "observation cleanliness" are different axes. Sigma collapse
is driven by the **action→reward** landscape being too flat at step 0 (any small action
works). It is NOT driven by observations being too clean. Conflating these led to
ramping observation corruption from step 0, which destroyed bootstrappability.

### What should be at step 0 (prevents sigma collapse)
These increase **task difficulty** without corrupting the observation→action mapping:
- Agent velocity ramp (forces tilt compensation)
- Safety/CBF (forces avoidance)
- Tracking initial state difficulty
- Moving target (forces prediction)
- Coordination reward (forces multi-agent geometry)
- Dynamics DR (filtered through controller, barely visible)
- Task level curriculum (reward shaping)

### What should NOT be at step 0 (destroys learning)
These corrupt observations and require an existing tracking skill to handle:
- **Noise**: bboxes are perturbed
- **Random delay**: bboxes are stale
- **Dropout**: bboxes are missing
- **Burst dropout**: sustained bbox blackouts
- **FP/FN**: bboxes are zeroed (FN) or invented (FP)

The policy must learn "the bbox tells me where the target is" before it can learn
"sometimes the bbox is wrong/missing." This is a fundamental ordering: signal → noise.

### The literature got it right (and we missed the nuance)

OpenAI/RMA/Rudin/Loquercio all apply DR from step 0. **They do not apply observation
corruption from step 0.** Their "DR from start" is mass/inertia/friction/gain
randomization — physics dynamics, not sensor models. Sensor noise/dropout/FN are
typically introduced after the base policy is learned (often via fine-tuning).

The principle "expose the policy to all difficulty from start" applies to **physics**,
not **observation corruption**. Observation corruption needs a learned base skill to
attach to.

## 5. Decision

**Abort and apply the corrected schedule.**

### The fix
Keep all task-difficulty phases at step 0 (preserves sigma growth + literature
alignment), but delay observation corruption phases to ~30-40k (after basic tracking
is established):

```
Phase                     Start    End      Duration
─────────────────────────────────────────────────────
Agent velocity            0k       40k      40k    (task difficulty)
Safety                    0k       40k      40k    (task difficulty)
Tracking                  0k       40k      40k    (task difficulty)
Moving target             0k       60k      60k    (task difficulty)
Coordination              0k       60k      60k    (task difficulty)
Dynamics DR               0k       100k     100k   (filtered physics)
Task L2                   0k       40k      40k    (reward shaping)
Task L3                   40k      60k      20k
─── observation corruption (after base skill) ──────
Noise                     30k      100k     70k
Random delay              30k      100k     70k
Dropout                   30k      100k     70k
Burst dropout             40k      100k     60k
FP/FN                     40k      100k     60k
─── all curriculum ends ─────────────────────────────
Post-curriculum           100k     320k     220k
─────────────────────────────────────────────────────
```

Rationale:
- **0-30k**: Policy learns clean bbox tracking with full task difficulty (velocity,
  safety, target motion). Sigma stays healthy because tasks are non-trivial. Dynamics DR
  is barely felt (controller-filtered).
- **30-40k**: Noise/delay/dropout introduced gradually. The policy has a base tracking
  skill to defend.
- **40-100k**: Burst and FP/FN added — these are the most disruptive (zero bboxes), so
  they need the most established baseline.
- **100k-320k**: Post-curriculum stabilization (220k).

### Validation criteria for next run
- Step 8k: pair_valid > 0.85 (matches previous successful runs)
- Step 16k: pair_valid > 0.90, episode_len > 490
- Step 30k: sigma > 0.4 (haven't collapsed yet)
- Step 60k: pair_valid > 0.75 through observation corruption ramp
- Step 100k: pair_valid > 0.70, sigma > 0.5 — survived full corruption
