# Experiment: 2026-04-15_18-02-07_mappo_rnn_torch_bda2639a71_min_steps2_baseline

**Commit**: `bda2639a71`
**Date**: 2026-04-15
**Experiment ID**: `bda2639a71_min_steps2_baseline`
**Base**: `087be7697c` with `min_steps` reverted to 2 (bisect fix) + observation redesign v1

**Status**: Ongoing at step 252k. **First run to survive the 120k delay mode transition.**

---

## 1. Hypothesis

The `087be7697c` post-fix run degraded from step 20k onwards (pair_valid peaked at 0.87 @
16k, dropped to 0.50 @ 36k). The bisect experiment on 2026-04-15
(`bisect_min_steps_vs_ticket029.md`) isolated the cause:

| Branch (50k bisect) | min_steps | Ticket 029 | Reward @ 40k | pair_valid @ 40k |
|---|---:|---:|---:|---:|
| 2695 pre-fix | 2 | no | 2587 | 0.88 |
| 087 post-fix | **0** | yes | 913 | 0.46 |
| **B1** (min=0 alone) | **0** | no | **−6265** | **0.03** |
| **B2** (ticket 029 alone) | 2 | yes | **3171** | **0.87** |

`min_steps: 2 → 0` alone caused a catastrophic failure (reward −6265, sigma maxed at 2.01).
Ticket 029 is innocent and is a net improvement.

This run reverts `min_steps: 0 → 2` to restore healthy training, keeping the ticket 029
dual-cache pipeline.

## 2. Configuration Delta

### vs 087be7697c (the fix)

```python
# delay_system_v3/delay_cfg_v3.py
LatencyCfg.min_steps: 0 → 2     # Restore pre-fix default
```

### Other changes in bda2639a71

- `iris_ma_env6_v1.py` (+2490 lines): new observation redesign v1 (not yet in use — the
  running experiment still uses `iris_ma_env6_test.py`)
- `attitude_controller.py`: 35 line changes (minor tuning)
- Documentation and experiment registry updates

The `min_steps` revert is the only runtime-relevant change for this training run.

### vs 2be3 (the target run that worked)

The current config differs from 2be3 on:
- **MAPPO**: kl_threshold=0.02, min_lr=1e-4 (matched to 2be3)
- **Action space**: max_yaw_rate=45°/s, gimbal=180°/s (matched to 2be3)
- **Penalties**: action_sum=-30, action_delta=-15 (matched to 2be3)
- **Still different**: PX4_MATCHED controller, 31D obs, ticket 029 dual-cache pipeline,
  DomainRandomization system (gated at 180k)

## 3. Results

### 3.1 Bootstrap & Phase 1 (0-60k) — Healthier than 2695

| Step | Reward | pair_valid | track_lost | sigma | LR | val_loss | ep_len |
|-----:|-------:|-----------:|-----------:|------:|----:|---------:|-------:|
| 4k | -5.69 | 0.393 | 0.695 | 0.493 | 0.009 | 0.041 | 334 |
| 8k | 2.31 | 0.772 | 0.192 | 0.322 | 0.006 | 0.000 | 479 |
| 16k | 4.96 | 0.861 | 0.030 | 0.240 | 0.003 | 0.000 | 494 |
| 20k | 5.98 | **0.923** | 0.005 | 0.211 | 0.002 | 0.000 | 498 |
| 40k | 6.44 | 0.881 | 0.006 | 0.144 | 0.0002 | 0.000 | 497 |
| 60k | 5.84 | 0.795 | 0.026 | 0.139 | 0.0001 | 0.000 | 492 |

Sigma stabilized at **~0.14** — close to 2be3's 0.11 and much tighter than 2695's 0.40.
The policy exploited the action space more efficiently. Clean bootstrap, excellent
pair_valid (0.92 peak at 20k), near-zero tracking_lost.

### 3.2 Phase 2 — Noise Ramp (100k-120k): Survived

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 100k | **7.56** | 0.805 | 0.034 | 0.145 | 491 |
| 108k | 7.58 | 0.793 | 0.028 | 0.148 | 491 |
| 116k | 7.52 | 0.790 | 0.031 | 0.149 | 491 |
| 120k | 7.49 | 0.777 | 0.034 | 0.151 | 490 |

Peak reward of 7.58 at 108k. **Unlike all prior post-2be3 runs, the noise ramp did not
collapse the policy.** Pair_valid held at 0.78+ through the entire noise ramp. This
confirms that the earlier "noise kills the policy" hypothesis was confounded by the
`min_steps=0` bug — with correct delay semantics, the policy at sigma=0.14 handles noise
just fine.

### 3.3 The 120k Delay Mode Transition: Clean Dip and Recovery

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 120k | 7.49 | 0.777 | 0.034 | 0.151 | 490 |
| 124k | **3.99** | **0.409** | **0.306** | 0.158 | **440** |
| 128k | 5.39 | 0.627 | 0.083 | 0.162 | 483 |
| 132k | 5.58 | 0.661 | 0.073 | 0.163 | 484 |
| 140k | 5.41 | 0.667 | 0.074 | 0.165 | 485 |

**Key milestone: first run to survive the 120k→124k delay mode transition.** Brief dip
at 124k (pair_valid 0.78 → 0.41, episode length 490 → 440) but recovered within 8k steps
to a stable plateau. Compare with `2695ffe1e3`:

| Run | 120k pair_valid | 124k pair_valid | 128k pair_valid | Recovery? |
|-----|----------------:|----------------:|----------------:|:--|
| 2695ffe1e3 (min_steps=2) | 0.752 | 0.157 | 0.146 | **No — frozen forever** |
| **bda2639a71 (min_steps=2)** | **0.777** | **0.409** | **0.627** | **Yes** |

Both runs had `min_steps=2`. The difference must come from the ticket 029 dual-cache
refactor — it made the pipeline robust enough that the progress=0 fixed-mode transition
no longer kills the policy.

### 3.4 Delay Ramp Continues (140k-200k) — Stable

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 140k | 5.41 | 0.667 | 0.074 | 0.165 | 485 |
| 160k | 5.43 | 0.648 | 0.081 | 0.171 | 481 |
| 180k | 5.31 | 0.657 | 0.062 | 0.181 | 483 |
| 200k | 4.88 | 0.603 | 0.092 | 0.194 | 472 |

Policy holds reward ~5 and pair_valid ~0.60-0.65 through all delay-system ramps (fixed
delay ends at 140k, random delay ends at 160k, dropout ramps 160k-180k). Sigma gently
rising (0.15 → 0.19) which is healthy — consistent with the `entropy_loss_scale=0.01`
bonus doing its job.

### 3.5 Late-Phase Degradation (200k-240k)

| Step | Reward | pair_valid | track_lost | sigma | ep_len |
|-----:|-------:|-----------:|-----------:|------:|-------:|
| 200k | 4.88 | 0.603 | 0.092 | 0.194 | 472 |
| 220k | 2.93 | 0.395 | 0.269 | 0.293 | 412 |
| 240k | 2.74 | 0.369 | 0.281 | 0.437 | 344 |

Degradation begins at ~200k. Active curriculum stages in this window:
- **Burst dropout**: ramps 200k → 220k
- **Dynamics DR**: ramps 180k → 220k

Sigma is growing (0.19 → 0.44 in 40k steps = +0.006/kstep, much faster than earlier
stages). The policy is actively exploring to handle the new disturbances — consistent
with Ch3 §A.1 "physics-side expand" and "observation-side expand" both active. Whether
this stabilizes or collapses is TBD.

## 4. Analysis

### Primary finding: min_steps=0 was the post-fix regression

The user's bisect conclusively proved that my earlier diagnosis (blaming `min_steps=2`
for the 120k cliff in 2695ffe1e3) was **wrong**. The actual behaviors are:

- `min_steps=2` + pre-ticket-029 pipeline → 120k cliff collapse (2695 failure mode)
- `min_steps=0` + pre-ticket-029 → catastrophic from step 0 (B1: reward −6265)
- `min_steps=0` + ticket 029 → 20k degradation (087 failure mode)
- **`min_steps=2` + ticket 029 → healthy through 120k and beyond (this run)**

The ticket 029 dual-cache refactor fixed the 120k cliff indirectly. The old pipeline's
"none" mode early-return created a stale-cache condition on mode transition; the new
pipeline's dual-cache semantics (raw/noisy × with/no dropout) handles the transition
cleanly.

### Secondary finding: sigma stabilizes at 0.14 in this config

Through 80-120k, sigma held at 0.14 — much tighter than the 0.40-0.48 stagnation seen in
949d/492f/025af4d/82364. This is closer to 2be3's 0.11 normalized sigma. The policy
found a compact, efficient strategy because:
- Action penalties are heavy (-30/-15, matched to 2be3)
- Action space is narrow (45°/s yaw, 180°/s gimbal, matched to 2be3)
- Observation pipeline is correct (ticket 029 fix)

The earlier sigma stagnation was not a fundamental problem with the wider action space —
it was a symptom of broken observation semantics interacting with the wider action
space. With correct observations, even the wider action space would likely produce a
tighter sigma.

### Late-phase stress (200k+)

Reward drops from ~5 to ~3 and sigma grows from 0.19 to 0.44 between 200k and 240k.
Burst dropout + dynamics DR are simultaneously ramping. This is two stages stacking
per Ch3 §A.4, but they're compatible (physics + observation expand). Watch for:
- If sigma stabilizes by 260k (curriculum complete at 220k for burst, 200k for DR), the
  run recovers and this is the final settling trajectory
- If sigma continues climbing past 0.5 and pair_valid falls below 0.3, a collapse is
  beginning

## 5. Decision

**Validated**: `min_steps=2` is the correct default. The ticket 029 dual-cache pipeline
is a net improvement and should stay.

**Current config** is the new working baseline. Re-introducing the wider action space
(90°/s yaw, 360°/s gimbal) or lighter penalties (-10/-5) on top of this baseline may
now be safe — the observation pipeline is no longer the bottleneck.

**Next steps**:
1. Let this run complete (~260k or 400k) to see if late-phase degradation stabilizes
2. If it stabilizes, bisect the remaining 2be3 vs current differences:
   - PX4_MATCHED vs TUNED controller
   - 31D vs 30D observation (`effective_hfov`)
   - DomainRandomization presence
3. Once baseline is verified stable, widen action space one dimension at a time to
   re-enable sim-to-real flexibility