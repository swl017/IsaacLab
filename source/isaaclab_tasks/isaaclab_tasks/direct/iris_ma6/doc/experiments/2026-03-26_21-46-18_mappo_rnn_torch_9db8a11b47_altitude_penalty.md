# Experiment: 2026-03-26_21-46-18_mappo_rnn_torch_9db8a11b47_altitude_penalty

**Commit**: `9db8a11b47`
**Date**: 2026-03-26
**Experiment ID**: `a1_with_aoi`
**Base**: `820eb7a5543aea4594c2b5dfbfdf5a12643c96c2`<2026-03-26_20-00-43_mappo_rnn_torch_820eb7a554_z_action_tuning>

Stopped at 50k due to `ray_w` observation bug discovered during run.

---

## 1. Hypothesis
Replace hard altitude termination with soft penalty (same approach that fixed collision
termination). The policy should learn altitude maintenance from penalty signal while
keeping full-length episodes for stable reward accumulation.

## 2. Configuration Delta
- Hard termination: `z < 2.0` → `z < 0.5` (physics safety net only)
- Added soft altitude penalty: `altitude_penalty_scale = -100.0` when `z < 2.0m`
- Reverted `action_weight[2]`: 1 → 5 (z-action penalty back to original)
- Added "altitude" to reward tracking channels

## 3. Results (stopped at 50k — ray_w bug invalidates later training)

### 3.1 Training Curves
**Metric: `Total timesteps (mean)`**
- **No more collapse.** Stays at 499 through 24k (base: collapsed at 4k).
- Occasional dips: 476 at 36k, 405 at 38k, 448 at 50k — but always recovers.
- Min drops to 1 sporadically (26k, 30k, 36-42k, 50k) but most envs survive full episode.
- This confirms the altitude penalty approach works.

**Metric: `pair_valid_rate`**
- Peaks at 0.85 at 32k, oscillates 0.32-0.85 range.
- The familiar dip around 36-48k is present but less severe than previous experiments,
  and the metric recovers (0.64 at 50k).
- More volatile than earlier experiments — likely due to full-length episodes giving
  the policy more freedom to explore aggressive strategies.

**Metric: `drone_0_bbox_center`**
- Steady rise: 1.0 → 21.2 at 50k. Continues improving throughout without peak

**Metric: `Reward / Total reward (mean)`**
- Stable around 1500-1800 from 6k onwards. No collapse.
- Compare: base z_action_tuning collapsed to 16 at 6k.

**Metric: `collision_fraction`**
- Decreasing over time: 0.28% at 4k → 0.0001% by 34k.
- The policy learned collision avoidance purely from penalty (no termination).

**Metric: `drone_0_cbf_penalty`**
- Very low throughout: -0.001 to -0.018. Negligible compared to other rewards.
- CBF at lambda=1.0 is well-calibrated.

**Metric: `drone_0_action_sum`**
- Higher magnitude than previous experiments: -5.2 → -12.4 at 46k.
- The z-weight=5 is contributing more due to full-length episodes
  (500 steps of action penalty vs 50 steps in crashed episodes).

**Metric: `Policy / Standard deviation (drone_0)`**
- Grows to 1.28 by 50k. Similar growth rate to base experiments.

**Metric: `drone_0_triangulation`**
- Zero throughout (coordination curriculum starts at 60k).

### 3.2 Behavior Observations
Not tested (stopped early due to ray_w bug).

## 4. Analysis
- **The altitude penalty approach works.** Episodes stay full-length, Total reward stays
  stable, and pair_valid_rate shows the familiar oscillation pattern without permanent collapse.
- **Combined with collision penalty (no termination)**, the env now runs full 499-step
  episodes consistently. This is the MA5-parity training regime we were targeting.
- Collision avoidance learned from penalty alone: collision_fraction drops from 0.28% to
  near-zero by 34k without any termination signal.
- The action_sum magnitude is higher (-12 vs -8) because episodes are longer. May need to
  monitor if this becomes problematic at later training stages.
- **Results invalidated by ray_w observation bug** — the policy was training on incorrect
  camera ray direction. Need to re-run with the fix.

## 5. Decision / Todo
- Fix ray_w observation bug and re-run with identical config.
- This config (soft altitude penalty + no collision termination + z-weight=5) is the
  new baseline for future experiments.
