# Experiment: 2026-03-27_00-30-27_mappo_rnn_torch_9db8a11b47_altitude_penalty

**Commit**: `9db8a11b47`
**Date**: 2026-03-27
**Experiment ID**: `a1_with_aoi`
**Base**: Same config as `2026-03-26_21-46-18` (stopped early due to ray_w bug). Re-run with ray_w bug fixed.

Ran to completion (200k).

---

## 1. Hypothesis
Same as the stopped run: soft altitude penalty + no collision termination should maintain
full-length episodes and enable stable training. This is a re-run (same commit).

## 2. Configuration Delta
(Same as stopped run 9db8a11b47)
- Hard termination: `z < 0.5` (physics safety net only)
- Soft altitude penalty: `-100` when `z < 2.0m`
- No collision termination (penalty only)
- `lambda_cbf`: 1.0, `collision_penalty_scale`: -100
- `action_weight[2]`: 5 (z-axis), `cylinder_height_min`: 20m

## 3. Results (200k complete)

### 3.1 Training Curves
**Metric: `Total timesteps (mean)`**
- Full 499 during 0-22k, then same step-function collapse to ~52 at 24-26k.
- Stabilizes at ~52 for the rest of training. Never recovers.
- The altitude penalty did NOT fix the crash problem — drones still hit z < 0.5.
- Altitude reward is zero throughout (drones rarely go below 2m), confirming they
  crash straight through the penalty zone to the hard floor.

**Metric: `pair_valid_rate`**
- Strong first phase: 0.09 → 0.95 at 26k. Best peak of any MA6 experiment.
- Gradual decline 26k-120k: oscillates 0.4-0.94 with high variance.
- Collapse at 140k: drops to 0.05-0.35 range and stays there through 200k.
- The 140k collapse coincides with noise curriculum starting (120k) and task level 3 (130k).

**Metric: `drone_0_bbox_center`**
- Excellent first phase: rises to 37.0 at 32k (best of any MA6 experiment).
- Maintains 18-37 range through 140k despite pair_valid_rate oscillations.
- Collapses to 4.6-11.8 after 140k. The policy loses tracking ability.

**Metric: `drone_0_triangulation`**
- Ramps from 0 at 60k to 29.4 at 112k (best ever).
- Declines to 2-13 range after 120k, matching the pair_valid_rate degradation.

**Metric: `collision_fraction`**
- Drops from 0.5% to near-zero by 50k. Collision avoidance learned from penalty.
- Occasional spikes (4% at 148k, 0.5% at 172k) during the late collapse phase.

**Metric: `all_invalid_rate`**
- Low (1-15%) during 0-140k. Jumps to 20-48% after 140k.
- Consistent with pair_valid_rate collapse.

**Metric: `drone_0_altitude`**
- Zero throughout except tiny blips at 146-148k.
- Drones never hover near the 2m threshold — they either fly safely or crash to z < 0.5.
- The soft penalty provides no gradient because the crash is too fast to intercept.

**Metric: `Policy / Standard deviation`**
- Grows continuously: 0.53 → 1.49 at 200k. Never saturates.
- This unbounded growth is a concern — by 140k (std=1.29), exploration overwhelms exploitation.

**Metric: `Reward / Total reward (mean)`**
- Peak: 2727 at 22k (full-length episodes). Drops to ~250-330 during 26-112k (short episodes).
- Further decline to 50-120 after 140k.

**Metric: `drone_0_action_sum`**
- Grows to -8 to -9 range by 100k, stays there. Not as extreme as z_action_tuning (-12).

### 3.2 Behavior Observations
Not tested yet.

## 4. Analysis
- **The altitude soft penalty approach failed.** Total timesteps still collapses to ~52
  at the same timing (~24k). The altitude reward is zero, meaning drones don't hover
  near the 2m threshold — they plunge straight past it to the 0.5m hard floor. The
  descent is too fast (one bad action sequence → unrecoverable dive) for a threshold
  penalty to intercept.
- **Despite the episode length collapse, the policy still learns well through ~120k.**
  pair_valid_rate hits 0.95, bbox_center hits 37, triangulation hits 29 — all best
  values seen in any MA6 experiment. The short episodes (52 steps ≈ 2s) are enough
  for the policy to learn tracking and coordination in the early portion of each episode.
- **The 140k collapse is new** and caused by the noise/delay curriculum phases
  (noise 120-140k, task level 3 at 130k). The policy cannot handle observation noise
  with only 52 steps per episode — there isn't enough time to recover from noisy
  observations before the episode terminates.
- **Policy std never stops growing** (1.49 at 200k). This suggests the entropy
  coefficient is too high or the reward signal is too noisy to drive std down.
  The std growth accelerates after 140k, coinciding with the performance collapse.
- The ray_w bug was fixed in this run, so these results reflect correct camera ray observations.

## 5. Decision / Todo
- The altitude penalty is ineffective (zero gradient). Consider:
  - Gradient-based altitude penalty: penalty proportional to `max(0, threshold - z)`
    instead of binary. Provides gradient before the crash floor.
  - Or simply remove altitude termination entirely (let physics handle it).
- The 140k collapse from noise curriculum needs investigation once ray_w is fixed.
- Despite high per-step metrics (pair_valid_rate, bbox_center), the Total timesteps
  collapse to 52 means Total reward is only ~250 vs ~2700 at peak. The policy learns
  to track well for 2 seconds but cannot sustain a full episode. The altitude crash
  problem remains the primary blocker.
- Monitor policy std growth — may need entropy coefficient reduction or std capping.
