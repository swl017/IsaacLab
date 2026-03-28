# Experiment: 2026-03-27_19-11-14_mappo_rnn_torch_395280af3d_tracking_lost_truncation

**Commit**: `395280af3d`
**Date**: 2026-03-27
**Experiment ID**: `a1_with_aoi`
**Base**: `9db8a11b47`<2026-03-27_00-30-27_mappo_rnn_torch_9db8a11b47_altitude_penalty>

Ran to completion (200k).

---

## 1. Hypothesis
Multiple changes combined:
1. **Tracking-lost truncation**: Truncate (not terminate) episodes when all agents lose
   detection for 3s consecutive. Prevents wasting training on "lost and wandering" episodes.
   Uses truncation so value function bootstraps V(s) instead of V=0.
2. **Gradient-based altitude penalty**: Continuous `clamp(threshold - z, 0) * scale`
   instead of binary, providing signal before the crash floor.
3. **ray_w bug fix**: Camera ray directions now computed from bbox center via
   `compute_ray_directions_from_bbox()` instead of gimbal boresight.
4. **Tighter spawn heights**: cylinder_height 20-40m (was 20-50m), range_max 5m (was 10m),
   symmetric z_offset around center.

## 2. Configuration Delta
- `compute_ray_directions_from_bbox()` replaces `gimbal_ray_direction_world()` for ray_w
- Altitude penalty: binary → gradient-based (`clamp(threshold - z, 0) * scale`)
- Tracking-lost truncation: `enable_tracking_truncation=True`, `timeout=3.0s`, `grace_steps=50`
- `cylinder_height_max`: 50→40, `cylinder_height_range_max`: 10→5
- z_offset: `[0, range]` → `[-range, +range]` (symmetric)

## 3. Results (200k complete)

### 3.1 Training Curves

#### Phase 1: Basic Tracking (0-60k)
**Metric: `Total timesteps (mean)`**
- **Episode length collapse is fixed.** Mean stays 485-496 during 0-50k.
- Gradual, gentle decline from 496 at 12k to 472 at 60k — not a step-function crash.
- Compare: previous experiments collapsed to 52 at 24k.

**Metric: `pair_valid_rate`**
- Peaks at 0.91 at 12k. Gradual decline to 0.62 at 60k.
- The familiar decline is present but much gentler and from a higher peak.

**Metric: `drone_0_bbox_center`**
- Peaks at 30.3 at 12k. Gentle decline to 24.6 at 60k. No collapse.

**Metric: `Reward / Total reward (mean)`**
- Peaks at 2631 at 12k. Dips to 1953 at 60k.

#### Phase 2: Coordination (60-120k)
**Metric: `pair_valid_rate`**
- **Recovery!** Rebounds from 0.62 at 60k to 0.80 at 120k.
- The coordination curriculum (60-100k) drives agents to improve geometry,
  which also improves detection as a side effect.

**Metric: `drone_0_triangulation`**
- Ramps from 0 at 60k to **32.9 at 120k** — best of any MA6 experiment.
- Steady, healthy growth through the entire phase.

**Metric: `Total timesteps (mean)`**
- Recovers to 493 by 120k. Episodes nearly full-length again.

**Metric: `Reward / Total reward (mean)`**
- Climbs from 1953 at 60k to **3516 at 120k** — new all-time high.
- The coordination reward stacks on top of tracking reward.

**Metric: `drone_0_bbox_center`**
- Recovers from 24.6 to 26.0 and stabilizes. Healthy.

**Metric: `collision_fraction`**
- Increases from 0.1% to 0.6% during 60-120k as agents move more aggressively
  for triangulation geometry. Still low.

#### Phase 3: Noise Introduction (120-142k)
**Metric: `pair_valid_rate`**
- Drops from 0.80 at 120k to 0.51 at 122k (noise curriculum starts).
- **Recovers** to 0.80 by 138k — the policy adapts to noise.

**Metric: `drone_0_triangulation`**
- Dip from 32.9 to 21.8 at 122k, recovers to 33.1 at 138k.

**Metric: `Total timesteps (mean)`**
- Jumps to 498 at 122k and stays there — tracking_lost_fraction drops to 0%.
- With noise, agents' detections flicker, resetting the tracking-lost counter.
  The truncation mechanism effectively stops triggering.

**Metric: `Reward / Total reward (mean)`**
- Dip to 2358 at 122k, recovers to 3342 by 138k.

#### Phase 4: Collapse (142k+)
**Metric: `pair_valid_rate`**
- **Sharp collapse at 142k**: 0.75 → 0.22. Partially recovers to 0.40-0.50 range
  but never returns to pre-collapse levels. Settles at 0.40-0.43 by 200k.

**Metric: `drone_0_bbox_center`**
- Collapses from 19.9 → 1.0 at 142k. Stabilizes at 2.9-4.5 range through 200k.
- The policy loses the ability to center targets in frame.

**Metric: `drone_0_triangulation`**
- Drops from 31.5 → 6.8 at 142k. Stabilizes at 10-15 range.
- Still producing some triangulation reward but far below peak.

**Metric: `Reward / Total reward (mean)`**
- Drops from 3179 → 1004 at 142k. Stabilizes at 1300-1600 range.

**Metric: `tracking_lost_fraction`**
- Drops to 0% at 122k and stays there through 200k.
- The truncation mechanism is no longer active — every episode runs to completion
  because noisy detections keep resetting the lost counter.

**Metric: `Total timesteps (mean)`**
- Stays at 498 through 200k. No episode length collapse.

**Metric: `collision_fraction`**
- Drops from 0.5% to 0.07% after 142k. Policy becomes conservative.

**Metric: `Policy / Standard deviation`**
- Stabilizes at 1.32 from 130k onwards. Much better than 9db8a11b47 (1.49).
- The std plateau suggests the policy has found a stable exploration level.

**Metric: `drone_0_action_sum`**
- Peaks at -8.7 at 120k, then DECREASES to -7.4 at 200k.
- Policy is moving less after 142k — consistent with conservative behavior.

### 3.2 Behavior Observations
- Checkpoint: `/home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6/2026-03-27_19-11-14_mappo_rnn_torch_395280af3d_tracking_lost_truncation/checkpoints/best_agent.pt`
- Play from 130k step: Smooth tracking, but as the drone gets far from the target, it easily loses tracking. Agressive and vibrating gimbal control.
- Play from 140k step: Violent drone body tilt and gimbal. The gimbal stabilization seems to not keep up with the drone's attitude. Gimbal turns AWAY from the target.
- Play from 200k step: Similar to 140k.

## 4. Analysis
- **Phases 1-3 (0-140k) are excellent.** Episode length stays near 499, Total reward
  reaches 3516 (all-time high), triangulation hits 32.9, pair_valid_rate recovers
  through coordination phase. This is the best sustained MA6 training to date.

- **The 142k collapse is a new pattern.** It coincides with:
  - fixed_delay_start_step = 140k (deterministic latency introduced)
  - Note: `curriculum_task_levels = False` and `task_reward_level = 1`, so task level 3
    is never activated — the triangulation reward is always FIM (Level 1).
  - The collapse is caused by fixed delay introduction alone.

- **Tracking-lost truncation has a critical flaw with noise.** When noise is introduced
  (120k+), noisy detections produce intermittent valid frames that reset the 3s lost
  counter. The truncation drops to 0% and never triggers again. This means after 120k,
  the mechanism provides no benefit — episodes run to completion regardless, including
  ones where the policy has effectively lost the target but gets occasional noise-induced
  "valid" frames.

- **The 142k collapse is NOT an episode length problem** (timesteps stays at 498).
  It's a policy quality collapse — the policy unlearns tracking when task_level_3
  and fixed_delay are introduced simultaneously. The bbox_center drop (19.9 → 1.0)
  indicates the policy stops actively centering targets.

- **Policy std stabilizes at 1.32** — much healthier than 9db8a11b47 (1.49).
  The collapse is not from exploration divergence but from curriculum shock.

- **Gimbal rate dilema** The gimbal command vibrate violently at later stage, but also the gimbal stabilization seems to not keep up with violent drone tilt. Should the physical maximum gimbal rate be lowered (12pi to 2.5pi), or kept as is? 

## 5. Decision / Todo
- **Fix tracking-lost truncation for noisy observations.** Options:
  - Use a smoothed/filtered detection signal instead of raw per-frame validity.
  - Require N consecutive valid frames to reset the lost counter (debouncing).
  - Or tie truncation to a different metric (e.g., bbox_center below threshold).
- **Investigate the 142k collapse.** Caused by fixed_delay alone (task_level_3 is inactive).
  Consider: (a) push fixed_delay_start_step later (e.g., 160k); (b) slower delay ramp;
  (c) start with very small delay and ramp more gradually.
- The 0-120k training is strong. If the late-stage curriculum can be fixed,
  this config is viable as the MA6 baseline.
