# Experiment: 2026-03-26_20-00-43_mappo_rnn_torch_820eb7a554_z_action_tuning

**Commit**: `820eb7a5543aea4594c2b5dfbfdf5a12643c96c2`
**Date**: 2026-03-26
**Experiment ID**: `a1_with_aoi`
**Base**: `3da98de79003823eddb75b8d74d081726bbfeb03`<2026-03-26_14-56-04_mappo_rnn_torch_3da98de790_no_collision_termination>

Accidentally stopped at 16k.

---

## 1. Hypothesis
The Total timesteps collapse in the base experiment (499 → 51 at 20k) is caused by drones
crashing into the ground (`z < 2.0` termination). Two mitigations attempted:
1. Raise minimum spawn altitude: `cylinder_height_min` 10 → 20m (more altitude buffer)
2. Reduce z-axis action penalty weight: `action_weight[2]` 5 → 1 (was disproportionately
   penalizing vertical movement, which may have discouraged altitude recovery maneuvers)

## 2. Configuration Delta
- `cylinder_height_min`: 10.0 → 20.0
- `action_weight`: [1, 1, **5**, 1, 0.5, 0.5, 0.3] → [1, 1, **1**, 1, 0.5, 0.5, 0.3]

## 3. Results (aborted at 16k)

### 3.1 Training Curves
**Metric: `pair_valid_rate`**
- Rising: 0.22 → 0.78 at 16k. Trajectory similar to base experiment's first 16k.

**Metric: `Total timesteps (mean)`**
- Collapse happened EARLIER: 499 at 2k → 247 at 4k → 5.7 at 6k.
- Base experiment stayed at 499 until 18k. This is 4x faster collapse.
- Recovery to ~48 at 14-16k, but far below 499.

**Metric: `Total timesteps (min/max)`**
- Min drops to 1 at 4k (base: at 20k). Max stays at 499.

**Metric: `drone_0_bbox_center`**
- Rising normally: 2.6 → 23.0 at 16k. Similar to base.
- Per-step quality is fine; the policy tracks when alive.

**Metric: `Reward / Total reward (mean)`**
- Collapsed: 747 → 16.6 at 6k. Partially recovers to 190 at 16k.
- Base experiment: 2374 at 18k (before its collapse).

**Metric: `Policy / Standard deviation (drone_0)`**
- Faster std growth: 0.58 → 1.26 in 16k (base: 0.53 → 0.97 in 20k).
- Reducing the z-action penalty removed a constraint that was slowing std growth.

**Metric: `drone_0_action_sum`**
- Lower magnitude than base at same steps (-6.1 vs -6.5 at 16k).
- Expected: z-weight=1 instead of 5 reduces total action penalty.

**Metric: `collision_fraction`**
- Similar to base (~0.1-0.2%).

### 3.2 Behavior Observations
Not tested (training aborted).

## 4. Analysis / Open question
- **Reducing z-action weight made the crash problem WORSE, not better.**
  The z-weight=5 was acting as implicit altitude protection — by penalizing vertical
  velocity commands, it discouraged the large downward actions that cause crashes.
  Removing it (5→1) freed the policy to explore vertical actions more aggressively,
  causing earlier and more frequent ground crashes.
- The raised spawn altitude (10→20m) was insufficient to compensate. 20m - 2m = 18m buffer,
  but at max_lin_vel ≈ 4.5 m/s (early ramp), that's ~4s to crash. With the z penalty
  removed, the policy readily commands max downward velocity.
- Policy std growth accelerated (1.26 at 16k vs 0.97 in base) because the reduced action
  penalty removed a regularizing signal, confirming that the z-weight=5 was serving
  double duty: action smoothness AND altitude safety.
- The altitude termination (`z < 2.0`) is the fundamental bottleneck. Neither
  raising spawn height nor tuning action weights addresses the root cause.
  The termination must be replaced with a soft penalty (same approach as collision).

## 5. Decision / Todo
- Revert `action_weight[2]` back to 5 — it serves as useful regularization.
- Replace `z < 2.0` hard termination with soft altitude penalty in `_get_dones()`.
- Keep `cylinder_height_min = 20.0` as additional buffer.
