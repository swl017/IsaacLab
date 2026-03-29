# Experiment: 2026-03-28_21-29-36_mappo_rnn_torch_4f7d9d0c34_gimbal_controller_fix

**Commit**: `4f7d9d0c34`
**Date**: 2026-03-28
**Experiment ID**: `a1_with_aoi`
**Base**: `395280af3d`<2026-03-27_19-11-14_mappo_rnn_torch_395280af3d_tracking_lost_truncation>

Training in process (324k/324k — extended past original 280k).

---

## 1. Hypothesis
Multiple changes from the base:
1. **Remove gimbal controller internal rate integration**: Controller now reads actual
   joint state from simulation each step instead of maintaining parallel integrated state.
   Eliminates state drift between controller and physics.
2. **Lower max_gimbal_rate**: 2π → 1π rad/s (360 → 180 deg/s). Reduces gimbal vibration
   from policy exploration while still allowing reasonable tracking speeds.
3. **Debounced tracking-lost truncation**: `tracking_reacquire_steps=5` (0.2s sustained
   reacquisition required to reset lost counter). Fixes noise-defeating-timeout bug.
4. **Extended curriculum**: 200k → 280k total. Delay phases have 2x longer ramps
   (fixed_delay 140-180k, random 180-220k, dropout 220-240k, dynamics 240-280k).
5. **Tighter spawn heights**: cylinder_height 20-40m (was 20-50m), range_max 5m.

## 2. Configuration Delta
- Gimbal controller: read actual joint state instead of internal integration
- `max_gimbal_rate`: 2π → 1π rad/s
- `tracking_reacquire_steps`: 5 (new)
- `fixed_delay_end_step`: 160k → 180k
- `random_delay_start/end`: 160-180k → 180-220k
- `dropout_start/end`: 180-200k → 220-240k
- `dynamics_start/end`: 180-200k → 240-280k
- `timesteps`: 200k → 280k (extended to 324k)
- `cylinder_height_max`: 50 → 40, `cylinder_height_range_max`: 10 → 5

## 3. Results (324k complete)

### 3.1 Training Curves

#### Phase 1: Basic Tracking (0-60k)
**Metric: `pair_valid_rate`**
- Peaks at **0.966 at 12k** — highest of any MA6 experiment.
- Stays above 0.90 through 40k, gradual decline to 0.74 at 60k.

**Metric: `drone_0_bbox_center`**
- Peaks at **51.2 at 36k** — far above any previous experiment (best was 37).
- Stays above 48 through 60k.

**Metric: `Total timesteps (mean)`**
- **Rock solid at 497-498** throughout entire 324k run. No episode length collapse at any point.

**Metric: `Reward / Total reward (mean)`**
- Rises from 1638 to 2798 at 20k.

#### Phase 2: Coordination (60-120k)
**Metric: `drone_0_triangulation`**
- Ramps from 0 at 60k to **44.6 at 120k** — far above previous best (32.9).
- Continues climbing through coordination phase.

**Metric: `pair_valid_rate`**
- Dips to 0.74 at 60k, recovers to 0.81 by 120k.

**Metric: `Reward / Total reward (mean)`**
- Climbs to **4277 at 140k** — all-time high (previous best: 3516).

#### Phase 3: Noise + Delay (120-240k)
**Metric: `pair_valid_rate`**
- Dip at noise onset (122k): 0.81 → 0.64 at 144k. **Recovers to 0.94 by 180k.**
- First experiment to fully recover from delay introduction.
- Stays 0.88-0.94 through 200k. Gradual decline to 0.81 at 280k.

**Metric: `drone_0_triangulation`**
- Dip to 28 at 144k, recovers to **50.4 at 140k** (pre-noise peak).
- Stabilizes at 46-48 through 200k. Gradual decline to 41 at 280k.

**Metric: `drone_0_bbox_center`**
- Dip to 16.8 at 144k, recovers to 24-28 by 180-200k.
- Gradual decline to 21 at 280k, then 13-15 at 312-324k.

**Metric: `tracking_lost_fraction`**
- Active (1-3%) during 0-120k. **Drops to 0% at 124k** and stays there.
- The debounced truncation still gets defeated by noise — same issue as before,
  5 consecutive valid frames is too easy to achieve with noisy detections.

**Metric: `collision_fraction`**
- Peaks at 0.7% at 12k (early coordination learning), declines to 0.04% by 160k.
- Stays at 0.03-0.07% through 280k. Slight uptick to 0.07% at 312-324k.

#### Phase 4: Late Training (240-324k)
**Metric: `pair_valid_rate`**
- Slow continuous decline: 0.88 at 240k → 0.76 at 324k.

**Metric: `drone_0_bbox_center`**
- Decline from 26 at 240k → 15 at 312k. Concerning downward trend.

**Metric: `drone_0_triangulation`**
- Decline from 48 at 240k → 38 at 324k. Still healthy absolute values.

**Metric: `Reward / Total reward (mean)`**
- Decline from 3945 at 240k → 3221 at 320k.

**Metric: `Policy / Standard deviation`**
- Much healthier: stabilizes at **1.08-1.10** from 220k onwards.
- Previous experiments reached 1.32-1.49. The lower max_gimbal_rate may help
  by reducing the effective action magnitude.

**Metric: `drone_0_action_sum`**
- Stabilizes at -7.2 to -7.3 from 140k onwards. Lower than previous (-8 to -9).

### 3.2 Behavior Observations
Not tested yet.

## 4. Analysis
- **Best MA6 experiment by far.** No episode length collapse, no catastrophic curriculum
  shock, sustained high performance through delay phases. Peak metrics:
  pair_valid_rate 0.97, bbox_center 51, triangulation 50, total_reward 4277.
- **The 144k dip recovered fully** — pair_valid_rate bounced from 0.64 back to 0.94
  at 180k. Previous experiment (395280af3d) collapsed to 0.22 at 142k and never
  recovered. The slower delay ramp (40k instead of 20k) gave the policy enough
  time to adapt.
- **Policy std stabilized at 1.08** — significantly lower than all previous experiments.
  The combination of lower max_gimbal_rate (less action magnitude) and full-length
  episodes (stable reward signal) keeps exploration bounded.
- **Tracking-lost truncation still defeated by noise** — drops to 0% at 124k. The 5-step
  reacquire requirement is too low. Need 20-50 steps, or a different approach.
- **Late-stage decline (240-324k) is likely from curriculum overfitting, not dynamics.**
  The policy spent 100k steps (140-240k) at full delay with no dynamics randomization,
  deeply specializing its value function for that regime. When dynamics kicks in at 240k,
  the value predictions become wrong (value loss rises 0.003 → 0.006) and the policy
  can't adapt. Loss analysis confirms:
  - Entropy loss keeps decreasing (-0.004 → -0.019): policy grows MORE exploratory,
    not converging. Suggests continued uncertainty from non-stationarity.
  - Value loss rises after 240k (0.003 → 0.006): value function can't keep up.
  - Policy loss near zero: PPO clipping activates frequently, updates are marginal.
  - Instantaneous reward peaked at 8.6 (140k), declined to 6.8 (356k): more training
    actively hurts.
  - Best checkpoint is around 140-200k, not the final one.
  **The curriculum phases are too long.** Each phase allows deep specialization that makes
  the next transition a shock. The policy learned tracking + coordination + noise recovery
  in 160k. Everything after was maintaining/declining. A compressed schedule (20k per phase,
  ~180k total) would keep the value function plastic by continuously changing the MDP.
- **Collision penalty rises from 92-104k** (-0.32 to -0.53) coinciding with the
  coordination curriculum driving agents closer for triangulation geometry.
  The agents learn to approach each other for better triangulation angles but sometimes
  get too close. **Need a proximity penalty for getting too close to the target** — currently
  there's inter-agent CBF but no target-proximity penalty. Agents diving toward the target
  for better bbox/triangulation signal risk collision with it.

## 5. Decision / Todo
- **Add target proximity penalty**: Penalize drones that get too close to the target.
  Currently only inter-agent CBF exists. Agents are incentivized to approach the target
  for larger bbox and better triangulation geometry, with no downside for getting
  dangerously close.
- **Increase tracking_reacquire_steps** from 5 to 25-50 (1-2s sustained reacquisition)
  to make the truncation robust to noise.
- **Compress curriculum to ~180k total**: Each phase gets 20k steps instead of 40-80k.
  Prevents value function overfitting to any single regime. Proposed schedule:
  ```
  AgentVel:  0-20k     Safety:    0-20k     Tracking:  0-40k
  Target:    20-60k    Coord:     40-80k    Noise:     80-100k
  FixDelay:  100-120k  RndDelay:  120-140k  Dropout:   140-160k
  Dynamics:  160-180k
  ```
  Evidence: noise recovery took 16k steps in this run, basic tracking learned by 12k —
  20k per phase is sufficient.
- This config is the strongest MA6 baseline. Use as the starting point for ablation studies.
