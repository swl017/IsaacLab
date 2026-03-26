# Experiment: 2026-03-26_04-39-35_mappo_rnn_torch_3737169494_tracking_curriculum

**Commit**: `37371694949b81ac5996d087a04dda5b2aa4fb03`
**Date**: YYYY-MM-DD
**Experiment ID**: `<registry_id>` (from experiment_registry.py)
**Base**: `4dc2f44b429fffe85c2d9e068a118934b3f56a46`2026-03-25_11-03-54_mappo_rnn_torch_4dc2f44b42_tilt_limit_30deg

---

## 1. Hypothesis
Extending the tracking curriculum would relieve 20k-80k collapse.

## 2. Configuration Delta
tracking_end_step: 60k -> 150k

## 3. Results
Failed catastrophically.

### 3.1 Training Curves
Metric: `pair_valid_rate`
- Starts small, never peaks, converges to zero.

Metric: `all_invalid_rate`
- 0.7-0.9

Metric: `Total timesteps (mean)`
- Starts small, never peaks, converges to zero.

Metric: `drone_0_triangulation`
- Starts small, never peaks, converges to zero.

Metric: `drone_0_bbox_center`
- Starts small, never peaks, converges to zero.

Metric: `collision_per_env`
- The only two metrics this experiment wins.

Metric: `drone_0_cbf_penalty`
- The only two metrics this experiment wins.

Metric: `drone_0_action_sum`
- Huge minus (-60) where as other experiments stays well above -10.

Metric: `drone_0_action_delta`
- Very steep peak and vallies.


### 3.2 Behavior Observations
Not yet tested(Training in process 110k)

## 4. Analysis / Open question
The initial conditions stays almost the same while vehicle max lin vel increase, making the policy act focus only on avoiding collisions.
May be we should shorten the tracking curriculum rather than extending it.

## 5. Decision / Todo
Sync the tracking curriculum with ego motion curriculum(0k -> 60k)