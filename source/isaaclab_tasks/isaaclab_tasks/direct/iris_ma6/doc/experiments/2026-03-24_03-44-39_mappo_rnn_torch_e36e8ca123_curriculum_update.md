# Experiment: <2026-03-24_03-44-39_mappo_rnn_torch_e36e8ca123_curriculum_update>

**Commit**: `e36e8ca12388369861d6f07d4c945f48eedb3718`
**Date**: 2026-03-24_03-44-39
**Experiment ID**: `a1_with_aoi`
**Base**: `67dfcdf4ced25e14a09ea56a6ffeba06aa20ac12`<2026-03-23_17-33-02_mappo_rnn_torch_67dfcdf4ce_obs_redesign>

---

## 1. Hypothesis
Seperating the curriculums for ego max lin vel to target max lin vel would accelerate learning after 20k steps.

## 2. Configuration Delta
- Added `agent_velocity_start_step/end` curriculum 0k->40k (ego agent max lin vel 3->10 m/s, was 5->10m/s in the 
`moving_target` curriculum)
- Moved `safety_start_step/end` from 60k->100k to the beginning 0k->20k to prevent the agents from ever learning
unsafe actions
- Extended `moving_target_start_step/end` from 20k->80k to 40k->120k

## 3. Results

### 3.1 Training Curves
**Metric: `pair_valid_rate`**
- Slightly outperforms `67dfcdf4ce_obs_redesign` 
- The metric maxes out at around 24k, decrease until around 76k, starts to regain performance from around 80k.

**Metric: `drone_0_cbf_penalty` and `drone_1_cbf_penalty`**
- Monotonic decrease until 42k(reaching -0.7), increase until 70k(reaching -0.5), decrease afterwards.

**Metric: `drone_0_triangulation` and `drone_1_triangulation`**
- Outperforms `67dfcdf4ce_obs_redesign` more than twice. (20 vs 8)
- Monotonic increase of performance, whereas `67dfcdf4ce_obs_redesign` makes as deep drop at 100k-110k.
(Thanks to the bug fix in fixed-delay curriculum)

**Metric: `drone_0_action_sum` and `drone_1_action_sum`**
- Outperform `67dfcdf4ce_obs_redesign` from 12k steps and onwards.
- The *trend* is what's interesting: the action reward increases(penalty decreases) until 20k then decrease, whereas 
the reward decreases(penalty increases) monotonically in `67dfcdf4ce_obs_redesign`.

**Metric: `Total timesteps (mean)`**
- Significantly underforms(<330 steps out of 500) than `67dfcdf4ce_obs_redesign` (~495 steps out of 500)

### 3.2 Behavior Observations
Not tested yet.(Training still in process)

## 4. Analysis / Open question
- `pair_valid_rate` and `action_sum` both peaks at 20k. What is causing the dramatic change in the trend, especially same trend for `pair_valid_rate` across multiple experiments(`2026-03-23_17-33-02_mappo_rnn_torch_67dfcdf4ce_obs_redesign`, `2026-03-21_21-20-19_mappo_rnn_torch_0004dcd9df_max_lin_vel_curriculum`)? 
- Why is the `Total timesteps (mean)`, `drone_*_action_sum` trend so different?
- Add analysis to the losses(entropy, policy, value) from the training log

## 5. Decision / Todo
WIP