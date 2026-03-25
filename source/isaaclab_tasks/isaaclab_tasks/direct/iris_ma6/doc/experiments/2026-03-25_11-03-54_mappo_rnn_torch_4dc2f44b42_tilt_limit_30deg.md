# Experiment: 2026-03-25_11-03-54_mappo_rnn_torch_4dc2f44b42_tilt_limit_30deg

**Commit**: `4dc2f44b429fffe85c2d9e068a118934b3f56a46`
**Date**: 2026-03-25_11-03-54
**Experiment ID**: `a1_with_aoi`
**Base**: `5d2b8f8eca95c374f1b9a9d92bcc7f3920c33b4f`<2026-03-24_14-51-20_mappo_rnn_torch_5d2b8f8eca_rpy_instead_of_quat>
---

## 1. Hypothesis
Reducing max tilt would prevent performance drop at 20k-80k.

## 2. Configuration Delta
`max_tilt` 45 -> 30 (deg)

## 3. Results

### 3.1 Training Curves
Metric: `pair_valid_rate`
- Slightly higher peak.
- But even steeper downhill from 20k to almost 150k. Downhills in other experiements stopped at around 80k.

Metric: `all_invalid_rate`
- Massive peak at 120k (0.35 vs 0.13).
- This is the biggest difference in trend from other experiments.

Metric: `Total timesteps (mean)`
- Even lower at 200 steps (vs 250 steps in base)

Metric: `drone_0_triangulation`
- 2~3x lower than previous runs.
- Even the learning progress is slower.
- This is the second biggest difference in trend from other experiments.

Metric: `drone_0_bbox_center`
- Scored 1.5x high at 20k, gap decreased and vanished at 110k.

Metric: `collision_per_env`
- Slightly reduced than the base experiment.

### 3.2 Behavior Observations
`./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_iris_mappo_rnn.py --task Isaac-Iris-MA6-Direct-Test-v0   --num_envs 16 --enable_cameras --checkpoint /home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6/2026-03-24_14-51-20_mappo_rnn_torch_5d2b8f8eca_rpy_instead_of_quat/checkpoints/best_agent.pt`
Agents track the target reasonably well, contrary to the training results.
We need quntified performance measures.

## 4. Analysis / Open question
- So `pair_valid_rate` reveals 20k-80k collapse is not becuase of tilt limits. The inherent lag by tilt dynamics could be the root cause. I need more ideas.
- `all_invalid_rate` at 100k-120k also hints that reduced manueverability from reduced max tilt make the agents lose track more easily... but how did `collision_per_env` improved(reduced) even with less manueverable drones?
- Still the `play.sh` script shows reasonable tracking performances. Something feels wrong.
- The tracking curriculum is the only variable left for investigation. Either extend it to 150k, or do it adaptively with performance-gated curriculum.

## 5. Decision / Todo
- Revert max tilt back to 45 degrees.
- Extend the tracking curriculum(formation) to 150k.