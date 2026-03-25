# Experiment: <2026-03-24_14-51-20_mappo_rnn_torch_5d2b8f8eca_rpy_instead_of_quat>

**Commit**: `5d2b8f8eca95c374f1b9a9d92bcc7f3920c33b4f`
**Date**: 2026-03-24_14-51-20
**Experiment ID**: `a1_with_aoi`
**Base**: `e36e8ca12388369861d6f07d4c945f48eedb3718`<2026-03-24_03-44-39_mappo_rnn_torch_e36e8ca123_curriculum_update>

---

## 1. Hypothesis
1-1. **RPY instead of quat in obs**: Performance collaps at 20k-80k might be because of tilt limits. Representing body attitudein Euler RPY instead of quat 
would help the policy to learn what happens if the drone is tilted too much, and stays away from it.
1-2. **Explicit collision penalty**: On top of just terminating the episode when a collision happens, give a large penalty.
1-3. **CBF param tuning**: It might have been too weak.

## 2. Configuration Delta
- **Explicit collision penalty**:
    - `collision_penalty_scale` = -100.0
- **CBF param tuning**:
    - `T`: 1.0 -> 3.0 (somewhat matching `iris_ma5`)
    - `lambda_cbf`: 1.0 -> 10.0 (matching `iris_ma5`)

## 3. Results(Training in process, now 70k/200k)

### 3.1 Training Curves
**Metric: `pair_valid_rate`**
- Still collaping at 20k-80k, disproving hypothesis 1-1 that RPY helps. We will keep it, however, since we can save one dimension.
- Peak value lower than the base `2026-03-24_03-44-39_mappo_rnn_torch_e36e8ca123_curriculum_update`

**Metric: `drone_0_cbf_penalty` and `drone_1_cbf_penalty`**
- Smoother decrease, similar values.

**Metric: `drone_0_action_sum` and `drone_1_action_sum`**
- Even higher than the base. (-6.5 vs -6.2)

**Metric: `Total timesteps (mean)`**
- Even lower than the base. (330 vs 250)

**Metric: `drone_0_bbox_center` and `drone_1_bbox_center`**
- It's almost half the base! (40 vs 20)

### 3.2 Behavior Observations
Somewhat OK? `./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_iris_mappo_rnn.py --task Isaac-Iris-MA6-Direct-Test-v0   --num_envs 16 --enable_cameras --checkpoint /home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6/2026-03-24_14-51-20_mappo_rnn_torch_5d2b8f8eca_rpy_instead_of_quat/checkpoints/best_agent.pt`
Agents track the target reasonably well, contrary to the training results.
We need quntified performance measure.

## 4. Analysis / Open question
- The policy already sees `ego_gimbal_yaw_body` and `ego_gimbal_pitch_body`, so there is not much can be done to inform the policy of the gimbal margin. 
That could mean 20k-80k collapse is not due to tilt limits.
- Stronger collision penalty seems to have pushed the `Total timesteps (mean)` metric lower.

## 5. Decision / Todo
- Keep RPY in obs since we can save one dimension.
- Reduce max tilt in velocity controller to 30 degrees. The gimbal controller now has around 15 degrees of margin. If the 20k-80k collapse persists, however, apply curriculum on max_tilt itself or limit max lin vel to 5 m/s flat.
