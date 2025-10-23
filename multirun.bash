#!/bin/bash

# Define sweep parameters
ACTION_SUM_SCALES=(-1.0)
# ACTION_WEIGHT=("'[1, 1, 5, 0.5, 0.03, 0.03, 0.01]'" "'[1, 1, 5, 0.5, 0.3, 0.3, 0.01]'" "'[1, 1, 5, 0.5, 3, 3, 0.01]'")
ACTION_WEIGHT=(0 1 2)
ACTION_DELTA_SCALES=(-0.01 -0.1 -1.0)
# ACTION_DELTA_WEIGHT=("[1, 1, 5, 0.5, 0.03, 0.03, 0.01]" "[1, 1, 5, 0.5, 0.3, 0.3, 0.01]" "[1, 1, 5, 0.5, 3, 3, 0.01]")
ACTION_DELTA_WEIGHT=(0 1 2)
bbox_reward_shape_width_list=(0.3 0.6 0.8)
# Base command
BASE_CMD="python /home/usrg/IsaacPX4/IsaacLab/scripts/reinforcement_learning/skrl/train.py --task Isaac-Iris-MA2-Direct-v0 --algorithm MAPPO --num_envs 1024 --headless --video --video_interval 20000 --video_length 1000 --enable_cameras"

# Loop through all combinations
for action_weight in "${ACTION_WEIGHT[@]}"; do
    for action_delta_weight in "${ACTION_DELTA_WEIGHT[@]}"; do
        echo "Running with action_weight=${action_weight}, action_delta_weight=${action_delta_weight}"
        
        $BASE_CMD \
            env.aw=$action_weight \
            env.adw=$action_delta_weight \
            agent.agent.experiment.experiment_name="sweep_05_aw${action_weight}_adw${action_delta_weight}" \
            # env.lin_vel_penalty_scale=$lin_vel \
            # env.action_sum_penalty_scale=$action_sum \
        
        sleep 5
    done
done