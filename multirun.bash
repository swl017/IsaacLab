#!/bin/bash

# Define sweep parameters
DETECTION_FPS=(50.0)
DETECTION_MEAN_LATENCY=(0.05 0.2)
DETECTION_STD_LATENCY=(0.01 0.1)
EXPERIMENT_CASE=(0)
# Base command
BASE_CMD="python /home/usrg/IsaacPX4/IsaacLab/scripts/reinforcement_learning/skrl/train.py --algorithm MAPPO --headless"
RNN_CMD="python /home/usrg/IsaacPX4/IsaacLab/scripts/reinforcement_learning/skrl/train_iris_mappo_rnn.py"

# Loop through all combinations
for exp_case in "${EXPERIMENT_CASE[@]}"; do
    # detection_mean_latency=${DETECTION_MEAN_LATENCY[$exp_case]}
    # detection_std_latency=${DETECTION_STD_LATENCY[$exp_case]}
    # detection_fps=${DETECTION_FPS[0]}
    # echo "Running with learning_rate=${learning_rate}, detection_mean_latency=${detection_mean_latency}, detection_std_latency=${detection_std_latency}"
    # $BASE_CMD --num_envs 1024 --task Isaac-Iris-MA2-Direct-Delay-v0 \
    #     agent.agent.experiment.experiment_name="sweep14_noise_o_delay_x_obsdim29" \
    #     agent.trainer.timesteps=200000 \
    #     env.enable_delay_system=False \
    #     # agent.trainer.timesteps=300 \
    #     # env.action_sum_penalty_scale=$action_sum \
    # sleep 5
    # $BASE_CMD --num_envs 1024 --task Isaac-Iris-MA2-Direct-Delay-v0 \
    #     agent.agent.experiment.experiment_name="sweep14_noise_o_delay_x_obsdim35" \
    #     agent.trainer.timesteps=200000 \
    #     env.enable_delay_system=False \
    #     # agent.trainer.timesteps=300 \
    #     # env.action_sum_penalty_scale=$action_sum \
    # sleep 5

    $BASE_CMD --num_envs 1024 --task Isaac-Iris-MA2-Direct-Delay-v0 \
        agent.agent.experiment.experiment_name="sweep14_noise_o_delay_o_obsdim29" \
        agent.trainer.timesteps=200000 \
        # agent.trainer.timesteps=300 \
        # env.action_sum_penalty_scale=$action_sum \
    sleep 5

    $RNN_CMD --task Isaac-Iris-MA2-Direct-Delay-v0 \
        
    sleep 5

    # $RNN_CMD --task Isaac-Iris-MA2-Direct-Delayed-v0 \

    # sleep 5

done