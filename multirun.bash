#!/bin/bash
EXP_DIR="iris_ma4_mappo_rnn_tuning6_$(date +%Y%m%d_%H%M%S)"
export IDE_DEBUG_MODE=True

# Define sweep parameters
DETECTION_FPS=(50.0)
DETECTION_MEAN_LATENCY=(0.05 0.2)
DETECTION_STD_LATENCY=(0.01 0.1)
EXPERIMENT_CASE=(0)
# Base command
BASE_CMD="python /home/usrg/IsaacPX4/IsaacLab/scripts/reinforcement_learning/skrl/train.py --algorithm MAPPO --headless"
RNN_CMD="python /home/usrg/IsaacPX4/IsaacLab/scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"

# Loop through all combinations
for exp_case in "${EXPERIMENT_CASE[@]}"; do
    $RNN_CMD --num_envs 1024 --task Isaac-Iris-MA4-Direct-v0 \
        agent.agent.experiment.directory=\"$EXP_DIR\" \
        agent.agent.experiment.experiment_name="reward_rework" \
        agent.trainer.timesteps=200000

    sleep 5


done