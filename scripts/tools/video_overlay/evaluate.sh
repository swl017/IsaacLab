./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/experiments/evaluate.py \
    --experiment a3_noisy_reward \
    --checkpoint /home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma5/2026-03-08_02-18-20_a3_noisy_reward_seed42/checkpoints/best_agent.pt \
    --enable_cameras \
    --num_episodes 1 \
    --num_envs 4 \
    --record-trajectory \
    --record-video raw_demo.mp4 \
    --camera-mode chase \
    --camera-smoothing 0.08 \
    --video-fps 25 \
    --video-resolution 1920 1080 \
    --output metrics.json


# --experiment a1_with_aoi \
# --checkpoint /home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma5_ablations_2026-02-28/2026-02-24_00-45-58_a1_with_aoi_seed42/checkpoints/best_agent.pt \
