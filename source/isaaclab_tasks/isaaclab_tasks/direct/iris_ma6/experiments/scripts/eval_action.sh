python3 ../evaluate.py \
  --checkpoint /home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6_action_penalty_20260503_070817/2026-05-03_07-08-48_mappo_rnn_torch_sum8_delta8/checkpoints/agent_120000.pt \
  --num_episodes 1 --num_envs 64 --trajectory-envs 8 \
  --no-timeseries --no-record-trajectory --headless \
  --output diag_sum8_delta8.json \
  --step 110000
