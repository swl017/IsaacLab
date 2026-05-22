python3 play_iris_mappo_rnn.py --task Isaac-Iris-MA6-Direct-Test-v0 --checkpoint /home/usrg/IsaacPX4/IsaacLab/logs/skrl/iris_ma6/2026-05-20_21-39-19_mappo_rnn_torch_a3f94fdc8b_ticket037_critic_priv_obs_no_dr/checkpoints/agent_80000.pt --step 40000 --enable_cameras --num_envs 2 env.episode_length_s=5.0

# My objective is sim-to-sim, or sim-to-real. The action, however, shows very jerky behavior, inappropriate for deployment.(Especially, the quadrotor attitude may get destabilized when the velocity command changes fast). I attributed this to a robust behavior, thinking that the noisy actions could be more robust to observation delay and noise than smooth, anticipating actions.

# The extended privileged observations doesn't seem to have helped.

# May be, the environment is just too difficult(Distance to the target, target speed, safety constraints, noise level, delay scale), or some params are not suitable(25 Hz policy rate), or it could even be that the optimal solution for the task not feasible.