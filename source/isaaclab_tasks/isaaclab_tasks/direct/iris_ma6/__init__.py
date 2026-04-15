# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-agent Iris drone environment V6 with rotor-level DroneController.

This module provides realistic quadcopter control with:
- Rotor-level physics (T = k_f * omega^2)
- First-order motor dynamics (tau=0.02s)
- Cascaded velocity -> attitude -> motor control
- Integrated gimbal control (tau=0.05s)
- Optical zoom control (tau=0.1s)
- Configurable aerodynamic effects
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Test environment for validating DroneController integration
gym.register(
    id="Isaac-Iris-MA6-Direct-Test-v0",
    entry_point=f"{__name__}.iris_ma_env6_test:IrisMA6TestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_ma_env6_test_cfg:IrisMA6TestEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IrisPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_mappo_rnn_cfg.yaml",
    },
)

# V1 environment for heading-frame observation redesign (ticket 022)
gym.register(
    id="Isaac-Iris-MA6-Direct-Test-v1",
    entry_point=f"{__name__}.iris_ma_env6_v1:IrisMA6V1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_ma_env6_v1_cfg:IrisMA6V1EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IrisPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_mappo_rnn_cfg.yaml",
    },
)
