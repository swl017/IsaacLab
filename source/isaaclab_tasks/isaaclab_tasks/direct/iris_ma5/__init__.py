# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-agent Iris drone environment V5 (N-agent support).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# V5 environment using DelaySystemV2 architecture with N-agent support
gym.register(
    id="Isaac-Iris-MA5-Direct-v0",
    entry_point=f"{__name__}.iris_ma_env5:IrisMAEnvV5",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_ma_env5_cfg:IrisMAEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IrisPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_mappo_rnn_cfg.yaml",
    },
)
