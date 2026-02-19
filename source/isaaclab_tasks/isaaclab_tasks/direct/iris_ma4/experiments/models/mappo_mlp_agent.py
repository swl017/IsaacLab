# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MAPPO_MLP agent — custom multi-agent PPO for MLP-only models.

Derived from MAPPO_RNN (mappo_rnn.py:29). When MLP models are used, the parent
class MAPPO_RNN detects no RNN specification and falls back to sequence_length=1
with flat minibatching (no BPTT). This class:
1. Enforces MLP-compatible defaults (no burn-in, no episode masking)
2. Provides a distinct class name for experiment logging
3. Validates that models have no RNN state
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence, Union

import gymnasium
import torch

from skrl import logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

# Import parent class — located in scripts/reinforcement_learning/skrl/
# The run_experiment.py script adds the correct sys.path
from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG


# MLP-specific default config
MAPPO_MLP_DEFAULT_CONFIG = copy.deepcopy(MAPPO_RNN_DEFAULT_CONFIG)
MAPPO_MLP_DEFAULT_CONFIG.update({
    "burn_in_steps": 0,
    "episode_start_mask_steps": 0,
    "sequence_length": 1,  # MLP: no sequence training
})


class MAPPO_MLP(MAPPO_RNN):
    """Multi-Agent PPO with MLP-only models (no recurrence).

    Inherits from MAPPO_RNN, which handles the no-RNN fallback:
    - get_specification() returns {} → sequence_length = 1
    - No RNN state stored/passed/reset
    - Standard (non-sequence) minibatch sampling
    """

    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        shared_observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
    ) -> None:
        # Override config with MLP defaults
        _cfg = copy.deepcopy(MAPPO_MLP_DEFAULT_CONFIG)
        if cfg is not None:
            _cfg.update(cfg)

        # Enforce no burn-in for MLP
        _cfg["burn_in_steps"] = 0
        _cfg["episode_start_mask_steps"] = 0

        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
            shared_observation_spaces=shared_observation_spaces,
        )

        # Validate no RNN state in models
        for uid in self.possible_agents:
            policy_spec = self.policies[uid].get_specification()
            if policy_spec.get("rnn"):
                raise ValueError(
                    f"MAPPO_MLP requires MLP-only models, but agent '{uid}' policy "
                    f"has RNN specification: {policy_spec['rnn']}"
                )
            value_spec = self.values[uid].get_specification()
            if value_spec.get("rnn"):
                raise ValueError(
                    f"MAPPO_MLP requires MLP-only models, but agent '{uid}' value "
                    f"has RNN specification: {value_spec['rnn']}"
                )

        logger.info("MAPPO_MLP: Initialized with MLP-only models (no recurrence)")
