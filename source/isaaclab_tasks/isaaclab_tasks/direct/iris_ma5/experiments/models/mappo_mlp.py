# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLP-only policy and value networks for A2 ablation (MAPPO-MLP vs MAPPO-RNN).

These models replace the GRU-based MAPPORNNPolicy/MAPPORNNValue with pure MLP
networks. They return empty RNN specifications so the MAPPO_RNN agent falls back
to sequence_length=1 (standard PPO minibatching).
"""

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class MAPPOMLPPolicy(GaussianMixin, Model):
    """MLP-only Gaussian policy for MAPPO.

    Architecture: Linear → ReLU → [Linear → ReLU] × (num_hidden_layers - 1) → Linear → actions
    Same total parameter budget as RNN variant when hidden_size=256, num_hidden_layers=3.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_size: int = 256,
        num_hidden_layers: int = 3,
        num_envs: int = 1,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 0.7,
        sequence_length: int = 1,  # Unused, kept for API compatibility
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.num_envs = num_envs

        # Build MLP
        layers = []
        input_size = self.num_observations
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.net = nn.Sequential(*layers)

        # Policy head
        self.policy_layer = nn.Linear(hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.ones(self.num_actions, device=device) * initial_log_std
        )

    def get_specification(self):
        """No RNN state — agent will use sequence_length=1."""
        return {}

    def compute(self, inputs, role):
        states = inputs["states"]
        # Handle 3D input (B, S, obs) by flattening to (B*S, obs)
        is_3d = states.dim() == 3
        if is_3d:
            B, S, F = states.shape
            states = states.reshape(B * S, F)

        x = self.net(states)
        mean = self.policy_layer(x)
        return mean, self.log_std_parameter, {}


class MAPPOMLPValue(DeterministicMixin, Model):
    """MLP-only deterministic value network for MAPPO.

    Architecture mirrors the policy network but outputs a single scalar value.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_size: int = 256,
        num_hidden_layers: int = 3,
        num_envs: int = 1,
        sequence_length: int = 1,  # Unused, kept for API compatibility
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.num_envs = num_envs

        # Build MLP
        layers = []
        input_size = self.num_observations
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.net = nn.Sequential(*layers)

        # Value head
        self.value_layer = nn.Linear(hidden_size, 1)

    def get_specification(self):
        """No RNN state — agent will use sequence_length=1."""
        return {}

    def compute(self, inputs, role):
        states = inputs["states"]
        # Handle 3D input (B, S, obs) by flattening to (B*S, obs)
        is_3d = states.dim() == 3
        if is_3d:
            B, S, F = states.shape
            states = states.reshape(B * S, F)

        x = self.net(states)
        value = self.value_layer(x)
        return value, {}
