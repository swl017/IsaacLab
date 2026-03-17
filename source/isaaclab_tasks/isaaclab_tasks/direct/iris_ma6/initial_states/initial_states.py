# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper class for initial states generation in iris_ma6.

This module provides a simple API for the environment's reset logic,
following the pattern established by iris_ma5's Randomizer class.
"""

from __future__ import annotations

import torch
from typing import Optional

from .initial_states_cfg import InitialStatesCfg, InitialStatesResult
from .initial_states_generator import InitialStatesGenerator


class InitialStates:
    """Wrapper class for initial states generation in iris_ma6.

    Provides a simplified API for generating agent and target initial states
    with curriculum-driven randomization.

    Usage in environment:
        ```python
        # Initialization
        self._initial_states = InitialStates(
            cfg=self.cfg.initial_states,
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
        )

        # In _reset_idx
        result = self._initial_states.generate(
            env_ids=env_ids,
            curriculum_progress=self._curriculum_progress,
        )

        # Apply to robots
        for idx, agent_id in enumerate(self.cfg.possible_agents):
            robot = self._robots[agent_id]
            root_pose = torch.cat([
                result.agent_positions[:, idx],
                result.agent_orientations[:, idx],
            ], dim=-1)
            root_vel = torch.cat([
                result.agent_linear_velocities[:, idx],
                result.agent_angular_velocities[:, idx],
            ], dim=-1)
            robot.write_root_pose_to_sim(root_pose, env_ids)
            robot.write_root_velocity_to_sim(root_vel, env_ids)

            # Apply gimbal joint states
            gimbal_pos = result.gimbal_joint_positions[:, idx]
            joint_vel = torch.zeros_like(gimbal_pos)
            robot.write_joint_state_to_sim(gimbal_pos, joint_vel, ...)

        # Apply to target
        target_pose = torch.cat([
            result.target_positions,
            result.target_orientations,
        ], dim=-1)
        self.target.write_root_pose_to_sim(target_pose, env_ids)
        self.target.write_root_velocity_to_sim(result.target_velocities, env_ids)
        ```
    """

    def __init__(
        self,
        cfg: InitialStatesCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the wrapper.

        Args:
            cfg: Initial states configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device for tensor operations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        self._generator = InitialStatesGenerator(
            cfg=cfg,
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
        )

    def generate(
        self,
        env_ids: Optional[torch.Tensor] = None,
        curriculum_progress: float = 0.5,
    ) -> InitialStatesResult:
        """Generate initial states for specified environments.

        This is the primary entry point for the environment's _reset_idx method.

        Args:
            env_ids: Environment indices to generate for (None = all).
            curriculum_progress: Curriculum progress [0, 1].
                - 0.0: Easy (close targets, low velocities, all pointing)
                - 1.0: Hard (far targets, high velocities, random gimbal)

        Returns:
            InitialStatesResult containing all state tensors.
        """
        return self._generator.generate(
            env_ids=env_ids,
            curriculum_progress=curriculum_progress,
        )

    def update_config(self, **kwargs):
        """Update configuration parameters dynamically.

        Useful for adjusting parameters during training without
        recreating the generator.

        Args:
            **kwargs: Configuration parameters to update.
                Example: update_config(target_distance_max=150.0)

        Raises:
            ValueError: If an unknown parameter is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
