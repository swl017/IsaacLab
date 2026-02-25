# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Main randomizer class for iris_ma5 environment.

This module provides the Randomizer class that wraps the distance-based
formation generator for use in the environment's reset logic.
"""

from __future__ import annotations

import math
import torch
from typing import Optional

from .formation_cfg import DistanceBasedFormationCfg, FormationResult
from .distance_based_generator import DistanceBasedFormationGenerator


class Randomizer:
    """
    Main randomizer class for iris_ma5 environment.

    Provides a simplified API for generating agent formations and target positions
    with distance as the primary curriculum difficulty factor.

    Usage in environment:
        ```python
        # Initialization
        self.randomizer = Randomizer(
            num_envs=self.num_envs,
            num_agents=len(self.cfg.possible_agents),
            device=self.device,
        )

        # In _reset_idx
        result = self.randomizer.generate_formation_and_target(
            env_ids=env_ids,
            scale_factor=self.progress_tracking,
            formation_type="planar",  # or None for random
        )

        formation_data = result.agent_root_states
        target_positions = result.target_position
        ```
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        cfg: Optional[DistanceBasedFormationCfg] = None,
        max_lin_vel: float = 10.0,
        max_yaw_rate: float = math.radians(90.0),
    ):
        """Initialize the randomizer.

        Args:
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Torch device for tensor operations.
            cfg: Formation configuration. If None, uses defaults.
            max_lin_vel: Maximum linear velocity for velocity initialization.
            max_yaw_rate: Maximum yaw rate for velocity initialization.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        if cfg is None:
            cfg = DistanceBasedFormationCfg()

        self.cfg = cfg

        self.formation = DistanceBasedFormationGenerator(
            cfg=cfg,
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
            max_lin_vel=max_lin_vel,
            max_yaw_rate=max_yaw_rate,
        )

    def generate_formation_and_target(
        self,
        env_ids: Optional[torch.Tensor] = None,
        scale_factor: float = 0.5,
        formation_type: Optional[str] = None,
    ) -> FormationResult:
        """
        Generate agent formation and target position.

        This is the primary entry point for the environment's _reset_idx method.

        Args:
            env_ids: Environment indices to generate for (None = all).
            scale_factor: Curriculum scale (0.0 to 1.0).
                - At 0.0: distance = distance_min (easy, close targets)
                - At 1.0: distance = distance_max (hard, far targets)
            formation_type: Formation type ("planar", "grid", "line") or None for random.

        Returns:
            FormationResult containing:
                - agent_root_states: [num_envs, num_agents, 13] tensor
                - target_position: [num_envs, 3] tensor
                - formation_center: [num_envs, 3] tensor
                - distances_to_target: [num_envs, num_agents] tensor
                - formation_types: list of formation type strings
        """
        return self.formation.generate(
            env_ids=env_ids,
            scale_factor=scale_factor,
            formation_type=formation_type,
        )

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update (e.g., distance_min=15.0).
        """
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
