# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision detection module for multi-agent environments."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict

from isaaclab.utils import configclass


@configclass
class CollisionDetectorCfg:
    """Configuration for collision detector."""

    min_safe_distance: float = 0.5
    """Minimum safe distance between agents (in meters)."""

    enable_velocity_based: bool = False
    """Whether to enable velocity-based collision prediction."""

    lookahead_time: float = 1.0
    """Time horizon for velocity-based collision prediction (seconds)."""

    collision_penalty_scale: float = -100.0
    """Scale factor for collision penalty in rewards."""


class CollisionDetector:
    """
    Collision detector for multi-agent drone environments.

    This class provides collision detection between agents, including:
    - Distance-based collision detection
    - Velocity-based collision prediction (optional)
    - Pairwise agent distance computation
    """

    def __init__(
        self,
        cfg: CollisionDetectorCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """
        Initialize the collision detector.

        Args:
            cfg: Configuration for the collision detector.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Preallocate distance matrix [N, A, A]
        self.distance_matrix = torch.zeros(
            num_envs, num_agents, num_agents, device=device
        )

        # Collision flags [N, A, A] - True if agents i and j are in collision
        self.collision_matrix = torch.zeros(
            num_envs, num_agents, num_agents, dtype=torch.bool, device=device
        )

    def compute_distances(
        self, agent_positions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pairwise distances between all agents.

        Args:
            agent_positions: Dictionary mapping agent_id to position tensor [N, 3].

        Returns:
            Distance matrix [N, A, A] where element [n, i, j] is the distance
            between agent i and agent j in environment n.
        """
        # Stack positions [N, A, 3]
        positions_stacked = torch.stack(list(agent_positions.values()), dim=1)

        # Compute pairwise distances using broadcasting
        # positions_stacked[:, :, None, :] -> [N, A, 1, 3]
        # positions_stacked[:, None, :, :] -> [N, 1, A, 3]
        diff = positions_stacked[:, :, None, :] - positions_stacked[:, None, :, :]  # [N, A, A, 3]
        distances = torch.norm(diff, dim=-1)  # [N, A, A]

        # Set diagonal to large value (self-distance)
        eye = torch.eye(self.num_agents, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        distances = torch.where(eye.bool(), torch.full_like(distances, float('inf')), distances)

        self.distance_matrix = distances
        return distances

    def detect_collisions(
        self, agent_positions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Detect collisions between agents based on minimum safe distance.

        Args:
            agent_positions: Dictionary mapping agent_id to position tensor [N, 3].

        Returns:
            Collision matrix [N, A, A] where element [n, i, j] is True if
            agents i and j are in collision in environment n.
        """
        distances = self.compute_distances(agent_positions)

        # Check if any pair is within minimum safe distance
        self.collision_matrix = distances < self.cfg.min_safe_distance

        return self.collision_matrix

    def compute_collision_penalties(
        self,
        agent_positions: Dict[str, torch.Tensor],
        agent_ids: list[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute collision penalties for each agent.

        Args:
            agent_positions: Dictionary mapping agent_id to position tensor [N, 3].
            agent_ids: List of agent identifiers.

        Returns:
            Dictionary mapping agent_id to collision penalty tensor [N].
        """
        # Detect collisions
        collision_matrix = self.detect_collisions(agent_positions)

        penalties = {}
        for i, agent_id in enumerate(agent_ids):
            # Check if this agent is in collision with any other agent
            # collision_matrix[:, i, :] gives collisions for agent i with all others
            # Exclude self-collision (diagonal)
            agent_collisions = collision_matrix[:, i, :].clone()
            agent_collisions[:, i] = False  # Exclude self

            # Penalty is -1 for each collision
            num_collisions = agent_collisions.sum(dim=1).float()
            penalties[agent_id] = -num_collisions

        return penalties

    def predict_collisions(
        self,
        agent_positions: Dict[str, torch.Tensor],
        agent_velocities: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict collisions based on current positions and velocities.

        Args:
            agent_positions: Dictionary mapping agent_id to position tensor [N, 3].
            agent_velocities: Dictionary mapping agent_id to velocity tensor [N, 3].

        Returns:
            Predicted collision matrix [N, A, A] at lookahead_time.
        """
        if not self.cfg.enable_velocity_based:
            return self.collision_matrix

        # Predict future positions
        future_positions = {}
        for agent_id in agent_positions.keys():
            future_positions[agent_id] = (
                agent_positions[agent_id]
                + agent_velocities[agent_id] * self.cfg.lookahead_time
            )

        # Detect collisions at predicted positions
        return self.detect_collisions(future_positions)

    def get_closest_agent_distance(
        self, agent_id: str, agent_ids: list[str]
    ) -> torch.Tensor:
        """
        Get the distance to the closest agent for a specific agent.

        Args:
            agent_id: Agent identifier.
            agent_ids: List of all agent identifiers.

        Returns:
            Distance to closest agent [N].
        """
        agent_idx = agent_ids.index(agent_id)

        # Get distances to all other agents
        distances_to_others = self.distance_matrix[:, agent_idx, :]  # [N, A]

        # Find minimum (excluding self)
        min_distance = distances_to_others.min(dim=1).values

        return min_distance

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset collision detection state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self.distance_matrix.zero_()
            self.collision_matrix.fill_(False)
        else:
            self.distance_matrix[env_ids] = 0.0
            self.collision_matrix[env_ids] = False
