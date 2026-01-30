# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified safety management for multi-agent drone environments."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Dict

from isaaclab.utils import configclass

from .collision_detector import CollisionDetector, CollisionDetectorCfg
from .ttc_computer import TTCComputer, TTCComputerCfg


@configclass
class SafetyManagerCfg:
    """Configuration for unified safety manager."""

    collision_cfg: CollisionDetectorCfg = CollisionDetectorCfg()
    """Configuration for collision detection."""

    ttc_cfg: TTCComputerCfg = TTCComputerCfg()
    """Configuration for TTC computation."""

    enable_collision_detection: bool = True
    """Whether to enable collision detection."""

    enable_ttc_computation: bool = True
    """Whether to enable TTC computation."""


class SafetyManager:
    """
    Unified safety manager for multi-agent drone environments.

    This class provides a unified interface for:
    - Inter-agent collision detection
    - Time-to-collision computation for target tracking
    - Safety-based reward penalties

    The manager coordinates both collision detection (for agent-agent safety)
    and TTC computation (for camera-target safety) in a single module.
    """

    def __init__(
        self,
        cfg: SafetyManagerCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """
        Initialize the safety manager.

        Args:
            cfg: Configuration for the safety manager.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Initialize collision detector
        if cfg.enable_collision_detection:
            self.collision_detector = CollisionDetector(
                cfg.collision_cfg, num_envs, num_agents, device
            )
        else:
            self.collision_detector = None

        # Initialize TTC computers (one per agent for camera-target TTC)
        if cfg.enable_ttc_computation:
            self.ttc_computers = {
                i: TTCComputer(cfg.ttc_cfg, num_envs, device)
                for i in range(num_agents)
            }
        else:
            self.ttc_computers = None

    def compute_collision_penalties(
        self,
        agent_positions: Dict[str, torch.Tensor],
        agent_ids: list[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute collision penalties for all agents.

        Args:
            agent_positions: Dictionary mapping agent_id to position tensor [N, 3].
            agent_ids: List of agent identifiers.

        Returns:
            Dictionary mapping agent_id to collision penalty tensor [N].
        """
        if not self.cfg.enable_collision_detection:
            return {agent_id: torch.zeros(self.num_envs, device=self.device) for agent_id in agent_ids}

        return self.collision_detector.compute_collision_penalties(
            agent_positions, agent_ids
        )

    def compute_ttc_penalties(
        self,
        agent_bboxes: Dict[str, torch.Tensor],
        agent_bbox_valid: Dict[str, torch.Tensor],
        agent_focal_lengths: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        agent_ids: list[str],
        dt: float,
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute TTC penalties for all agents.

        Args:
            agent_bboxes: Dictionary mapping agent_id to bbox tensor [N, 4] (x, y, w, h normalized).
            agent_bbox_valid: Dictionary mapping agent_id to validity mask [N].
            agent_focal_lengths: Dictionary mapping agent_id to tuple (fx, fy) [N].
            agent_ids: List of agent identifiers.
            dt: Timestep (seconds).

        Returns:
            Dictionary mapping agent_id to tuple (phi, tau) where:
            - phi: TTC penalty in [0, 1] [N]
            - tau: Estimated TTC in seconds [N]
        """
        if not self.cfg.enable_ttc_computation:
            zero_penalty = torch.zeros(self.num_envs, device=self.device)
            inf_ttc = torch.full((self.num_envs,), float('inf'), device=self.device)
            return {agent_id: (zero_penalty, inf_ttc) for agent_id in agent_ids}

        ttc_results = {}
        for i, agent_id in enumerate(agent_ids):
            bbox = agent_bboxes[agent_id]
            bbox_w = bbox[:, 2]  # width
            bbox_h = bbox[:, 3]  # height
            valid = agent_bbox_valid[agent_id]
            fx, fy = agent_focal_lengths[agent_id]

            phi, tau = self.ttc_computers[i].compute_ttc(
                bbox_w, bbox_h, valid, fx, fy, dt
            )
            ttc_results[agent_id] = (phi, tau)

        return ttc_results

    def compute_all_safety_penalties(
        self,
        agent_positions: Dict[str, torch.Tensor],
        agent_bboxes: Dict[str, torch.Tensor],
        agent_bbox_valid: Dict[str, torch.Tensor],
        agent_focal_lengths: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        agent_ids: list[str],
        dt: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute all safety penalties (collision + TTC) for all agents.

        Args:
            agent_positions: Dictionary mapping agent_id to position tensor [N, 3].
            agent_bboxes: Dictionary mapping agent_id to bbox tensor [N, 4].
            agent_bbox_valid: Dictionary mapping agent_id to validity mask [N].
            agent_focal_lengths: Dictionary mapping agent_id to tuple (fx, fy) [N].
            agent_ids: List of agent identifiers.
            dt: Timestep (seconds).

        Returns:
            Dictionary mapping agent_id to dictionary of penalties:
            - "collision": Collision penalty [N]
            - "ttc_penalty": TTC penalty phi [N]
            - "ttc_value": TTC estimate tau [N]
        """
        penalties = {}

        # Compute collision penalties
        collision_penalties = self.compute_collision_penalties(agent_positions, agent_ids)

        # Compute TTC penalties
        ttc_results = self.compute_ttc_penalties(
            agent_bboxes, agent_bbox_valid, agent_focal_lengths, agent_ids, dt
        )

        # Combine results
        for agent_id in agent_ids:
            phi, tau = ttc_results[agent_id]
            penalties[agent_id] = {
                "collision": collision_penalties[agent_id],
                "ttc_penalty": phi,
                "ttc_value": tau,
            }

        return penalties

    def get_distance_matrix(self) -> torch.Tensor | None:
        """
        Get the current pairwise distance matrix between agents.

        Returns:
            Distance matrix [N, A, A] or None if collision detection is disabled.
        """
        if self.collision_detector is None:
            return None
        return self.collision_detector.distance_matrix

    def get_collision_matrix(self) -> torch.Tensor | None:
        """
        Get the current collision matrix between agents.

        Returns:
            Collision matrix [N, A, A] or None if collision detection is disabled.
        """
        if self.collision_detector is None:
            return None
        return self.collision_detector.collision_matrix

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset safety manager state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if self.collision_detector is not None:
            self.collision_detector.reset(env_ids)

        if self.ttc_computers is not None:
            for ttc_computer in self.ttc_computers.values():
                ttc_computer.reset(env_ids)

    def reset_ttc_with_state(
        self,
        env_ids: torch.Tensor,
        agent_bboxes: Dict[str, torch.Tensor],
        agent_bbox_valid: Dict[str, torch.Tensor],
        agent_focal_lengths: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        agent_ids: list[str],
    ):
        """
        Reset TTC computers with current bbox and focal length state.

        Args:
            env_ids: Environment indices to reset [K].
            agent_bboxes: Dictionary mapping agent_id to bbox tensor [N, 4].
            agent_bbox_valid: Dictionary mapping agent_id to validity mask [N].
            agent_focal_lengths: Dictionary mapping agent_id to tuple (fx, fy) [N].
            agent_ids: List of agent identifiers.
        """
        if self.ttc_computers is None:
            return

        for i, agent_id in enumerate(agent_ids):
            bbox = agent_bboxes[agent_id]
            bbox_w = bbox[:, 2]
            bbox_h = bbox[:, 3]
            valid = agent_bbox_valid[agent_id]
            fx, fy = agent_focal_lengths[agent_id]

            self.ttc_computers[i].reset_with_current_state(
                env_ids, bbox_w, bbox_h, valid, fx, fy
            )
