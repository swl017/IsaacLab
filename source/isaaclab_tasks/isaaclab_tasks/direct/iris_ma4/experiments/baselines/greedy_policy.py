# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted equiangular placement policy for greedy baseline.

This policy positions agents in a fixed-radius orbit around the target,
maintaining equal angular separation. No learning involved — purely geometric.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


class GreedyEquiangularPolicy:
    """Greedy heuristic: orbit target at fixed distance with equiangular spacing.

    Each agent moves toward its assigned position on a circle of radius
    ``orbit_radius`` centered on the target at ``orbit_height``. Agents
    are spaced evenly around the circle (2*pi/N apart).

    The policy outputs normalized actions in [-1, 1] compatible with the
    iris_ma4 environment action space: [vx, vy, vz, yaw_rate,
    gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate].
    """

    def __init__(
        self,
        num_agents: int,
        orbit_radius: float = 20.0,
        orbit_height: float = 15.0,
        approach_gain: float = 1.0,
        max_lin_vel: float = 10.0,
        device: str = "cuda:0",
    ):
        self.num_agents = num_agents
        self.orbit_radius = orbit_radius
        self.orbit_height = orbit_height
        self.approach_gain = approach_gain
        self.max_lin_vel = max_lin_vel
        self.device = device

        # Pre-compute equiangular positions (relative angles)
        self.angles = [2.0 * math.pi * i / num_agents for i in range(num_agents)]

    def compute_actions(
        self,
        agent_positions: Dict[str, torch.Tensor],
        target_position: torch.Tensor,
        agent_orientations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute scripted actions for all agents.

        Args:
            agent_positions: {agent_id: [N, 3]} current agent world positions.
            target_position: [N, 3] current target world position.
            agent_orientations: {agent_id: [N, 4]} current agent quaternions.

        Returns:
            {agent_id: [N, 7]} normalized actions in [-1, 1].
        """
        actions = {}
        for idx, (agent_id, pos) in enumerate(agent_positions.items()):
            N = pos.shape[0]
            target_xy = target_position[:, :2]  # [N, 2]

            # Desired position on orbit circle
            angle = self.angles[idx]
            offset = torch.tensor(
                [math.cos(angle), math.sin(angle)],
                device=self.device, dtype=pos.dtype,
            ).unsqueeze(0) * self.orbit_radius  # [1, 2]
            desired_xy = target_xy + offset  # [N, 2]
            desired_z = torch.full(
                (N, 1), self.orbit_height, device=self.device, dtype=pos.dtype
            )
            desired_pos = torch.cat([desired_xy, desired_z], dim=-1)  # [N, 3]

            # Proportional velocity command toward desired position
            error = desired_pos - pos  # [N, 3]
            vel_cmd = self.approach_gain * error

            # Normalize to [-1, 1] action space
            vel_cmd = vel_cmd / self.max_lin_vel
            vel_cmd = torch.clamp(vel_cmd, -1.0, 1.0)

            # Build full action: [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
            action = torch.zeros(N, 7, device=self.device, dtype=pos.dtype)
            action[:, 0:3] = vel_cmd
            # Other actions stay at 0 (no yaw, no gimbal control, no zoom)

            actions[agent_id] = action

        return actions
