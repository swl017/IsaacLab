# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CBF Manager module - unified facade for CBF safety components.

Coordinates:
- CPA reward shaper (training-time soft penalty)
- Deployment filter (deployment-time hard constraint)
- Diagnostics (training metrics)
- Collision detection (episode termination)

Ref: safety_spec.md Section 3, 6
"""

from __future__ import annotations

import torch
from typing import Dict

from .cbf_cfg import CBFManagerCfg
from .cbf_diagnostics import CBFDiagnostics
from .cpa_reward_shaper import CPARewardShaper
from .deploy_filter import RobustDeploymentFilter


class CBFManager:
    """Unified facade for CBF safety components.

    The CBF safety architecture has two distinct modes:

    **Training mode** (enable_training_penalty=True, enable_deployment_filter=False):
    - Actions are UNFILTERED - policy experiences full consequences
    - CPA penalty provides soft reward shaping using GT state
    - Episode termination on actual collision provides backstop

    **Deployment mode** (enable_training_penalty=False, enable_deployment_filter=True):
    - Actions pass through hard CBF filter before reaching actuators
    - Filter uses delayed observations with inflated margin
    - MAPPO-RNN policy is primary collision avoidance, filter is safety net

    Usage (training):
        cbf_manager = CBFManager(cfg, num_envs, num_agents, device)

        # In _get_rewards():
        penalty = cbf_manager.compute_training_penalty(gt_positions, cmd_velocities, dt)
        reward = task_reward - cfg.cpa_cfg.lambda_cbf * penalty

        # In _get_dones():
        collided = cbf_manager.check_collisions(gt_positions)
        terminated = terminated | collided

        # In _reset_idx():
        cbf_manager.reset(env_ids)

    Usage (deployment):
        v_safe, info = cbf_manager.filter_actions(v_nom, delayed_positions, delayed_velocities)
    """

    def __init__(
        self,
        cfg: CBFManagerCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the CBF manager.

        Args:
            cfg: Configuration for the CBF manager.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Initialize components based on configuration
        if cfg.enable_training_penalty:
            self.cpa_shaper = CPARewardShaper(
                cfg.cpa_cfg, num_envs, num_agents, device
            )
        else:
            self.cpa_shaper = None

        if cfg.enable_deployment_filter:
            self.deploy_filter = RobustDeploymentFilter(
                cfg.deploy_cfg, num_envs, num_agents, device
            )
        else:
            self.deploy_filter = None

        # Always initialize diagnostics
        self.diagnostics = CBFDiagnostics(
            cfg.diagnostics_cfg, num_envs, num_agents, device
        )

        # Pre-compute collision threshold squared
        self.collision_dist_sq = cfg.collision_distance**2

    def compute_training_penalty(
        self,
        gt_positions: torch.Tensor,
        commanded_velocities: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute CPA penalty from GT state for reward shaping.

        This penalty is used ONLY during training. It provides dense,
        velocity-aware gradient information about collision geometry.

        Args:
            gt_positions: Ground-truth positions (E, N, 3).
            commanded_velocities: Commanded velocities (E, N, 3).
            dt: Timestep duration in seconds.

        Returns:
            penalty: Summed CPA violation per environment (E,).
            Returns zeros if CPA shaper is disabled.
        """
        if self.cpa_shaper is None:
            return torch.zeros(self.num_envs, device=self.device)

        return self.cpa_shaper.compute_penalty(
            gt_positions, commanded_velocities, dt
        )

    def filter_actions(
        self,
        v_nom: torch.Tensor,
        delayed_positions: torch.Tensor,
        delayed_velocities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Apply hard CBF filter (deployment only).

        The deployment filter ensures forward invariance of the safe set
        under worst-case delay assumptions.

        Args:
            v_nom: Nominal velocities from policy (E, N, 3).
            delayed_positions: Latest received positions (E, N, 3).
            delayed_velocities: Latest received velocities (E, N, 3).
                If None, assumes zero (most conservative).

        Returns:
            v_safe: Filtered safe velocities (E, N, 3).
            info: Dictionary with filter diagnostics.

        Raises:
            RuntimeError: If deployment filter is not enabled.
        """
        if self.deploy_filter is None:
            raise RuntimeError(
                "Deployment filter is not enabled. "
                "Set enable_deployment_filter=True in CBFManagerCfg."
            )

        return self.deploy_filter.filter(
            v_nom, delayed_positions, delayed_velocities
        )

    def check_collisions(
        self,
        gt_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Check for actual collisions using GT positions.

        An environment has a collision if any pair of agents is closer
        than the collision distance threshold.

        Args:
            gt_positions: Ground-truth positions (E, N, 3).

        Returns:
            collided: Boolean tensor (E,) indicating collision per environment.
        """
        if not self.cfg.enable_collision_termination:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Compute pairwise distances
        # (E, N, 1, 3) - (E, 1, N, 3) -> (E, N, N, 3)
        diff = gt_positions[:, :, None, :] - gt_positions[:, None, :, :]
        dist_sq = (diff * diff).sum(dim=-1)  # (E, N, N)

        # Set diagonal to large value (self-distance)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        dist_sq = torch.where(
            eye.unsqueeze(0),
            torch.full_like(dist_sq, float("inf")),
            dist_sq
        )

        # Check if any pair is in collision
        # min_dist_sq: (E, N, N) -> min over all pairs -> (E,)
        min_dist_sq = dist_sq.view(self.num_envs, -1).min(dim=-1).values
        collided = min_dist_sq < self.collision_dist_sq

        return collided

    def get_diagnostics(
        self,
        gt_positions: torch.Tensor,
    ) -> Dict[str, float]:
        """Get training diagnostics from GT positions.

        Args:
            gt_positions: Ground-truth positions (E, N, 3).

        Returns:
            Dictionary of diagnostic metrics.
        """
        return self.diagnostics.compute_metrics(gt_positions)

    def get_cpa_info(
        self,
        gt_positions: torch.Tensor,
        commanded_velocities: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get detailed CPA information for debugging.

        Args:
            gt_positions: Ground-truth positions (E, N, 3).
            commanded_velocities: Commanded velocities (E, N, 3).

        Returns:
            Dictionary with CPA metrics. Empty if CPA shaper disabled.
        """
        if self.cpa_shaper is None:
            return {}

        return self.cpa_shaper.get_cpa_info(gt_positions, commanded_velocities)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset internal state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if self.cpa_shaper is not None:
            self.cpa_shaper.reset(env_ids)

        if self.deploy_filter is not None:
            self.deploy_filter.reset(env_ids)

        self.diagnostics.reset(env_ids)

    @property
    def D_s(self) -> float:
        """Physical safety distance (training)."""
        return self.cfg.cpa_cfg.D_s

    @property
    def D_deploy(self) -> float:
        """Inflated safety distance (deployment)."""
        return self.cfg.deploy_cfg.D_deploy

    @property
    def lambda_cbf(self) -> float:
        """CPA penalty weight."""
        return self.cfg.cpa_cfg.lambda_cbf
