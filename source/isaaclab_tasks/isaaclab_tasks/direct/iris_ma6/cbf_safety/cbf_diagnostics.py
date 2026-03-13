# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CBF Diagnostics module for training metrics.

Tracks safety metrics during training using ground-truth state.
All computations are fully batched for GPU efficiency.

Ref: safety_spec.md Section 4.3
"""

from __future__ import annotations

import torch
from typing import Dict

from .cbf_cfg import CBFDiagnosticsCfg


class CBFDiagnostics:
    """Tracks safety metrics during training. All computed from GT.

    Metrics tracked:
    - safety/min_separation_mean: Mean of per-env minimum pairwise distance
    - safety/min_separation_min: Global minimum separation
    - safety/collision_fraction: Fraction of envs with collision (dist < D_s)

    Usage:
        diagnostics = CBFDiagnostics(cfg, num_envs, num_agents, device)
        metrics = diagnostics.compute_metrics(gt_positions)
    """

    def __init__(
        self,
        cfg: CBFDiagnosticsCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        """Initialize the diagnostics tracker.

        Args:
            cfg: Configuration for diagnostics.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: Device for tensor computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        self.step_counter = 0

        # Pre-compute mask for diagonal elements (self-distances)
        # Shape: (1, num_agents, num_agents)
        self._eye_mask = torch.eye(
            num_agents, dtype=torch.bool, device=device
        ).unsqueeze(0)

    @staticmethod
    def compute_pairwise_distances(
        positions: torch.Tensor,
        num_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute all pairwise distances efficiently using broadcasting.

        Args:
            positions: Ground-truth positions (E, N, 3).
            num_agents: Number of agents per environment.
            device: Device for computation.

        Returns:
            Distance matrix (E, N, N) where element [e, i, j] is the distance
            between agent i and agent j in environment e. Diagonal is inf.
        """
        # positions: (E, N, 3)
        # Expand for pairwise computation:
        # positions[:, :, None, :] -> (E, N, 1, 3)
        # positions[:, None, :, :] -> (E, 1, N, 3)
        diff = positions[:, :, None, :] - positions[:, None, :, :]  # (E, N, N, 3)
        distances = torch.norm(diff, dim=-1)  # (E, N, N)

        # Set diagonal to inf (self-distance should not be considered)
        eye = torch.eye(num_agents, dtype=torch.bool, device=device)
        distances = torch.where(eye.unsqueeze(0), torch.full_like(distances, float("inf")), distances)

        return distances

    def compute_metrics(
        self,
        positions: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all safety diagnostics from GT positions.

        Args:
            positions: Ground-truth positions (E, N, 3).

        Returns:
            Dictionary of metric name -> value.
        """
        self.step_counter += 1

        metrics: Dict[str, float] = {}

        # Compute pairwise distances
        distances = self.compute_pairwise_distances(
            positions, self.num_agents, self.device
        )  # (E, N, N)

        # Minimum separation per environment (excluding self-distance)
        # Take min over all pairs: min over both agent dimensions
        min_sep_per_env = distances.view(self.num_envs, -1).min(dim=-1).values  # (E,)

        if self.cfg.track_min_separation:
            metrics["safety/min_separation_mean"] = min_sep_per_env.mean().item()
            metrics["safety/min_separation_min"] = min_sep_per_env.min().item()

        if self.cfg.track_collision_fraction:
            # Collision when any pair is closer than D_s
            collided = min_sep_per_env < self.cfg.D_s  # (E,) bool
            metrics["safety/collision_fraction"] = collided.float().mean().item()

        return metrics

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset diagnostics state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        # Currently no per-env state to reset, but interface provided
        # for consistency with other components.
        pass
