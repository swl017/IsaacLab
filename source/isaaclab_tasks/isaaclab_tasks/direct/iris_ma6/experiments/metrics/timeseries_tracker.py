# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-timestep timeseries tracking for iris_ma6 evaluation rollouts.

Collects statistics at each episode timestep across all environments and
episodes, enabling visualization of how metrics evolve within an episode.

Metrics tracked:
    - triangulation_rmse — RMSE vs ground truth (valid tri steps only)
    - sqrt_trace_sigma — sqrt(trace(Sigma_X)) uncertainty (valid tri steps only)
    - visibility — mean fraction of agents detecting the target
    - tri_valid — binary triangulation validity (0 or 1)
    - distance_to_target — mean agent-to-target Euclidean distance
    - target_speed — target ground-truth speed (m/s)
    - viewing_angle — mean pairwise viewing angle between agents (degrees)
    - cbf_penalty — CBF penalty magnitude
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


_ALWAYS_METRICS = ["visibility", "tri_valid", "distance_to_target", "target_speed",
                    "viewing_angle", "cbf_penalty"]

_VALID_ONLY_METRICS = ["triangulation_rmse", "sqrt_trace_sigma"]

_ALL_METRICS = _ALWAYS_METRICS + _VALID_ONLY_METRICS


class TimeseriesTracker:
    """Accumulates per-timestep statistics across evaluation episodes."""

    def __init__(
        self,
        max_episode_steps: int,
        num_envs: int,
        device: torch.device | str,
    ):
        self.max_steps = max_episode_steps
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.env_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self._count = torch.zeros(max_episode_steps, dtype=torch.long)
        self._valid_count = torch.zeros(max_episode_steps, dtype=torch.long)

        self._sum: Dict[str, torch.Tensor] = {}
        self._sumsq: Dict[str, torch.Tensor] = {}
        for m in _ALL_METRICS:
            self._sum[m] = torch.zeros(max_episode_steps, dtype=torch.float64)
            self._sumsq[m] = torch.zeros(max_episode_steps, dtype=torch.float64)

    def step(
        self,
        rmse: torch.Tensor,
        sqrt_trace: torch.Tensor,
        visibility: torch.Tensor,
        tri_valid: torch.Tensor,
        distance: torch.Tensor,
        target_speed: torch.Tensor,
        viewing_angle: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        tri_valid_mask: Optional[torch.Tensor] = None,
        cbf_penalty: Optional[torch.Tensor] = None,
    ) -> None:
        """Record one simulation step."""
        t = self.env_step.cpu()

        if active_mask is not None:
            active = active_mask.cpu()
        else:
            active = torch.ones(self.num_envs, dtype=torch.bool)

        t_active = t[active].clamp(max=self.max_steps - 1)
        n_active = t_active.numel()
        if n_active == 0:
            return

        ones = torch.ones(n_active, dtype=torch.long)
        self._count.scatter_add_(0, t_active, ones)

        vals_cpu = {
            "visibility": visibility.float().cpu()[active],
            "tri_valid": tri_valid.float().cpu()[active],
            "distance_to_target": distance.float().cpu()[active],
            "target_speed": target_speed.float().cpu()[active],
            "viewing_angle": viewing_angle.float().cpu()[active],
            "cbf_penalty": cbf_penalty.float().cpu()[active] if cbf_penalty is not None else torch.zeros(n_active),
        }
        for name in _ALWAYS_METRICS:
            v = vals_cpu[name].double()
            self._sum[name].scatter_add_(0, t_active, v)
            self._sumsq[name].scatter_add_(0, t_active, v * v)

        if tri_valid_mask is not None:
            valid = tri_valid_mask.cpu()[active]
        else:
            valid = tri_valid.bool().cpu()[active]

        if valid.any():
            t_valid = t_active[valid]
            valid_ones = torch.ones(t_valid.numel(), dtype=torch.long)
            self._valid_count.scatter_add_(0, t_valid, valid_ones)

            rmse_v = rmse.float().cpu()[active][valid].double()
            self._sum["triangulation_rmse"].scatter_add_(0, t_valid, rmse_v)
            self._sumsq["triangulation_rmse"].scatter_add_(0, t_valid, rmse_v * rmse_v)

            st_v = sqrt_trace.float().cpu()[active][valid].double()
            self._sum["sqrt_trace_sigma"].scatter_add_(0, t_valid, st_v)
            self._sumsq["sqrt_trace_sigma"].scatter_add_(0, t_valid, st_v * st_v)

        if active_mask is not None:
            self.env_step[active_mask] += 1
        else:
            self.env_step += 1

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Reset step counters for environments that completed an episode."""
        self.env_step[env_ids] = 0

    def compute_timeseries(self) -> Dict:
        """Compute per-timestep mean and std, return as JSON-serializable dict."""
        count = self._count.numpy()
        valid_count = self._valid_count.numpy()

        nonzero = np.nonzero(count)[0]
        max_t = int(nonzero[-1]) + 1 if len(nonzero) > 0 else 0

        result: Dict = {
            "timesteps": list(range(max_t)),
            "count": count[:max_t].tolist(),
        }

        for name in _ALL_METRICS:
            if name in _VALID_ONLY_METRICS:
                c = valid_count[:max_t].astype(np.float64)
            else:
                c = count[:max_t].astype(np.float64)

            c_safe = np.maximum(c, 1.0)
            s = self._sum[name].numpy()[:max_t]
            sq = self._sumsq[name].numpy()[:max_t]

            mean = s / c_safe
            var = sq / c_safe - mean * mean
            std = np.sqrt(np.maximum(var, 0.0))

            result[name] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "count": c.astype(int).tolist(),
            }

        return result
