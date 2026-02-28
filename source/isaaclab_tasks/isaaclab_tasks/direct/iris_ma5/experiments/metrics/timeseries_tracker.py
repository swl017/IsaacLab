# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-timestep timeseries tracking for evaluation rollouts.

Collects statistics at each episode timestep across all environments and
episodes, enabling visualization of how metrics evolve within an episode
(convergence, transient behavior, steady-state performance).

Uses ``scatter_add_`` for efficient vectorized accumulation: each environment
maintains its own episode step counter, and metric values are accumulated into
per-timestep buckets (count, sum, sum-of-squares) on CPU.

Metrics tracked:
    - ``triangulation_rmse`` — RMSE vs ground truth (valid tri steps only)
    - ``sqrt_trace_sigma`` — √trace(Σ_X) uncertainty (valid tri steps only)
    - ``visibility`` — mean fraction of agents detecting the target
    - ``tri_valid`` — binary triangulation validity (0 or 1)
    - ``distance_to_target`` — mean agent-to-target Euclidean distance
    - ``target_speed`` — target ground-truth speed (m/s)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


# Metrics recorded at every step (all environments)
_ALWAYS_METRICS = ["visibility", "tri_valid", "distance_to_target", "target_speed"]

# Metrics recorded only when triangulation is valid
_VALID_ONLY_METRICS = ["triangulation_rmse", "sqrt_trace_sigma"]

_ALL_METRICS = _ALWAYS_METRICS + _VALID_ONLY_METRICS


class TimeseriesTracker:
    """Accumulates per-timestep statistics across evaluation episodes.

    Usage::

        ts = TimeseriesTracker(max_episode_steps=500, num_envs=4096, device="cuda:0")

        while not done:
            obs, rew, term, trunc, info = env.step(actions)
            done_mask = term | trunc  # [N, 1]

            ts.step(
                rmse=rmse_tensor,            # [N]
                sqrt_trace=sqrt_trace,       # [N]
                visibility=vis_ratio,        # [N]
                tri_valid=tri_valid_float,   # [N]
                distance=mean_agent_dist,    # [N]
                target_speed=tgt_speed,      # [N]
                active_mask=~done_mask,      # [N] — exclude post-reset envs
                tri_valid_mask=is_valid,     # [N] bool — for RMSE/trace gating
            )

            done_envs = torch.where(done_mask)[0]
            if done_envs.numel() > 0:
                ts.record_episode_end(done_envs)

        timeseries_dict = ts.compute_timeseries()
    """

    def __init__(
        self,
        max_episode_steps: int,
        num_envs: int,
        device: torch.device | str,
    ):
        self.max_steps = max_episode_steps
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Per-env step counter within current episode (on GPU for masking ops)
        self.env_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Per-timestep accumulators (on CPU for numerical precision)
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
        active_mask: Optional[torch.Tensor] = None,
        tri_valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Record one simulation step.

        Args:
            rmse: Triangulation RMSE per env [N].
            sqrt_trace: √trace(Σ_X) per env [N].
            visibility: Mean bbox validity fraction per env [N].
            tri_valid: Binary tri validity per env [N] (float 0/1).
            distance: Mean agent-target distance per env [N].
            target_speed: Target ground-truth speed per env [N] (m/s).
            active_mask: Boolean [N] — True for envs in an active episode
                (False for envs that just terminated/truncated and were reset).
                If None, all envs are considered active.
            tri_valid_mask: Boolean [N] — True where triangulation is valid.
                Used to gate RMSE and sqrt_trace accumulation.
        """
        t = self.env_step.cpu()  # [N]

        # Apply active mask
        if active_mask is not None:
            active = active_mask.cpu()
        else:
            active = torch.ones(self.num_envs, dtype=torch.bool)

        t_active = t[active].clamp(max=self.max_steps - 1)
        n_active = t_active.numel()
        if n_active == 0:
            return

        # -- "Always" metrics (all active envs) --
        ones = torch.ones(n_active, dtype=torch.long)
        self._count.scatter_add_(0, t_active, ones)

        vals_cpu = {
            "visibility": visibility.float().cpu()[active],
            "tri_valid": tri_valid.float().cpu()[active],
            "distance_to_target": distance.float().cpu()[active],
            "target_speed": target_speed.float().cpu()[active],
        }
        for name in _ALWAYS_METRICS:
            v = vals_cpu[name].double()
            self._sum[name].scatter_add_(0, t_active, v)
            self._sumsq[name].scatter_add_(0, t_active, v * v)

        # -- "Valid-only" metrics (only envs with valid triangulation) --
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

        # Advance step counter for active envs
        if active_mask is not None:
            self.env_step[active_mask] += 1
        else:
            self.env_step += 1

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Reset step counters for environments that completed an episode."""
        self.env_step[env_ids] = 0

    def compute_timeseries(self) -> Dict:
        """Compute per-timestep mean and std, return as JSON-serializable dict.

        Returns:
            Dictionary with keys:
                ``timesteps``: list of timestep indices [0, 1, ..., T-1]
                ``count``: number of environments contributing at each timestep
                ``<metric_name>``: dict with ``mean``, ``std``, ``count`` arrays
        """
        count = self._count.numpy()
        valid_count = self._valid_count.numpy()

        # Find last timestep with data
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
