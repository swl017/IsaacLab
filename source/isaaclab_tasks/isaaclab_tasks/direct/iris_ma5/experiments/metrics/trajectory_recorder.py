# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step trajectory recorder for evaluation visualization.

Records agent, target, and triangulated-estimate positions for a small subset
of environments during evaluation rollouts. Designed for bird's-eye view
trajectory plotting (not for statistical aggregation — see TimeseriesTracker
for that).

Positions are stored in **local coordinates** (relative to ``env_origins``) so
trajectories from different parallel environments are directly comparable.
Recording stops per-environment after the first episode completes, capturing
exactly one full trajectory per sampled environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class TrajectoryRecorder:
    """Record per-timestep XYZ positions for a subset of environments.

    Usage::

        rec = TrajectoryRecorder(
            num_sample_envs=8,
            num_total_envs=4096,
            agent_ids=["drone_0", "drone_1"],
            max_steps=2500,
        )

        while not done:
            obs, rew, term, trunc, info = env.step(actions)
            done_mask = term | trunc

            rec.step(
                agent_positions={"drone_0": pos0, "drone_1": pos1},
                target_pos=target_pos,
                tri_pos=triangulated_pos,
                tri_valid=is_valid,
                viewing_angle=angle_deg,
                env_origins=terrain_origins,
            )

            done_envs = torch.where(done_mask)[0]
            if done_envs.numel() > 0:
                rec.record_episode_end(done_envs)

        traj_dict = rec.to_dict()
    """

    def __init__(
        self,
        num_sample_envs: int,
        num_total_envs: int,
        agent_ids: List[str],
        max_steps: int,
    ):
        """Initialize trajectory recorder.

        Args:
            num_sample_envs: How many environments to sample (e.g. 8).
            num_total_envs: Total parallel environments (e.g. 4096).
            agent_ids: List of agent identifiers (e.g. ["drone_0", "drone_1"]).
            max_steps: Maximum episode length in steps.
        """
        self.agent_ids = list(agent_ids)
        self.max_steps = max_steps

        # Evenly-spaced sample indices across [0, num_total_envs)
        n = min(num_sample_envs, num_total_envs)
        self.sample_env_ids: List[int] = (
            torch.linspace(0, num_total_envs - 1, n).long().tolist()
        )
        self._sample_set = set(self.sample_env_ids)

        # Active flag per sampled env (True = still recording)
        self._active: Dict[int, bool] = {eid: True for eid in self.sample_env_ids}

        # Trajectory storage: {env_id: {"agents": ..., "target": ..., ...}}
        self._data: Dict[int, dict] = {}
        for eid in self.sample_env_ids:
            self._data[eid] = {
                "agents": {
                    aid: {"x": [], "y": [], "z": []} for aid in self.agent_ids
                },
                "target": {"x": [], "y": [], "z": []},
                "tri_estimate": {"x": [], "y": [], "z": []},
                "tri_valid": [],
                "rmse": [],
                "viewing_angle": [],
            }

    def step(
        self,
        agent_positions: Dict[str, torch.Tensor],
        target_pos: torch.Tensor,
        tri_pos: torch.Tensor,
        tri_valid: torch.Tensor,
        rmse: torch.Tensor,
        viewing_angle: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Record one simulation step for all active sampled environments.

        Args:
            agent_positions: ``{agent_id: [N, 3]}`` world-frame positions.
            target_pos: ``[N, 3]`` target world-frame position.
            tri_pos: ``[N, 3]`` triangulated position estimate.
            tri_valid: ``[N]`` or ``[N, 1]`` boolean validity mask.
            rmse: ``[N]`` per-env triangulation RMSE in meters.
            viewing_angle: ``[N]`` mean pairwise viewing angle in degrees.
            env_origins: ``[N, 3]`` terrain environment origins.
        """
        # Squeeze validity to 1-D
        if tri_valid.dim() > 1:
            tri_valid = tri_valid.squeeze(-1)

        for eid in self.sample_env_ids:
            if not self._active[eid]:
                continue

            d = self._data[eid]
            origin = env_origins[eid]  # [3]

            # Agent positions (local)
            for aid in self.agent_ids:
                local = agent_positions[aid][eid] - origin  # [3]
                d["agents"][aid]["x"].append(float(local[0]))
                d["agents"][aid]["y"].append(float(local[1]))
                d["agents"][aid]["z"].append(float(local[2]))

            # Target position (local)
            tgt_local = target_pos[eid] - origin
            d["target"]["x"].append(float(tgt_local[0]))
            d["target"]["y"].append(float(tgt_local[1]))
            d["target"]["z"].append(float(tgt_local[2]))

            # Triangulated estimate (local, None if invalid)
            valid = bool(tri_valid[eid])
            d["tri_valid"].append(valid)
            d["rmse"].append(float(rmse[eid]) if valid else None)
            if valid:
                tri_local = tri_pos[eid] - origin
                d["tri_estimate"]["x"].append(float(tri_local[0]))
                d["tri_estimate"]["y"].append(float(tri_local[1]))
                d["tri_estimate"]["z"].append(float(tri_local[2]))
            else:
                d["tri_estimate"]["x"].append(None)
                d["tri_estimate"]["y"].append(None)
                d["tri_estimate"]["z"].append(None)

            # Viewing angle
            d["viewing_angle"].append(float(viewing_angle[eid]))

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Mark sampled environments as done after their first episode ends.

        Args:
            env_ids: ``[K]`` tensor of environment indices that just finished.
        """
        for eid in env_ids.tolist():
            if eid in self._sample_set and self._active.get(eid, False):
                self._active[eid] = False

    def is_done(self) -> bool:
        """Return True if all sampled environments have completed one episode."""
        return not any(self._active.values())

    def to_dict(self) -> dict:
        """Return JSON-serializable trajectory data.

        Returns:
            Dictionary with structure::

                {
                    "sample_env_ids": [0, 512, ...],
                    "agent_ids": ["drone_0", "drone_1"],
                    "envs": {
                        "0": {
                            "num_steps": 2500,
                            "agents": {"drone_0": {"x": [...], ...}, ...},
                            "target": {"x": [...], ...},
                            "tri_estimate": {"x": [1.5, null, ...], ...},
                            "tri_valid": [true, false, ...],
                            "viewing_angle": [45.2, ...]
                        },
                        ...
                    }
                }
        """
        envs = {}
        for eid in self.sample_env_ids:
            d = self._data[eid]
            envs[str(eid)] = {
                "num_steps": len(d["target"]["x"]),
                "agents": d["agents"],
                "target": d["target"],
                "tri_estimate": d["tri_estimate"],
                "tri_valid": d["tri_valid"],
                "rmse": d["rmse"],
                "viewing_angle": d["viewing_angle"],
            }

        return {
            "sample_env_ids": self.sample_env_ids,
            "agent_ids": self.agent_ids,
            "envs": envs,
        }
