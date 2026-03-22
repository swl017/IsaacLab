# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step trajectory recorder for iris_ma6 evaluation visualization.

Records agent, target, and triangulated-estimate positions for a small subset
of environments. Includes gimbal angles and zoom levels (new for iris_ma6).

Positions are stored in local coordinates (relative to env_origins).
Recording stops per-environment after the first episode completes.
"""

from __future__ import annotations

from typing import Dict, List

import torch


class TrajectoryRecorder:
    """Record per-timestep XYZ positions and gimbal state for a subset of environments."""

    def __init__(
        self,
        num_sample_envs: int,
        num_total_envs: int,
        agent_ids: List[str],
        max_steps: int,
    ):
        self.agent_ids = list(agent_ids)
        self.max_steps = max_steps

        n = min(num_sample_envs, num_total_envs)
        self.sample_env_ids: List[int] = (
            torch.linspace(0, num_total_envs - 1, n).long().tolist()
        )
        self._sample_set = set(self.sample_env_ids)

        self._active: Dict[int, bool] = {eid: True for eid in self.sample_env_ids}

        self._data: Dict[int, dict] = {}
        for eid in self.sample_env_ids:
            self._data[eid] = {
                "agents": {
                    aid: {"x": [], "y": [], "z": [], "gimbal_yaw": [], "gimbal_pitch": [], "zoom": []}
                    for aid in self.agent_ids
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
        gimbal_angles: Dict[str, torch.Tensor] | None = None,
        zoom_levels: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Record one simulation step for all active sampled environments.

        Args:
            agent_positions: {agent_id: [N, 3]} world-frame positions.
            target_pos: [N, 3] target world-frame position.
            tri_pos: [N, 3] triangulated position estimate.
            tri_valid: [N] or [N, 1] boolean validity mask.
            rmse: [N] per-env triangulation RMSE.
            viewing_angle: [N] mean pairwise viewing angle (degrees).
            env_origins: [N, 3] terrain environment origins.
            gimbal_angles: {agent_id: [N, 2]} gimbal yaw/pitch per agent.
            zoom_levels: {agent_id: [N]} zoom level per agent.
        """
        if tri_valid.dim() > 1:
            tri_valid = tri_valid.squeeze(-1)

        for eid in self.sample_env_ids:
            if not self._active[eid]:
                continue

            d = self._data[eid]
            origin = env_origins[eid]

            for aid in self.agent_ids:
                local = agent_positions[aid][eid] - origin
                d["agents"][aid]["x"].append(float(local[0]))
                d["agents"][aid]["y"].append(float(local[1]))
                d["agents"][aid]["z"].append(float(local[2]))

                if gimbal_angles is not None and aid in gimbal_angles:
                    ga = gimbal_angles[aid][eid]
                    d["agents"][aid]["gimbal_yaw"].append(float(ga[0]))
                    d["agents"][aid]["gimbal_pitch"].append(float(ga[1]))
                if zoom_levels is not None and aid in zoom_levels:
                    d["agents"][aid]["zoom"].append(float(zoom_levels[aid][eid]))

            tgt_local = target_pos[eid] - origin
            d["target"]["x"].append(float(tgt_local[0]))
            d["target"]["y"].append(float(tgt_local[1]))
            d["target"]["z"].append(float(tgt_local[2]))

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

            d["viewing_angle"].append(float(viewing_angle[eid]))

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Mark sampled environments as done after their first episode ends."""
        for eid in env_ids.tolist():
            if eid in self._sample_set and self._active.get(eid, False):
                self._active[eid] = False

    def is_done(self) -> bool:
        """Return True if all sampled environments have completed one episode."""
        return not any(self._active.values())

    def to_dict(self) -> dict:
        """Return JSON-serializable trajectory data."""
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
