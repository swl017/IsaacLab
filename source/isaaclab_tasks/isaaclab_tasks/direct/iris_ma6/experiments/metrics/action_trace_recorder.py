# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step physical-unit action recorder with per-axis RMS, ΔRMS, and FFT diagnostics.

Records the env's `cmd_vel` after each step for a subset of envs. Dims 0-3 are
already in physical units; dims 4-6 (gimbal yaw rate, gimbal pitch rate, zoom
rate) are stored normalized in [-1, 1] in the env, so they are scaled here by
`max_gimbal_rate` / per-env `max_zoom_rate` so that all reported stats are in
the units listed in `AXIS_UNITS`.

Recording stops per-environment after the first episode completes (mirrors
`TrajectoryRecorder`).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch


AXIS_NAMES_BODY: List[str] = [
    "vx_body",
    "vy_body",
    "vz_body",
    "yaw_rate",
    "gimbal_yaw_rate",
    "gimbal_pitch_rate",
    "zoom_rate",
]
"""7-axis layout for v0 (`iris_ma_env6_test.py`). vx/vy/vz are body-frame."""

AXIS_UNITS: List[str] = ["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s", "1/s"]
"""Units corresponding to AXIS_NAMES_BODY after scaling."""


class ActionTraceRecorder:
    """Record cmd_vel per step for a subset of envs; emit per-axis RMS, ΔRMS, peak-freq, and trace."""

    def __init__(
        self,
        num_sample_envs: int,
        num_total_envs: int,
        agent_ids: List[str],
        max_steps: int,
        policy_rate_hz: float,
        gimbal_rate_scale: float,
        zoom_rate_scale: torch.Tensor,
        record_trace: bool = True,
        min_active_steps_for_fft: int = 16,
    ):
        self.agent_ids = list(agent_ids)
        self.max_steps = int(max_steps)
        self.policy_rate_hz = float(policy_rate_hz)
        self.record_trace = bool(record_trace)
        self.min_active_steps_for_fft = int(min_active_steps_for_fft)

        n = min(num_sample_envs, num_total_envs)
        self.sample_env_ids: List[int] = (
            torch.linspace(0, num_total_envs - 1, n).long().tolist()
        )
        self._sample_set = set(self.sample_env_ids)

        self._active: Dict[int, bool] = {eid: True for eid in self.sample_env_ids}
        self._step_count: Dict[int, int] = {eid: 0 for eid in self.sample_env_ids}

        # Per-(env, agent), per-axis scale: dims 0-3 already physical; dims 4,5
        # use gimbal scale (scalar); dim 6 uses per-(env, agent) zoom scale,
        # which is DR-perturbed independently per drone. The env's controller
        # is batched across agents, so its `_max_zoom_rate` is shape
        # (num_envs * num_agents,) with agent-major layout
        # [agent_0_envs..., agent_1_envs..., ...]; reshape to (A, num_envs)
        # then transpose to (num_envs, A).
        num_agents = len(self.agent_ids)
        device = zoom_rate_scale.device
        z = zoom_rate_scale.to(dtype=torch.float32)
        if z.dim() == 2 and z.shape == (num_total_envs, num_agents):
            zoom_per_env_agent = z
        elif z.dim() <= 1:
            z_flat = z.flatten()
            if z_flat.numel() == num_total_envs * num_agents:
                # Env's controller layout is agent-major:
                # [agent_0_envs..., agent_1_envs..., ...]. Reshape to (A, N) → transpose.
                zoom_per_env_agent = z_flat.view(num_agents, num_total_envs).T
            elif z_flat.numel() == num_total_envs:
                zoom_per_env_agent = z_flat.unsqueeze(-1).expand(num_total_envs, num_agents)
            elif z_flat.numel() == 1:
                zoom_per_env_agent = z_flat.expand(num_total_envs, num_agents)
            else:
                raise ValueError(
                    f"zoom_rate_scale has {z_flat.numel()} elements; expected "
                    f"{num_total_envs * num_agents} (per-(env, agent) flat), "
                    f"{num_total_envs} (per-env), or 1 (scalar)."
                )
        else:
            raise ValueError(
                f"zoom_rate_scale has shape {tuple(z.shape)}; expected 1D, "
                f"or 2D ({num_total_envs}, {num_agents})."
            )

        scale = torch.ones(num_total_envs, num_agents, 7, device=device, dtype=torch.float32)
        scale[:, :, 4] = gimbal_rate_scale
        scale[:, :, 5] = gimbal_rate_scale
        scale[:, :, 6] = zoom_per_env_agent
        self._scale = scale

        # Preallocate per-env, per-agent buffers as (max_steps, 7), NaN-filled
        # so that uninitialized rows are clearly distinguishable.
        self._data: Dict[int, Dict[str, torch.Tensor]] = {}
        for eid in self.sample_env_ids:
            self._data[eid] = {
                aid: torch.full((self.max_steps, 7), float("nan"), device=device, dtype=torch.float32)
                for aid in self.agent_ids
            }

    def step(self, cmd_vel: torch.Tensor) -> None:
        """Record one step.

        Args:
            cmd_vel: (num_total_envs, num_agents, 7) tensor — env's `unwrapped.cmd_vel`
                read after `env.step()` returns.
        """
        for eid in self.sample_env_ids:
            if not self._active[eid]:
                continue
            t = self._step_count[eid]
            if t >= self.max_steps:
                continue
            for aidx, aid in enumerate(self.agent_ids):
                self._data[eid][aid][t, :] = cmd_vel[eid, aidx, :] * self._scale[eid, aidx]
            self._step_count[eid] = t + 1

    def record_episode_end(self, env_ids: torch.Tensor) -> None:
        """Mark sampled envs as done after their first completed episode."""
        for eid in env_ids.tolist():
            if eid in self._sample_set and self._active.get(eid, False):
                self._active[eid] = False

    def is_done(self) -> bool:
        """True iff every sampled env has completed its first episode."""
        return not any(self._active.values())

    def to_dict(self) -> dict:
        """JSON-serializable diagnostics block.

        Aggregates stats across recorded envs by mean (NaN-safe). Includes raw
        per-step trace if `record_trace` was True at construction.
        """
        agents_out: Dict[str, dict] = {}
        for aid in self.agent_ids:
            # Per-env stats: list of dict[axis] -> dict of 4 floats
            per_env_stats = [self._compute_per_env_stats(eid, aid) for eid in self.sample_env_ids]

            action_rms_agg: Dict[str, float] = {}
            action_delta_rms_agg: Dict[str, float] = {}
            dom_freq_agg: Dict[str, float] = {}
            dom_freq_delta_agg: Dict[str, float] = {}

            for ax_idx, ax_name in enumerate(AXIS_NAMES_BODY):
                rms_list = torch.tensor([s[ax_name]["action_rms"] for s in per_env_stats])
                drms_list = torch.tensor([s[ax_name]["action_delta_rms"] for s in per_env_stats])
                df_list = torch.tensor([s[ax_name]["dominant_freq_hz"] for s in per_env_stats])
                dfd_list = torch.tensor([s[ax_name]["dominant_freq_delta_hz"] for s in per_env_stats])

                action_rms_agg[ax_name] = float(torch.nanmean(rms_list))
                action_delta_rms_agg[ax_name] = float(torch.nanmean(drms_list))
                dom_freq_agg[ax_name] = float(torch.nanmean(df_list))
                dom_freq_delta_agg[ax_name] = float(torch.nanmean(dfd_list))

            agent_block: dict = {
                "action_rms": action_rms_agg,
                "action_delta_rms": action_delta_rms_agg,
                "dominant_freq_hz": dom_freq_agg,
                "dominant_freq_delta_hz": dom_freq_delta_agg,
            }
            if self.record_trace:
                agent_block["trace"] = [
                    self._data[eid][aid][0:self._step_count[eid], :].cpu().tolist()
                    for eid in self.sample_env_ids
                ]
            agents_out[aid] = agent_block

        return {
            "step_dt_seconds": 1.0 / self.policy_rate_hz,
            "policy_rate_hz": self.policy_rate_hz,
            "axis_names": list(AXIS_NAMES_BODY),
            "axis_units": list(AXIS_UNITS),
            "sample_env_ids": list(self.sample_env_ids),
            "agents": agents_out,
        }

    def _compute_per_env_stats(self, eid: int, aid: str) -> Dict[str, Dict[str, float]]:
        """Per-axis stats for a single (env, agent). NaN if too few active steps."""
        T = self._step_count[eid]
        out: Dict[str, Dict[str, float]] = {ax: {
            "action_rms": float("nan"),
            "action_delta_rms": float("nan"),
            "dominant_freq_hz": float("nan"),
            "dominant_freq_delta_hz": float("nan"),
        } for ax in AXIS_NAMES_BODY}

        if T < 2:
            return out

        a = self._data[eid][aid][0:T, :]  # (T, 7)
        da = a[1:, :] - a[:-1, :]         # (T-1, 7)

        rms_a = torch.sqrt(torch.mean(a * a, dim=0))                # (7,)
        rms_da = torch.sqrt(torch.mean(da * da, dim=0)) * self.policy_rate_hz  # (7,) units/sec

        do_fft = T >= self.min_active_steps_for_fft
        for ax_idx, ax_name in enumerate(AXIS_NAMES_BODY):
            out[ax_name]["action_rms"] = float(rms_a[ax_idx])
            out[ax_name]["action_delta_rms"] = float(rms_da[ax_idx])
            if do_fft:
                out[ax_name]["dominant_freq_hz"] = self._peak_freq_hz(a[:, ax_idx], self.policy_rate_hz)
                out[ax_name]["dominant_freq_delta_hz"] = self._peak_freq_hz(da[:, ax_idx], self.policy_rate_hz)
        return out

    @staticmethod
    def _peak_freq_hz(signal_1d: torch.Tensor, policy_rate_hz: float) -> float:
        """Peak rFFT bin (excluding DC) → Hz. Returns NaN if signal is essentially constant.

        The constant-signal guard uses the AC stddev rather than `mag == 0` because
        FFT round-off leaves O(1e-7) noise in the magnitude bins, which would
        otherwise produce a meaningless argmax over noise.
        """
        n = signal_1d.numel()
        if n < 2:
            return float("nan")
        # AC component must be non-negligible relative to the DC level
        ac_std = float(signal_1d.std(unbiased=False))
        scale = max(float(signal_1d.abs().max()), 1.0)
        if ac_std < 1e-6 * scale:
            return float("nan")
        mag = torch.fft.rfft(signal_1d).abs()
        if mag.numel() < 2:
            return float("nan")
        mag[0] = 0.0
        k = int(mag.argmax())
        return float(k) / float(n) * float(policy_rate_hz)
