# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SIYI A8 mini gimbal rate loop — first-order lag with hard saturation.

Sits between the policy-emitted (yaw_rate, pitch_rate) command and the
existing world-frame setpoint integration inside the gimbal controller.

Architecture invariant: the rate loop applies ONLY to the user-LOS-rate
command path. Body-motion compensation downstream (the IK that uses
``q_body_NOW`` every step) bypasses this loop entirely and remains
instantaneous. See mas/035 ticket for the full diagram.

Per-axis state (yaw and pitch are independent at the policy interface).
Joint-level coupling continues to live inside ``J^{-1}`` downstream and is
not touched by this loop.

mas/036 adds a per-env command-to-first-move dead-time buffer at the loop
input. The buffer's depth is sampled per env at reset from
``N(dead_time_mean_s, dead_time_std_s) * dead_time_curriculum_scale`` and
clipped to ``[0, dead_time_max_s]``. While the buffer is "cold" (fewer
samples pushed than the env's dead-time depth) the popped value is zero,
matching real hardware cold-start behavior.
"""

from __future__ import annotations

import math

import torch

from .gimbal_rate_loop_cfg import GimbalRateLoopCfg


class GimbalRateLoop:
    """First-order rate loop modeling the SIYI hardware user-command path.

    Inputs and outputs are in rad/s. Saturation, deadband, dead-time
    buffer (mas/036), and the per-axis first-order lag are applied in
    that order. Per-env τ tensors support DR via :meth:`set_tau_per_env`.

    Curriculum behavior:
      - τ blended via :meth:`set_progress` (0 = pass-through, 1 = full τ).
      - Dead time blended via :meth:`set_dead_time_curriculum_scale`
        (0 = zero delay, 1 = full measured distribution). Resets resample
        per env using the current scale.
    """

    def __init__(
        self,
        cfg: GimbalRateLoopCfg,
        num_envs: int,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device

        # Per-env τ for yaw and pitch (DR can override via set_tau_per_env)
        self._tau_yaw = torch.full(
            (num_envs,), cfg.tau_yaw_s, dtype=torch.float32, device=self.device
        )
        self._tau_pitch = torch.full(
            (num_envs,), cfg.tau_pitch_s, dtype=torch.float32, device=self.device
        )

        # Tracked rate output state — [N, 2] = (yaw_rate, pitch_rate)
        self._omega_actual = torch.zeros(
            num_envs, 2, dtype=torch.float32, device=self.device
        )

        # Curriculum scale on the lag (1.0 = full configured τ; 0.0 = pass-through)
        self._tau_progress: float = 1.0

        # mas/036: per-env dead-time buffer at the rate-loop input.
        # The ring depth is bounded by `dead_time_max_s / dt` — but `dt`
        # is per-step, so we lazily allocate the ring on the first
        # `step()` call when we know `dt`. The Gaussian samples are
        # stored in seconds and converted to integer step counts at use.
        self._dead_time_curr_scale: float = float(
            max(0.0, min(1.0, cfg.dead_time_curriculum_scale))
        )
        self._dead_time_seconds = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self._dead_time_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        # Counter of how many cmds have been pushed since last reset (per
        # env). Used to gate the cold-start zero output.
        self._dead_time_n_pushed = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._dead_time_buffer: torch.Tensor | None = None
        self._dead_time_buffer_idx: int = 0
        self._dead_time_dt: float | None = None  # cached `dt` at allocation

    # ------------------------------------------------------------------ properties

    @property
    def tau_yaw(self) -> torch.Tensor:
        """Per-env yaw τ tensor [N]."""
        return self._tau_yaw

    @property
    def tau_pitch(self) -> torch.Tensor:
        """Per-env pitch τ tensor [N]."""
        return self._tau_pitch

    @property
    def omega_actual(self) -> torch.Tensor:
        """Last computed (yaw_rate, pitch_rate) tensor [N, 2] in rad/s."""
        return self._omega_actual

    @property
    def tau_progress(self) -> float:
        """Current curriculum progress on τ ∈ [0, 1]."""
        return self._tau_progress

    @property
    def dead_time_curriculum_scale(self) -> float:
        """Current curriculum scale on the dead-time distribution ∈ [0, 1]."""
        return self._dead_time_curr_scale

    @property
    def dead_time_seconds(self) -> torch.Tensor:
        """Per-env dead time [N] in seconds (last sampled at reset)."""
        return self._dead_time_seconds

    @property
    def dead_time_steps(self) -> torch.Tensor:
        """Per-env dead time [N] in integer simulation steps (set lazily on
        the first :meth:`step` call once ``dt`` is known)."""
        return self._dead_time_steps

    # ------------------------------------------------------------------ control

    def step(self, omega_cmd_per_axis: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the rate loop one simulation step.

        Args:
            omega_cmd_per_axis: ``[N, 2]`` tensor in rad/s with columns
                ``(yaw_rate_cmd, pitch_rate_cmd)``.
            dt: simulation timestep [s] (typically the physics dt).

        Returns:
            ``[N, 2]`` tensor of tracked rates ``(yaw_rate, pitch_rate)``
            after saturation, optional deadband, optional dead-time, and
            first-order lag with the per-env τ.
        """
        if not self.cfg.enabled:
            self._omega_actual.copy_(omega_cmd_per_axis)
            return self._omega_actual.clone()

        if omega_cmd_per_axis.shape != (self.num_envs, 2):
            raise ValueError(
                f"omega_cmd_per_axis must be [N={self.num_envs}, 2], got "
                f"{tuple(omega_cmd_per_axis.shape)}"
            )

        cmd = omega_cmd_per_axis.to(self.device)

        # 1) Hard saturation per axis
        cmd = torch.clamp(
            cmd, -self.cfg.max_rate_per_axis, self.cfg.max_rate_per_axis
        )

        # 2) Optional deadband
        if self.cfg.deadband_rad_s > 0.0:
            mag = cmd.abs()
            cmd = torch.where(
                mag < self.cfg.deadband_rad_s,
                torch.zeros_like(cmd),
                cmd,
            )

        # 3) Per-env dead-time buffer (mas/036).
        cmd = self._apply_dead_time(cmd, dt)

        # 4) First-order lag toward (saturated, deadbanded, delayed) command,
        #    with per-axis τ scaled by the curriculum progress.
        tau_yaw_eff = self._tau_yaw * self._tau_progress
        tau_pitch_eff = self._tau_pitch * self._tau_progress

        # Pass-through when curriculum progress is 0 (or τ is effectively 0).
        # alpha = 1 - exp(-dt / tau); guard against τ ≈ 0 → alpha → 1 (instant).
        eps = 1e-6
        alpha_yaw = 1.0 - torch.exp(-dt / torch.clamp(tau_yaw_eff, min=eps))
        alpha_pitch = 1.0 - torch.exp(-dt / torch.clamp(tau_pitch_eff, min=eps))
        # When τ_eff is below eps (i.e. progress=0), alpha → 1 → pass-through.
        alpha_yaw = torch.where(
            tau_yaw_eff < eps, torch.ones_like(alpha_yaw), alpha_yaw
        )
        alpha_pitch = torch.where(
            tau_pitch_eff < eps, torch.ones_like(alpha_pitch), alpha_pitch
        )

        prev = self._omega_actual
        new_yaw = prev[:, 0] + alpha_yaw * (cmd[:, 0] - prev[:, 0])
        new_pitch = prev[:, 1] + alpha_pitch * (cmd[:, 1] - prev[:, 1])
        self._omega_actual = torch.stack([new_yaw, new_pitch], dim=-1)

        return self._omega_actual.clone()

    # ------------------------------------------------------------------ hooks

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset tracked rate state for the specified envs (default: all).

        Freshly-spawned envs start holding still (omega_actual = 0). The
        dead-time buffer entries for the affected envs are also cleared
        so the cold-start zero-output behavior is preserved, and a fresh
        per-env dead-time depth is sampled from the configured Gaussian
        scaled by the current curriculum scale.
        """
        if env_ids is None:
            self._omega_actual.zero_()
            self._dead_time_n_pushed.zero_()
            if self._dead_time_buffer is not None:
                self._dead_time_buffer.zero_()
                self._dead_time_buffer_idx = 0
            self._sample_dead_time(env_ids=None)
        else:
            self._omega_actual[env_ids] = 0.0
            self._dead_time_n_pushed[env_ids] = 0
            if self._dead_time_buffer is not None:
                self._dead_time_buffer[:, env_ids, :] = 0.0
            self._sample_dead_time(env_ids=env_ids)

    def set_progress(self, p: float) -> None:
        """Curriculum hook: scale the lag τ by ``p ∈ [0, 1]``.

        At p=0 the loop is pass-through (no lag); at p=1 the loop uses the
        configured τ_yaw / τ_pitch. Saturation, deadband, and dead-time
        buffer are unaffected by this scale.
        """
        self._tau_progress = float(max(0.0, min(1.0, p)))

    def set_dead_time_curriculum_scale(self, scale: float) -> None:
        """Curriculum hook: scale the per-env dead-time samples by
        ``scale ∈ [0, 1]``.

        At scale=0 each env's dead time is 0 (the buffer is a no-op); at
        scale=1 dead times are sampled from the full measured Gaussian.
        Takes effect at the *next* :meth:`reset` call (per-env dead times
        are constant within an episode by design — the dead time is a
        structural firmware/motor property, not a per-step variable).
        """
        self._dead_time_curr_scale = float(max(0.0, min(1.0, scale)))

    def set_tau_per_env(
        self,
        tau_yaw: torch.Tensor,
        tau_pitch: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """DR hook: write per-env τ values for yaw and pitch.

        Both inputs must be 1-D tensors. When ``env_ids`` is None they must
        have length ``num_envs``; otherwise they must match ``len(env_ids)``.
        """
        if env_ids is None:
            if tau_yaw.shape != (self.num_envs,) or tau_pitch.shape != (self.num_envs,):
                raise ValueError(
                    f"tau tensors must be [{self.num_envs}], got "
                    f"yaw={tuple(tau_yaw.shape)}, pitch={tuple(tau_pitch.shape)}"
                )
            self._tau_yaw.copy_(tau_yaw.to(self.device))
            self._tau_pitch.copy_(tau_pitch.to(self.device))
        else:
            self._tau_yaw[env_ids] = tau_yaw.to(self.device)
            self._tau_pitch[env_ids] = tau_pitch.to(self.device)

    # ------------------------------------------------------------------ internals

    def _sample_dead_time(self, env_ids: torch.Tensor | None) -> None:
        """Draw new per-env dead-time samples (in seconds) from
        ``N(mean, std) * curriculum_scale``, clipped to
        ``[0, dead_time_max_s]``. Steps cache is refreshed lazily on the
        next :meth:`step` once ``dt`` is known.
        """
        scale = self._dead_time_curr_scale
        mean_s = self.cfg.dead_time_mean_s * scale
        std_s = self.cfg.dead_time_std_s * scale
        max_s = self.cfg.dead_time_max_s

        if env_ids is None:
            n = self.num_envs
            # If both effective parameters are zero, force exact zero
            # rather than sampling — preserves the bit-exact pre-curriculum
            # behavior at scale=0.
            if scale <= 0.0:
                self._dead_time_seconds.zero_()
            else:
                samples = torch.randn(n, device=self.device) * std_s + mean_s
                samples.clamp_(min=0.0, max=max_s)
                self._dead_time_seconds.copy_(samples)
        else:
            n = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)
            if scale <= 0.0:
                self._dead_time_seconds[env_ids] = 0.0
            else:
                samples = torch.randn(n, device=self.device) * std_s + mean_s
                samples.clamp_(min=0.0, max=max_s)
                self._dead_time_seconds[env_ids] = samples

        # If the ring is already allocated, refresh the integer step count
        # for the affected envs immediately so the next `step()` uses the
        # new depth without waiting for the lazy path.
        if self._dead_time_dt is not None:
            self._refresh_dead_time_steps(env_ids)

    def _refresh_dead_time_steps(self, env_ids: torch.Tensor | None) -> None:
        """Convert ``dead_time_seconds`` to integer step counts (rounded,
        clipped to the pre-allocated ring depth) for the affected envs."""
        assert self._dead_time_dt is not None
        max_buf_steps = self._dead_time_buffer.shape[0] if self._dead_time_buffer is not None else 0
        if env_ids is None:
            steps = torch.round(self._dead_time_seconds / self._dead_time_dt).long()
            steps.clamp_(min=0, max=max_buf_steps)
            self._dead_time_steps.copy_(steps)
        else:
            steps = torch.round(
                self._dead_time_seconds[env_ids] / self._dead_time_dt
            ).long()
            steps.clamp_(min=0, max=max_buf_steps)
            self._dead_time_steps[env_ids] = steps

    def _ensure_dead_time_buffer(self, dt: float) -> None:
        """Lazy-allocate (or re-allocate on dt change) the dead-time ring
        buffer. The depth is ``ceil(dead_time_max_s / dt)`` to bound any
        per-env sampled dead time at run time."""
        # Allocation is gated on whether the cfg has any dead-time budget
        # at all. ``dead_time_max_s == 0`` short-circuits the buffer path
        # to a no-op, matching the pre-mas/036 contract.
        if self.cfg.dead_time_max_s <= 0.0:
            self._dead_time_buffer = None
            self._dead_time_dt = dt
            self._dead_time_steps.zero_()
            return

        max_steps = max(1, int(math.ceil(self.cfg.dead_time_max_s / max(dt, 1e-9))))
        if (
            self._dead_time_buffer is None
            or self._dead_time_dt != dt
            or self._dead_time_buffer.shape[0] != max_steps
        ):
            self._dead_time_buffer = torch.zeros(
                max_steps, self.num_envs, 2, device=self.device, dtype=torch.float32
            )
            self._dead_time_buffer_idx = 0
            self._dead_time_dt = dt
            # Refresh integer step counts for the new dt.
            self._refresh_dead_time_steps(env_ids=None)

    def _apply_dead_time(self, cmd: torch.Tensor, dt: float) -> torch.Tensor:
        """Push ``cmd`` through the per-env dead-time ring; return the
        per-env delayed command. Cold-start envs (n_pushed < dead_time_steps)
        receive zero — matching real hardware behavior on first power-up.
        """
        # Skip the buffer entirely if no env has any dead time AND no
        # budget is configured. This preserves bit-exactness when
        # `dead_time_curriculum_scale = 0` (nothing was sampled, all zero).
        if self.cfg.dead_time_max_s <= 0.0:
            return cmd
        if self._dead_time_curr_scale <= 0.0 and bool(
            (self._dead_time_steps == 0).all().item()
        ):
            return cmd

        self._ensure_dead_time_buffer(dt)
        if self._dead_time_buffer is None:  # max_s == 0 path
            return cmd
        max_steps = self._dead_time_buffer.shape[0]

        # Push current cmd into the ring.
        self._dead_time_buffer[self._dead_time_buffer_idx] = cmd
        self._dead_time_n_pushed.add_(1)

        # Pop per-env delayed cmd. For env i with dead_time_steps[i] = d,
        # the delayed command came from slot
        #   (idx - d) mod max_steps          if d > 0
        #   idx (the just-pushed cmd)        if d == 0
        # mod arithmetic on a long tensor.
        d = self._dead_time_steps  # [N]
        idx = (
            (self._dead_time_buffer_idx - d) % max_steps
        ).long()  # [N]
        env_arange = torch.arange(self.num_envs, device=self.device)
        delayed = self._dead_time_buffer[idx, env_arange]  # [N, 2]

        # Cold-start: while fewer cmds have been pushed than the env's
        # dead-time depth, the popped value is stale zero (correct, since
        # the buffer was zeroed at reset). Defensive override for envs
        # where the ring's pre-fill state would otherwise leak across
        # episodes — replace with explicit zero when n_pushed <= d.
        cold = self._dead_time_n_pushed <= d  # [N]
        delayed = torch.where(cold.unsqueeze(-1), torch.zeros_like(delayed), delayed)

        # Advance write index (modulo).
        self._dead_time_buffer_idx = (self._dead_time_buffer_idx + 1) % max_steps

        return delayed
