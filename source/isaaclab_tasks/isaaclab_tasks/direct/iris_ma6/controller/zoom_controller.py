# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zoom controller — selectable between legacy first-order lag and SIYI A8 model.

Model selection is per-controller-construction via ``cfg.model``:

- ``"first_order"`` (default — bit-exact pre-mas/037): single first-order lag
  toward the integrator state. Stages run: action denorm → integrator clamp →
  first-order lag. ``zoom`` and ``zoom_internal`` properties return the same
  continuous post-lag state.

- ``"siyi_a8"`` (mas/037 fitted SIYI A8 mini): four-stage pipeline ::

      cmd ∈ [-1, 1]
        × max_zoom_rate (action scale, deployment-matched, ≤ v_max)
        clamp ±v_max_levels_per_s (lens slew, symmetric)
        input-side dead-time buffer (per-env Gaussian @ reset, curriculum-scaled)
        integrator: target ← clamp(target + rate·dt, [zoom_min, zoom_max])
        first-order lag: state ← state + (1 − exp(-dt/τ₁)) · (target − state)
        zoom_internal = state            (continuous; for tests/debug)
        zoom_published = round(state / quantum) * quantum    (← what env sees)

  See mas/037 ticket for the bench fit and the architectural rationale. The
  dead-time buffer is the 1-D analog of ``GimbalRateLoop._apply_dead_time``
  (mas/036 pattern).
"""

from __future__ import annotations

import math

import torch

from .zoom_controller_cfg import ZoomControllerCfg


# Measured zoom curve from the SIYI camera bench fit:
#   z_eff = 1 + a * (exp(b * (cmd - 1)) - 1)
# Source: /home/usrg/mas/src/scripts/camera_calibration/zoom_curve.json
# Trustworthy domain: cmd in [1.0, 5.0]. cmd > 5 is clamped because the 6x
# calibration uncertainty exceeds the trust threshold (mas/028).
ZOOM_CURVE_A: float = 0.32489
ZOOM_CURVE_B: float = 0.4767
ZOOM_CURVE_CMD_MAX: float = 5.0


def compute_z_eff(zoom_cmd: torch.Tensor) -> torch.Tensor:
    """Operator zoom command -> effective focal-length multiplier.

    Maps the operator-facing zoom command (1.0 = 1x, 5.0 = 5x) to the
    actual focal-length multiplier the real SIYI camera applies. The
    relationship is sub-linear: cmd=5 corresponds to z_eff ≈ 2.86, not 5.

    Inputs above ZOOM_CURVE_CMD_MAX (5.0) are clamped before the exponential
    is applied. cmd ∈ [1, 5] is the trust region of the underlying mrcal
    calibration.
    """
    cmd = torch.clamp(zoom_cmd, min=1.0, max=ZOOM_CURVE_CMD_MAX)
    return 1.0 + ZOOM_CURVE_A * (torch.exp(ZOOM_CURVE_B * (cmd - 1.0)) - 1.0)


class ZoomController:
    """Zoom controller with selectable dynamics (``cfg.model``)."""

    def __init__(
        self,
        cfg: ZoomControllerCfg,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """Initialize zoom controller.

        Args:
            cfg: Zoom controller configuration.
            num_envs: Number of parallel environments.
            device: Torch device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        if cfg.model not in ("first_order", "siyi_a8"):
            raise ValueError(
                f"ZoomControllerCfg.model must be 'first_order' or 'siyi_a8', "
                f"got {cfg.model!r}"
            )

        # State: integrator target (continuous; pre-lag).
        self._zoom_target = torch.ones(num_envs, dtype=torch.float32, device=self.device)

        # State: post-lag continuous level. Both modes treat this as the
        # observable ``zoom_internal``. In siyi_a8 mode, the published level
        # is a quantized read of this state.
        self._zoom = torch.ones(num_envs, dtype=torch.float32, device=self.device)

        # Per-env scalar gains.
        self._tau_zoom = torch.full(
            (num_envs,), cfg.tau_zoom, dtype=torch.float32, device=self.device
        )
        self._max_zoom_rate = torch.full(
            (num_envs,), cfg.max_zoom_rate, dtype=torch.float32, device=self.device
        )
        self._v_max = torch.full(
            (num_envs,), cfg.v_max_levels_per_s, dtype=torch.float32, device=self.device
        )

        # ------------------------------------------------------------------
        # mas/037 dead-time buffer state (siyi_a8 only). Same shape pattern
        # as GimbalRateLoop but 1-D (no axis dim). Buffer is allocated lazily
        # on the first compute_control call when dt becomes known.
        # ------------------------------------------------------------------
        self._dead_time_curr_scale: float = float(
            max(0.0, min(1.0, cfg.dead_time_curriculum_scale))
        )
        self._dead_time_seconds = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self._dead_time_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._dead_time_n_pushed = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._dead_time_buffer: torch.Tensor | None = None
        self._dead_time_buffer_idx: int = 0
        self._dead_time_dt: float | None = None
        # Bug-3: Python-side mirror of "are all envs at d=0?". Updated
        # whenever _dead_time_steps is rewritten (only in
        # _refresh_dead_time_steps). _dead_time_steps starts at zeros so the
        # cache starts True and the per-call early-exit in _apply_dead_time
        # can avoid a CUDA->CPU sync on every controller call.
        self._dead_time_all_zero: bool = True

    # ------------------------------------------------------------------ properties

    @property
    def zoom(self) -> torch.Tensor:
        """Published zoom level (N,).

        - ``first_order``: identical to the continuous internal state (no
          quantization; preserves pre-mas/037 obs).
        - ``siyi_a8``: quantized to ``cfg.quantum_levels`` (default 0.1)
          (mas/037 Tip 1: published path snaps; internal stays continuous).
        """
        if self.cfg.model == "siyi_a8":
            q = self.cfg.quantum_levels
            return torch.round(self._zoom / q) * q
        return self._zoom

    @property
    def zoom_internal(self) -> torch.Tensor:
        """Continuous post-lag zoom state (N,) — same in both modes. For
        tests / debug / DR. Use :attr:`zoom` for the env-facing observation.
        """
        return self._zoom

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
        """Per-env dead time [N] in integer simulation steps. Set lazily on
        the first :meth:`compute_control` call once ``dt`` is known.
        """
        return self._dead_time_steps

    # ------------------------------------------------------------------ control

    def compute_control(
        self,
        zoom_rate_cmd: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance the zoom controller one step.

        Args:
            zoom_rate_cmd: (N,) policy zoom rate command in [-1, 1].
            dt: Simulation timestep [s].

        Returns:
            zoom_level: (N,) published zoom level (quantized in siyi_a8 mode,
            continuous in first_order mode). Always clamped to
            [zoom_min, zoom_max].
        """
        # Stage 1: action denorm → physical levels/s.
        rate = zoom_rate_cmd * self._max_zoom_rate

        if self.cfg.model == "siyi_a8":
            # Stage 2: lens slew clip (symmetric).
            rate = torch.clamp(rate, min=-self._v_max, max=self._v_max)
            # Stage 3: input-side dead-time buffer.
            rate = self._apply_dead_time(rate, dt)

        # Stage 4: integrator → continuous target, clamped to [zoom_min, zoom_max].
        self._zoom_target = self._zoom_target + rate * dt
        self._zoom_target = torch.clamp(
            self._zoom_target, self.cfg.zoom_min, self.cfg.zoom_max
        )

        # Stage 5: first-order lag. Curriculum-gated tau via per-env tensor.
        # eps guard so tau≈0 collapses to pass-through (instant), matching
        # the legacy controller's curriculum-bootstrap behavior.
        eps = 1e-6
        tau_eff = torch.clamp(self._tau_zoom, min=eps)
        alpha = 1.0 - torch.exp(-dt / tau_eff)
        # When tau_eff is at the eps floor, alpha → 1 → pass-through.
        alpha = torch.where(self._tau_zoom < eps, torch.ones_like(alpha), alpha)
        self._zoom = self._zoom + alpha * (self._zoom_target - self._zoom)

        # Defensive clamp on post-lag state (matches legacy controller).
        self._zoom = torch.clamp(self._zoom, self.cfg.zoom_min, self.cfg.zoom_max)

        return self.zoom  # property; quantized in siyi_a8, continuous in first_order

    def get_fov(self, base_fov: float = 90.0) -> torch.Tensor:
        """Get current field of view based on the published zoom level.

        Assumes simple relationship: FOV = base_fov / zoom

        Args:
            base_fov: Field of view at 1x zoom [degrees].

        Returns:
            fov: (N,) field of view in degrees.
        """
        return base_fov / self.zoom

    def get_focal_length_multiplier(self) -> torch.Tensor:
        """Get focal length multiplier relative to base lens.

        Applies the measured SIYI zoom curve so that zoom command 5.0 maps
        to ~2.86x effective focal multiplier (not 5.0x). Use the raw
        operator command via `self.zoom` when feeding observations.

        Returns:
            multiplier: (N,) effective focal length multiplier (z_eff).
        """
        return compute_z_eff(self.zoom)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset zoom state.

        Both modes reset target and post-lag state to 1.0. In ``siyi_a8`` mode,
        the dead-time buffer entries for the affected envs are cleared and a
        fresh per-env τ_d depth is sampled from the configured Gaussian
        scaled by the current curriculum scale.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._zoom.fill_(1.0)
            self._zoom_target.fill_(1.0)
        else:
            self._zoom[env_ids] = 1.0
            self._zoom_target[env_ids] = 1.0

        if self.cfg.model != "siyi_a8":
            return

        # mas/037 dead-time buffer reset.
        if env_ids is None:
            self._dead_time_n_pushed.zero_()
            if self._dead_time_buffer is not None:
                self._dead_time_buffer.zero_()
                self._dead_time_buffer_idx = 0
            self._sample_dead_time(env_ids=None)
        else:
            self._dead_time_n_pushed[env_ids] = 0
            if self._dead_time_buffer is not None:
                self._dead_time_buffer[:, env_ids] = 0.0
            self._sample_dead_time(env_ids=env_ids)

    def set_tau_zoom(self, tau_zoom: torch.Tensor | float):
        """Set zoom time constant (for randomization / curriculum).

        Args:
            tau_zoom: Time constant [s]. Can be scalar or (N,) tensor.
        """
        if isinstance(tau_zoom, torch.Tensor):
            self._tau_zoom = tau_zoom.to(self.device)
        else:
            self._tau_zoom = torch.full_like(self._tau_zoom, float(tau_zoom))

    def set_max_zoom_rate(self, max_zoom_rate: torch.Tensor | float):
        """Set policy action scale (for randomization).

        Args:
            max_zoom_rate: Max zoom rate [levels/s]. Scalar or (N,) tensor.
                Should remain ≤ ``v_max_levels_per_s`` to keep the policy in
                the linear regime (mas/037 Tip 5).
        """
        if isinstance(max_zoom_rate, torch.Tensor):
            self._max_zoom_rate = max_zoom_rate.to(self.device)
        else:
            self._max_zoom_rate = torch.full_like(self._max_zoom_rate, float(max_zoom_rate))

    def set_v_max(self, v_max: torch.Tensor | float):
        """Set per-env lens slew clip [levels/s] (forward-compat hook for DR).

        Per design 6 + Tip 6, v_max DR is intentionally NOT enabled by default.
        This setter exists so future tickets can flip it on without touching
        the controller.
        """
        if isinstance(v_max, torch.Tensor):
            self._v_max = v_max.to(self.device)
        else:
            self._v_max = torch.full_like(self._v_max, float(v_max))

    def set_zoom(self, zoom_level: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Directly set zoom level (for initialization / teleop).

        Writes both the integrator target and the post-lag state, so there is
        no transient on initial-state randomization. The published level is
        derived from the post-lag state on read.

        Args:
            zoom_level: Zoom level to set.
            env_ids: Environment indices. If None, set all.
        """
        zoom_clamped = torch.clamp(zoom_level, self.cfg.zoom_min, self.cfg.zoom_max)
        if env_ids is None:
            self._zoom = zoom_clamped.clone()
            self._zoom_target = zoom_clamped.clone()
        else:
            self._zoom[env_ids] = zoom_clamped
            self._zoom_target[env_ids] = zoom_clamped

    def set_dead_time_curriculum_scale(self, scale: float) -> None:
        """Curriculum hook: scale the per-env dead-time samples by
        ``scale ∈ [0, 1]``.

        At scale=0 each env's dead time is 0 (the buffer is a no-op); at
        scale=1 dead times are sampled from the full measured Gaussian. Takes
        effect at the *next* :meth:`reset` call (per-env dead times are
        constant within an episode by design — the dead time is a structural
        firmware/motor property, not a per-step variable).

        No-op when ``cfg.model != "siyi_a8"``.
        """
        if self.cfg.model != "siyi_a8":
            return
        self._dead_time_curr_scale = float(max(0.0, min(1.0, scale)))

    # ------------------------------------------------------------------ internals (siyi_a8 only)

    def _sample_dead_time(self, env_ids: torch.Tensor | None) -> None:
        """Draw new per-env dead-time samples from
        ``N(mean, std) * curriculum_scale``, clipped to [0, dead_time_max_s].
        """
        scale = self._dead_time_curr_scale
        mean_s = self.cfg.dead_time_mean_s * scale
        std_s = self.cfg.dead_time_std_s * scale
        max_s = self.cfg.dead_time_max_s

        if env_ids is None:
            n = self.num_envs
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

        if self._dead_time_dt is not None:
            self._refresh_dead_time_steps(env_ids)

    def _refresh_dead_time_steps(self, env_ids: torch.Tensor | None) -> None:
        """Convert dead_time_seconds to integer step counts.

        Clamps to ``[0, depth - 1]`` because the apply-path writes the new rate
        first and then reads ``(write_idx - d) mod depth``: at ``d == depth``
        the read wraps back onto the just-written slot. With the buffer
        allocated at ``ceil(dead_time_max_s / dt) + 1``, the ``-1`` here
        leaves the natural physical max (``ceil(dead_time_max_s / dt)`` steps)
        as the largest valid delay.
        """
        assert self._dead_time_dt is not None
        max_d = (
            self._dead_time_buffer.shape[0] - 1
            if self._dead_time_buffer is not None
            else 0
        )
        if env_ids is None:
            steps = torch.round(self._dead_time_seconds / self._dead_time_dt).long()
            steps.clamp_(min=0, max=max_d)
            self._dead_time_steps.copy_(steps)
        else:
            steps = torch.round(
                self._dead_time_seconds[env_ids] / self._dead_time_dt
            ).long()
            steps.clamp_(min=0, max=max_d)
            self._dead_time_steps[env_ids] = steps
        # Bug-3: cache the all-zeros predicate so the per-call early-exit in
        # _apply_dead_time can read a Python bool instead of forcing a CUDA
        # sync on every controller call. _dead_time_steps is only written
        # here, so refreshing the cache after each write is sufficient.
        self._dead_time_all_zero = bool((self._dead_time_steps == 0).all().item())

    def _ensure_dead_time_buffer(self, dt: float) -> None:
        """Lazy-allocate (or re-allocate on dt change) the 1-D dead-time ring.

        Depth = ``ceil(dead_time_max_s / dt) + 1``. The +1 is required because
        the apply-path writes the new rate at ``_dead_time_buffer_idx`` *before*
        reading the popped value at ``(_dead_time_buffer_idx - d) mod depth``.
        Without the +1, ``d == ceil(dead_time_max_s / dt)`` wraps back onto the
        just-written slot and silently produces 0 delay instead of the maximum.

        Skipped when ``dead_time_max_s <= 0`` — the buffer path becomes a no-op.
        """
        if self.cfg.dead_time_max_s <= 0.0:
            self._dead_time_buffer = None
            self._dead_time_dt = dt
            self._dead_time_steps.zero_()
            return

        max_steps = max(1, int(math.ceil(self.cfg.dead_time_max_s / max(dt, 1e-9)))) + 1
        if (
            self._dead_time_buffer is None
            or self._dead_time_dt != dt
            or self._dead_time_buffer.shape[0] != max_steps
        ):
            self._dead_time_buffer = torch.zeros(
                max_steps, self.num_envs, device=self.device, dtype=torch.float32
            )
            self._dead_time_buffer_idx = 0
            self._dead_time_dt = dt
            self._refresh_dead_time_steps(env_ids=None)

    def _apply_dead_time(self, rate: torch.Tensor, dt: float) -> torch.Tensor:
        """Push ``rate`` through the per-env dead-time ring; return the
        per-env delayed rate. Cold-start envs (n_pushed ≤ depth) emit zero.

        1-D analog of ``GimbalRateLoop._apply_dead_time``.
        """
        if self.cfg.dead_time_max_s <= 0.0:
            return rate
        # Bug-3: read the cached all-zeros bool instead of doing a per-call
        # GPU->CPU sync on (steps == 0).all(). The cache is refreshed inside
        # _refresh_dead_time_steps, which is the only writer of
        # _dead_time_steps.
        if self._dead_time_curr_scale <= 0.0 and self._dead_time_all_zero:
            return rate

        self._ensure_dead_time_buffer(dt)
        if self._dead_time_buffer is None:
            return rate
        max_steps = self._dead_time_buffer.shape[0]

        # Push current rate into the ring.
        self._dead_time_buffer[self._dead_time_buffer_idx] = rate
        self._dead_time_n_pushed.add_(1)

        # Per-env pop index = (write_idx - depth) mod max_steps.
        d = self._dead_time_steps  # [N]
        idx = ((self._dead_time_buffer_idx - d) % max_steps).long()  # [N]
        env_arange = torch.arange(self.num_envs, device=self.device)
        delayed = self._dead_time_buffer[idx, env_arange]  # [N]

        # Cold-start: while fewer rates have been pushed than the env's
        # dead-time depth, emit zero (correct, since the buffer was zeroed
        # at reset). Defensive override against pre-fill leaks across
        # episodes.
        cold = self._dead_time_n_pushed <= d
        delayed = torch.where(cold, torch.zeros_like(delayed), delayed)

        # Advance write index (modulo).
        self._dead_time_buffer_idx = (self._dead_time_buffer_idx + 1) % max_steps

        return delayed
