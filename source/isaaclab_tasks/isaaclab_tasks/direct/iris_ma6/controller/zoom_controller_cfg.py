# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for zoom controller.

Two model variants:

- ``model = "first_order"`` (default — backward-compatible, pre-mas/037 behavior):
  pure first-order lag toward an integrator that tracks ``cmd * max_zoom_rate``.
  Only ``tau_zoom``, ``zoom_min``, ``zoom_max``, ``max_zoom_rate`` are read in
  this mode. The other fields are present but unused.

- ``model = "siyi_a8"`` (mas/037 fitted SIYI A8 mini model): four-stage pipeline
  (action denorm → ±v_max slew clip → input-side dead-time buffer → integrator
  → first-order lag), with the integrator's continuous post-lag state
  quantized at 0.1 levels on the *published* path (internal state stays
  continuous so sub-quantum momentum is preserved). Reference:
  ``/home/usrg/mas/src/scripts/sim2real_model_fitting/output/zoom_model.json``
  (mas/037 ticket).
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ZoomControllerCfg:
    """Configuration for zoom controller.

    The default ``model="first_order"`` preserves bit-exact pre-mas/037 behavior
    on existing checkpoints, training runs, and tests. Opt in to the measured
    SIYI A8 model by setting ``model="siyi_a8"`` (or via yaml override).
    """

    model: str = "siyi_a8"
    """Zoom dynamics model selector. ``"first_order"`` (default) or ``"siyi_a8"``.

    - ``"first_order"``: legacy continuous first-order lag — only ``tau_zoom``,
      ``zoom_min``, ``zoom_max``, ``max_zoom_rate`` are consumed.
    - ``"siyi_a8"``: full mas/037 four-stage model — slew clip, input-side
      dead-time buffer, integrator, first-order lag, with output quantization
      at 0.1 levels on the published path.
    """

    tau_zoom: float = 0.1
    """First-order lag time constant on the post-integrator zoom level [s].

    Default 0.1s preserves the legacy ``first_order`` model's behavior. When
    using ``model="siyi_a8"``, override to the mas/037 fit (0.091 s) via cfg
    or yaml (left as 0.1 in the dataclass default to keep backward compat).
    """

    zoom_min: float = 1.0
    """Minimum zoom level (1x = no zoom). Hard clamp on integrator state."""

    zoom_max: float = 5.0
    """Maximum zoom level (5x). Hard clamp on integrator state."""

    max_zoom_rate: float = 2.0
    """Policy action scale: ``[-1, 1]`` rate command → physical levels/s.

    Matches the deployed ``mas_policy.action_publisher.max_zoom_rate``. Default
    2.0 levels/s, kept BELOW the lens v_max (3.16 levels/s) so the policy's
    full action range stays in the linear regime where the SIYI model is
    faithful (mas/037 Tip 5).
    """

    # =========================================================================
    # mas/037 SIYI A8 model parameters (consumed only when model="siyi_a8")
    # =========================================================================

    v_max_levels_per_s: float = 3.16
    """Lens-level slew clip [levels/s], applied after the action scaler and
    before the dead-time buffer. Symmetric in/out (mas/037 fit reports 0.2%
    asymmetry, well below the 10% acceptance). Only consumed when
    ``model="siyi_a8"``.
    """

    quantum_levels: float = 0.1
    """Output quantum on the *published* zoom level [levels]. The integrator's
    internal state stays continuous; only the published value (consumed by
    obs / intrinsics / bbox raycaster) snaps to this quantum (mas/037 Tip 1).
    Only consumed when ``model="siyi_a8"``.
    """

    dead_time_mean_s: float = 0.100
    """Mean of the input-side dead-time Gaussian [s] (mas/037 fit). Sampled
    per env at episode reset; effective mean is scaled by the curriculum.
    Only consumed when ``model="siyi_a8"``.
    """

    dead_time_std_s: float = 0.018
    """Std of the input-side dead-time Gaussian [s] (mas/037 fit). Used at
    episode reset to draw a per-env sample, scaled by the curriculum.
    Only consumed when ``model="siyi_a8"``.
    """

    dead_time_max_s: float = 0.150
    """Hard cap on per-env dead time [s] (≈ mean + 3σ). Bounds the ring-buffer
    depth (``ceil(dead_time_max_s / dt)`` slots pre-allocated) and clips
    Gaussian samples that would exceed it. Only consumed when
    ``model="siyi_a8"``.
    """

    dead_time_curriculum_scale: float = 0.0
    """Curriculum-driven scaling on the sampled dead time ∈ [0, 1].
    0 = no delay (preserves pre-mas/037 behavior; the dead-time stage is a
    no-op), 1 = full measured Gaussian. Driven by the env's per-step call
    to ``set_dead_time_curriculum_scale`` (mas/036 pattern). Only consumed
    when ``model="siyi_a8"``.
    """

    randomize_v_max: bool = False
    """[RESERVED — DO NOT enable.] mas/037 Tip 6: v_max is a measured
    firmware constant, asymmetric DR risks tipping the policy into the
    saturation-stall regime that mas/037 explicitly engineered around.
    Field present for forward-compat only; no DR pathway consumes it yet.
    """
