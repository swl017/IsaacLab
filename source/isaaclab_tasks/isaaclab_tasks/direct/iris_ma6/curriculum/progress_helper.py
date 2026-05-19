# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-(env, agent) effective curriculum progress sampler.

For a global curriculum progress ``p ∈ [0, 1]`` (e.g.
``CurriculumCfg.get_noise_progress(current_step)``), this module produces a
per-(env, agent) tensor of effective progress values drawn i.i.d. uniformly
from ``[0, p]``. The env distribution then always retains positive mass on
the easy regime (``eff_p ≈ 0``) at every global ``p``, which is the
anti-forgetting invariant enforced by ticket 034.

Stateless. Consumers (env._reset_idx) call ``sample_per_env_progress`` once
per axis per reset and cache the resulting tensor on the env until the next
reset overwrites it.
"""

from __future__ import annotations

import torch


def sample_per_env_progress(
    global_progress: float,
    num_envs: int,
    num_agents: int,
    device: torch.device,
    env_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw per-(env, agent) effective progress ~ Uniform(0, global_progress).

    Args:
        global_progress: Scalar curriculum progress in ``[0, 1]``. Clamped
            internally so callers can pass through values outside the range
            without worrying about it.
        num_envs: Returned tensor's first-dim size when ``env_ids`` is
            ``None``. Otherwise must match ``env_ids.numel()`` (callers can
            use this to be explicit) — when ``env_ids`` is provided, the
            returned tensor has shape ``(len(env_ids), num_agents)``.
        num_agents: Number of agents per env.
        device: Output tensor device.
        env_ids: Optional 1-D tensor of env indices the caller is resampling.
            Only its ``numel()`` is consulted for the output shape; the
            indices themselves are not used (the caller is expected to slice
            its destination tensor with the same ``env_ids``).
        generator: Optional :class:`torch.Generator`. When provided, sampling
            uses it for reproducibility. When ``None``, sampling uses the
            default global generator.

    Returns:
        ``Tensor[E, A]`` of values in ``[0, clamp(global_progress, 0, 1)]``
        drawn i.i.d. uniformly per element, where ``E = num_envs`` (or
        ``env_ids.numel()`` when provided) and ``A = num_agents``.

        At ``global_progress = 0`` the result is exactly zero. At
        ``global_progress = 1`` the result spans the full ``[0, 1]`` range
        per element.
    """
    p = max(0.0, min(1.0, float(global_progress)))

    if env_ids is not None:
        rows = int(env_ids.numel())
    else:
        rows = int(num_envs)

    if p == 0.0:
        return torch.zeros((rows, num_agents), device=device, dtype=torch.float32)

    u = torch.rand(
        (rows, num_agents),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return u * p
