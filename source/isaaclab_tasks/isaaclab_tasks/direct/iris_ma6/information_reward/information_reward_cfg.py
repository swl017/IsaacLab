# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the team / difference information reward (ticket 050, Slice B)."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class InformationRewardCfg:
    """Configuration for :class:`InformationReward`.

    Builds a Fisher-information matrix with a prior from per-agent bearings and produces a team
    quality (A-optimality, trace-of-covariance readout — matches the existing reward form) and a
    per-agent DIFFERENCE (counterfactual) reward = each agent's marginal information contribution.
    Reward-side only; uses privileged GT range/target (never enters obs).
    """

    enabled: bool = False
    """Whether the information reward is active. Default False = bit-exact baseline."""

    # ===== Measurement / prior (estimation constants) =====

    sigma_theta: float = 0.005
    """Bearing angular noise [rad]. Sets a bearing's information weight w = 1/(r^2 sigma_theta^2).
    Placeholder; the env computes the calibrated value from live intrinsics (sigma_pix / f_eff)."""

    sigma_prior: float = 40.0
    """Prior position std [m] ~ scenario position scale. Sets the no-measurement uncertainty floor
    (FIM_prior = diag(1/sigma_prior^2, ...)) so the matrix is always invertible (defined at 0/1/N
    bearings) and a second bearing produces a large covariance collapse."""

    z_prior_scale: float = 4.0
    """Extra z (vertical) prior information multiplier for the full 3x3 readout: the z prior is
    z_prior_scale / sigma_prior^2. Stabilizes the weakly-observed vertical axis. Unused if
    ``in_plane_only``."""

    # ===== Readout =====

    quality_const_c: float = 10.0
    """Constant in the quality readout J = sqrt(c / trace(Sigma)). 10.0 matches the existing
    _compute_fim_reward()."""

    in_plane_only: bool = False
    """If True, the quality readout uses only the xy (2x2) covariance block (vertical excluded).
    If False, full 3x3 trace with the z_prior_scale stabilizer."""

    # ===== Scaling =====

    info_scale_max: float = 8.0
    """Scale applied to the per-agent difference reward at full curriculum strength (~ the existing
    triangulation_reward_scale). The env's rebalance curriculum ramps the effective weight 0->this."""
