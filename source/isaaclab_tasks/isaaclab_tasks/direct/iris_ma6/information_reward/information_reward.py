# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Team / difference (counterfactual) information reward from per-agent bearings.

Ticket 050 (cooperative track re-acquisition), Slice B.

A Fisher-information matrix with a prior over the target position, built from each agent's
bearing measurement:

    FIM = Λ_prior + Σ_{i: valid}  w_i · (I − d_i d_iᵀ),     w_i = 1 / (r_i² · σθ²)

where d_i is agent i's unit world-frame bearing to the target and r_i its (privileged GT) range.
Λ_prior (PD) makes FIM invertible for 0, 1, or N bearings, so the team estimate — and the reward —
is defined even during a single-agent deficit (unlike the LS triangulation which is NaN below 2).

- Team quality (A-optimality, matches the existing reward form): J(Σ) = sqrt(c / trace(Σ)),
  Σ = FIM⁻¹.
- Per-agent DIFFERENCE reward (marginal information contribution / counterfactual):
  r_i = J(Σ) − J(Σ_{−i}),  Σ_{−i} = (FIM − w_i (I − d_i d_iᵀ))⁻¹.
  r_i is large exactly when agent i's bearing is what collapses Σ — i.e. when it re-acquires a
  target a peer already holds. Agents without a valid detection contribute nothing ⇒ r_i = 0.

Calling contract: compute() is a pure, READ-only function of the current geometry — call once per
step in _get_rewards. No internal state.
"""

from __future__ import annotations

import torch

from .information_reward_cfg import InformationRewardCfg


class InformationReward:
    """Bearing-only FIM-with-prior team quality + per-agent difference reward."""

    def __init__(
        self,
        cfg: InformationRewardCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        self._cfg = cfg
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device
        self._eye = torch.eye(3, device=device)
        self._prior_mat = self._prior()  # [3, 3]

    def _prior(self) -> torch.Tensor:
        """Λ_prior = diag(1/σ_prior², 1/σ_prior², z_scale/σ_prior²) [3, 3] (PD)."""
        inv_var = 1.0 / (self._cfg.sigma_prior ** 2)
        diag = torch.tensor(
            [inv_var, inv_var, inv_var * self._cfg.z_prior_scale], device=self._device
        )
        return torch.diag(diag)

    def _quality(self, sigma: torch.Tensor) -> torch.Tensor:
        """A-optimality readout J = sqrt(c / trace(Σ)). sigma: [..., 3, 3] -> [...]."""
        if self._cfg.in_plane_only:
            trace = sigma[..., 0, 0] + sigma[..., 1, 1]
        else:
            trace = sigma[..., 0, 0] + sigma[..., 1, 1] + sigma[..., 2, 2]
        safe_trace = torch.clamp(trace, min=1e-6)
        return torch.clamp(torch.sqrt(self._cfg.quality_const_c / safe_trace), max=100.0)

    def compute(
        self,
        bearings_w: torch.Tensor,    # [N, A, 3] unit world-frame bearings to target
        cam_pos_w: torch.Tensor,     # [N, A, 3] camera/agent world positions
        target_pos_w: torch.Tensor,  # [N, 3] GT target world position (privileged, reward-only)
        valid: torch.Tensor,         # [N, A] bool — agent has a valid detection this step
    ) -> dict[str, torch.Tensor]:
        """Return {"team_quality": [N], "r_diff": [N, A]}. Pure / READ-only."""
        N, A = self._num_agents, self._num_agents  # noqa: F841 (clarity)
        N = bearings_w.shape[0]
        A = self._num_agents

        # Per-agent range to target (privileged GT), clamped.
        r = (target_pos_w.unsqueeze(1) - cam_pos_w).norm(dim=-1)  # [N, A]
        # Floor at a physically meaningful range (agents never legitimately sit on the target;
        # CBF/proximity keep them away). This also caps the weight's dynamic range.
        r = torch.clamp(r, min=0.5)
        w = 1.0 / (r ** 2 * self._cfg.sigma_theta ** 2)  # [N, A]

        # Per-bearing projection P_i = I − d d^T  [N, A, 3, 3], zeroed where invalid.
        d = torch.nn.functional.normalize(bearings_w, dim=-1)  # [N, A, 3]
        P = self._eye - d.unsqueeze(-1) * d.unsqueeze(-2)  # [N, A, 3, 3]
        info_terms = (w * valid.float()).unsqueeze(-1).unsqueeze(-1) * P  # [N, A, 3, 3]

        prior = self._prior_mat.unsqueeze(0)  # [1, 3, 3]
        reg = 1e-6 * self._eye  # tiny diagonal floor for robustness
        fim = prior + info_terms.sum(dim=1) + reg  # [N, 3, 3]

        # Invert in float64: a near-target bearing makes w huge, so the FIM is PD but very
        # ill-conditioned (~w/prior ≈ 1e8) — float32 inv reports it singular; float64 resolves it.
        sigma = torch.linalg.inv(fim.double()).to(fim.dtype)  # [N, 3, 3]
        team_quality = self._quality(sigma)  # [N]

        # Counterfactual (leave-one-out): drop each agent's information term. Computed by SUMMING
        # the OTHER agents' terms via a mask — NOT by subtracting from `fim` — because when an agent
        # is near the target w_i is enormous and `fim - info_terms[i]` would lose the prior to
        # float cancellation, yielding a singular matrix.
        loo_mask = 1.0 - torch.eye(A, device=self._device)  # [A, A], 0 on the diagonal
        loo_sum = torch.einsum("ij,njab->niab", loo_mask, info_terms)  # [N, A, 3, 3] = Σ_{j≠i} term_j
        fim_minus = prior.unsqueeze(1) + loo_sum + reg  # [N, A, 3, 3]
        sigma_minus = torch.linalg.inv(fim_minus.double()).to(fim_minus.dtype)  # [N, A, 3, 3]
        quality_minus = self._quality(sigma_minus)  # [N, A]

        r_diff = team_quality.unsqueeze(1) - quality_minus  # [N, A]
        # Invalid agents contribute a zero term ⇒ fim_minus == fim ⇒ r_diff ≈ 0; force exact 0.
        r_diff = torch.where(valid, r_diff, torch.zeros_like(r_diff))
        return {"team_quality": team_quality, "r_diff": r_diff}
