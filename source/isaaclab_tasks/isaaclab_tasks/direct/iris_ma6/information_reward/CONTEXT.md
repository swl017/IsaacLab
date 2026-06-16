# information_reward — Module Context

## Purpose
Team / difference (counterfactual) information reward for cooperative active perception (ticket 050,
Slice B). Pays each agent its MARGINAL information contribution to the team's target estimate, so an
agent that re-acquires a lost track (restoring a second bearing that collapses Σ) is directly
rewarded. Replaces the shared, NaN-below-2-detections trace reward with an FIM-with-prior that is
defined at 0/1/N bearings.

## Inputs (per step, via compute)
- `bearings_w` [N, A, 3] — per-agent unit world-frame bearing to target.
- `cam_pos_w` [N, A, 3] — per-agent (camera) world positions.
- `target_pos_w` [N, 3] — privileged GT target position (reward-side only).
- `valid` [N, A] bool — per-agent valid-detection mask.

## Outputs (via compute → dict)
- `team_quality` [N] — A-optimality readout J = sqrt(c / trace(Σ)), Σ = FIM⁻¹.
- `r_diff` [N, A] — per-agent difference reward J(Σ) − J(Σ₋ᵢ); 0 for agents without a valid detection.

## Math
`FIM = Λ_prior + Σ_valid w_i (I − d_i d_iᵀ)`, `w_i = 1/(r_i² σθ²)`. Λ_prior PD ⇒ always invertible.
A bearing informs ⊥ to its ray (rank-2 term); two well-separated bearings localize, collinear do not
(GDOP). Readout = A-optimality (trace); D-optimality (logdet) is a possible ablation.

## Dependencies
None (standalone; torch + isaaclab.utils.configclass). Consumed by iris_ma_env6_test.py `_get_rewards`.

## Key Files
- `information_reward_cfg.py` — InformationRewardCfg (sigma_theta, sigma_prior, z_prior_scale,
  quality_const_c, in_plane_only, info_scale_max, enabled).
- `information_reward.py` — InformationReward (compute, _build via _prior/_quality).
- `tests/run_tests.py` — standalone suite.

## Calling Contract (§4.4)
- `compute(...)`: **READ-only / pure** — no internal state mutation; safe to call any number of times.
  Call once per sim step in `_get_rewards`, before the per-agent reward assembly.
- Privileged GT target/range is used for the reward only and must NOT enter observations.
