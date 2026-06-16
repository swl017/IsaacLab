# Stage I — Design Document (Slice B): Team / difference (information) reward

## Problem statement
The cooperative reward is shared/identical across agents and is exactly ZERO during a single-agent
deficit (LS covariance is NaN→0 below 2 detections) — no credit for the agent that re-acquires, and
no signal precisely when recovery matters. Slice B introduces an information reward: a per-agent
DIFFERENCE (counterfactual) reward built from a Fisher-information matrix with a prior, defined for
0/1/N detections, plus a curriculum that rebalances away from bbox-90 dominance (C2).

## Proposed approach
New standalone module `information_reward/` computing a 3×3 information matrix from per-agent
bearings, plus an env-side curriculum rebalance and the privileged-critic flag.

**(1) FIM-with-prior (always defined).** Each agent i with a valid detection contributes a
bearing-only information term about the target position. For a bearing of angular noise σθ from
camera position p_i at GT range r_i, with unit ray d_i, the information is concentrated ⊥ to the ray:

    FIM = Λ_prior + Σ_{i: valid}  w_i · (I − d_i d_iᵀ),     w_i = 1 / (r_i² · σθ²)

`Λ_prior = Σ_prior⁻¹` (PD) makes `FIM` invertible always ⇒ `Σ = FIM⁻¹` defined at 0/1/N bearings.
Readout matches the current reward form (trace / A-optimality): `J(Σ) = sqrt(c / trace(Σ))`.

**(2) Per-agent difference (counterfactual) reward.** Agent i's marginal information contribution:

    r_i = J(Σ)  −  J(Σ_{−i}),     Σ_{−i} = (FIM − w_i (I − d_i d_iᵀ))⁻¹

Drop-one-rank per valid agent (N+1 small inverses/step, vectorized). r_i is large exactly when i's
bearing is what collapses Σ — i.e. when it re-acquires a target a peer already holds. Agents without a
valid detection get r_i = 0 (no contribution to remove).

**(3) Derived defaults (Q#2).** σθ from detector calibration: `σθ ≈ σ_pix / f_eff`, σ_pix≈7 px
(delay/replicator bbox noise), `f_eff = img_w / (2 tan(hfov_eff/2))` → σθ ≈ a few mrad; I'll compute
the exact number in S from the live intrinsics. `Σ_prior = σ0² I`, σ0 ~ scenario position scale
(≈ target_distance_max = 40 m) so the no-measurement state is high-uncertainty and a second bearing
gives a large collapse. All exposed as cfg, marked tunable; D-optimality (logdet) left as an ablation.

**(4) Rebalance curriculum (Q#4 — proposed schedule).** New phase `reward_rebalance_{start,end}_step`
driving `p_rebal∈[0,1]`; bootstrap single-agent tracking first, then shift to the team objective:

| phase | steps | bbox_center | bbox_size | info-difference scale |
|---|---|---|---|---|
| bootstrap | 0 – 80k | 90 | 30 | 0 |
| ramp | 80k – 200k | 90→30 | 30→20 | 0→full |
| team | 200k – 400k | 30 | 20 | full |

`bbox_center_eff = 90 + p_rebal·(30−90)`, `info_scale_eff = p_rebal·info_scale_max`. bbox kept
nonzero throughout (still need the target framed to *get* a bearing). Rides the existing `progress_*`
machinery; the info term replaces/augments the current shared `triangulation` slot (which stays as the
bootstrap-era team signal until the ramp).

**(5) Privileged critic (Q#7).** Set `enable_critic_gt_target=True` for Slice-B runs — GT target
position to the centralized critic only (actor obs unchanged), stabilizing value through the deficit.
Must be set pre-`__post_init__` (R gap #5 — same Hydra caveat as Slice A); handled in the env-cfg
default for the Slice-B task/experiment, not via late `from_dict`.

## Key interfaces and data flow
```
information_reward/  (standalone; torch + configclass only)
  InformationRewardCfg: sigma_theta, sigma_prior, quality_const c, info_scale_max, enabled
  InformationReward(cfg, num_envs, num_agents, device)
    compute(bearings_w[N,A,3], cam_pos_w[N,A,3], target_pos_w[N,3], valid[N,A])
      -> { team_quality[N], r_diff[N,A] }     # READ-only, pure function of current geometry

_get_rewards():
  bearings/cam_pos/target/valid already gathered (mirrors _gather_reacq_signals)
  out = self._info_reward.compute(...)
  rewards["triangulation"] blended: (1-p_rebal)*shared_trace + p_rebal*(r_diff[i]*info_scale)  ×progress_coord
  bbox_center scale -> curriculum-scaled by p_rebal
```
- No obs change; reward-side only; GT target privileged (reward + critic), consistent with current
  `use_gt_target=True`.
- Curriculum: add `reward_rebalance_*` + `get_reward_rebalance_progress()` to `curriculum_cfg.py`.

## What this does NOT include
Explicit peer-bearing obs channel / learned observer (Slice C — peer bearing already in obs); GNN/
attention (Slice D); obs changes; scenario changes (Slice A reused); sysid envelope; potential-based
shaping (we WANT the info objective to shape the optimum); logdet/GDOP (trace chosen; logdet = ablation).

## Open risks (for engineer review)
1. σθ / Σ_prior set the reward shape; reward sensitivity to them is real — derive from calibration +
   keep as cfg; ablate. (Estimation-flavored tuning — your domain.)
2. Difference reward could in principle incentivize an agent to let a teammate's view degrade to inflate
   its own marginal — counterfactual rewards are generally robust to this, but worth a diagnostic.
3. Vertical axis is weakly observed by near-coplanar bearings; the 3×3 `Σ` may be ill-conditioned in z.
   Option (S): restrict the readout to the in-plane 2×2 or add a stronger z-prior. Flag.
4. Scale matching: r_diff magnitude vs bbox must be comparable at the ramp end — the rebalance numbers
   are first proposals; the gate run calibrates them.
5. Reward uses GT range r_i (privileged, reward-only) — must NOT leak to actor/obs (it doesn't).
6. Bootstrap length: if 80k is too short under the raised scenario, single-agent tracking may not
   consolidate before the team reward ramps — adjustable.
