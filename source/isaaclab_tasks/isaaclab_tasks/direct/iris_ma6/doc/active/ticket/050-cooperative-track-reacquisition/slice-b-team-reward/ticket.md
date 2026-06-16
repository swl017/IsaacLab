# Ticket 050 / Slice B — Reward the team, not the individual

**Parent:** [../ticket.md](../ticket.md) (050 — Cooperative track re-acquisition)
**Predecessor:** [../slice-a-scenario-trigger/](../slice-a-scenario-trigger/) (DONE; gate verified)
**Status:** Stage Q (proposed)
**Flow:** Full QRISPY

## What
Slice A manufactured single-agent peer-assisted track-loss (2.05 events/ep) and measured the
selfish-policy baseline: **reacq_success_rate = 0.385** (the C2 baseline). But the reward is still
~purely individual (bbox_center=90 dominates), so the policy has little incentive to recover a lost
track using peer information. Slice B adds a **team / difference (counterfactual) reward in
estimation-theoretic terms** — pay each agent for its *marginal information contribution* to the
team's target estimate — and rebalances away from bbox-90 dominance (cause C2).

## Why
With the cooperation trigger in place (Slice A), the C2 reward structure is now the binding
constraint: the most direct "where is it" signal vanishes at single-agent loss, and recovery is
sparse, delayed-credit, multi-agent exploration. A covariance/FIM team reward makes the cooperative
signal **dense, principled, and well-defined**, and the per-agent difference reward directly pays an
agent for re-acquiring (its restored bearing collapses Σ). This rides on the centralized critic
(CTDE) already in the env.

## Scope boundary
- IN: team reward (−trace Σ / +logdet FIM of the fused estimate); per-agent difference/counterfactual
  reward (marginal information contribution); rebalance of bbox/triangulation weights; optional
  privileged-critic GT-target stabilization; multi-seed (≥3) evaluation of the headline.
- OUT: explicit peer-bearing channel / learned observer (Slice C — peer bearing is already in obs as
  other_ray_w); architecture (Slice D); the scenario itself (Slice A, reused as-is); sysid envelope
  (max_lin_vel/slew LOCKED); obs changes.

## Acceptance criteria (gate)
- reacq_success_rate and time-to-reacquire **improve vs the Slice-A C2 baseline** (0.385 / 0.46 s),
  seed-confirmed (≥3 seeds), under the same track_loss_scenario.
- No collapse of base tracking quality (pair_valid_rate, triangulation) vs the t048 wide baseline.

## Build-on
t048 wide lead checkpoint (warm-start), with enable_track_loss_scenario=True +
cooperation_metrics.enable=True. Sysid LOCKED. Multi-seed per methodology.

## Key references
- C2 baseline: [../slice-a-scenario-trigger/eval_baseline_results.md](../slice-a-scenario-trigger/eval_baseline_results.md)
- Parent Open Question #1 (difference-reward form) + the estimation-theoretic framing section.
