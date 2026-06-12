## Ticket 050 — Cooperative track re-acquisition (the MARL contribution)

**Status**: Proposed (research thread / parent ticket — to be sliced into child tickets). No scheduling commitment.
**Created**: 2026-06-12
**Type**: Research program. Dec-POMDP cooperative active-perception. This is the *scientific* centerpiece of the Iris project and the intended high-value-journal contribution (T-RO / RA-L / Autonomous Robots), not an incremental tuning ticket.

**Goal**: Agents that **hand off and re-acquire a target track using peer bearing information** under realistic perception loss (FOV exit, occlusion, detection dropout). Concretely: when agent A loses its detection while peer B still sees the target, A re-points and re-acquires using B's information — and an ablation that removes the peer channel fails to recover. That ablation is the paper's central claim; the existing PX4 deploy-bag harness is the sim2real validation backbone.

**Observed gap (2026-06-12)**: Across the t048 width sweep (64/128/256) and at deploy, the policy shows **limited cooperative behavior** — the current policy *cannot regain a lost track using peer information*. Width does not fix it (t048: capacity is not the binding constraint). mid's degenerate "fly close to maximise own bbox" deploy strategy (cov 15× worse) is the same pathology from the other side: the policy optimises *individual* framing, never *team* geometry.

### Diagnosis — why cooperation is absent (grounded in the obs topology)

Per-agent observation = **26D ego + 14D per peer + 6D triangulation**. The peer block carries:
`position(3), velocity(3), gimbal_azimuth_world(1), gimbal_elevation_world(1), gimbal_az_rate(1), gimbal_el_rate(1), zoom(1), bbox_empty(1), data_age(1), bbox_age(1)`.

So the information for re-acquisition is *mostly present* (peer position + peer **world-frame gimbal ray** + a binary "peer sees target"). The behavior is blocked by **three structural causes — none of them network capacity**:

| # | Cause | Mechanism | Fix axis |
|---|---|---|---|
| **C1** | **The task never creates the trigger.** | t046 tuned spawn so the target is "near-permanently in view." Single-agent track-loss events — the *only* situation cooperation is for — barely occur in training, so there is near-zero gradient for "recover using peer." | Scenario / curriculum (composes with t045/t046 difficulty-revert) |
| **C2** | **Reward is ~purely individual, and the cooperative signal vanishes when needed.** | bbox_center=90 vs triangulation=8 (>10× selfish). And triangulation needs ≥2 valid detections, so the 6D fused-estimate channel goes **NaN the instant one agent loses the target** — the most direct "where is it" signal disappears exactly at loss. Recovery becomes sparse, delayed-credit, multi-agent exploration. | Reward shaping (team / difference reward) |
| **C3** | **Peer signal is a proxy and the fusion is hard.** | Agent receives peer *gimbal pointing*, not peer *target bearing/estimate*; must infer "peer gimbal az/el when peer.bbox_empty=0 ≈ bearing to target," then intersect with its own stale belief — an implicit recurrent Bayesian inference with no scaffolding. | Communication channel + (optional) learned cooperative observer |

C1 is load-bearing: until track-loss events exist, no reward or architecture can teach recovery.

### Estimation-theoretic framing (the controls-flavored contribution)

The team objective is minimizing the target's **posterior covariance** (equivalently maximizing FIM determinant / minimizing GDOP). An agent that re-acquires a lost target restores the second bearing → collapses the covariance. This makes the cooperative reward **dense, principled, and well-defined**, and distinguishes the work from generic MARL:

- **Team reward**: −trace(Σ_target) or +logdet(FIM) of the fused estimate.
- **Difference / counterfactual reward** (per-agent credit): `r_i = J(team with measurement i) − J(team without i)` = agent i's *marginal information contribution*. This directly pays an agent for re-acquiring, because its restored bearing is what collapses Σ. Rides on the **centralized critic already in place** (CTDE / asymmetric actor-critic plumbing in this env).
- **Belief-state view**: re-acquisition is belief propagation through a measurement gap (Dec-POMDP). The RNN hidden state is the belief; the peer bearing is the update. A **learned cooperative observer** (recurrent belief / differentiable filter over target position fed by own + peer bearings) is the natural architecture story — and a far better novelty than width/depth.

### Program of work (slices → child tickets, in priority)

**Slice A — Make the scenario demand cooperation (prerequisite).**
- Raise task difficulty so the target leaves individual FOVs (composes with the t045/t046 difficulty-revert: raise the curriculum *ceiling*, keep the floor easy).
- Exploit the existing **dropout/occlusion curriculum** in the delay system to force single-agent detection loss while the peer retains the target — i.e., *manufacture the hand-off event*.
- Instrument track-loss / re-acquisition events (onset, duration, recovery) as first-class metrics.
- **Gate**: track-loss events occur at a measurable, non-trivial rate in training. Without this, A/B below are untrainable.

**Slice B — Reward the team, not the individual.**
- Add the covariance/FIM team reward and the per-agent difference reward (Slice-B child ticket owns the exact form + weights). Rebalance away from bbox-90 dominance.
- Leverage the centralized critic for credit assignment.
- **Gate**: re-acquisition success rate and time-to-reacquire improve vs the C2-baseline reward.

**Slice C — Sharpen the cooperative channel.**
- Broadcast each agent's target *bearing/estimate* explicitly (the deploy stack already publishes `chosen_target_ray_w` / `target_rays_w` — the exact signal). Add to the peer obs block.
- Optional: learned cooperative observer (recurrent belief / differentiable filter).
- **Gate**: ablating the peer-bearing channel measurably degrades re-acquisition (the paper's central ablation).

**Slice D — Architecture (LAST).**
- Attention / GNN comm over peers, primarily for the >2-agent scaling figure. Deprioritized: t048 showed capacity is not the binding constraint.

### Evaluation / the paper claim

The killer figure + ablation:
- Occlude the target from agent A; show A re-acquires in *t* s using B's bearing.
- **Ablation removing the peer channel fails to recover.**
- Metrics: re-acquisition success rate, time-to-reacquire, team-track-maintenance under occlusion, team covariance trace through the loss window.
- **Reproduce on a PX4 deploy bag** (Pegasus + PX4 SITL + mas) using the existing bag→metrics pipeline (`experiments/outputs/t048_netwidth/deploy_tracking_compare.py` + ticket-049 `bag_to_csv.py`).

### Scope boundary

- **DO**: manufacture track-loss/hand-off scenarios; team/difference reward in estimation terms; explicit peer-bearing channel; the re-acquisition ablation; sim2real validation on deploy bags.
- **DO NOT (here)**: loosen the agent control envelope (`max_lin_vel`, slew) — that re-opens sysid and is explicitly out (see the t045/049 sysid-lock rationale). Higher *task* difficulty via initial-conditions/target-behaviour only.
- **DO NOT**: lead with architecture (comm/attention) — t048 ruled out capacity as the bottleneck.
- **DO NOT**: treat this as one PR — it is a multi-ticket research arc (months).

### Coupling / dependencies

- **Difficulty-revert thread (t045/t046 reversal)** — Slice A *is* that revert, doing double duty (sim2real generalization + creating the cooperation trigger). Schedule together.
- **t047 reward retune** — Slice B extends it with the covariance difference-reward; the bbox-90 dominance it left is exactly C2.
- **t048 width sweep** — supplies the "capacity is not the bottleneck" result that justifies deprioritizing architecture; wide (256) is the current default to build on.
- **t049 gimbal diagnosis** — shares the deploy-bag harness; cooperative re-acquisition stresses the gimbal slew under loss, so the two evals share instrumentation.
- **Delay/dropout system** — Slice A reuses its occlusion/dropout curriculum to manufacture single-agent loss.
- **Centralized critic (asymmetric actor-critic)** — already in the env; the CTDE substrate for the difference reward.

### Open questions (to resolve in child tickets)

1. Exact difference-reward form: covariance-trace vs logdet-FIM vs GDOP; potential-based shaping to keep it dense without changing the optimum.
2. Whether to share peer *bearing* (ray) vs peer *unilateral estimate* (ray + range prior) vs both.
3. Whether the learned cooperative observer (Slice C option) beats letting the RNN fuse implicitly — ablation.
4. Occlusion model fidelity: scripted FOV-exit vs detector-dropout vs real-clutter occlusion, and which transfers.
5. Multi-seed budget: this thread needs seed-confirmed claims for publication, not single-seed.

### References

- [iris_ma_env6_test.py:160-189](../../../../iris_ma_env6_test.py#L160) — observation topology (ego / per-peer 14D / triangulation 6D); the peer block is the cooperative channel.
- [iris_ma_env6_test.py:1813](../../../../iris_ma_env6_test.py#L1813) — `pair_valid` requires ≥2 valid detections → the fused-estimate channel is NaN at single-agent loss (C2).
- [doc/experiments/2026-06-11_ticket048_network_width_sweep.md](../../../experiments/2026-06-11_ticket048_network_width_sweep.md) — capacity-not-the-bottleneck result + deploy behavioural analysis (mid's selfish strategy).
- [Ticket 047](../047-reward-retune-bbox-vs-triangulation/ticket.md) — reward retune predecessor (bbox-90 dominance = C2).
- [Ticket 045](../045-task-difficulty-calibration/ticket.md), [Ticket 046](../046-closer-spawn-and-2d-target-motion/ticket.md) — the difficulty that Slice A partially reverts.
- [Ticket 049](../049-gimbal-oscillation-timeseries-diagnosis/ticket.md) — shared deploy-bag harness.

**Flow**: High (research arc). Sequence: Slice A (scenario, prerequisite) → Slice B (team/difference reward) → Slice C (peer-bearing channel + ablation = the paper claim) → Slice D (architecture, optional). Each slice is its own child ticket with seed-confirmed evaluation. Months, not days — but it is the actual contribution.
