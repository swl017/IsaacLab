## Ticket 051 — Smooth σ-bound + final-phase LR anneal (final-policy commitment)

**Status**: Proposed.
**Created**: 2026-06-12
**Type**: Agent-side fix (model + agent-cfg). No env-side change. Final-policy *quality/commitment* only — explicitly NOT a strategy/cooperation change (that is [Ticket 050](../050-cooperative-track-reacquisition/ticket.md)).
**Target setup**: iris_ma6 MAPPO-RNN, post-048 (wide 256/256 default). Warm-start validation off the wide checkpoint.

**What**: The policy never commits — its action-noise σ is pinned at the `max_log_std` cap for ~all of training, and the learning rate is pinned at its floor. Both are artifacts, not optima. Fix the σ pin with a *smooth* log_std bound (so the existing entropy schedule can actually anneal σ in the commitment phase), and let the LR anneal post-curriculum by lowering `min_lr`. Result: a crisper, smoother, more deployable final policy for sim2real.

### Evidence (wide@123 seed-confirm run, 196k, `2026-06-12_01-35-21_..._wide_seed123_confirm`)

| step | LR | KL | σ | entropy_loss_scale | reward |
|---|---|---|---|---|---|
| 2k | 1.46e‑3 | 0.022 | 0.67 | 0.010 | 1152 |
| 40k | **3.0e‑4** | 0.037 | **1.492** | 0.010 | **5885** |
| 120k | 3.0e‑4 | 0.041 | 1.492 | 0.010 | 4949 |
| 160k | 3.0e‑4 | 0.043 | 1.492 | 0.0056 | 3914 |
| 195k | 3.0e‑4 | 0.044 | **1.492** | **0.0018** | 4138 |

- **σ pinned at the cap**: σ = 1.492 = e^0.4 (the `max_log_std=0.4` cap) for 94% of training. The entropy schedule ramps `entropy_loss_scale` 0.010 → 0.0018, **but σ never moves** — because the bound is a hard `torch.clamp` with **zero gradient** at the boundary. The schedule is therefore *inert* on σ. (The cfg comment at [skrl_mappo_rnn_cfg.yaml:22-25](../../../agents/skrl_mappo_rnn_cfg.yaml#L22) already flagged this and logged a "smooth-bound fix" follow-up — **this is that ticket**.)
- **LR pinned at the floor**: `min_lr = 3e-4 = initial learning_rate`, and KL is above the scheduler target (0.02) **100%** of training, so KLAdaptiveLR perpetually wants to lower LR and is clamped at the floor 83% of the time. LR is effectively a constant 3e-4; the scheduler can never anneal.
- **Coupling**: σ-at-cap → wide action distribution → large per-update KL → scheduler floors LR. **The two observations are one phenomenon.** Relaxing σ should also relax KL and un-floor the LR.
- **Consequence**: reward *peaks at 40k (5885)* and ends lower (~4100); the policy is optimized under permanent max action-noise and never sharpens μ. Deploy/eval uses the mean action, so σ-at-cap doesn't inject deploy noise directly — but it caps how precise/smooth the learned μ can become.

### Root cause

`MAPPORNNPolicy` builds on skrl's `GaussianMixin(clip_log_std=True, min_log_std, max_log_std)` ([mappo_rnn.py:1279-1280](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1279)), which applies `torch.clamp` to the state-independent `log_std_parameter` ([mappo_rnn.py:1283](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1283)). `torch.clamp` has zero gradient outside [min,max], so once log_std saturates at `max_log_std` the entropy gradient that would pull it back is killed.

### Slices

**Slice 1 — Smooth log_std bound.**
- Set `clip_log_std=False` and apply a smooth saturating map in `compute()` so log_std ∈ (min_log_std, max_log_std) with **nonzero gradient near both bounds**, e.g.
  `log_std = min + (max - min) * sigmoid(raw_param / s)` or a softplus-based soft-clamp.
- Preserve the envelope σ ∈ [~0.007, 1.49] but make the boundary differentiable so the entropy schedule can anneal σ down as `entropy_loss_scale` drops.
- Unit-check: gradient of log_std wrt raw param is > 0 at log_std = max_log_std − ε.

**Slice 2 — Final-phase LR anneal.**
- Lower `min_lr` (e.g., 3e-4 → 1e-4) so KLAdaptiveLR can reduce LR once the curriculum is done. Keep `max_lr` for adaptation capacity *during* the curriculum.
- Optional: step the LR floor down at curriculum end (~200k) rather than globally, to keep early-curriculum adaptation intact.

**Slice 3 — Warm-start validation (cheap, ~+50–100k).**
- Warm-start from the wide `best_agent.pt` / `agent_180000.pt` with Slices 1+2 and check:
  - σ trace **anneals below the cap** once `entropy_loss_scale` is small (no longer flat at 1.492).
  - KL tail relaxes back into band (≤ ~0.03); LR floats below 3e-4 in the final phase.
  - Smoothness improves: `total_rms` / `cmd_vel_delta` down vs the pinned-σ wide baseline.
  - Task quality preserved (triangulation / bbox_center / pair_valid within ±5%).
- Warm-start (not cold 200k) keeps this a fast turn and de-risks σ collapse.

### Acceptance criteria

| Criterion | Result |
|---|---|
| log_std bound differentiable at the cap (unit test) | pending |
| σ anneals below cap in the commitment phase (warm-start TB) | pending |
| KL tail ≤ ~0.03 and LR drops below 3e-4 post-curriculum | pending |
| `total_rms` / `cmd_vel_delta` reduced vs pinned-σ wide baseline | pending |
| Task quality (triangulation, bbox_center, pair_valid) within ±5% | pending |
| Deploy bag: smoother gimbal/cmd_vel, no tracking regression | pending |

### Scope boundary

- **DO**: smooth σ bound, lower `min_lr` for post-curriculum anneal, warm-start validation on wide.
- **DO NOT**: change reward weights, task difficulty, or the entropy *schedule values* (the schedule is fine — the clamp defeats it). Slice 2 may let `entropy_loss_scale` → 0 at the very end, but that is downstream of the smooth bound.
- **DO NOT**: touch the agent control envelope (`max_lin_vel`, slew) — sysid-locked.
- **DO NOT**: expect a strategy/cooperation change. Max-σ exploration already failed to find cooperation (see t050); this ticket sharpens the *chosen* policy, it does not change *which* policy is chosen.

### Risk

- **σ collapse if annealed too early.** Cautionary precedent: t047's flat `entropy_loss_scale=0.001` collapsed σ to 0.24 by 16k and lost grip on later curriculum phases. Mitigation: the smooth bound only takes effect *with* the existing 120k–200k entropy ramp (post-curriculum), and warm-start validation catches collapse immediately. Keep `min_log_std` floor.
- **Lower min_lr slowing mid-curriculum adaptation.** Mitigation: only lower the floor after curriculum end, or validate via warm-start (already post-curriculum).

### Coupling

- **Ticket 047** (entropy curriculum + `max_log_std` 0.7→0.4) — this completes t047's intent; the schedule it added is currently defeated by the hard clamp.
- **Ticket 048** (wide default) — validate the fix on the shipped wide policy.
- **Ticket 050** (cooperation) — orthogonal; 051 is quality/commitment, 050 is strategy. Do not conflate.
- σ-cap diagnostic (`project_iris_ma6_sigma_cap_diagnostic`).

### Affected files

- [scripts/reinforcement_learning/skrl/mappo_rnn.py](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py) — `MAPPORNNPolicy`: smooth log_std bound (replace `clip_log_std=True` clamp).
- [agents/skrl_mappo_rnn_cfg.yaml](../../../agents/skrl_mappo_rnn_cfg.yaml) — `learning_rate_scheduler_kwargs.min_lr`; update the `max_log_std` comment that flagged this follow-up; optional `smooth_log_std_bound` flag.

### References

- wide@123 confirm run TB: `logs/skrl/iris_ma6/2026-06-12_01-35-21_mappo_rnn_torch_ticket048_net_width_wide_seed123_confirm`.
- [skrl_mappo_rnn_cfg.yaml:22-25](../../../agents/skrl_mappo_rnn_cfg.yaml#L22) — the "smooth-bound fix is logged as a follow-up ticket" note this ticket discharges.
- [mappo_rnn.py:1279-1283](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1279) — GaussianMixin clip_log_std + log_std_parameter.

**Flow**: Low–medium. One model change (smooth bound) + one cfg change (min_lr) + a warm-start A/B. The fix is small; the load-bearing part is the warm-start validation showing σ anneals without collapsing and smoothness improves with task quality held.
