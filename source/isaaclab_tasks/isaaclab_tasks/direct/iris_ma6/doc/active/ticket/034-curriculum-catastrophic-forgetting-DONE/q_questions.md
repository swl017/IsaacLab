# Stage Q — Questions and Answers

**Ticket**: 034-curriculum-catastrophic-forgetting
**Date**: 2026-05-17
**Engineer**: Seungwook Lee
**Gate status**: Q closed. Engineer to confirm → `Start R`.

---

## Resolved questions

### Assumptions

- **Q1. Reproduce the regression first?**
  **A:** Yes — quantify the gap before changing curriculum code. Stage R (or an explicit pre-R measurement task) must load the 400k checkpoint and evaluate it under 40k-step env settings, logging per-component reward, before any code change.

- **Q2. Failure attribution — fix all clusters at once?**
  **A:** Yes, fix all. No time for per-axis ablation.
  **Constraint added:** the current delay pipeline has a bug such that the **safe minimum latency is 2 steps of the 25 Hz policy rate (= 80 ms)**. The difficulty samplers for the `fixed_delay` / `random_delay` axes must respect this floor — `latency ~ Uniform(2*policy_dt, cfg.latency * progress)`, not `Uniform(0, ...)`.

- **Q3. `all_end_step=400000` schedule horizon.**
  **A:** The fix must hold across the entire curriculum range (progress 0 → 1). At every progress value, including `progress=1`, the env distribution must still include the easy task. This rules out idioms that only ramp the *upper bound* without also containing the lower bound; it requires the sampler form `Uniform(0, progress * max)` (or equivalent) that always has support at 0.

- **Q4. `per_env_curriculum_jitter` enable flag default.**
  **A:** Moot — superseded by Q12. The `Uniform(0, progress * max)` sampler form already collapses to a point mass at 0 when `progress=0`, so there is no need for a separate enable knob. No new flag introduced.

- **Q5. Validation bar.**
  **A:** `40k-settings eval of 400k ckpt ≥ 80%` of the 40k-ckpt's reward on those same settings is the starting bar. May be adjusted up/down after the first measurement (Q1).

- **Q6. FP/FN scope.**
  **A:** Separate ticket. Defer FP/FN to later. The `fp_fn_background_start_step=2_200_000` quirk is **not** in scope for 034.

- **Q7. Task-signal curricula (`progress_coord`, `progress_safety`, `progress_tracking`) — keep as is?**
  **A:** Mostly yes — but **investigate `progress_tracking` further**. The initial-states distributions triggered by `progress_tracking` must also include the easy-phase samples in the later phase (no contraction over time). Audit the `initial_states_generator` `uniform(min, min + p * (max - min))` family to confirm none of them silently lose the easy-phase support; flag any that do. `progress_coord` and `progress_safety` remain unchanged.

- **Q8. Per-env "personality" persistence (i.i.d. per reset vs quasi-stable).**
  **A:** i.i.d. across episodes is the correct default. Rationale (recorded for the design doc): episodes are ~500 steps, RNN `sequence_length=32`, so one episode ≈ 15 sequences all at one difficulty. Cross-episode i.i.d. provides per-(env,agent) difficulty diversity at the sequence-batch level; quasi-stable would only matter if we wanted *within-episode* difficulty changes for hidden-state adaptation, which is not a design goal for this ticket.

- **Q9. Heterogeneous-agent compatibility.**
  **A:** Orthogonal. For randomization/robustifying purposes, the fix **targets per-agent randomization where possible**. See Q17 below for the granularity follow-up.

- **Q10. Observation-shape compatibility.**
  **A:** No observation-shape change — existing checkpoints (`agent_400000.pt`) must remain loadable. Adding optional **critic-only** privileged obs dims (current per-env latency/noise/dropout for value function only) is a separate optional design that can be considered later but is not required for 034.

### Architectural decisions

- **Q11. Anti-forgetting idiom.**
  **A: (a) Uniform-over-history per (env, agent).** I.e. for each (env, agent) at reset, draw `eff_p ~ Uniform(0, global_progress)` and use this as the per-(env, agent) effective progress for the dynamics/observability axes. The τ_zoom path's `zoom_scale_range=(0.01, 1.0)` was a working precedent but was a multiplicative scale around a single curriculum value; the chosen (a) form is the more general uniform-over-history idiom and is consistent with `initial_states`.

- **Q12. Per-axis vs global jitter knob.**
  **A:** Neither — refactor the sampler form itself per-axis:
  - **Difficulty axes (zero-bounded):** `value ~ Uniform(0, cfg.value * progress)`. Examples: latency means/stds, noise stds, dropout rates, burst onset prob, gimbal/zoom dead-time means, `max_lin_vel` above its nominal floor.
  - **Non-difficulty axes (nominal-centered):** `value ~ Uniform(low, high) * cfg.nominal`. Examples: controller gains (Kp_vel, Ki_vel, …), motor τ.
  Result: nominal cfg values stay as nominal-system parameters; the randomization spans (1) around them when difficulty is not defined and (2) from zero when difficulty is defined.

- **Q13. Plumbing route for per-env effective rates.**
  **A: Route 1 (extend delay-system APIs to accept per-env tensors).** Compute is parity (the existing scalar multiplication was already a broadcast). Memory is ~`num_envs * num_agents * num_noise_types` floats ≈ <1 MB at 16k envs. The current delay-system inefficiency the engineer flagged is in a different code path (ringbuffer queries) and is outside ticket 034 — captured as an "Observations" item for a follow-up ticket.

- **Q14. New `CurriculumAxisCfg` abstraction vs axis-specific fields.**
  **A: Stay with axis-specific fields.** With the Q12 sampler-form change, **only ~1 new config field is required** (a `min_latency_steps: int = 2` on `DelaySystemKeyParams` to honor the 2-step latency floor noted in Q2). The `CurriculumAxisCfg` abstraction would cost more boilerplate than it saves; revisit if the count grows past ~5–10 new fields.

- **Q15. Slice ordering.**
  **A: Start with cluster A (dynamics-side) — and within A, start with `_max_lin_vel`.** It is the simplest axis (single per-env scalar, no delay-system plumbing), has the smallest blast radius, and verifies the end-to-end pattern that the obs-side cluster will then reuse.

- **Q16. Validation methodology.**
  **A: (β) Final-eval at 400k against a fixed eval suite** (env configurations at e.g. 40k / 100k / 200k / 400k). No training-script changes — runs as an eval harness against saved checkpoints. Lower runtime cost; discrete signal; preserves comparability with the existing fix2 baseline ckpt.

---

## Follow-up question raised during gate

- **Q17. Per-(env, agent) vs per-env granularity for cross-env sampling.**
  Triggered by Q9 ("target per-agent randomization where possible"). The current `per_agent_randomization=True` already gives intra-env, per-agent heterogeneity (drone_0 and drone_1 have different latency scales within the same env). The new cross-env sampling can either:
  - **(i)** Per-env: all agents in env_E share one effective progress `eff_p[E]`.
  - **(ii)** Per-(env, agent): every `(E, a)` pair has its own `eff_p[E, a]`. Maximum diversity; aligned with the "per-agent where possible" guidance from Q9.

  **A (engineer, recorded inline above for Q11): (ii) per-(env, agent).** All per-axis samplers operate at `(num_envs, num_agents)` granularity.

---

## Cross-cutting constraints extracted from answers

These must be enforced in Stage I (Design) and Stage S (Structure):

1. **Latency floor.** `min_latency_steps = 2` (= 80 ms at 25 Hz). Every delay sampler clamps `eff_latency ≥ min_latency_steps * policy_dt`. Capture as a new field on `DelaySystemKeyParams`.
2. **Sampler-form rule.** Difficulty axes: `Uniform(0, progress * nominal)`. Non-difficulty axes: `Uniform(low, high) * nominal`. Both at per-(env, agent) granularity where the underlying system supports it.
3. **Anti-forgetting at every progress value.** Easy-task support must be present in the distribution at `progress ∈ [0, 1]`, including `progress = 1`. No collapse onto a hard-only distribution at any time.
4. **No obs-shape change.** Existing `agent_drone_*_final.pt` checkpoints must remain loadable. Critic-only privileged obs additions are deferred to a separate optional design.
5. **`progress_tracking` audit.** `initial_states_generator` distributions triggered by `progress_tracking` must be checked for the same anti-forgetting property; this is part of Stage R rather than a code change yet.

---

## Out-of-scope items captured for follow-up tickets

- FP/FN curriculum (`fp_fn_background_start_step=2_200_000` quirk and FP/FN ramp design) — deferred per Q6.
- Delay-system runtime inefficiency in the ringbuffer-query path — deferred per Q13.
- Optional critic-only privileged-obs extension (per-env effective progress as critic input) — deferred per Q10.

---

**Next action (gate after Q):** Engineer confirms answers above → `Start R`.

For Stage R, the agent will:
1. Document the existing curriculum-related code paths in the env, controller, and delay system (what exists today, fact-only, no recommendations).
2. Measure the regression (per Q1) — load fix2's 400k ckpt, run it under 40k-step env settings, log per-component reward, compare to fix2's 40k checkpoint on the same settings.
3. Audit `initial_states_generator` distributions for the anti-forgetting property (per Q7 / constraint #5).
