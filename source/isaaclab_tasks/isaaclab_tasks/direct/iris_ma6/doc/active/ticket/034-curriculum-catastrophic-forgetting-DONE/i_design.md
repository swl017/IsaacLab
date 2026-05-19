# Stage I — Design Document

**Ticket**: 034-curriculum-catastrophic-forgetting
**Date**: 2026-05-17
**Status**: Draft, awaiting engineer approval
**Inputs**: [ticket.md](ticket.md), [q_questions.md](q_questions.md),
[r_research.md](r_research.md), [r_research/regression_eval.md](r_research/regression_eval.md).

---

## Problem statement

The fix2 baseline's 400 k checkpoint achieves 38–42 % lower
`visibility_mean` on early-phase env settings (39 k / 99 k) than the
checkpoints trained at those same settings (measured in R0). The cause
is that ~14 of 16 curriculum axes use a deterministic `progress · value`
sampler that collapses the env distribution onto the hardest setting
once `progress = 1`, leaving the policy with no gradient signal on
earlier-phase regimes during the last ~180 k steps of training.

## Proposed approach

We refactor the **sampler form** for every dynamics / observability
curriculum axis so the env distribution at any global progress `p`
spans `[easy, p · max]` rather than collapsing to `p · max` alone.
Two axis classes:

- **Difficulty axes** (zero-bounded). Today: effective value `= p · nominal`.
  Proposed: at each reset, draw `eff_p ~ Uniform(0, p)` per (env, agent)
  and write `eff_p · nominal` into that (env, agent) slot.
- **Non-difficulty axes** (nominal-centered, no notion of "harder"). Today:
  effective value `~ Uniform(1 − p(1−lo), 1 + p(hi−1)) · nominal`. **Keep
  unchanged** — this form already contains `nominal` at every `p`.

The change is structurally identical to the anti-forgetting sampler the
`initial_states` module already uses (`Uniform(min, min + p(max−min))`)
and to the validated `zoom_scale_range = (0.01, 1.0)` randomization on
τ_zoom. We propagate that idiom to the remaining axes.

Granularity is **per-(env, agent)** for every axis whose backing system
supports it. The current delay system stores per-axis state as Python
scalars (or per-agent dicts of scalars) that broadcast to all envs; this
is the API change with the largest blast radius. The delay system's
per-axis APIs are extended to accept `(num_envs,)` or
`(num_envs, num_agents)` tensors. Internal queries (`_get_noise_std`,
`_apply_dropout`, ringbuffer lookup) already perform broadcasts, so the
upgrade is a tensor-shape change at the API surface with no compute
penalty.

Two cross-cutting constraints flow into every axis:

1. **Latency floor.** A new field `min_latency_steps: int = 2` on
   `DelaySystemKeyParams`. Every per-(env, agent) latency sampler
   clamps `eff_latency_s ≥ min_latency_steps · policy_dt = 80 ms`.
2. **Anti-forgetting invariant.** For every axis at every `p ∈ [0, 1]`,
   the support of the per-(env, agent) effective value distribution
   must contain the `p = 0` setting. Captured as a unit-test
   property in the per-axis test suites.

`_max_lin_vel`'s curriculum and randomization paths currently *replace*
each other (env:2446-2465); they will be composed instead — the
curriculum supplies the per-(env, agent) base, and the existing
`±20 %` randomization (`max_lin_vel_scale_range`) multiplies it.

`progress_tracking` (initial-states scalar) is already
anti-forgetting-correct under the default config (R2.8); the design
adds a unit test pinning that invariant but no code change to
`initial_states_generator`.

The two task-signal curricula `progress_coord` and `progress_safety`
remain step-at-episode-boundary / per-env ramp scalars, unchanged.

Validation is offline: an eval suite re-runs ticket 034's 7-cell
matrix against the new training-run checkpoint, computes
`visibility_mean` ratios at the 4 eval steps, and passes when every
ratio is ≥ 0.80.

## Key interfaces and data flow

### New module: `curriculum/progress_helper.py`

A thin helper exposed off `CurriculumCfg` (or as a standalone class).

- **Inputs**: `global_progress: float`, `num_envs: int`, `num_agents: int`,
  `device`. Optionally `env_ids: Tensor`.
- **Output**: `eff_p: Tensor[num_envs, num_agents]` with values in
  `[0, global_progress]`, drawn i.i.d. uniform per (env, agent).
- **Stateless**. Called once per axis per reset (or once per reset
  shared across axes — Stage S decision).

### Env (`iris_ma_env6_test.py`)

Touchpoints (per R1):

- `_get_rewards` step-time hooks (env:1407-1458) — keep the global
  progress reads, but the values passed down to the controller and
  delay system become per-(env, agent) tensors.
- `_reset_idx` (env:2236-2516) — for each axis listed in R1.2, replace
  the deterministic `progress · value` (or single-value) computation
  with `Uniform(0, progress)` per (env, agent), then apply the
  existing mapping function. Compose with existing per-axis
  randomization where it already exists (gain randomization,
  `max_lin_vel` scale).

### Delay system (`delay_system_v3/multi_agent_wrapper.py`)

API upgrades (Route 1, per Q13):

- `set_noise_scale(scale)` → `set_noise_scale(scale: float | Tensor[N, A])`
- `set_dropout_rate(rate)` → `set_dropout_rate(rate: float | Tensor[N, A])`
- `set_delay_mode(mode, progress)` → `set_delay_mode(mode, progress: float | Tensor[N, A])`
- `set_burst_params(p_onset, p_recovery)` → both args accept per-(env, agent) tensors
- `set_field_dropout_rate`, `set_field_delay_mode` — already per-agent, extend to per-(env, agent)
- `_per_agent_*` dicts (Python scalars) become `(N, A)` tensors

The internal application sites (`_get_noise_std`, dropout sampling,
ringbuffer query) already broadcast; the change is structural only. The
2-step latency floor is enforced inside the delay-system setters.

### Controllers (`controller/gimbal_rate_loop.py`, `controller/zoom_controller.py`)

- `gimbal_rate_loop.set_progress(p)` → accepts per-(env, agent) tensor
  for the τ scaling factor.
- `gimbal_rate_loop.set_dead_time_curriculum_scale(scale)` and
  `zoom_controller.set_dead_time_curriculum_scale(scale)` → accept
  per-(env, agent) tensor; the existing reset-time `_sample_dead_time`
  draws `N(mean · scale, std · scale)` already, the change is letting
  `scale` vary per env/agent.

### Eval (`experiments/evaluate.py`)

No code change — the existing harness was sufficient for R0. A new
shell wrapper under `r_research/` re-runs the 7-cell suite against the
new training checkpoint at validation time.

## What this does NOT include

- **No observation-shape change.** Per Q10. Critic-only privileged obs
  (current per-(env, agent) effective progress as value-function input)
  is deferred to a future ticket.
- **No FP/FN curriculum redesign.** Per Q6 — separate ticket.
- **No delay-system runtime-efficiency rework.** Per Q13 — captured as
  a follow-up.
- **No `CurriculumAxisCfg` abstraction.** Per Q14 — only ~1 new config
  field is added (`min_latency_steps`).
- **No change to task-signal curricula** (`progress_coord`,
  `progress_safety`, `progress_tracking`). Per Q7.
- **No change to `initial_states` samplers.** R2.8 verified they
  already preserve anti-forgetting under the default config.
- **No change to MAPPO trainer, reward functions, or termination
  logic.** Ticket 033's fixes stay as the baseline.

## Open risks

- **Per-env API change blast radius in the delay system.** Callers
  outside `iris_ma_env6_test.py` (tests, other envs that share the
  module) may pass scalars; we'll add scalar-accepting overloads so
  existing callers don't break. To be verified in Stage S.
- **Resampling per-(env, agent) per reset increases variance.** With
  rollouts of 32 envs × 32 sequence steps, each minibatch may see a
  wider distribution of difficulties than the deterministic regime.
  Mitigation: this is the *point* of the fix; we expect the entropy
  band-aligned KL controller from ticket 033 to absorb it. If KL
  spikes, fall back to a narrower jitter window (e.g.
  `eff_p ~ Uniform(0.5 · global_p, global_p)`). Stage P will define
  the rollback knob.
- **2-step latency floor (80 ms) may be too coarse** at early
  curriculum progress where the policy expects near-zero delay. The
  floor only matters once `delay_mode` switches off `"none"` at 120 k.
  Risk is moderate; will measure in the first training run.
- **τ-curriculum interaction with zoom-τ randomization.** The new
  `progress_dynamics` jitter feeds the gain-randomization base; we
  must verify the composition with `zoom_scale_range = (0.01, 1.0)`
  doesn't collapse the zoom-τ floor below `1e-4 s`. Cheap to check
  numerically.
- **Eval-time interpretation of `visibility_mean`.** The 80 % bar was
  chosen against a single witness metric (ticket 034 §0). If the
  re-run shows the fix improves `visibility_mean` but degrades a
  different metric (e.g. `collision_rate_mean`), the bar may need a
  second-witness rule. Will be addressed in Stage P's validation step.
