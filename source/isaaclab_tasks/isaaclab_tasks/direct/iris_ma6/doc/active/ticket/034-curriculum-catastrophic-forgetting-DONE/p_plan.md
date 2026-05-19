# Stage P — Implementation Plan

**Ticket**: 034-curriculum-catastrophic-forgetting
**Date**: 2026-05-17
**Status**: Draft, awaiting engineer approval
**Inputs**: [i_design.md](i_design.md), [s_structure.md](s_structure.md).
**Slice ordering rule**: per Q15, start with cluster A (dynamics-side) →
`_max_lin_vel` first. Cluster B (observability) follows.

Each slice is vertically testable: it lands code + unit test + a
headless smoke-eval verification of the per-(env, agent) distribution
for the axes touched. The full training-run validation lives in Slice 5.

---

## Slice 1: `progress_helper` module + `_max_lin_vel` anti-forgetting

End-to-end behavior: at any `progress_agent_velocity ∈ [0, 1]`, the
distribution of `env._max_lin_vel` over envs has positive mass on the
slow regime (≤ 5 m/s) — proving the new sampler form works for the
simplest axis.

- **Step 1.1**: Create [`curriculum/progress_helper.py`](../../../curriculum/progress_helper.py)
  with `sample_per_env_progress(...)` implemented.
- **Step 1.2**: Create [`curriculum/tests/test_progress_helper.py`](../../../curriculum/tests/test_progress_helper.py)
  with the 6 tests listed in S, run via the standalone runner pattern
  from `iris_ma6/CLAUDE.md`.
- **Step 1.3**: Update [`curriculum/CONTEXT.md`](../../../curriculum/CONTEXT.md)
  to register the new helper.
- **Step 1.4**: Add `self._curriculum_generator: torch.Generator` to
  [`iris_ma_env6_test.py`](../../../iris_ma_env6_test.py)
  `__init__`, seeded from `cfg.seed`.
- **Step 1.5**: Init `self._eff_progress_agent_velocity: Tensor[N, A]`
  in `__init__` (zeros).
- **Step 1.6**: Resample `_eff_progress_agent_velocity[env_ids]` via
  `sample_per_env_progress` in `_reset_idx`, after `progress_agent_velocity`
  is computed (env:2297).
- **Step 1.7**: Rewrite the `_max_lin_vel` curriculum block (env:2444-2448)
  to use the per-(env, agent) tensor as the base.
- **Step 1.8**: Modify the `_max_lin_vel` randomization block (env:2459-2465)
  to **compose** with (not overwrite) the curriculum value.
- **Test checkpoint**:
  - `./isaaclab.sh -p source/isaaclab_tasks/.../curriculum/tests/run_tests.py`
    reports all `test_progress_helper` cases passing.
  - Smoke verification script (writes `slice1_smoke.json`): create env
    with `num_envs=256`, `debug_initial_step=399000`, run `env.reset()`,
    dump `env._max_lin_vel` to file. Pass: `min(_max_lin_vel) < 5.0 m/s`
    AND `max(_max_lin_vel) > 9.0 m/s` (i.e. the distribution genuinely
    spans the curriculum range, not just the ±20% scale band).

---

## Slice 2: Gimbal rate-loop τ + gimbal dead-time per-env jitter

End-to-end behavior: at full `progress_dynamics=1`, the distribution of
`gimbal_rate_loop._tau_yaw_eff` includes near-pass-through envs
(τ_eff ≈ 0); the distribution of sampled dead times has positive mass
near 0 s and at the max.

- **Step 2.1**: Add `_tau_progress_per_env: Tensor[N]` and
  `_dead_time_curr_scale_per_env: Tensor[N]` fields in
  [`controller/gimbal_rate_loop.py`](../../../controller/gimbal_rate_loop.py)
  `__init__`.
- **Step 2.2**: Modify `set_progress(p)` to accept `float | Tensor[N]`
  and write into the per-env tensor; keep the scalar `_tau_progress`
  field for back-compat / logging.
- **Step 2.3**: Modify `step()` to use per-env `_tau_progress_per_env`
  in the `tau_yaw_eff = _tau_yaw * _tau_progress_per_env` path
  (elementwise instead of scalar).
- **Step 2.4**: Modify `set_dead_time_curriculum_scale(scale)` to
  accept `float | Tensor[N]`; modify `_sample_dead_time(env_ids)` to
  read `_dead_time_curr_scale_per_env[env_ids]`.
- **Step 2.5**: Create [`controller/tests/test_gimbal_rate_loop_per_env.py`](../../../controller/tests/test_gimbal_rate_loop_per_env.py)
  asserting scalar/tensor equivalence under uniform input and
  per-env divergence under non-uniform input.
- **Step 2.6**: Init `self._eff_progress_dynamics: Tensor[N, A]` and
  `self._eff_progress_gimbal_dead_time: Tensor[N, A]` in env `__init__`.
- **Step 2.7**: Resample both tensors for `env_ids` in `_reset_idx`
  (alongside Slice 1's resample block).
- **Step 2.8**: Rewire step-time hooks at env:1442-1444 and env:1449-1451
  to pass the cached per-(env, agent) tensors (collapse to per-env
  via mean or per-axis selection — Stage Y decision; default mean).
- **Test checkpoint**: unit tests pass; smoke verification at
  `debug_initial_step=399000` confirms `gimbal_rate_loop._tau_yaw_eff`
  span across envs includes values below 0.1 × nominal.

---

## Slice 3: Zoom dead-time per-env jitter

End-to-end behavior: at full `progress_zoom_dead_time=1`, the
distribution of `zoom_controller._dead_time_seconds` includes both
near-0 and near-max envs.

- **Step 3.1**: Add `_dead_time_curr_scale_per_env: Tensor[N]` in
  [`controller/zoom_controller.py`](../../../controller/zoom_controller.py)
  `__init__`.
- **Step 3.2**: Modify `set_dead_time_curriculum_scale(scale)` to
  accept `float | Tensor[N]`; modify `_sample_dead_time(env_ids)` to
  read the per-env tensor.
- **Step 3.3**: Create [`controller/tests/test_zoom_controller_per_env.py`](../../../controller/tests/test_zoom_controller_per_env.py)
  with scalar/tensor equivalence + per-env divergence tests
  (Slice 2 pattern).
- **Step 3.4**: Init `self._eff_progress_zoom_dead_time: Tensor[N, A]`
  in env `__init__`.
- **Step 3.5**: Resample in `_reset_idx` (alongside Slices 1–2).
- **Step 3.6**: Rewire step-time hook at env:1456-1458.
- **Test checkpoint**: unit tests pass; smoke verification at
  `debug_initial_step=399000` confirms `_dead_time_seconds` span across
  envs covers `[0, dead_time_max_s]` rather than `N(mean, std)` only.

---

## Slice 4: Delay-system per-env API + obs-side axes (noise, dropout, delay mode, burst)

End-to-end behavior: at full progress on each obs-side axis, the
distribution of per-(env, agent) noise std, dropout rate, latency
realization, and burst onset prob all include zero / near-zero envs.
The 2-step latency floor invariant holds across all sampled latencies.

- **Step 4.1**: Add `min_latency_steps: int = 2` to `DelaySystemKeyParams`
  in [`delay_system_v3/delay_cfg_v3.py`](../../../delay_system_v3/delay_cfg_v3.py).
- **Step 4.2**: In [`delay_system_v3/multi_agent_wrapper.py`](../../../delay_system_v3/multi_agent_wrapper.py)
  `__init__`, allocate the 6 per-(env, agent) state tensors listed in S
  (`_noise_scale`, `_base_dropout_rate`, `_progress`, `_per_agent_*`).
- **Step 4.3**: Add `_enforce_latency_floor(eff_p, base_latency_s)` helper
  in `multi_agent_wrapper.py`; call from `set_delay_mode` and from each
  field-level setter.
- **Step 4.4**: Modify `set_noise_scale`, `set_dropout_rate`,
  `set_delay_mode`, `set_burst_params` to accept `float | Tensor[N, A]`
  with broadcast semantics.
- **Step 4.5**: Modify `_get_noise_std(noise_type, agent_id)` to
  return `Tensor[N]` (broadcasts at call site).
- **Step 4.6**: Modify `randomize_per_agent_params(env_ids)` to sample
  per-(env, agent) into the new tensor state (replaces the per-agent
  Python dict population).
- **Step 4.7**: Modify the corresponding setters in
  [`delay_system_v3/delay_system_v3.py`](../../../delay_system_v3/delay_system_v3.py)
  (`set_delay_mode`, `set_field_delay_mode`, `set_dropout_rate`,
  `set_field_dropout_rate`, `set_dropout_rate_by_perspective`) to accept
  per-env tensors.
- **Step 4.8**: Create [`delay_system_v3/tests/test_per_env_inputs.py`](../../../delay_system_v3/tests/test_per_env_inputs.py)
  with: scalar/tensor equivalence under uniform input; per-(env, agent)
  divergence under non-uniform input; latency-floor invariant
  (`min_latency_steps * policy_dt ≤ all_sampled_latencies`).
- **Step 4.9**: Init `_eff_progress_noise`, `_eff_progress_dropout`,
  `_eff_progress_delay`, `_eff_progress_burst_dropout` (each
  `Tensor[N, A]`) in env `__init__`.
- **Step 4.10**: Resample all four in `_reset_idx` and rewire the
  step-time hooks at env:1411, 1414, 1417, 1421, 1428-1430, 1435-1438
  to pass the cached per-(env, agent) tensors.
- **Test checkpoint**: unit tests pass; smoke verification at
  `debug_initial_step=399000` confirms (i) noise-std distribution per
  (env, agent) includes near-0 envs, (ii) dropout-rate distribution
  includes 0-rate envs, (iii) min sampled latency ≥ `2 * policy_dt`
  (80 ms at 25 Hz) for all envs that have `delay_mode != "none"`.

---

## Slice 5: Full-training validation

End-to-end behavior: a fresh 400 k-step training under the new
curriculum produces a checkpoint whose ticket-034 regression-eval
visibility ratios at all 4 eval steps satisfy the ≥ 0.80 bar from Q5.

- **Step 5.1**: Update [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) with the new
  `curriculum/progress_helper` → env dependency arrow.
- **Step 5.2**: Update [`doc/active/feature_list.json`](../../../doc/active/feature_list.json)
  curriculum entry to note ticket-034 changes.
- **Step 5.3**: Smoke training run: 10 k steps with the same
  `skrl_mappo_rnn_cfg.yaml` as fix2. Pass: no NaN in loss, KL parks in
  `[0.012, 0.028]` band (matches fix2 early-training), no entropy
  drift > 0.30.
- **Step 5.4**: Full training run: 400 k steps. Same cfg, same commit
  except the ticket-034 changes. Wall-clock ~24 h.
- **Step 5.5**: Run ticket 034's 7-cell regression eval against the new
  `agent_400000.pt` via
  [`r_research/run_regression_eval.sh`](r_research/run_regression_eval.sh)
  (pointed at the new checkpoint directory).
- **Step 5.6**: Generate the post-fix visibility ratio table; compare
  side-by-side with the pre-fix table from `r_research/regression_eval.md`.
  Pass: every ratio ≥ 0.80; fail: any ratio < 0.80.
- **Step 5.7**: Write `iris_ma6/doc/experiments/2026-MM-DD_..._ticket034_validation.md`
  documenting the pre/post comparison and closing the ticket.
- **Test checkpoint**: ratio table from Step 5.6 satisfies ≥ 0.80 at
  39 k, 99 k, 199 k, 399 k eval steps. If any ratio fails, see fallback
  below.

---

## Fallback (if Slice 5 fails ≥ 0.80 at any eval step)

Three triage paths, in order of cheapness, **not** part of the plan
unless the test checkpoint fails:

- **F1.** Narrow the per-(env, agent) jitter window: replace
  `eff_p ~ Uniform(0, global_p)` with `eff_p ~ Uniform(α · global_p, global_p)`
  for some `α ∈ (0, 1)`. Adds one config field, no other code change.
  Re-runs Slice 5 only.
- **F2.** Add critic-only privileged obs (per-env effective progress
  fed to value head only) — deferred per Q10 but available if the
  policy can't disambiguate easy / hard envs without it. Requires
  ticket-033 asymmetric-critic plumbing already in place.
- **F3.** Reopen Stage Q for re-scoping (the regression isn't pure
  curriculum-distribution forgetting and we need a new hypothesis).

---

## Cross-slice dependencies

```
Slice 1 ──> Slice 2
        └─> Slice 3
        └─> Slice 4 ──> Slice 5
```

Slices 2, 3, 4 are mutually independent after Slice 1 lands. They could
be parallelized by separate workers; the prescribed order (2 → 3 → 4)
follows Q15's "start cluster A, then cluster B" rule.

Slice 5 cannot start until all four code slices have landed because the
training-time stress test exercises every axis.

---

## Out of plan

- Critic-only privileged obs (Q10, deferred).
- FP/FN curriculum (Q6, separate ticket).
- Delay-system runtime-efficiency rework (Q13, separate ticket).
- `CurriculumAxisCfg` abstraction (Q14, deferred unless field count grows).
- `initial_states` code changes (R2.8 verified already correct under
  default config).
