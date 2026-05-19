# Stage S — Structure Outline

**Ticket**: 034-curriculum-catastrophic-forgetting
**Date**: 2026-05-17
**Status**: Draft, awaiting engineer approval
**Inputs**: [i_design.md](i_design.md), [r_research.md](r_research.md), [q_questions.md](q_questions.md).

Signatures and shapes only — no implementation, no logic.
Notation: `N = num_envs`, `A = num_agents`. Tensors are `torch.Tensor`
unless noted.

---

## New files

### `iris_ma6/curriculum/progress_helper.py`

```python
def sample_per_env_progress(
    global_progress: float,
    num_envs: int,
    num_agents: int,
    device: torch.device,
    env_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return Tensor[len(env_ids or N), A] with values ~ Uniform(0, global_progress)."""
```

Module-level constant `EPS_PROGRESS: float` (lower bound of returned
samples to avoid divide-by-zero downstream — value TBD in Stage P).

### `iris_ma6/curriculum/CONTEXT.md`

[update] Adds a "Per-env effective progress" section documenting
`progress_helper.sample_per_env_progress` and its consumers.

### `iris_ma6/curriculum/tests/test_progress_helper.py`

```python
def test_support_includes_zero_at_p_one()
def test_support_does_not_exceed_global_progress()
def test_p_zero_yields_zeros()
def test_env_ids_subset_returns_subset_shape()
def test_generator_reproducible()
def test_anti_forgetting_invariant_across_axes()  # property test
```

Test runner: extend existing
`iris_ma6/curriculum/test/run_tests.py` (existing pattern in this
project) to include the new file. No new runner.

---

## Modified files

### `iris_ma6/delay_system_v3/delay_cfg_v3.py`

- [existing] `DelaySystemKeyParams` — fields unchanged
- [add]      `DelaySystemKeyParams.min_latency_steps: int = 2`
             — minimum latency floor in policy-rate steps (80 ms at 25 Hz)

### `iris_ma6/delay_system_v3/multi_agent_wrapper.py`

- [existing] `MultiAgentDelaySystemWrapper.__init__` — extends per-axis
             state shapes (see field-shape changes below)
- [modify]   `_per_agent_latency_scale: Dict[AgentID, float]`
             → `_per_agent_latency_scale: Tensor[N, A]`
- [modify]   `_per_agent_noise_scale:   Dict[AgentID, float]`
             → `_per_agent_noise_scale:   Tensor[N, A]`
- [modify]   `_per_agent_dropout_offset: Dict[AgentID, float]`
             → `_per_agent_dropout_offset: Tensor[N, A]`
- [modify]   `_noise_scale: float`
             → `_noise_scale: Tensor[N, A]`
- [modify]   `_base_dropout_rate: float`
             → `_base_dropout_rate: Tensor[N, A]`
- [modify]   `_progress: float`
             → `_progress: Tensor[N, A]`
- [modify]   `set_noise_scale(scale: float)`
             → `set_noise_scale(scale: float | Tensor[N, A])`
- [modify]   `set_dropout_rate(rate: float)`
             → `set_dropout_rate(rate: float | Tensor[N, A])`
- [modify]   `set_delay_mode(mode: str, progress: float)`
             → `set_delay_mode(mode: str, progress: float | Tensor[N, A])`
- [modify]   `set_burst_params(p_onset: float, p_recovery: float)`
             → `set_burst_params(p_onset: float | Tensor[N, A], p_recovery: float | Tensor[N, A])`
- [modify]   `_get_noise_std(noise_type: str, agent_id: AgentID | None = None) -> float`
             → `_get_noise_std(noise_type: str, agent_id: AgentID | None = None) -> Tensor[N]`
             (per-env std, broadcast over the noise tensor at the call site)
- [modify]   `randomize_per_agent_params(env_ids: Tensor | None = None)`
             — signature unchanged; samples per-(env, agent) into the
             three `_per_agent_*` tensors above
- [add]      `_enforce_latency_floor(eff_progress: Tensor[N, A], base_latency_s: float) -> Tensor[N, A]`
             — clamp `eff_progress * base_latency_s ≥ cfg.min_latency_steps * policy_dt`

### `iris_ma6/delay_system_v3/delay_system_v3.py`

- [modify]   `set_delay_mode(mode: str, progress: float)`
             → `set_delay_mode(mode: str, progress: float | Tensor[N, A])`
- [modify]   `set_field_delay_mode(field_id: str, mode: str, progress: float)`
             → `set_field_delay_mode(field_id: str, mode: str, progress: float | Tensor[N])`
- [modify]   `set_dropout_rate(rate: float)`
             → `set_dropout_rate(rate: float | Tensor[N, A])`
- [modify]   `set_field_dropout_rate(field_id: str, rate: float)`
             → `set_field_dropout_rate(field_id: str, rate: float | Tensor[N])`
- [modify]   `set_dropout_rate_by_perspective(ego_rate: float, other_rate: float)`
             → `set_dropout_rate_by_perspective(ego_rate: float | Tensor[N], other_rate: float | Tensor[N])`
- [modify]   per-field latency / dropout / staleness internal tensors —
             extend from `(A,)` or scalar to `(N, A)` where the API now
             accepts per-env input. No new public methods.

### `iris_ma6/controller/gimbal_rate_loop.py`

- [existing] `_tau_progress: float` — kept; broadcast to per-env at step
             time when needed
- [add]      `_tau_progress_per_env: Tensor[N]` — backing tensor for the
             new per-env path (used when `set_progress` is called with
             a tensor)
- [add]      `_dead_time_curr_scale_per_env: Tensor[N]`
- [modify]   `set_progress(p: float)`
             → `set_progress(p: float | Tensor[N])`
- [modify]   `set_dead_time_curriculum_scale(scale: float)`
             → `set_dead_time_curriculum_scale(scale: float | Tensor[N])`
- [modify]   `_sample_dead_time(env_ids: Tensor | None)`
             — reads `_dead_time_curr_scale_per_env[env_ids]` instead of
             the scalar
- [modify]   `step(...)` — applies `_tau_progress_per_env` per row when
             populated; the existing `_tau_yaw_eff = _tau_yaw * _tau_progress`
             becomes elementwise

### `iris_ma6/controller/zoom_controller.py`

- [add]      `_dead_time_curr_scale_per_env: Tensor[N]`
- [modify]   `set_dead_time_curriculum_scale(scale: float)`
             → `set_dead_time_curriculum_scale(scale: float | Tensor[N])`
- [modify]   `_sample_dead_time(env_ids: Tensor | None)` — reads per-env
             scale tensor as in `gimbal_rate_loop`

### `iris_ma6/iris_ma_env6_test.py`

New per-env attributes initialized in `__init__`:

- [add]      `self._eff_progress_dynamics: Tensor[N, A]`
- [add]      `self._eff_progress_zoom_tau: Tensor[N, A]`
- [add]      `self._eff_progress_delay: Tensor[N, A]`
- [add]      `self._eff_progress_noise: Tensor[N, A]`
- [add]      `self._eff_progress_dropout: Tensor[N, A]`
- [add]      `self._eff_progress_burst_dropout: Tensor[N, A]`
- [add]      `self._eff_progress_gimbal_dead_time: Tensor[N, A]`
- [add]      `self._eff_progress_zoom_dead_time: Tensor[N, A]`
- [add]      `self._eff_progress_agent_velocity: Tensor[N, A]`

Method-level changes:

- [modify]   `_reset_idx(env_ids: Tensor)` (env:2236-2516)
  - [add]      Resample each `self._eff_progress_*` for `env_ids`
                via `progress_helper.sample_per_env_progress(...)`.
                The `global_progress` is the existing per-axis scalar
                (`get_dynamics_progress`, `get_noise_progress`, etc.).
  - [modify]   `_max_lin_vel` block (env:2444-2448) — derive
                per-(env, agent) effective progress from
                `self._eff_progress_agent_velocity`. Compose with
                (do not replace) the existing
                `max_lin_vel_scale_range` randomization at env:2459-2465.
  - [modify]   τ_zoom curriculum block (env:2408-2411) — use
                `self._eff_progress_zoom_tau` for the base.
  - [modify]   DR scale-application blocks (env:2467-2515) — gate on
                `self._eff_progress_dynamics` rather than the scalar.
- [modify]   `_get_rewards` step-time hooks (env:1407-1458) — pass the
                cached `self._eff_progress_*` tensors to delay system
                / controllers instead of the scalar global progress
                values returned by `curr.get_*_progress(current_step)`.
                Specifically rewrite the following call sites:
  - env:1411, 1414, 1417 → `set_delay_mode(mode, self._eff_progress_delay)`
  - env:1421 → `set_noise_scale(self._eff_progress_noise)`
  - env:1428-1430 → `set_dropout_rate(self._eff_progress_dropout * cfg.delay_system_params.dropout_prob)`
  - env:1435-1438 → `set_burst_params(p_onset=self._eff_progress_burst_dropout * cfg.delay_system_params.burst_p_onset, p_recovery=...)`
  - env:1442-1444 → `gimbal_rate_loop.set_progress(self._eff_progress_dynamics)`
  - env:1449-1451 → `gimbal_rate_loop.set_dead_time_curriculum_scale(self._eff_progress_gimbal_dead_time)`
  - env:1456-1458 → `zoom_controller.set_dead_time_curriculum_scale(self._eff_progress_zoom_dead_time)`

No other method bodies in this file change. `_get_observations`,
`_pre_physics_step`, `_get_dones`, `_apply_action` are untouched.

### `iris_ma6/iris_ma_env6_test_cfg.py`

- [existing] All current fields unchanged. No new fields here —
             `min_latency_steps` lives on `DelaySystemKeyParams` (above).

### `iris_ma6/ARCHITECTURE.md`

- [update]   Add `curriculum/progress_helper` to the curriculum-module
             section. Add the new env → progress_helper dependency arrow
             (env's `_reset_idx` consumes it; existing curriculum_cfg
             dependency unchanged).

### `iris_ma6/doc/active/feature_list.json`

- [update]   `curriculum` feature entry — add note for ticket 034
             (anti-forgetting jitter on dynamics / observability axes,
             per-(env, agent) granularity, latency floor enforcement).

---

## Files explicitly NOT modified

- `iris_ma6/initial_states/*` — R2.8 verified anti-forgetting already
  preserved under default config. Only the unit test in
  `progress_helper/tests/` will assert the invariant on the helper
  itself, not on `initial_states`.
- `iris_ma6/curriculum/curriculum_cfg.py` — no schedule changes; the
  `start_step` / `end_step` fields stay as is.
- `iris_ma6/controller/drone_controller.py` — `randomize_gains` is a
  non-difficulty axis (Q12) and stays on `Uniform(low, high) · nominal`.
- `iris_ma6/cbf_safety/*`, `iris_ma6/triangulation/*`,
  `iris_ma6/bbox_raycaster_v2/*`, `iris_ma6/target_controller/*`,
  `iris_ma6/visualization/*` — out of scope.
- All MAPPO trainer files, reward terms, and termination logic.
- All experiment registry entries except the optional new
  `validate_034` eval suite (Stage P decision).

---

## Test scaffolding (Stage Y, listed here for completeness)

- `iris_ma6/curriculum/tests/test_progress_helper.py` (new, listed above).
- `iris_ma6/delay_system_v3/tests/test_per_env_inputs.py` (new) — asserts
  scalar and tensor inputs produce identical outputs under broadcast
  semantics, and that `min_latency_steps` floor is enforced.
- `iris_ma6/controller/tests/test_gimbal_rate_loop_per_env.py` (new) —
  asserts `set_progress(scalar)` and `set_progress(tensor[N])` are
  equivalent when the tensor is uniform.

No other test files modified.

---

## Open structure questions for engineer

1. **`_eff_progress_*` storage shape.** Listed as `Tensor[N, A]` for
   every axis. Some axes (gimbal dead time, zoom dead time) are
   per-env-scalar in the underlying controller, not per-agent. Should
   we collapse those to `Tensor[N]` to save memory, or keep `Tensor[N, A]`
   for API uniformity? Memory cost is negligible (~9 × N × A × 4 bytes
   ≈ 200 KB at N=4096, A=2); recommending uniform `Tensor[N, A]`.
2. **Where to put the latency-floor enforcement.** Two reasonable spots:
   (i) inside `multi_agent_wrapper.set_delay_mode`, (ii) inside
   `delay_system_v3.set_field_delay_mode`. Recommending (i) so the
   wrapper owns curriculum semantics and the inner module stays
   curriculum-agnostic.
3. **Random number generator ownership.** `progress_helper` accepts an
   optional `torch.Generator`. Should the env create one dedicated
   generator (deterministic per seed) or use the global generator?
   Recommending env-owned generator stored on the env, seeded from
   `cfg.seed`.
