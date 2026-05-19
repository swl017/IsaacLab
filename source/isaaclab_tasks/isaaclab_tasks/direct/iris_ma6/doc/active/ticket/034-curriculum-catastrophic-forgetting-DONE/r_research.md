# Stage R — Codebase Research Report

**Ticket**: 034-curriculum-catastrophic-forgetting
**Date**: 2026-05-17
**Scope**: Map every curriculum touchpoint in `iris_ma6` and audit
`progress_tracking` (initial-states) distributions for anti-forgetting
support. Facts only — no recommendations.

R0 (regression measurement on fix2's 400k ckpt at 39k/99k/199k/399k env
settings) is **blocked** on a numpy/scipy ABI mismatch in the working
Isaac Lab Python env; see §R3.

---

## R1 — Curriculum touchpoint inventory

This section enumerates every code path that reads a `CurriculumCfg`
field or a `progress_*` value. Two state classes are distinguished:

- **Step-time hook** (executed every `_pre_physics_step`/`_get_rewards`,
  reads `current_step` directly).
- **Reset-time hook** (executed in `_reset_idx`, reads `current_step`
  once per reset and caches a per-env scalar or tensor).

Source of `current_step`:
[iris_ma_env6_test.py:1395](../../../iris_ma_env6_test.py#L1395)
and [iris_ma_env6_test.py:2252](../../../iris_ma_env6_test.py#L2252):
```python
current_step = (
    self.cfg.debug_initial_step
    if self.cfg.use_debug_initial_step
    else self.common_step_counter
)
```

### R1.1 Step-time hooks (in `_get_rewards`)

Implemented in [iris_ma_env6_test.py:1390-1458](../../../iris_ma_env6_test.py#L1390-L1458).
Each hook is called every step with the current global progress value.

| Hook | File:line | Curriculum source | Underlying call | Per-axis state shape | Phase window |
|---|---|---|---|---|---|
| Delay mode (`none`/`fixed`/`random`) | env:1407-1417 | `curr.get_delay_mode`, `get_fixed_delay_progress`, `get_random_delay_progress` | `delay_system.set_delay_mode(mode, progress: float)` | scalar (broadcast to all envs via per-agent in [multi_agent_wrapper.py:920-942](../../../delay_system_v3/multi_agent_wrapper.py#L920-L942)) | 120 k → 160 k |
| Noise scale | env:1419-1421 | `curr.get_noise_progress` | `delay_system.set_noise_scale(scale: float)` | scalar held in `_noise_scale` ([multi_agent_wrapper.py:982-990](../../../delay_system_v3/multi_agent_wrapper.py#L982-L990)), multiplied at query time in `_get_noise_std` (line 444-447) | 100 k → 120 k |
| FP/FN scale | env:1423-1424 | `curr.get_fp_fn_progress` | (stored on env as `_curriculum_fp_fn_scale`; consumed downstream) | scalar | 2.2 M → 3.0 M (effectively disabled in 400 k runs) |
| i.i.d. dropout rate | env:1426-1430 | `curr.get_dropout_progress * cfg.delay_system_params.dropout_prob` | `delay_system.set_dropout_rate(rate: float)` | scalar (per-agent offset added in [multi_agent_wrapper.py:964-968](../../../delay_system_v3/multi_agent_wrapper.py#L964-L968)) | 160 k → 180 k |
| Burst dropout | env:1432-1438 | `curr.get_burst_dropout_progress * cfg.delay_system_params.burst_p_onset` | `delay_system.set_burst_params(p_onset, p_recovery)` | scalar | 200 k → 220 k |
| Gimbal rate-loop τ | env:1442-1444 | `curr.get_dynamics_progress` | `gimbal_rate_loop.set_progress(p: float)` ([gimbal_rate_loop.py:232-239](../../../controller/gimbal_rate_loop.py#L232-L239)) — multiplies the per-env tensor `_tau_yaw`, `_tau_pitch` by scalar progress at step time | per-env tensor `(num_envs,)` × scalar progress | 60 k → 100 k |
| Gimbal dead-time scale | env:1449-1451 | `curr.get_gimbal_dead_time_progress` | `gimbal_rate_loop.set_dead_time_curriculum_scale(scale: float)` ([gimbal_rate_loop.py:241-251](../../../controller/gimbal_rate_loop.py#L241-L251)). Stored as scalar `_dead_time_curr_scale`; applied next reset in `_sample_dead_time` ([gimbal_rate_loop.py:278+](../../../controller/gimbal_rate_loop.py#L278)) as `N(mean*scale, std*scale)` clipped to `[0, max_s]` per env. | per-env tensor `(num_envs,)` sampled at reset | 180 k → 220 k |
| Zoom dead-time scale | env:1456-1458 | `curr.get_zoom_dead_time_progress` | `zoom_controller.set_dead_time_curriculum_scale(scale: float)` (no-op when `cfg.zoom.model != "siyi_a8"`). Same sampling pattern as gimbal dead time. | per-env tensor `(num_envs,)` | 180 k → 220 k |

### R1.2 Reset-time hooks (in `_reset_idx`)

Implemented in [iris_ma_env6_test.py:2236-2516](../../../iris_ma_env6_test.py#L2236-L2516).
Per-env progress values are computed once and cached for the lifetime of
each episode.

| Hook | File:line | Curriculum source | Per-axis state | Phase window |
|---|---|---|---|---|
| `progress_dynamics` | env:2253-2257 | `curr.dynamics_start_step, dynamics_end_step` via `_linear_progress` | scalar (single shared float) | 60 k → 100 k |
| `progress_zoom_tau` | env:2262-2266 | `curr.zoom_tau_start_step, zoom_tau_end_step` | scalar | 20 k → 200 k |
| `progress_coord` (per-env) | env:2274-2275 | `current_step >= curr.coordination_start_step` (step-at-episode-boundary) | `(num_envs,)` tensor; 0 → 1 step function | 20 k step |
| `progress_safety` (per-env) | env:2281-2285 | `curr.safety_start_step, safety_end_step` via `_linear_progress` | `(num_envs,)` tensor | 20 k → 40 k |
| `progress_tracking` (scalar) | env:2291-2293 | `curr.tracking_start_step, tracking_end_step` | scalar (consumed by `initial_states.generate(curriculum_progress=...)`) | 20 k → 60 k |
| `progress_moving_target` (scalar) | env:2294-2296 | `curr.moving_target_start_step, moving_target_end_step` | scalar | 40 k → 80 k |
| `progress_agent_velocity` (scalar) | env:2297 | `curr.get_agent_velocity_progress` | scalar | 0 k → 0 k (window is empty; immediately 1.0) |
| `_max_lin_vel` (per-env) | env:2444-2448 | `progress_agent_velocity` × `(cfg.max_lin_vel - cfg.max_lin_vel_min)` | `(num_envs,)` tensor; **deterministic** | follows `progress_agent_velocity` |
| Controller gain randomization | env:2451-2456 | `progress_dynamics` × `cfg.gain_randomization.scale_range` | per-env tensors via `controller.randomize_gains` ([drone_controller.py:444-512](../../../controller/drone_controller.py#L444-L512)) | follows `progress_dynamics` |
| `_max_lin_vel` randomization | env:2459-2465 | `progress_dynamics`, `gain_randomization.max_lin_vel_scale_range=(0.8, 1.2)` | overrides the previous deterministic value with `Uniform(low, high) * cfg.max_lin_vel` (note: replaces the curriculum value, does **not** layer on top of it) | follows `progress_dynamics` |
| Zoom τ curriculum | env:2408-2411 | `progress_zoom_tau × cfg.drone_controller.zoom.tau_zoom`, floored to 1e-4 | per-(env, agent) scalar via batch indexing into `_controller._zoom._tau_zoom` | follows `progress_zoom_tau` |
| Zoom τ randomization | invoked inside `randomize_gains` | `progress_dynamics`, `gain_randomization.zoom_scale_range=(0.01, 1.0)` ([gain_randomization_cfg.py:49-60](../../../controller/gain_randomization_cfg.py#L49-L60)) | layered on top of the curriculum value (in-place multiplication) | follows `progress_dynamics` |
| Domain randomization (FOV, gimbal offsets, target scale, physics, gimbal dynamics) | env:2467-2515 | `progress_dynamics` gates each randomization individually via `1 + p * (raw - 1)` | per-env tensors | follows `progress_dynamics` |
| Delay system per-agent randomization | env:2440-2442 | `delay_system.randomize_per_agent_params(env_ids)` ([multi_agent_wrapper.py:992-1019](../../../delay_system_v3/multi_agent_wrapper.py#L992-L1019)) | per-agent (Python dict, **not** per-env) — `latency_scale ~ Uniform(0.5, 2.0)`, `noise_scale ~ Uniform(0.5, 2.0)`, `dropout_offset ~ Uniform(0.0, 0.05)` ([delay_cfg_v3.py:252-261](../../../delay_system_v3/delay_cfg_v3.py#L252-L261)) | called every reset; values shared across all envs |

### R1.3 Sampler form per axis (consolidated)

For each axis listed above, the form of the value-of-interest as a
function of `progress p`:

| Axis | Sampler form at `p` | Per-env shape | Contains `p=0` setting at `p=1`? |
|---|---|:---:|:---:|
| `_max_lin_vel` curriculum | `cfg.max_lin_vel_min + p * (cfg.max_lin_vel - cfg.max_lin_vel_min)` | per-env tensor | scalar floor: no — single value `cfg.max_lin_vel` at `p=1` |
| `_max_lin_vel` randomization | `Uniform(0.8, 1.2) * cfg.max_lin_vel` (overrides curriculum value when `progress_dynamics>0`) | per-env tensor | `Uniform(0.8, 1.2)` does not include the curriculum-min ratio `cfg.max_lin_vel_min / cfg.max_lin_vel = 0.3` |
| Controller gains (`Kp_vel`, `Ki_vel`, `Kp_att`, `Kp_rate`, `Ki_rate`, `Kd_rate`, `tau_motor`) | `Uniform(1 - p*(1 - scale_range[0]), 1 + p*(scale_range[1] - 1)) * nominal` (scale_range=(0.8, 1.2)) | per-env tensor | nominal is at the center of the interval; `p=0` setting is `1.0 * nominal` which is the center of the `p=1` distribution |
| `tau_zoom` curriculum | `max(cfg.zoom.tau_zoom * progress_zoom_tau, 1e-4)` | per-(env, agent) scalar | scalar floor: no — single value `cfg.zoom.tau_zoom` at `p=1` |
| `tau_zoom` randomization (composed on top) | `Uniform(0.01, 1.0) * tau_zoom_curriculum` | per-env tensor | yes — sample as low as `0.01 * cfg.zoom.tau_zoom ≈ 9.1e-4 s` is near-instant; `p=0` setting `≈ 1e-4` is just outside the range floor by an order of magnitude |
| Gimbal rate-loop τ | `_tau_yaw[env] * progress` (scalar progress, multiplied at step time) | per-env tensor multiplied by scalar | scalar progress: no — all envs share the same scalar at `p=1` |
| Gimbal dead-time samples | `N(cfg.dead_time_mean_s * scale, cfg.dead_time_std_s * scale)` clipped to `[0, dead_time_max_s]` | per-env tensor sampled at reset | at `scale=1`, `mean=0.066 s`, `std=0.016 s` → 99.7 % of mass in `[0.018, 0.114] s`, mass below `~0.018 s` exists only via clipping; `scale=0` setting `0 s` not contained except via lower tail clipping |
| Zoom dead-time samples | `N(0.100, 0.018)` clipped to `[0, 0.150]` at `scale=1` | per-env tensor sampled at reset | same shape as gimbal — `0 s` not in distribution at `scale=1` |
| Detection / comm delay (when `delay_mode != "none"`) | `delay_system.set_field_delay_mode(agent, mode, p)` with `p ≤ 1`; per-agent `effective_progress = min(p * per_agent_latency_scale, 1.0)` where `per_agent_latency_scale ~ Uniform(0.5, 2.0)` | scalar per-agent, shared across envs | `p=0` setting is "no delay" only when `delay_mode == "none"`; once switched to `"fixed"`/`"random"`, all envs share the same progress (modulo per-agent scaling), so `p=1` setting does not contain `p=0` |
| Noise scale | `_noise_scale = p`; effective per-agent std `= base_std * _noise_scale * per_agent_noise_scale` ([multi_agent_wrapper.py:447](../../../delay_system_v3/multi_agent_wrapper.py#L447)) | scalar `_noise_scale` × per-agent dict | `p=0` setting `std=0` not in `p=1` distribution; `per_agent_noise_scale ~ Uniform(0.5, 2.0)` does not include 0 |
| i.i.d. dropout rate | `rate = p * dropout_prob`; per-agent `effective_rate = min(rate + offset, 1.0)` with `offset ~ Uniform(0.0, 0.05)` | scalar × per-agent dict | `p=0` setting (no drops) not present at `p=1` |
| Burst dropout `p_onset` | `p * burst_p_onset` (scalar) | scalar | `p=0` setting not present at `p=1` |
| Target scale (DR) | `1 + p * (Uniform(xy_lo, xy_hi) - 1)` for xy and similarly for z | per-env tensor `(N, 1, 3)` | per-env support contains 1.0 at every `p` (the `1 +` lower bound); the `p=0` setting `= 1.0` is contained at `p=1` since `Uniform(lo, hi)` includes 1.0 if `lo ≤ 1 ≤ hi` |
| Camera FOV scale (DR) | `1 + p * (1/Uniform(low, high) - 1)` | per-env tensor | same — contains 1.0 |
| Gimbal mechanical offsets (DR) | `p * Uniform(...)` | per-env tensor | contains 0 at every `p` |
| Physics mass/inertia (DR) | applied only when `progress_dynamics > 0`; per-env multipliers | per-env tensor (read from `DomainRandomizer`) | depends on randomizer distribution; not audited in this report |
| Gimbal joint stiffness/damping (DR) | same gating as physics | per-env tensor | not audited |

---

## R2 — `progress_tracking` audit (initial-states distributions)

`progress_tracking` is computed at env:2291-2293 and passed verbatim into
`InitialStates.generate(curriculum_progress=...)`
([initial_states.py:102-119](../../../initial_states/initial_states.py#L102-L119)),
which forwards to `InitialStatesGenerator.generate(curriculum_progress=...)`
([initial_states_generator.py:69-199](../../../initial_states/initial_states_generator.py#L69-L199)).

Per-axis facts:

### R2.1 `_sample_curriculum_parameters` — [initial_states_generator.py:205-250](../../../initial_states/initial_states_generator.py#L205-L250)

Anti-forgetting sampler family `Uniform(min, min + p * (max - min))`:

| Axis | Sampler at progress `p` | Contains `p=0` value at `p=1`? | Notes |
|---|---|:---:|---|
| `diameters` | `Uniform(cfg.cylinder_diameter_min=20, cfg.cylinder_diameter_min + p * (cfg.cylinder_diameter_max=100 - 20))` | yes | At `p=1` support is `[20, 100]` which contains `p=0` support `{20}`. |
| `target_distances` | `Uniform(10, 10 + p * 30)` | yes | At `p=1` support `[10, 40]` contains `p=0` support `{10}`. |
| `agent_velocity_scales` | `Uniform(0, p * cfg.agent_velocity_scale_max=1)` | yes | Support always includes 0. |
| `target_velocity_scales` | `Uniform(0, p * 1)` | yes | Support always includes 0. |

### R2.2 `_generate_agent_positions_in_cylinder` — [initial_states_generator.py:282-353](../../../initial_states/initial_states_generator.py#L282-L353)

`height_range = cfg.cylinder_height_range_min=2 + p * (cfg.cylinder_height_range_max=5 - 2)`

This is **a single scalar per call**, not per-env. At `p=1` `height_range=5`;
at `p=0` `height_range=2`. Each agent's z is then sampled as
`Uniform(-height_range, +height_range)` within the cylinder.

Verdict: at `p=1`, the per-env vertical spread distribution covers
`Uniform(-5, +5)` which includes `Uniform(-2, +2)` — so the easy
distribution is contained as a subset. **Anti-forgetting preserved**, but
the *expected* spread is larger at `p=1` than at `p=0`. If the policy is
sensitive to the *modal* spread (not the support), this axis still
shifts upward over training.

### R2.3 `_generate_target_positions` — [initial_states_generator.py:393-429](../../../initial_states/initial_states_generator.py#L393-L429)

Bearing `Uniform(0, 2π)` — not curriculum-controlled. Height offset
`Uniform(cfg.target_height_offset_min=0, cfg.target_height_offset_max=4)`
— not curriculum-controlled. Distance is consumed from
`_sample_curriculum_parameters` (R2.1, audited).

### R2.4 `_generate_agent_orientations` — [initial_states_generator.py:498-560](../../../initial_states/initial_states_generator.py#L498-L560)

Default `other_agents_orientation_mode = "face_target"`
([initial_states_cfg.py:148-155](../../../initial_states/initial_states_cfg.py#L148-L155)).
The designated observer always faces target. Both behaviors are
mode-driven, not progress-driven. Noise on yaw is `~N(0, 0.2 rad)`,
constant. **Not a curriculum axis under the default mode.**

The `"curriculum"` mode (line 542-545) blends between face-target and
random uniformly with `p` — at `p=1` it becomes uniform random yaw. The
`p=0` setting (face-target) is **only present with probability `1-p`**
at progress `p`, so at `p=1` the easy support is measure-zero.
**Anti-forgetting fails under `"curriculum"` mode.** Default mode is not
affected.

### R2.5 `_generate_agent_velocities` — [initial_states_generator.py:566-602](../../../initial_states/initial_states_generator.py#L566-L602)

- Linear: magnitude `Uniform(0, scale * cfg.agent_max_velocity=10)` with
  `scale ~ Uniform(0, p * 1)` from R2.1. At `p=1`, magnitude lies in
  `Uniform(0, 10 m/s)`. Support includes 0. **Preserved.**
- Angular yaw rate: `Uniform(-1, 1) * p * cfg.max_yaw_rate=π/4 rad/s`.
  At `p=1`, support `[-π/4, π/4]`. Always includes 0.
  **Preserved.**

### R2.6 `_generate_gimbal_states` — [initial_states_generator.py:608-700](../../../initial_states/initial_states_generator.py#L608-L700)

Default `gimbal_curriculum_mode = "always_pointing"`
([initial_states_cfg.py:179-185](../../../initial_states/initial_states_cfg.py#L179-L185)).
All agents point at target, regardless of `progress`. **Not a curriculum
axis under the default mode.**

`"gradual"` mode (line 670-674): same structure as the orientation
`"curriculum"` mode (R2.4); easy support disappears at `p=1`.
`"threshold"` mode (line 676-682): step from pointing to random at
`p > 0.5`; binary loss of easy support past the threshold.
**Anti-forgetting fails under these two modes.** Default is not affected.

### R2.7 `_generate_zoom_levels` — [initial_states_generator.py:742-767](../../../initial_states/initial_states_generator.py#L742-L767)

`zoom_max = cfg.zoom_initial_max_start=3 + p * (cfg.zoom_initial_max_end=4 - 3)`
then `zoom_levels ~ Uniform(cfg.zoom_initial_min=1, zoom_max)`.

At `p=1`: `Uniform(1, 4)`. At `p=0`: `Uniform(1, 3)`. The `p=1` support
contains the `p=0` support. **Preserved.**

### R2.8 Verdict for `progress_tracking`

Under the **default config** (`face_target`, `always_pointing`, etc.),
the initial-states module already satisfies the anti-forgetting property
on every axis: the support at `progress=1` contains the support at
`progress=0`.

Two non-default modes (`other_agents_orientation_mode = "curriculum"`
and `gimbal_curriculum_mode ∈ {"gradual", "threshold"}`) would violate
the property; these are not enabled in the current cfg.

---

## R3 — R0 regression measurement status

**Status**: complete. Full analysis in
[r_research/regression_eval.md](r_research/regression_eval.md). Headline
verdict (visibility_mean ratio, 400k-ckpt / reference-ckpt):

| Eval step | Ratio | Pass ≥0.80? |
|---:|---:|:---:|
| 39 000  | **0.616** | **FAIL** |
| 99 000  | **0.577** | **FAIL** |
| 199 000 | 0.987     | PASS |
| 399 000 | 1.00      | n/a |

Catastrophic forgetting confirmed; monotonic recovery as eval-step → 400 k.

Defect captured: the eval-harness triangulation metrics
(`triangulation_rmse`, `tri_valid_ratio`, `task_success_rate`,
`track_maintenance_rate`, `accuracy_rate`) are uniformly 0.0 in every
cell including reference cells. Root cause in
[evaluate.py:186-212](../../../experiments/evaluate.py#L186-L212) —
`_triangulation_result_obs/gt` are `None` at the point
`_collect_step_metrics` is called under `--no-timeseries`. Does not
affect the regression verdict (visibility is independent). Out of scope
for this ticket; flagged for follow-up.

### Earlier blocker (resolved)

The first invocation failed with numpy 2.2.6 / scipy 1.10.1 ABI mismatch
in the default `_isaac_sim/python.sh`. Resolved by activating
`conda env_isaaclab` (numpy 1.26.0) before invoking `isaaclab.sh`. The
wrapper script now sources `/home/usrg/miniconda3/etc/profile.d/conda.sh`
and activates `env_isaaclab` before each cell.

A secondary defect — the wrapper's `2>&1 | tail` pipe swallowed exit
codes — was fixed in the same iteration: per-cell logs now go to disk
and exit codes are checked explicitly.

### Original status notes (kept for context)


7 `evaluate.py` invocations were launched
([r_research/run_regression_eval.sh](r_research/run_regression_eval.sh))
to load `agent_400000.pt` and the corresponding intermediate checkpoints
at curriculum-pinned env steps. **All 7 failed identically** before
reaching the eval logic with:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
            Expected 96 from C header, got 88 from PyObject
```

Probe (`./isaaclab.sh -p -c "import numpy; ..."`) confirms:

```
numpy version: 2.2.6
numpy path: /home/usrg/IsaacPX4/IsaacLab/_isaac_sim/kit/python/lib/python3.10/site-packages/numpy/__init__.py
scipy version: 1.10.1
warnings.warn(f"A NumPy version >=1.19.5 and <1.27.0 is required for this version of SciPy (detected version 2.2.6)")
```

scipy 1.10.1 was compiled against numpy 1.x; numpy 2.x has incompatible
C ABI. The failure hits at `import gymnasium → import numpy.random`
inside the omni.kit.pip_archive's numpy distribution, before any
isaaclab/iris_ma6 code runs.

Unblock requires either downgrading numpy to `<2.0` (returns env to the
state the fix2 training ran under) or upgrading scipy to `>=1.13`. The
fix is environment-altering and is held for engineer approval before
proceeding.

**Secondary defect in the wrapper script**: `... 2>&1 | tail` swallowed
each invocation's exit code, defeating `set -e` and letting the loop
march through all 7 cells while none produced output. Fix queued (use
`PIPESTATUS` / `set -o pipefail` and write logs to disk with `tee`)
together with the env unblock.

---

## R4 — Cross-references for Stage I

These facts are the inputs to the design stage:

- The two axes already proven to preserve anti-forgetting under
  randomization (R1.3): `zoom_tau` and `tau_motor`/controller gains.
  Both use multiplicative scale randomization centered or asymmetric
  around the nominal value.
- All other dynamics/observability axes use **deterministic** scaling
  by global progress and lose the `p=0` setting at `p=1`.
- The delay system's `per_agent_randomization` is per-**agent** but
  not per-**env** (R1.2 last row); per-(env, agent) granularity is
  not currently supported by the underlying API.
- `_max_lin_vel` curriculum and `_max_lin_vel` randomization
  **replace** each other rather than compose; the randomization
  branch (env:2461-2465) overwrites the curriculum branch
  (env:2446-2448).
- `progress_tracking` (initial states) **already** satisfies
  anti-forgetting under the default mode; no `initial_states_*` code
  change is required by the ticket scope, but two non-default
  config-mode options would violate it.
- Latency floor constraint from Q2 (2 policy steps = 80 ms at 25 Hz)
  is not currently enforced anywhere in
  [delay_cfg_v3.py](../../../delay_system_v3/delay_cfg_v3.py) or
  the wrappers.

---

## R5 — Open items carried into Stage I

- R0 measurement values (deferred — blocked on env fix). Stage I can
  proceed in parallel using the audit above as the failure-mode
  justification; the regression numbers replace the ticket's currently
  qualitative claim once the env is unblocked.
- Comparison of `MetricTracker` JSON output fields versus the
  per-component reward decomposition we want in the regression table —
  to be confirmed once one successful `evaluate.py` run exists.
