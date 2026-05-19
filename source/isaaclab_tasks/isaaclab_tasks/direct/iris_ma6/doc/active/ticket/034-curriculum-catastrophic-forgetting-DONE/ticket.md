## Ticket 034 — Curriculum-driven catastrophic forgetting of easy/clean settings

**Status**: Open
**Created**: 2026-05-17
**Observed in run**: `2026-05-15_12-00-44_mappo_rnn_torch_7fd070ee09_mappo_rnn_shared_model_scheduler_param_fix2`
(current best baseline — see [experiment doc](../../../experiments/2026-05-15_12-00-44_mappo_rnn_torch_7fd070ee09_mappo_rnn_shared_model_scheduler_param_fix2.md))
**Target files**:
- [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) — `_reset_idx`, `_pre_physics_step` curriculum hooks
- [curriculum/curriculum_cfg.py](../../../../curriculum/curriculum_cfg.py) — phase schedule
- [controller/gain_randomization_cfg.py](../../../../controller/gain_randomization_cfg.py) — scale ranges
- [controller/zoom_controller.py](../../../../controller/zoom_controller.py) — `_sample_dead_time`
- [controller/gimbal_rate_loop.py](../../../../controller/gimbal_rate_loop.py) — dead-time sampling
- [delay_system_v3/multi_agent_wrapper.py](../../../../delay_system_v3/multi_agent_wrapper.py) — `set_noise_scale`, `set_dropout_rate`, `set_delay_mode`

---

### What

The 400k checkpoint from the new baseline (ticket-033 fix2) is the strongest training run on record (last-10-avg 2714, peak 5406 @ 28k) but **fails to replay at the 40k-step difficulty regime** — i.e., it has forgotten the easy task (slow agents, slow target, clean fast observations). Walk back through the curriculum schedule and the per-env randomization that backs it: at full progress, the **distribution over envs has collapsed onto the hardest setting for several axes**, leaving no envs at "easy" so the policy never re-sees the early-phase regime and drifts off it.

The previously-reported "slow decline from the 28k peak (5406 → 2700 plateau)" in the fix2 experiment doc (§3) is the same phenomenon viewed from the training side. We can now attribute it to specific curriculum knobs rather than treating it as a generic late-training plateau.

### Why this matters

Two distinct symptoms:

1. **Loss of robustness across the input distribution.** The policy at 400k can't operate at 40k-step env settings even though those settings are physically inside the action/obs space. This is catastrophic forgetting: the empirical training distribution at 220k–400k has zero mass on the early-phase regime, so the gradient signal from those settings vanishes.
2. **Late-training plateau below peak.** The peak at 28k (when only the easy regime is in the distribution) is 2× the 400k plateau. If the late-training distribution kept some envs at the easy regime, the policy should be able to *retain* peak-level performance on the easy slice while still learning the hard slice. The fact that mean reward drops by 50% from peak instead of being averaged ~70/30 between hard and easy slices is consistent with the easy slice being absent from the training distribution.

The `initial_states` generator (cylinder/target distance, velocities, gimbal pointing, zoom levels) already uses the **anti-forgetting sampler** `uniform(min, min + progress * (max - min))`, so those axes preserve easy samples by design. The failure is on a different set of axes where the env writes a **single curriculum-gated value to every env** (or randomizes around a curriculum-shifted mean), not a uniform-over-history sample.

### Audit — which axes preserve "easy" and which collapse it

Inventory of every reset-time / step-time curriculum hook in `iris_ma_env6_test.py`. "Preserves easy" means the post-curriculum env distribution still contains envs at or near the progress=0 setting.

| Axis | Where | Sampler form at progress=1 | Preserves easy? | Notes |
|---|---|---|---|---|
| Agent placement diameter | `initial_states_generator._sample_curriculum_parameters` L221–226 | `Uniform(20, 100) m` | ✓ | Anti-forgetting form. |
| Target distance | same, L229–233 | `Uniform(10, 40) m` | ✓ | Anti-forgetting form. |
| Agent initial velocity scale | same, L236–238 | `Uniform(0, 1) * 10 m/s` | ✓ | Spans 0. |
| Target initial velocity scale | same, L241–243 | `Uniform(0, 1) * 2 m/s` | ✓ | Spans 0. |
| Cylinder vertical spread | `_generate_agent_positions_in_cylinder` L307–309 | `2 + p*(5-2) = 5 m` flat | ✗ | Single value, no uniform — collapses to 5 m. Minor (small range). |
| Body yaw | `_generate_agent_orientations` L539–550 | `face_target` default — not curriculum-randomized | n/a | Mode-driven, not progress-driven. |
| Gimbal pointing | `_generate_gimbal_states` L666 | `always_pointing` default | n/a | Mode-driven (designated observer always points). |
| Initial zoom | `_generate_zoom_levels` L757–765 | `Uniform(1, 3 + p*(4-3)) = Uniform(1, 4)` | ✓ | Anti-forgetting form. |
| **`_max_lin_vel` curriculum** | `_reset_idx` L2446–2448 | `3 + p*(10-3) = 10 m/s` flat | **✗** | All envs locked at 10 m/s. |
| **`_max_lin_vel` randomization** | `_reset_idx` L2459–2465 | `Uniform(0.8, 1.2) * 10 = [8, 12] m/s` | **✗** | Centered on 10; never re-enters 3-7 m/s. |
| **Controller gains (vel/att/rate/motor)** | `controller.randomize_gains` via `GainRandomizationCfg.scale_range=(0.8, 1.2)` | `Uniform(0.8, 1.2) * nominal` | △ | Symmetric around nominal — nominal is in the distribution, but it's *only* the trained nominal: there is no "easy" version of nominal gains, so this isn't a forgetting axis. OK. |
| **τ_zoom curriculum** | `_reset_idx` L2408–2411 | `max(p * 0.091, 1e-4) = 0.091 s` flat | **✗** at the base | But then... |
| **τ_zoom randomization** | `GainRandomizationCfg.zoom_scale_range=(0.01, 1.0)` | `Uniform(0.01, 1.0) * 0.091 = [9e-4, 0.091] s` | ✓ | The wide asymmetric scale_range **explicitly** preserves near-instant zoom. This is the model for the other axes. |
| **Gimbal rate-loop τ (`progress_dynamics`)** | `controller.gimbal_rate_loop.set_progress(p)` L1442 | scale=1.0 applied to every env | ✗ | No matching asymmetric randomization. |
| **Gimbal dead time** | `controller.gimbal_rate_loop.set_dead_time_curriculum_scale(p)` L1450 | per-env sample from `N(66 ms, 16 ms)` clipped to `[0, 120 ms]` | ✗ | Mean is 66 ms — near-zero dead time effectively absent. |
| **Zoom dead time** | `controller.zoom_controller.set_dead_time_curriculum_scale(p)` L1457 | per-env sample from `N(100 ms, 18 ms)` clipped to `[0, 150 ms]` | ✗ | Mean is 100 ms — near-zero dead time effectively absent. |
| **Observation noise** | `delay_system.set_noise_scale(p)` L1421 | scale=1.0 — every env at full noise | ✗ | No envs see clean obs. |
| **Fixed delay (`fixed_delay_*` 120k–140k)** | `delay_system.set_delay_mode("fixed", p)` L1414 | `p=1` → full configured latency means | ✗ | All envs at full latency. |
| **Random delay (`random_delay_*` 140k–160k)** | `set_delay_mode("random", p)` L1417 | `p=1` → full configured latency std | ✗ | All envs at full jitter. |
| **i.i.d. dropout (`dropout_*` 160k–180k)** | `delay_system.set_dropout_rate(p * 0.05)` L1428 | every env at 5% drop | ✗ | No envs see complete obs streams. |
| **Burst dropout (`burst_dropout_*` 200k–220k)** | `set_burst_dropout_params(p_onset=p*..., …)` L1435 | every env at full burst onset prob | ✗ | No envs see uninterrupted obs. |
| **FP/FN (calibrated YOLO miss/FP)** | `_curriculum_fp_fn_scale` set L1424 | **Disabled in 400k** — `fp_fn_background_start_step=2_200_000` is past `all_end_step=400000` | n/a | Not active in this run; irrelevant here. |
| **Per-agent delay/noise/dropout heterogeneity** | `delay_system.randomize_per_agent_params` L2442 | Only varies between agents *within* an env, not between envs | △ | Adds intra-env spread (helpful) but the env-aggregate distribution is still locked to the hard setting. |

The two clusters of failure are:

- **Cluster A — speed/dynamics-side.** `_max_lin_vel` and the gimbal rate-loop τ collapse the env distribution onto a fast/laggy regime. The policy never sees a slow agent or an instant-gimbal env after the relevant phase ends.
- **Cluster B — observation-side.** Noise, fixed/random delay, i.i.d. dropout, burst dropout, gimbal dead-time, zoom dead-time all reach progress=1 well before 400k (last is 220k for burst dropout) and stay there. The env distribution after 220k contains **zero** envs with clean/instant observations.

Cluster B is the most consequential because it's the largest set of axes and they all become non-zero simultaneously. The policy's response between 100k (clean obs end) and 220k (final hardening) is to specialize against the worst-case observation stream — and that specialization is what the 40k-step regression is showing.

### Root cause statement

The curriculum implementation has **two ramping idioms** mixed together:

1. **Uniform-over-history** (`uniform(min, min + p * (max - min))`). Used by `initial_states` axes. Preserves easy.
2. **Deterministic gating** (`progress * value`). Used by all dynamics and observability axes. **Does not** preserve easy — at p=1 every env is at the full hard setting.

The τ_zoom axis has a follow-on randomization step (`zoom_scale_range=(0.01, 1.0)`) which converts idiom (2) into something equivalent to idiom (1), and the experiment record confirms this axis does *not* forget. Every other dynamics/observability axis uses idiom (2) alone.

This is structurally the same anti-forgetting bug `initial_states` was already designed to avoid. It was solved on the placement side; it was not propagated to the dynamics/observability side.

### Desired semantics

For every curriculum axis, the env-aggregate distribution at `current_step ≥ phase_end_step` must include positive mass on the easy setting (`progress=0` equivalent). Concretely, replace idiom (2) with one of:

- **(a) Uniform-over-history per env.** For each env at reset, sample `effective_progress ~ Uniform(0, current_global_progress)` and apply to that env. Identical to `initial_states`'s approach. Cleanest, mirrors existing code.
- **(b) Asymmetric per-env scale.** Like `zoom_scale_range`. For each env, sample a per-env multiplicative scale on the curriculum's effective magnitude so that some envs get scale≈0 (= easy) and others get scale=1 (= hard). The τ_zoom path already does this — it's a working reference implementation.
- **(c) Mixture-of-difficulties.** With probability `keep_easy_p`, force the env to the easy setting; otherwise apply the curriculum at full strength. Trivial and explicit but coarse.

For the noise/delay/dropout/burst-dropout/dead-time axes the cleanest path is (a) applied at per-env resolution, because the underlying systems already accept per-env effective rates via `set_field_*` methods or per-env tensors.

For `_max_lin_vel` the cleanest path is also (a): per-env `eff_progress ~ Uniform(0, progress_agent_velocity)` → `max_lin_vel_min + eff_progress * (max_lin_vel - max_lin_vel_min)`. The existing ±20% randomization then layers on top unchanged.

### Validation criterion

The fix is validated when the same 400k checkpoint, evaluated on the **40k-step env settings** (target_velocity_scale≈0, agent_velocity_scale≈0, clean obs, instant gimbal, fast zoom), achieves reward within a clearly-defined margin (proposal: ≥ 80%) of the 40k checkpoint's reward on the same settings. The current state of the world: 0% (the policy fails outright on those settings).

A weaker secondary criterion: the late-training mean reward at 400k should reach a plateau ≥ 70% of peak rather than the current ~50%. The expectation is that this is the easy-slice contribution returning to the average once the easy slice is back in the distribution.

### Implementation sketch (not binding)

1. **Add `per_env_curriculum_jitter` to `CurriculumCfg`**, default 0.0 (off) for behavioral parity, 1.0 (full uniform-over-history) when enabling the fix.
2. **In `_reset_idx`**, for each env-id being reset, draw `eff_p ~ Uniform(0, global_p)` once per env and use it as the per-env effective progress for the dynamics/observability hooks. Where the underlying system only accepts a scalar (e.g. `set_noise_scale(scale: float)`), promote the API to take a per-env tensor (existing patterns in `set_field_*` and `randomize_per_agent_params` show this is structurally supported).
3. **For `_max_lin_vel`**, replace L2446–2448 with the per-env eff-progress computation. Keep the ±20% randomization layer.
4. **For the τ_zoom path**, no change required (already correct via `zoom_scale_range`).
5. **For controller gains (`scale_range=(0.8, 1.2)`)**, no change required — the symmetric range around nominal already covers the easy ≡ nominal point.
6. **Test plan**: a short ablation run (~50k steps) restarting from `agent_drone_*_final.pt` with `per_env_curriculum_jitter=1.0` should show reward stay near 2700 (not crash) and an eval at 40k-step settings should recover. A from-scratch run is the real test but is gated on the shorter-restart smoke first.

### Open questions

- Should `per_env_curriculum_jitter` be the same for every axis, or per-axis (e.g. heavier easy-slice for observation-side than dynamics-side)? Default per-axis values can be deferred to the experiment that turns the flag on.
- Per-env jitter implies per-env state in the delay/noise systems that today share scalars. The plumbing change is mechanical but non-trivial — list every `set_*_scale(scalar)` call site before starting.
- Interaction with `progress_safety`, `progress_coord`, and `progress_tracking` (reset-only scalar curricula): these are *task-signal* curricula, not difficulty curricula. They should remain step-at-episode-boundary as today. The fix only applies to dynamics/observability axes.
- The `fp_fn_background_start_step=2_200_000` default disables FP/FN in 400k runs. Is this intentional, or a leftover from a longer-horizon training plan? Worth flagging separately because the FP/FN-on case is the one where the policy is most likely to need exposure to bbox-empty states during late training.

### Related

- Ticket 030 (yaw distribution diversity) — same family of issue: a distribution-shape concern that hurts robustness. Likely shares fixture scaffolding for the per-env-aware delay/noise plumbing.
- Anti-forgetting design note in `initial_states_cfg.py` L32–48 — already articulates the exact rule that this ticket needs to propagate to other axes.
- Experiment doc §5 "Notes for future work" — independently lists "extend curriculum window" as a candidate next experiment. This ticket frames *why* extending the window would help (gives easy-slice more wall-clock at the hardened end of curriculum if the distribution is reshaped) and proposes a more direct fix.
