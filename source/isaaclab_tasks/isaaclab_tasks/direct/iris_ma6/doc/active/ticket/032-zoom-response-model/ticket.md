## Ticket 032 — SIYI A8 zoom response model (sim port of mas/037)

**Status**: Open
**Created**: 2026-05-05
**Target env**: v0 ([iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) + [iris_ma_env6_test_cfg.py](../../../../iris_ma_env6_test_cfg.py)).
**Source ticket**: [mas/037](file:///home/usrg/mas/src/doc/active/tickets/037-zoom-response-characterization-DONE/ticket.md) — characterization complete, fitted parameters in [zoom_model.json](file:///home/usrg/mas/src/scripts/sim2real_model_fitting/output/zoom_model.json).
**Architectural pattern**: mas/035 (rate-loop τ) + mas/036 (input-side dead-time buffer), already implemented for the gimbal axes in [controller/gimbal_rate_loop.py](../../../../controller/gimbal_rate_loop.py). This ticket applies the same shape to the zoom path.

**What**: Replace the current first-order-lag-only zoom model in [controller/zoom_controller.py](../../../../controller/zoom_controller.py) with the four-stage measured model from mas/037:

```
zoom_rate_cmd ∈ [-1, 1]
   │
   ├─ × action_scale (= cfg.zoom.max_zoom_rate, default 2.0 levels/s)  ← policy denorm; DEPLOYMENT-MATCHED, kept ≤ v_max
   │
   ├─ rate clip ±v_max  (default 3.16 levels/s; symmetric; lens-level slew limit)
   │
   ├─ input-side dead-time buffer  (per-env N(τ_d_mean, τ_d_std), clipped to [0, τ_d_max], curriculum-scaled — mas/036 pattern)
   │
   ├─ integrator         target ← target + rate · dt;   clamp [zoom_min, zoom_max]   (continuous float32 state)
   │
   ├─ first-order lag    state  ← state + (1 − exp(-dt/τ_1)) · (target − state)      (continuous float32 state; τ_1 ≈ 0.091 s)
   │
   ├─ self.zoom_internal     ← state    (continuous; this is what compute_z_eff and obs use TODAY)
   │
   └─ self.zoom_published    ← round(state / 0.1) · 0.1   (NEW; this is what the policy SHOULD see in obs)
```

**Why**: The current `ZoomController.compute_control` ([zoom_controller.py:83-115](../../../../controller/zoom_controller.py#L83-L115)) implements only the first-order lag stage. The deployed SIYI A8 mini lens has a measured **~100 ms dead-time** (std 18 ms), a hard **3.16 levels/s slew limit**, and a **0.1-level output quantum** — none of which appear in the current sim. The policy currently trains against a continuous, infinite-bandwidth zoom that it can drive with arbitrarily fast rate commands; deployed it will see chunky, delayed, slew-limited motion. mas/037's bench data + fit ([zoom_model.json](file:///home/usrg/mas/src/scripts/sim2real_model_fitting/output/zoom_model.json)) gives us the measurement; this ticket consumes it.

**Blocked on**: nothing. mas/037 closed `compare_zoom.py` validation (median |error| ≤ 0.10 levels on 3/4 bench runs in the linear regime). The rate-loop architecture pattern (mas/035) and the input-side dead-time buffer pattern (mas/036) are both already deployed for the gimbal axes — this ticket is a direct application of the same hooks to the zoom path.

**Depends on**: mas/035, mas/036 — same `set_progress` + `set_dead_time_curriculum_scale` curriculum-hook pattern. Already shipped for the gimbal in [iris_ma_env6_test.py:1423-1434](../../../../iris_ma_env6_test.py#L1423-L1434); we add the matching pair for zoom.

### Truth values (from mas/037 [zoom_model.json](file:///home/usrg/mas/src/scripts/sim2real_model_fitting/output/zoom_model.json), 2026-05-05 bench)

| Parameter | Value | Field name in cfg |
|---|---|---|
| Dead-time mean | **0.100 s** | `dead_time_mean_s` |
| Dead-time std | **0.018 s** | `dead_time_std_s` |
| Dead-time max (cap) | **0.150 s** (= mean + ~3σ ceiling for buffer pre-allocation) | `dead_time_max_s` |
| Max slew rate v_max | **3.16 levels/s** (symmetric in/out, 0.2% asymmetry; well under the 10% acceptance) | `v_max_levels_per_s` |
| First-order lag τ₁ | **0.091 s** (canonical: rate_sine 0.25 Hz xcorr) | `tau_zoom_s` |
| Output quantum | **0.1 levels** (A8 protocol-level constant) | `quantum_levels` |
| Action scale (policy → rate, deployment-matched) | **2.0 levels/s** (kept BELOW v_max so the policy's `[-1, 1]` lives in the linear regime — see mas/037 Tip 5) | `max_zoom_rate` (already exists) |

### Background — what already exists in the env

- **First-order lag model**, `tau_zoom = 0.1 s`, `max_zoom_rate = 2.0 levels/s` ([zoom_controller_cfg.py](../../../../controller/zoom_controller_cfg.py)).
- **`max_zoom_rate` is dual-purpose today**: it serves both as the policy action scale (`zoom_rate ← cmd * max_zoom_rate`) AND as the implicit hardware slew limit. mas/037 forces us to disambiguate these: the action scale stays 2.0 (deployment), the lens v_max (3.16) becomes a separate cfg field. See "Action / observation interface" below.
- **Curriculum-gated tau_zoom**: env writes `_zoom._tau_zoom = max(cfg.zoom.tau_zoom * progress_dynamics, 1e-4)` at every reset ([iris_ma_env6_test.py:2272-2275](../../../../iris_ma_env6_test.py#L2272-L2275)), giving instant zoom at curriculum bootstrap and the configured τ at full progress. Carries over verbatim with the new τ₁ value.
- **DR on tau_zoom**: `GainRandomizationCfg.zoom_scale_range = (0.01, 1.0)` multiplied onto the curriculum-gated base ([gain_randomization_cfg.py:49-60](../../../../controller/gain_randomization_cfg.py#L49-L60)). Carries over with τ₁.
- **DR on max_zoom_rate**: same multiplicative hook ([drone_controller.py:507-508](../../../../controller/drone_controller.py#L507-L508)). With the new action-scale / v_max split, this ticket randomizes the **action scale** (stays ≤ v_max so DR can't push the policy into saturation) — see decision 6.
- **Curriculum-gated zoom initial state**: in `_reset_idx`, `result.zoom_levels[:, idx]` initializes both the env tracking tensor and the controller's internal `_zoom` / `_zoom_target` ([iris_ma_env6_test.py:2253-2263](../../../../iris_ma_env6_test.py#L2253-L2263)). Carries over.
- **Gimbal rate-loop + dead-time buffer**, both already plumbed: [GimbalRateLoop](../../../../controller/gimbal_rate_loop.py), [GimbalRateLoopCfg](../../../../controller/gimbal_rate_loop_cfg.py), curriculum hooks at [iris_ma_env6_test.py:1423-1434](../../../../iris_ma_env6_test.py#L1423-L1434). The zoom path will mirror this verbatim.
- **Zoom curve** `z_eff = 1 + a · (exp(b · (cmd − 1)) − 1)` ([zoom_controller.py:25-37](../../../../controller/zoom_controller.py#L25-L37)) — separate calibration (mas/028, NOT touched by this ticket). Maps the operator-zoom command to the focal-length multiplier; consumed downstream by [_compute_zoomed_intrinsics](../../../../iris_ma_env6_test.py#L1118) and [bbox_raycaster](../../../../bbox_raycaster_v2/) to render zoom-aware images. The new model fits the **operator zoom command**'s transient (zoom levels in [1.0, 5.0]); `z_eff` continues to apply to whatever the model emits.

### Design choices (decided up front; do not re-litigate during implementation)

1. **Disambiguate `max_zoom_rate` from `v_max`.** Two physical quantities, two cfg fields:
   - `cfg.zoom.max_zoom_rate` (existing, default 2.0) = **policy action scale**. The `[-1, 1]` rate command is multiplied by this. Matches `mas_policy.action_publisher.max_zoom_rate = 2.0`. Keep it ≤ `v_max_levels_per_s` so the policy never asks for sustained super-v_max rates (mas/037 saturation-stall regime, intentionally avoided per Tip 5).
   - `cfg.zoom.v_max_levels_per_s` (NEW, default 3.16) = **lens-level slew clip**. Applied AFTER the action scaler, BEFORE the dead-time buffer. With default action_scale=2.0 < v_max=3.16, the clip is a no-op in nominal operation; it becomes load-bearing if a future task / DR ever drives action_scale toward v_max.

2. **Output quantization on the published level only.** The integrator's internal `_zoom` state stays continuous float32. We add a `_zoom_published = round(_zoom / quantum) * quantum` accessor. The env's observation pipeline (and the env's `self.zoom_level[:, idx]` mirror that feeds [_compute_zoomed_intrinsics](../../../../iris_ma_env6_test.py#L1118) and the bbox raycaster) reads `_zoom_published`. **Internal `_zoom` is NEVER quantized** — quantizing the integrator state corrupts the sub-quantum momentum and produces jitter that does not exist on hardware (mas/037 Tip 1). DR / curriculum / set_zoom continue to write the continuous state.

3. **Dead-time buffer is per-env, sampled at reset, constant within an episode.** Same Gaussian-sample-then-clip mechanism as [GimbalRateLoop._sample_dead_time](../../../../controller/gimbal_rate_loop.py#L278-L313). Curriculum scale ∈ [0, 1]: 0 = no delay (preserves pre-mas/037 behavior), 1 = full measured Gaussian. Per mas/037 scope-boundary "reuse the gimbal-axis dead-time curriculum scale unless the fit forces otherwise" — the fit gives zoom 100 ms vs gimbal 66 ms (mean), distinguishably different, so we add a **separate** zoom curriculum knob. The implementation pattern is shared; the curriculum knob is not.

4. **Curriculum staging.** Two new knobs in [CurriculumCfg](../../../../curriculum/curriculum_cfg.py), aligned with the existing dynamics / gimbal-dead-time cadence:
   - `zoom_tau_start_step / zoom_tau_end_step` (NEW knob, distinct from `dynamics_*`) — gates the τ₁ ramp on the zoom rate-loop. Default = `(20000, 200000)`. Rationale: bootstrap (< 20k) keeps τ₁ near-instant (`progress=0` → `tau_zoom_eff = max(cfg.tau_zoom * 0, 1e-4) = 1e-4`) so the policy can learn bbox / zoom alignment with infinite-bandwidth zoom; after 20k it ramps **slowly** over a ~180k window so the policy gradually adapts to the measured τ₁ = 0.091 s without a step change. This **decouples** the zoom τ ramp from the gimbal-rate-loop τ ramp (`dynamics_*` 60k / 100k) — zoom and gimbal have different timescales, mas/037 fits zoom independently of mas/035, and the policy benefits from a slower zoom-τ ramp in particular because zoom misalignment cascades into bbox-size and triangulation noise. The env's existing reset-time write becomes `_tau_zoom = max(cfg.zoom.tau_zoom * progress_zoom_tau, 1e-4)` at [iris_ma_env6_test.py:2272-2275](../../../../iris_ma_env6_test.py#L2272-L2275), reading `curr.get_zoom_tau_progress(current_step)` instead of `progress_dynamics`.
   - `zoom_dead_time_start_step / zoom_dead_time_end_step` — gates the dead-time buffer ramp (mas/036 pattern). Default = `(180000, 220000)`, mirroring `gimbal_dead_time_*`. Independent of the τ₁ ramp.

5. **First-order lag stays.** mas/037's fit has `first_order_tau_s = 0.091 s` (non-zero). The fit's acceptance test (compare_zoom.py median |err| ≤ 0.10 on 3/4 linear-regime runs) requires it. Default `cfg.zoom.tau_zoom = 0.091` (down from 0.1, within DR range so no behavior shock).

6. **DR scope (per-env, per-episode).** Three knobs, each multiplicative on the configured nominal:
   - **τ₁ DR** (existing, repurposed): `GainRandomizationCfg.zoom_scale_range = (0.01, 1.0)` continues to scale the curriculum-gated τ₁ base. Carry over verbatim — wide range preserves the bootstrap-time near-instant zoom.
   - **action_scale DR** (existing, renamed-in-place): the existing `randomize_zoom` second `scale2` ([drone_controller.py:507-508](../../../../controller/drone_controller.py#L507-L508)) continues to randomize what is now the policy action scale (`_max_zoom_rate`). DR range stays `(0.8, 1.2)` (one knob shared with τ₁). Crucial: the multiplied range must keep `action_scale * scale2 ≤ v_max` — at default 2.0 × 1.2 = 2.4 < 3.16, still safe.
   - **v_max DR**: NOT randomized in this ticket. v_max is a measured firmware/lens constant; DR'ing it asymmetrically without a v_max symmetric-saturation-stall model is asking for trouble (mas/037 Tip 6). Add as cfg field `randomize_v_max: bool = False` for forward compatibility; default off.
   - **τ_d DR**: covered by the curriculum scale + per-episode Gaussian draw. No additional multiplicative DR knob.

7. **Architecture = single `ZoomController`, internal stages.** No new module. The four stages live inside `compute_control`, in the order shown in the diagram. Reusing `GimbalRateLoop`'s buffer code via copy-and-adapt is appropriate (zoom is 1-D, gimbal is 2-D yaw+pitch — different shapes, identical ring-buffer logic). The shared logic is small enough that wrapping it in a generic `DeadTimeBuffer1D` helper is overkill; copy the ~80 lines into `ZoomController` and keep both call-sites readable. Document the parallel in both modules' top comments.

8. **Reset semantics carry over from the existing controller.** `set_zoom(z)` writes both the integrator state and the post-lag state to `z` (no transient on initial-state randomization). Episode reset clears the dead-time buffer for the affected envs and resamples per-env τ_d from the current curriculum scale (matches `GimbalRateLoop.reset` at [gimbal_rate_loop.py:209-230](../../../../controller/gimbal_rate_loop.py#L209-L230)).

9. **`compute_z_eff` is unchanged.** mas/028's calibration curve maps the operator zoom level to the focal-length multiplier; mas/037 fits the operator zoom level's transient. Composable. The env continues to read `self.zoom_level[:, idx]` (= controller's published level) and feed it through `compute_z_eff` for the intrinsics path.

10. **Backward-compatible model selector.** Add `cfg.zoom.model: str = "first_order"` to `ZoomControllerCfg`. Two values:
    - `"first_order"` (default — preserves bit-exact pre-mas/037 behavior on existing checkpoints, training runs, and tests): only stages 1 + 4 + 5 run (action denorm → integrator → first-order lag). Stages 2 (v_max clip), 3 (dead-time), and the output quantization (design 2) are bypassed. The `zoom` and `zoom_internal` properties return the same continuous post-lag state. Curriculum hook `set_dead_time_curriculum_scale` becomes a no-op. v_max / quantum / dead_time cfg fields are present but unread. This makes flipping `model` a pure-additive change with no risk to in-flight runs or trained policies.
    - `"siyi_a8"`: all four stages active, output quantized at 0.1, dead-time buffer engaged when curriculum scale > 0. The full mas/037 model.
    Implementation: a single `if self.cfg.model == "siyi_a8"` branch inside `compute_control` selects the pipeline. Both modes share the integrator + lag stages — only the surrounding pre-clip / dead-time / post-quantization wrapping differs. Selection is per-controller-construction (cfg-time), not per-step. Env's `_init_<task>_v0` (or equivalent) chooses the mode; YAML configs override via `agent_cfg.cfg_overrides` if present.

    Default = `"first_order"` so this ticket's landing does not change behavior for any current task / yaml / training run. New runs that want the deployment-faithful model set `model: siyi_a8` (or wire it from a yaml). The bit-exact regression gate (`test_zoom_response_model_stages.py::test_first_order_mode_unchanged`) compares `model="first_order"` outputs against a reference trace captured before this ticket lands — locking in zero behavior drift for the default mode.

### Workflow

1. **Write [zoom_response_spec.md](../../../zoom_response_spec.md)** — per [iris_ma6/CLAUDE.md](../../../../CLAUDE.md) "Specs live in doc/*_spec.md". Document the four-stage model, the calling contract (per the §Stateful Component Rules in [/home/usrg/IsaacPX4/IsaacLab/CLAUDE.md](../../../../../../../../CLAUDE.md)), the DR/curriculum hooks, and the published-vs-internal split. Reference mas/037 truth values.

2. **Edit [controller/zoom_controller_cfg.py](../../../../controller/zoom_controller_cfg.py)** — add the new fields:
   ```python
   model: str = "first_order"          # NEW (design 10); "first_order" | "siyi_a8". Default preserves pre-mas/037.
   tau_zoom: float = 0.1               # UNCHANGED default to preserve "first_order" mode bit-exactness.
                                       # When model="siyi_a8", the env / yaml should override to 0.091 (mas/037 fit).
   v_max_levels_per_s: float = 3.16    # NEW; lens slew clip; only consumed when model="siyi_a8".
   quantum_levels: float = 0.1         # NEW; output quantum; only consumed when model="siyi_a8".
   dead_time_mean_s: float = 0.100     # NEW; mas/037 deadtime_s; only consumed when model="siyi_a8".
   dead_time_std_s: float = 0.018      # NEW; mas/037 deadtime_std_s; only consumed when model="siyi_a8".
   dead_time_max_s: float = 0.150      # NEW; per-env clip cap (≈ mean + 3σ); only consumed when model="siyi_a8".
   dead_time_curriculum_scale: float = 0.0   # NEW; 0 → no delay (preserves pre-mas/037); only consumed when model="siyi_a8".
   randomize_v_max: bool = False       # NEW; reserved (DO NOT enable per design 6).
   ```
   `max_zoom_rate` stays as is (it's now explicitly the policy action scale; clarify in docstring).

3. **Edit [controller/zoom_controller.py](../../../../controller/zoom_controller.py)** — replace `compute_control`'s body with the four-stage pipeline. Add:
   - `_zoom_target` becomes the integrator's pre-lag target (was already named this; now its semantics are pinned).
   - `_zoom_post_lag` (renamed from `_zoom`; the post-lag continuous state).
   - `_zoom_published` property: `torch.round(self._zoom_post_lag / quantum) * quantum`.
   - `zoom` property: returns `_zoom_published` (= what env consumes for obs / intrinsics; behaviorally equivalent to the deployed lens-published value).
   - `zoom_internal` property: returns `_zoom_post_lag` (continuous; for tests / debug only).
   - `_dead_time_buffer`, `_dead_time_seconds`, `_dead_time_steps`, `_dead_time_n_pushed`, `_dead_time_curr_scale` — copied from `GimbalRateLoop` with shape `[max_steps, num_envs]` (1-D, no axis dim).
   - `_apply_dead_time(rate_after_clip, dt) -> rate_delayed` — the 1-D analog of [gimbal_rate_loop.py:358-405](../../../../controller/gimbal_rate_loop.py#L358-L405).
   - `_sample_dead_time(env_ids)` — Gaussian sample, curriculum scale, clip — analog of [gimbal_rate_loop.py:278-313](../../../../controller/gimbal_rate_loop.py#L278-L313).
   - `set_dead_time_curriculum_scale(scale)` — analog of [gimbal_rate_loop.py:241-251](../../../../controller/gimbal_rate_loop.py#L241-L251).
   - `set_v_max(v)` — write per-env v_max tensor (DR hook; default unused per design 6).

   Stage order inside `compute_control`:
   ```python
   # 1) Action denorm: policy [-1, 1] → physical levels/s
   rate = zoom_rate_cmd * self._max_zoom_rate                        # action scale (= cfg.max_zoom_rate)
   # 2) Lens-level slew clip (symmetric)
   rate = torch.clamp(rate, -self._v_max, self._v_max)
   # 3) Input-side dead-time buffer (mas/036 pattern, 1-D)
   rate = self._apply_dead_time(rate, dt)
   # 4) Integrator: continuous target, clamped to [zoom_min, zoom_max]
   self._zoom_target = torch.clamp(self._zoom_target + rate * dt,
                                    self.cfg.zoom_min, self.cfg.zoom_max)
   # 5) First-order lag toward target (curriculum-gated τ₁ via _tau_zoom)
   alpha = 1.0 - torch.exp(-dt / torch.clamp(self._tau_zoom, min=1e-6))
   self._zoom_post_lag = self._zoom_post_lag + alpha * (self._zoom_target - self._zoom_post_lag)
   # 6) Clamp post-lag (defensive; matches existing controller)
   self._zoom_post_lag = torch.clamp(self._zoom_post_lag, self.cfg.zoom_min, self.cfg.zoom_max)
   return self.zoom   # property → quantized published level
   ```
   `reset(env_ids)` clears the dead-time buffer for those envs and resamples τ_d. `set_zoom(z)` writes both `_zoom_target` and `_zoom_post_lag` to `z` (no published-level transient).

4. **Edit [controller/drone_controller.py](../../../../controller/drone_controller.py)** — pass-through changes only:
   - In `randomize_gains`, the existing `_zoom._tau_zoom` and `_zoom._max_zoom_rate` writes carry over (same field names). No new DR knob (per design 6).
   - In `_nominal_gains` ([drone_controller.py:203-204](../../../../controller/drone_controller.py#L203-L204)), `tau_zoom` continues to track `_zoom._tau_zoom` (now τ₁); `max_zoom_rate` continues to track action scale.
   - Add `_zoom._v_max` to `_nominal_gains` for symmetry / DR forward compatibility (write `cfg.zoom.v_max_levels_per_s` at construction; never read by the active randomize path while design 6 holds).

5. **Edit [curriculum/curriculum_cfg.py](../../../../curriculum/curriculum_cfg.py)** — add four new fields and two getters:
   - `zoom_tau_start_step: int = 20000` and `zoom_tau_end_step: int = 200000`. Bootstrap (< 20k) keeps τ₁ near-instant; slow ramp through 200k.
   - `zoom_dead_time_start_step: int = 180000` and `zoom_dead_time_end_step: int = 220000`. Mirrors `gimbal_dead_time_*`.
   - `get_zoom_tau_progress(current_step) -> float` — used by the env's reset-time write of `_tau_zoom`. Replaces `progress_dynamics` for the zoom path only.
   - `get_zoom_dead_time_progress(current_step) -> float` — used by the env's per-step `set_dead_time_curriculum_scale` hook. Mirror the gimbal-dead-time block at [curriculum_cfg.py:266-285](../../../../curriculum/curriculum_cfg.py#L266-L285).

6. **Edit [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py)** — three small additions:
   - **Curriculum hook for zoom dead-time scale**: at the same site as the gimbal-dead-time hook ([iris_ma_env6_test.py:1429-1434](../../../../iris_ma_env6_test.py#L1429-L1434)), add:
     ```python
     self._controller.zoom_controller.set_dead_time_curriculum_scale(
         curr.get_zoom_dead_time_progress(current_step)
     )
     ```
   - **`self.zoom_level[:, idx] = zoom_batch[s:s + N]`** at [iris_ma_env6_test.py:807](../../../../iris_ma_env6_test.py#L807): when `model="siyi_a8"`, `zoom_batch` arrives quantized (controller's `zoom` property returns `_zoom_published`). When `model="first_order"`, `zoom_batch` is the continuous post-lag state — bit-exact pre-mas/037. No code change at the env site; the controller's property handles the mode switch.
   - **Reset path — τ₁ curriculum source**: change `tau_zoom_curriculum = max(self.cfg.drone_controller.zoom.tau_zoom * self.progress_dynamics, 1e-4)` at [iris_ma_env6_test.py:2272-2274](../../../../iris_ma_env6_test.py#L2272-L2274) to read `self.curriculum_progress.get_zoom_tau_progress(current_step)` (or use a cached `self.progress_zoom_tau` mirror written next to `self.progress_dynamics`). The existing `_zoom._tau_zoom[batch_ids] = tau_zoom_curriculum` write itself is unchanged. With the default `zoom_tau_*=(20000, 200000)`, this keeps `tau_zoom ≈ 1e-4` for steps < 20k (bootstrap-instant zoom) and slowly ramps to `cfg.zoom.tau_zoom` thereafter.
   - **Reset path — `set_zoom`**: existing `_zoom.set_zoom(...)` at [iris_ma_env6_test.py:2259-2261](../../../../iris_ma_env6_test.py#L2259-L2261) carries over (writes the continuous internal state; published quantization is computed on read).

7. **No agents/yaml changes required.** The action / observation interfaces are unchanged: action[6] still ∈ [-1, 1], obs zoom field still emits a single float. The float is now quantized to 0.1, but the obs dim stays 1. Policies trained on the old continuous-zoom env will see slightly chunkier zoom obs after this change — that is intentional (mas/037 Tip 2). No reward/scaling change.

8. **Tests** at [controller/tests/](../../../../controller/tests/) — add the following, runnable via the existing [run_tests.py](../../../../controller/tests/run_tests.py) AppLauncher harness:

   - **`test_zoom_response_model_stages.py`** — pure-PyTorch unit test of the four stages on a small (num_envs=8) `ZoomController` with `model="siyi_a8"`. Adds an explicit **mode-selection** test (`test_first_order_mode_unchanged`) that compares `model="first_order"` outputs across a 200-step random-rate rollout against the pre-mas/037 controller's analytic formula (`alpha = 1 - exp(-dt/τ); state += alpha * (state + cmd*max_rate*dt - state)`) — must match within FP noise (atol=1e-6). This is the hard bit-exact backward-compatibility gate. Tests in this file:
     - **Stage 2 — slew clip**: feed sustained `zoom_rate_cmd = 1.0` at `max_zoom_rate = 4.0` (= 4 levels/s commanded). With `v_max = 3.16`, post-clip rate is 3.16. Integrate 1 s with τ₁ = 0 (set via `set_tau_zoom(0)`), τ_d = 0 (curriculum scale = 0). Assert `_zoom_target` advanced by **3.16 ± 0.05**, not 4.0.
     - **Stage 3 — dead-time**: with `dead_time_curriculum_scale = 1.0`, set per-env `_dead_time_seconds = 0.1`, `dt = 0.04` (env step). Feed `zoom_rate_cmd = 1.0` for 5 steps. The first 2 steps (≥ 100 / 40 = 2.5 → 3 steps cold) emit zero rate; integrator stays at 1.0. Step 3+ emits 1.0 · max_zoom_rate. Assert cold-start zero output, then non-zero after the dead-time depth.
     - **Stage 4 — integrator clamp**: zoom_target stays in [zoom_min, zoom_max] under sustained max-rate command (existing test, carry over).
     - **Stage 5 — first-order lag**: with τ₁ = 0.091, step a unit target jump and assert post-lag state at t = τ₁ is **0.632 ± 0.01** (analytic).
     - **Output quantization**: feed continuous internal state of 1.234567; assert `controller.zoom` returns 1.2 exactly (not 1.234567). Confirm `controller.zoom_internal` returns 1.234567.
     - **Bit-exact regression at default scales**: with `dead_time_curriculum_scale = 0`, `tau_zoom = 1e-4`, `max_zoom_rate * 1.0 = 2.0`, the controller must produce the same `zoom_internal` trace as the pre-mas/037 first-order-only model (within FP error). This is the curriculum-bootstrap regression gate.

   - **`test_zoom_compare_to_bench.py`** — replay-style integration test, **runs only if [zoom_model.json](file:///home/usrg/mas/src/scripts/sim2real_model_fitting/output/zoom_model.json) and the four bench dataset directories are reachable** (skip with explanatory message otherwise — keeps CI portable). For each of the four mas/037 bench runs (`level_step_quantum`, `level_step_paired`, `rate_chirp`, `rate_sine_saturated`):
     - Load the recorded `cmd` timeseries.
     - Drive a single-env `ZoomController` (cfg loaded from `zoom_model.json`, action scale = 1.0 so cmd is in physical units, dead_time_curriculum_scale = 1.0, all DR off) at the recorded sample rate.
     - Compare the controller's `zoom` (published, quantized) against the bench `level_observed` trace.
     - Acceptance (matches mas/037's `compare_zoom.py`): median absolute level error **≤ 0.15 levels** on `level_step_quantum`, `level_step_paired`, `rate_chirp`. `rate_sine_saturated` is informational only (saturation-stall regime, mas/037 §"Saturation stall" — not in the deployment-relevant operating envelope).

   - **`test_zoom_curriculum_hooks.py`** — confirms the env-side wiring:
     - Construct `ZoomController(num_envs=4)`. Call `set_dead_time_curriculum_scale(0.0)`; call `reset(env_ids=None)`. Assert `_dead_time_seconds.max() == 0.0` (force-zero path).
     - Call `set_dead_time_curriculum_scale(1.0)`; call `reset(env_ids=None)`. Assert `_dead_time_seconds.mean()` is within **±2σ of 0.100** (Gaussian-sample, num_envs small so loose bound).
     - Call `reset(env_ids=torch.tensor([0, 1]))` with scale=1.0 after a scale=0 reset. Assert envs 0,1 have non-zero τ_d, envs 2,3 retain τ_d = 0 (per-env reset semantics).
     - Bit-exact regression: with scale=0, the new controller's `compute_control` over a 100-step rollout must produce the same output as a control run that bypasses the dead-time stage entirely (the `_apply_dead_time` early-out path).

   - **`test_zoom_published_vs_internal.py`** — locks in design 2 (no internal quantization):
     - Drive a 1-env controller with `zoom_rate_cmd = 0.05` (very slow), τ₁ = 0, τ_d = 0, action_scale = 1.0, for 20 steps at dt = 0.04.
     - Assert `zoom_internal[t]` is monotonically increasing by ~0.002 per step (continuous) — proves momentum is preserved.
     - Assert `zoom[t]` snaps to 0.1 boundaries — proves the published value is quantized.
     - Assert the published trace transitions from 1.0 → 1.1 → 1.2 at the integrator-state crossings, without rebinning back to 1.0.

9. **Iterative test harness** per [/home/usrg/IsaacPX4/IsaacLab/CLAUDE.md](../../../../../../../../CLAUDE.md) §"Iterative Test-Fix Workflow":
   - Run `./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py > .../tests/test_result.txt 2>&1`.
   - Append per-iteration error analysis to `controller/tests/error_log.txt`.
   - Loop until all four new test files pass + the existing `controller/tests/` suite stays green.

10. **Sim2real regression**: rerun [controller/sysid_output/gimbal/compare_gimbal.py](../../../../controller/sysid_output/gimbal/compare_gimbal.py) (gimbal regression — should be unchanged; this ticket touches zoom only). Add a `compare_zoom.py` symlink-or-fork next to it that points at the bench data and reuses `test_zoom_compare_to_bench.py`'s comparison logic — gives the on-vehicle / sim agreement plot (PDF / PNG) for the experiment lab notebook.

### Scope boundary

- DO: implement all four stages (rate clip, dead-time, integrator, lag), output quantization on the published path, per-env DR / curriculum hooks for τ_d and τ₁, behind the `cfg.zoom.model` selector.
- DO: default `cfg.zoom.model = "first_order"` so this ticket's landing is a no-op for every existing yaml / training run / checkpoint resume. Opt-in to `"siyi_a8"` by cfg or yaml override.
- DO: keep `tau_zoom` as the field name for the first-order lag's time constant (preserves DR / curriculum compatibility with the existing `randomize_zoom` path and the `_tau_zoom` per-env tensor name).
- DO: preserve bit-exact behavior at curriculum-bootstrap (`dead_time_curriculum_scale = 0`, `tau_zoom ≈ 1e-4`, `max_zoom_rate * scale ≤ v_max`) vs the pre-mas/037 first-order-only model. Locked in by `test_zoom_response_model_stages.py`.
- DO: place the dead-time buffer at the **input to the integrator** (matches mas/036 architecture for the gimbal axes; matches the on-vehicle integrator's input).
- DO: clip the integrator target to `[zoom_min, zoom_max]` at every sim step (matches the existing controller and the on-vehicle integrator).
- DO: quantize at the **output stage**, not at the integrator state — the integrator carries continuous float momentum; only the published level snaps to 0.1 (mas/037 Tip 1).
- DO NOT: model the SIYI `0x05 MANUAL_ZOOM` direction-only protocol behavior. The sim policy interface is the rate command; the protocol-level binary command is firmware-internal and not part of the sim2real boundary (mas/037 scope-boundary).
- DO NOT: model camera frame-rate drop / image artifacts during zoom transitions (out of scope; vision-side ticket if needed).
- DO NOT: enable v_max DR. v_max is a measured firmware constant, asymmetric DR risks tipping the policy into saturation-stall (mas/037 Tip 6). Cfg field added for forward compatibility only; default `False`.
- DO NOT: introduce asymmetric `v_max⁺ / v_max⁻`. The fit's in/out asymmetry is 0.2% — well below the 10% acceptance from mas/037, so symmetric is correct (mas/037 fit table).
- DO NOT: filter or smooth the observation between samples. The 0.1-quantum is real; smoothing it teaches the policy to expect precision the lens cannot deliver (mas/037 Tip 2).
- DO NOT: change action / observation dimensions. Action[6] stays a single float ∈ [-1, 1]; obs zoom field stays a single float per agent. Policy networks built for the existing env load and run unchanged.

### Acceptance criteria

- **All four new test files pass.** Pure-PyTorch tests in `controller/tests/` run under the existing AppLauncher harness; the bench-replay test skips cleanly when the mas dataset is unreachable, runs the median-error gate when it is.
- **Bit-exact regression at `dead_time_curriculum_scale = 0`, `tau_zoom = 1e-4`** vs the pre-mas/037 first-order-only model on a 100-step rollout (allowing FP-noise tolerance of `1e-6`). Wired-but-off gate, mirrors the gimbal mas/036 regression test.
- **Bench replay**: median |level error| ≤ **0.15 levels** on `level_step_quantum`, `level_step_paired`, `rate_chirp`. (`rate_sine_saturated` informational only.)
- **No regression in the existing `controller/tests/run_tests.py` suite** — particularly `test_dynamics_curriculum_fix.py` which already exercises `randomize_gains` over `tau_zoom` / `max_zoom_rate`.
- **Single training run** (`Isaac-Iris-MA6-Direct-Test-v0`, default 2 agents, 200k env-steps, default config) with the new model active. Compare to a paired run with `dead_time_curriculum_scale = 0` (= old behavior except 0.1-quantum quantization, which IS active):
  - Mean episode return at 200k is **within ±10%** of the paired baseline (do-no-harm gate; new dead-time during the curriculum-active phase is expected to slightly slow target acquisition during the first ramp-in window, that's fine).
  - No NaN / Inf in any controller output across the rollout. Locked in by env's existing reward-NaN guards.
  - Zoom trace from a sample episode shows the expected 0.1-step staircase. Visual-inspection only; not gated.

### Risk

Low to medium.

1. **Action-scale / v_max disambiguation** — touches DR (`drone_controller.randomize_gains`) and the controller's internal field naming. Risk: DR run-time error if a stale codepath reads `_max_zoom_rate` expecting it to be the lens v_max. Mitigation: keep `_max_zoom_rate` as the action-scale field (existing semantics), add `_v_max` as the new lens-clip field. No rename.
2. **Dead-time buffer 1-D port** — copying ~80 lines of GimbalRateLoop's ring-buffer logic and dropping the per-axis dim is mechanical but easy to off-by-one. Mitigation: `test_zoom_curriculum_hooks.py` directly probes the cold-start, partial-reset, and bit-exact-at-scale-0 invariants — same shape as the gimbal's mas/036 regression tests.
3. **Quantization side-effects in obs / triangulation** — the env's `self.zoom_level[:, idx]` feeds `compute_z_eff`, which feeds intrinsic K, which feeds bbox raycaster. Risk: if a downstream consumer assumed continuous zoom (e.g. a bbox covariance term that takes a derivative), the 0.1-quantum step could surface as discontinuity-induced numerical pain. Mitigation: code search shows all consumers compose `zoom → z_eff → fx`, and `compute_z_eff` is well-defined at quantized inputs. Locked in by the bench-replay test indirectly (tests the published path end-to-end).

### Coupling

- **mas/035, mas/036** (gimbal rate-loop / dead-time): same architecture pattern, this ticket is a direct application. The dead-time curriculum knobs are independent (gimbal uses `gimbal_dead_time_*`, zoom uses `zoom_dead_time_*`).
- **mas/028** (camera intrinsic calibration): independent. The intrinsic curve `compute_z_eff` is upstream of the zoom dynamics; this ticket changes when the lens reaches a given level, not what focal-length multiplier that level induces.
- **mas/029** (sim2real measured-model impl): the zoom path was deferred there; this ticket is the deferred deliverable.
- **iris_ma6 Ticket 031** (policy triangulation head): independent. Aux supervision is on target position, not on the zoom path.
- **Curriculum cadence** ([curriculum_spec.md](../../../curriculum_spec.md)): adds `zoom_dead_time_*` to the dynamics ramp window. No phase reorder; mas/037 Tip 4 prefers `gimbal_dead_time_*`-aligned timing so the policy first learns the rate-loop lag, then the dead time.

### Affected files

**New**:
- NEW: [doc/zoom_response_spec.md](../../../zoom_response_spec.md) — calling contract, four-stage diagram, DR / curriculum hooks, published-vs-internal split.
- NEW: [controller/tests/test_zoom_response_model_stages.py](../../../../controller/tests/test_zoom_response_model_stages.py)
- NEW: [controller/tests/test_zoom_compare_to_bench.py](../../../../controller/tests/test_zoom_compare_to_bench.py) (skippable when dataset unreachable)
- NEW: [controller/tests/test_zoom_curriculum_hooks.py](../../../../controller/tests/test_zoom_curriculum_hooks.py)
- NEW: [controller/tests/test_zoom_published_vs_internal.py](../../../../controller/tests/test_zoom_published_vs_internal.py)
- NEW (optional, parallels [compare_gimbal.py](../../../../controller/sysid_output/gimbal/compare_gimbal.py)): [controller/sysid_output/zoom/compare_zoom.py](../../../../controller/sysid_output/zoom/compare_zoom.py).

**Edits**:
- EDIT: [controller/zoom_controller_cfg.py](../../../../controller/zoom_controller_cfg.py) — add `v_max_levels_per_s`, `quantum_levels`, `dead_time_*` fields. Update `tau_zoom` default to 0.091.
- EDIT: [controller/zoom_controller.py](../../../../controller/zoom_controller.py) — replace `compute_control` body with the four-stage pipeline. Add 1-D dead-time buffer (port of `GimbalRateLoop`). Add `zoom` (published) and `zoom_internal` properties. Add `set_dead_time_curriculum_scale`, `set_v_max`.
- EDIT: [controller/drone_controller.py](../../../../controller/drone_controller.py) — `_nominal_gains` adds `v_max`; existing `randomize_gains` path unchanged in semantics (still scales `_tau_zoom` and `_max_zoom_rate`). Doc-comment update only at [drone_controller.py:494-501](../../../../controller/drone_controller.py#L494-L501).
- EDIT: [curriculum/curriculum_cfg.py](../../../../curriculum/curriculum_cfg.py) — add `zoom_dead_time_start_step`, `zoom_dead_time_end_step`, `get_zoom_dead_time_progress`. Mirror gimbal block at [curriculum_cfg.py:266-285](../../../../curriculum/curriculum_cfg.py#L266-L285).
- EDIT: [iris_ma_env6_test.py](../../../../iris_ma_env6_test.py) — add `set_dead_time_curriculum_scale` call at the same site as the gimbal-dead-time hook ([iris_ma_env6_test.py:1429-1434](../../../../iris_ma_env6_test.py#L1429-L1434)).
- EDIT: [doc/active/feature_list.json](../../feature_list.json) — flip the `controller` feature's notes to mention the mas/037 zoom model integration (does not flip status; controller stays "done").
- EDIT: [doc/sim-to-real/modeling_checklist.md](../../../sim-to-real/modeling_checklist.md) — mark zoom-response item complete (mas/037 + this ticket).

### References

- [mas/037 ticket](file:///home/usrg/mas/src/doc/active/tickets/037-zoom-response-characterization-DONE/ticket.md) — characterization data, fitted parameters, deployment behavior tips.
- [zoom_model.json](file:///home/usrg/mas/src/scripts/sim2real_model_fitting/output/zoom_model.json) — canonical fit (`τ_d`, `v_max`, `τ₁`, `quantum`, provenance).
- [/home/usrg/mas/datasets/zoom_response/](file:///home/usrg/mas/datasets/zoom_response/) — four bench runs for `compare_zoom.py` replay.
- [controller/gimbal_rate_loop.py](../../../../controller/gimbal_rate_loop.py) — mas/035 + mas/036 reference impl; the zoom path is the 1-D analog.
- [controller/gimbal_rate_loop_cfg.py](../../../../controller/gimbal_rate_loop_cfg.py) — pattern for cfg field naming and curriculum-scale defaults.
- [controller/zoom_controller.py](../../../../controller/zoom_controller.py) — current first-order-only controller, replaced in this ticket.
- [iris_ma_env6_test.py:1423-1434](../../../../iris_ma_env6_test.py#L1423-L1434) — env-side curriculum-hook pattern (gimbal); zoom hook lands here.
- [iris_ma_env6_test.py:2253-2275](../../../../iris_ma_env6_test.py#L2253-L2275) — reset path that already writes per-env tau_zoom and `set_zoom`; carries over.

**Flow**: Low-medium. Three load-bearing pieces: (1) action-scale / v_max disambiguation in cfg + DR, (2) 1-D port of the dead-time ring buffer, (3) published-vs-internal accessor split. All others are plumbing. Estimated 2 commits, with vertical slices: (a) cfg + controller + stage tests; (b) curriculum + env hook + replay test; full training validation as a separate pass.
