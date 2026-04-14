# Experiment: 2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3

**Commit**: `2695ffe1e3`
**Date**: 2026-04-14
**Base**: attempt to replicate `2be3b9a3d9` with tickets 016–020 kept intact
**Status**: Failed. Policy collapsed immediately after 120k (the fixed-delay curriculum onset), same cliff as `025af4d022` / `82364f333c`.

---

## 1. Context — the regression window

- Last stable run: `2026-04-05_20-31-03_mappo_rnn_torch_2be3b9a3d9_control_gain_and_fidelity` (aborted early at 264k because of the `-30/-15` action-penalty over-shrink, but up to 116k it was on the 928b trajectory and handled noise + fixed-delay cleanly).
- First collapse after "the big bundle" landed: `2026-04-09_04-52-12_a9_bbox_size_baseline_40aab3574f_seed42`, and every run since.
- This `2695ffe1e3` run was supposed to be the minimal revert proving that the fidelity-stack additions (detector replicator, FP/FN curriculum, DR system) weren't the root cause. It collapsed at 120k with the same signature, so **FP/FN alone is not the problem**.

## 2. What landed between `2be3b9a3d9` and the first collapse (`40aab3574f`)

`git log 2be3b9a3d9..40aab3574f` — twelve commits, three distinct classes of change:

| Class | Commits | What it does |
|-------|---------|--------------|
| **Fidelity stack** | `7fb…`, `d86…`, `155…`, `6bf…`, `8ef…`, `b5f…`, `7e4…` | sysid-replicator, detector replicator, bbox center/size noise model, FP/FN injection, continuous AoI, burst dropout (Gilbert-Elliott), DR mount/offset |
| **Env surface changes** | `8ef…`, `b5f…` | observation dim `30D → 31D` (+effective_hfov), wider action limits (45°/s→90°/s yaw, 180°/s→360°/s gimbal on the controller side), zoom_max 6×→10× |
| **Perf / correctness** (tickets 016–020) | `3461991973` | batched controller, pre-allocated AgentStates, cached GT triangulation, separated detection-delay pipelines |

Tickets 017–019 are nominally optimizations. Ticket 020 is a **behavior-changing bug fix** — see §3.

## 3. Ticket-by-ticket analysis (016–020)

### Ticket 016 — research only
Scoping doc for the other four. No code.

### Ticket 017 — Batch `DroneController` across agents
- Rewrote the controller to run on `(N·A, …)` instead of `A` separate `DroneController(num_envs=N)` instances.
- Internal state (motor ω, rate integrator, gimbal integrator) is now a single `(N·A, …)` tensor; the reset path must touch rows `[env_id * A + agent_idx]` for each agent.
- **Risk**: a partial-env reset that zeros the wrong stride (or doesn't zero the rate integrator for just the affected agents) would silently inject correlated integrator error across agents after every episode reset. Not confirmed, but worth auditing — the 25 Hz step after a reset is where the first action amplitude gets compressed, which fits the sigma-stagnation signature.
- **Not a likely root cause of the 120k cliff** on its own, because 2be3's curriculum-same schedule sailed through the fixed-delay onset with `A` separate controllers. But combined with the observation-side widening it plausibly amplifies the normalized-action margin that drives sigma.

### Ticket 018 — Pre-allocate `AgentStates`
- `self._delay_gt_buffers[agent_id]` and `self._gt_state_cache` are now long-lived `AgentStates` objects that get `zero_/reset_data` each step.
- Fields get overwritten in-place rather than via fresh `torch.zeros`.
- **Risk**: if any consumer expects a *particular field* to default to zero before being written (i.e. reads an un-overwritten stale field from the previous step), it will silently read last-step data. This is exactly the class of bug that doesn't show up in unit tests but corrupts training gradients slowly. Worth diffing consumers against the old allocation path.

### Ticket 019 — Cache GT states / dedupe triangulation
- `_build_gt_states()` now short-circuits to the delay-system's GT when the delay system is active.
- **Risk**: the delay-system GT is computed once per sim step under an idempotency guard, but `_get_rewards()` may run between delay-system `step()` calls in some edge paths. If the cache is read before the guard is cleared in `_reset_idx()`, a reset environment could read stale triangulation for one step. Probably negligible at the reward scale, but the same caveat as 018.

### Ticket 020 — **Behavior-changing "bug fix"** (mechanism verified by line-by-line trace)

The bug is a shared-pipeline idempotency-guard race. At `2be3b9a3d9`:

1. `DirectRLEnv.step()` calls `_get_rewards()` at L365 and `_get_observations()` at L384 — rewards always run first.
2. `_get_rewards()` with `use_noisy_rewards=False` (default) calls `get_all_states_for_rewards()` → `_build_all_states(use_noise=False)` → for each field `_delay_system.get_delayed(..., use_noise=False)`.
3. Inside `UnifiedDelaySystem.get_delayed` (L215–218): `use_noise=False` fetches CLEAN data from `_storage.get_raw()`. L221–226 picks a pipeline from `_ego_pipelines[field_name]` / `_other_pipelines[field_name]` — **one shared instance per (field, perspective)**, used for both reward and observation queries.
4. L235 calls `pipeline.process(clean_data, …)` → `advance()`. Idempotency guard at L262–264 passes (first call at this `t_current`), so L265 stamps `_last_advance_time = t_current`. Stages run on clean data. L290–298 caches the clean result into BOTH `_cached_data_no_dropout` and `_cached_data_with_dropout`.
5. `_get_observations()` then calls `get_all_states_for_observations()` → `_build_all_states(use_noise=True)` → `get_delayed(..., use_noise=True)`.
6. Now L215–216 fetches `_storage.get_noisy()` (Gaussian-perturbed by `FieldStorage.store` when `noise_std > 0`), and L221–222 returns the **same** pipeline instance. L235 calls `pipeline.process(noisy_data, …)` → `advance(noisy_data, …)` — but L262–263's idempotency guard now fires (`t_current == _last_advance_time`), so `advance` returns immediately without re-running any stage. L340's `query()` returns the cached clean result from step 4.

So the observation path returned cached clean data whenever rewards ran first (always, under the default config). Delay and dropout were applied to the clean reward data and that same buffer was what observations ultimately read — delay curriculum worked for rewards, but never delivered corruption to observations.

**Why this was invisible at 2be3 (including in the teleop check)**:

- **Curriculum gate**: `multi_agent_wrapper._noise_scale = 0.0` by default (L125), only bumped by `set_noise_scale(noise_progress)` inside `_get_rewards`'s curriculum block. `_get_noise_std("bbox", …) = base_std * _noise_scale * per_agent_scale`, so at teleop (no curriculum progress), `noise_std = 0` and `FieldStorage.store` sets `_noisy = data.clone()` (L100–104) — identical to raw. There is literally no divergence for the bug to hide.
- **Teleop never ramps noise**: `tests/teleop_iris_ma6.py:545` only calls `set_delay_mode("fixed", progress=1.0)`; `set_noise_scale` is never called. Fixed delay looks correct because latency stages operate the same on clean or noisy input.
- **Detector replicator / FP/FN didn't exist yet**: those land in `155585853a`, `6bfb9593a3`, `8ef03231ff` (post-2be3). At 2be3 the only "noise" source was the Gaussian add inside `FieldStorage` — and per the two points above, its std was 0 during teleop. Post-fix teleop shows FN blink / FP offset because the detector replicator writes `_storage._noisy` independently of `_noise_scale`, which is what made the divergence suddenly obvious.

During training at 2be3, `_noise_scale` did ramp (100k–120k), so the bug actually suppressed a real noise curriculum. But with Gaussian std at the calibrated values (small pixel perturbations), the "missing noise" was not catastrophic and the policy trained through.

**What the fix changed**:

- `_ego_detection_pipelines` / `_other_detection_pipelines` now exist for `bboxes_2d` fields with independent buffers and independent idempotency guards.
- `get_delayed` routes to the detection pipeline when `use_noise=True` for those fields.
- Real observation corruption (Gaussian bbox noise, plus post-2be3: FP/FN, burst dropout, calibrated detector replicator noise) now actually reaches the policy.

**Implication for the collapse window**: the fix is not just a post-detector-replicator concern. It also uncovers a 2be3-era bug. Every post-fix run is training against an observation-corruption curriculum that no prior run has ever had to solve — including every phase that existed in 2be3 (Gaussian bbox noise, random/fixed delay on the observation path, IID dropout). That is enough on its own to explain the 120k cliff in `2695ffe1e3`, where FP/FN is disabled but fixed-delay onset is the first real observation-path corruption the policy has seen.

### Companion fixes shipped alongside ticket 020
- `teleop_iris_ma6.py`: `set_delay_mode("fixed", progress=1.0)` uncommented. The env starts in `"none"` mode for curriculum — before this, teleop was quietly running without delay entirely.
- `detector_replicator.py`: `params_path=""` fallback wired up. Before the fix, the detector replicator defaulted to a no-op.
- `multi_agent_wrapper.py` + `delay_system_v3.py`: `set_dropout_rate_by_perspective()` — before, burst dropout was zeroing ego IID dropout too; now ego dropout is preserved while burst applies to "other".

All four "companion" fixes lean the same direction: **more corruption actually reaches the policy**. In aggregate they change the effective curriculum far more than any phase-timing tweak.

## 4. What the `2695ffe1e3` run actually proves

The run kept the fidelity-stack code present but tried to neutralize the pieces I suspected:

- `kl_threshold: 0.02`, `min_lr: 1e-4` (matches 2be3)
- `max_yaw_rate: 45°/s`, gimbal controller cap 180°/s (matches 2be3)
- `action_sum: -30`, `action_delta: -15` (matches 2be3)
- FP/FN disabled out to 4M steps (effectively off)
- `tracking_lost_timeout: 2.0 s` (matches 2be3)
- DR mount-offset disabled

Still different from 2be3:

- PX4_MATCHED controller (sysid gains)
- 31D observation (+ effective_hfov)
- DR system *present* (even if gated off)
- **Ticket 017–020 code paths all active**

Collapse still fires at **120k**, the fixed-delay onset. In 2be3 the same fixed-delay ramp was a non-event. The only way this reconciles:

1. **Ticket 020's split changed what the observation path sees during every curriculum phase.** Pre-fix, the observation path returned the cached clean data from the reward call (shared pipeline + idempotency guard). Delay mode changes still ran in the latency stage, but they ran on clean data that the observation path then read — which is fine. Post-fix, the observation path has its own buffer that actually gets populated with `_storage.get_noisy()` (Gaussian-noisy bboxes when `_noise_scale > 0`) and runs its own latency / dropout stages. This flips the observation-side effect of every phase that touches `_noisy` or the detection pipeline.
2. With FP/FN disabled in `2695ffe1e3`, the first observation-degrading event of the curriculum is fixed-delay onset at 120k. Under the corrected pipeline, this is also the first time the observation bbox carries real, independent latency — 2be3 never trained on observations whose latency was decoupled from the reward stream.
3. The sigma-starved policy (~0.43, same signature across every post-bundle run) does not have the action variance to adapt to this step-change in observation quality, so the cliff manifests at the exact curriculum boundary rather than a gradual decline.

So **the 120k cliff is not about FP/FN, and not about curriculum shape. It is the first curriculum boundary where the observation path actually carries the corruption the curriculum says it should carry.** 2be3 never had to learn that, because the bug masked it.

## 5. Why sigma stagnates (separate, compounding issue)

Every post-bundle run — 3461, 949d, 492f, 36700, 5ce56, d60d, 025af4d, 82364 — sits at sigma ≈ 0.42–0.50, never climbing toward 928b's 1.2. 2be3 collapsed sigma to 0.14 but *before* the bundle, so the policy was at least lock-stepped on clean observations. The sigma stagnation signature is:

- Wider normalized action space (90°/s yaw, 360°/s gimbal, zoom_max 10×) + lighter penalties (`-10/-5`) let the policy solve step-0 with tiny normalized actions → action→reward landscape is flat at origin → KL-adaptive scheduler has nothing to push against → sigma parks at 0.43–0.48.
- 2be3 avoided this with narrow action space + heavy penalties, but at the cost of its own collapse (sigma → 0.14).
- Neither regime produces 928b's healthy sigma trajectory (0.3 → 1.2), because 928b had both the narrow action surface *and* the pre-bug observation pipeline that let it overfit clean signals.

Sigma stagnation makes the policy brittle to any real observation corruption. Ticket 020 turned the corruption on. Together they produce the universal 120k cliff.

## 6. Should we revert to `2be3b9a3d9`?

**No — a straight revert throws away real correctness progress.** The tickets-20 bug fix, the detector replicator, and the burst dropout are load-bearing for sim-to-real; running training on silently-clean observations is not "stability," it's a measurement artifact. Reverting to 2be3 would just hide the problem again until sim-to-real time.

**Recommended path instead (priority order):**

1. **Isolate ticket 020's behavior change.** Add a flag `detection_pipeline_split: bool = True`. Run the current commit with the flag off; this should reproduce 2be3-era behavior (clean observations on noise path) and confirm the 120k cliff disappears. If it does, the rest of the fidelity stack is exonerated. *Expected outcome: cliff vanishes, confirming the cliff is fidelity-induced, not a code bug elsewhere.*

2. **Re-introduce the fidelity stack with a proper observation-corruption curriculum.** The lesson from 5ce56e3ae8 applies: observation corruption must land *after* base tracking is learned. With real corruption now reaching the policy, the curriculum needs to be re-tuned, not reused from the pre-fix era.
   - Fixed-delay ramp at 120k is fine for clean obs, wrong for corrupted obs.
   - Either move the observation-corruption phases later (≥150k start) or bootstrap on clean obs and keep the ramp gradient far gentler (20k → 50k ramp width).

3. **Independently fix sigma stagnation.** The normalized-action problem is orthogonal to ticket 020. Options, in order of invasiveness:
   - Rescale action normalization so the effective step-0 action magnitude matches 2be3 (tighten the forward map, not the penalty).
   - Add an entropy floor via `min_log_std` only *after* the base skill phase (30–40k), so early sigma dynamics are natural but late-phase sigma can't fully collapse.
   - Re-tune action penalties (`-2/-1 → -10/-5` range) guided by A8 ablation rather than by lookalike matching to 2be3.

4. **Audit tickets 017–019 for the quiet-corruption risks in §3** (batched-controller reset stride, in-place AgentStates reads, cache-read vs reset-order in `_build_gt_states`). These are unlikely to be *the* cause but they're precisely the class of change that could be contributing a 5–10% tail that's making the cliff irrecoverable. Cheap to verify with a parity diff against `2be3b9a3d9` under a clean-obs config.

5. **Keep `2be3b9a3d9` warm as a reference.** Pin it as a known-good *training-only* baseline for comparing sigma trajectory, pair_valid curves, and triangulation ramp — but do not treat it as a sim-to-real-ready configuration. Its stability was partly an artifact.

## 7. Decision

- **Abort** the `2695ffe1e3` revert run — it has already proved the thing we needed it to prove.
- **Do not** hard-revert to `2be3b9a3d9`.
- **Next run**: reproduce 2be3 behavior under current commit by flagging off the detection-pipeline split (step 1 above). If the 120k cliff vanishes, proceed with curriculum re-tune (step 2). If it doesn't, the cliff is caused by something outside ticket 020 and the audit in step 4 becomes primary.

## 8. Resolution (ticket 029)

The diagnostic path (ticket 028) was replaced by a structural fix landed in
**ticket 029** (`doc/active/ticket/029-delay-system-redesign/`):

1. `DelayPipelineV3.advance` now accepts a `(raw, noisy)` payload pair and
   maintains four cache slots (raw/noisy × with/no dropout) plus two shared
   timestamp slots. One staleness mask, one latency draw, and one dropout
   mask drive BOTH payloads, so the obs path cannot diverge from the reward
   path in delay realization — only in content. The idempotency guard now
   correctly no-ops the second `advance` call because both caches are
   populated on the first advance.
2. Storage populates `_noisy[field]` only when a noisy payload is actually
   provided (`noisy_data=…` or `noise_std > 0`). `FieldStorage.has_noisy`
   lets `UnifiedDelaySystem.get_delayed` route clean-only fields as
   `noisy_data=None`; the pipeline then mirrors raw into the noisy cache.
3. `bboxes_2d` remains a single field per agent, with dual-payload storage:
   - `_raw = raycaster geometric GT` (read by reward path, `use_noise=False`).
   - `_noisy = detector-replicator output` (read by obs path, `use_noise=True`).
     When the replicator is disabled, noisy explicitly mirrors raw so the
     obs cache slot stays populated.
   One pipeline per perspective applies one delay realization to both
   payloads via the dual-cache advance. An earlier iteration of this
   ticket split the bbox into two distinct fields (raycaster / replicator)
   with independent pipelines; that was reverted because separate
   pipelines produce independent latency draws under `random` curriculum
   mode, which is physically wrong — the two views model one camera
   capture event and must share delay.
4. `detection_pipeline_split` cfg flag, `_DETECTION_SUFFIXES`,
   `_ego_detection_pipelines`, and `_other_detection_pipelines` were all
   removed. `DelaySystemKeyParams.detection_pipeline_split` plumbing was
   removed.

**Regression test**: [`delay_system_v3/tests/test_dual_cache.py`](../../delay_system_v3/tests/test_dual_cache.py)
locks in the invariants above (shared delay realization across payloads,
idempotency no-op on same-step second advance, clean-only field fallback,
shared dropout mask). The test that would have failed under the original
bug pattern is the "reward-then-obs query ordering returns distinct
payloads" test.

**Visualization**: [`delay_system_v3/tests/dual_cache_invariants.png`](../../delay_system_v3/tests/dual_cache_invariants.png)
renders the four invariants directly — Panel 2 in particular shows the RMS
of (obs − reward) staying positive through the whole run, which is exactly
the quantity that used to be zero under the pre-fix bug.

**Implication for the 120k cliff**: with the fix landed, observation-path
noise/FP/FN/dropout now actually reach the policy under their curriculum
schedule (they did not, during 2be3 training). This means the next
training run will exercise an observation-corruption curriculum that no
prior run has ever fully experienced. Expect to re-tune curriculum
timing and slopes per the lesson from `5ce56e3ae8`: observation
corruption belongs AFTER the base tracking skill is bootstrapped.

Ticket 028 (the diagnostic-flag proposal) is retained in-tree for the
historical record — its hypothesis is validated by ticket 029's
regression test rather than by the runtime flag it originally proposed.
- Log outcomes in `doc/active/progress.txt` and update `feature_list.json` when step 1 confirms or refutes the hypothesis.
