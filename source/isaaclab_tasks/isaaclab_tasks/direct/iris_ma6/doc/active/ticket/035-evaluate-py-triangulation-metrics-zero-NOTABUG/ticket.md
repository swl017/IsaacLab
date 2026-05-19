## Ticket 035 — `evaluate.py` reports zero triangulation metrics for all checkpoints

**Status**: **Closed — NOT A BUG (2026-05-17).**
`enable_triangulation = False` is the intentional deploy-time setting: the
triangulation solver is not part of the deployable observation pipeline.
Zero-valued triangulation metrics in `evaluate.py` output are expected under
this design. Downstream consumers (including ticket 034 validation) should
use `visibility_mean` as the task-quality witness. No fix planned.
Ticket retained as documentation of the design choice.

**Created**: 2026-05-17
**Discovered in**: Stage R0 measurement for ticket 034 (regression eval of fix2's 400k checkpoint at 39k/99k/199k/399k env settings).
**Target files**:
- [scripts/`evaluate.py`](../../../../experiments/evaluate.py) — `_collect_step_metrics`
- [`iris_ma_env6_test.py`](../../../../iris_ma_env6_test.py) — `_reset_idx`, `_get_observations`
- [`iris_ma_env6_test_cfg.py`](../../../../iris_ma_env6_test_cfg.py) — `enable_triangulation` default
- [`experiments/experiment_registry.py`](../../../../experiments/experiment_registry.py) — `a1_with_aoi` and siblings

---

### What

Running `experiments/evaluate.py --experiment a1_with_aoi --checkpoint <any> --step <any>` returns a JSON in which **every triangulation-derived metric is exactly 0.0**, including for checkpoints that are known to triangulate well at the env settings being evaluated:

- `triangulation_rmse_mean = 0.0`
- `triangulation_rmse_std = 0.0`
- `trace_sigma_mean = 0.0`
- `trace_sigma_std = 0.0`
- `task_success_rate = 0.0`
- `track_maintenance_rate = 0.0`
- `accuracy_rate = 0.0`
- `tri_valid_ratio_mean = 0.0`

The non-triangulation metrics (`visibility_mean`, `collision_rate_mean`, `cbf_violation_rate_mean`, `min_separation_mean`, `max_track_gap_mean`) populate correctly.

Reproduced in all 7 cells of ticket 034's regression eval (see [`r_research/regression_eval.md`](../034-curriculum-catastrophic-forgetting/r_research/regression_eval.md) §3). Reference checkpoints (`agent_40000.pt`, `agent_80000.pt`, `agent_200000.pt`) all show `tri_valid_ratio = 0` even at their own training-step env settings, where they are known to score `triangulation` reward components of ~24 during training.

### Why this matters

`evaluate.py` is the canonical evaluation harness for iris_ma6 paper metrics and for all gated eval suites. The triangulation metrics it nulls out are exactly the metrics the paper's Level-1 task definition is built on:

- `task_success_rate` (defined as RMSE ≤ threshold AND trace ≤ threshold)
- `track_maintenance_rate` (defined as `tri_valid_ratio ≥ 0.5`)
- `accuracy_rate` (defined as RMSE ≤ threshold)

All three are derived from triangulation outputs. As shipped, **every evaluation reports 0 % task success and 0 % track maintenance**, regardless of policy quality. This affects:

- Ticket 034 — the regression-validation visibility-based proxy works as a witness, but the cleaner triangulation-based 80 % bar cannot be computed without first fixing this.
- Any cross-checkpoint or cross-experiment comparison in the existing eval suite.
- Any paper-table generation downstream that calls into `evaluate.py`.

### Root cause (two compounding defects)

**Defect 1 (primary):** [`_collect_step_metrics`](../../../../experiments/evaluate.py#L186-L212) reads `env._triangulation_result_obs` for `tri_valid` gating ([line 210](../../../../experiments/evaluate.py#L210)) and never reads `env._triangulation_result_gt` for the same purpose:

```python
tri_obs = env._triangulation_result_obs
...
if tri_obs is not None:
    tri_pos = tri_obs.position[:, 0, :]
    tri_valid_obs = tri_obs.is_valid[:, 0]
else:
    tri_pos = torch.zeros(env.num_envs, 3, device=env.device)
    tri_valid_obs = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
...
tri_valid = tri_valid_obs                       # used by MetricTracker.step
tri_valid_with_cov = tri_valid_obs & tri_valid_gt
```

But `_triangulation_result_obs` is **only populated when `cfg.enable_triangulation == True`**, gated at [`iris_ma_env6_test.py:2035`](../../../../iris_ma_env6_test.py#L2035):

```python
if self.cfg.enable_triangulation:
    ...
    self._triangulation_result_obs = self._compute_triangulation(...)
```

And `enable_triangulation` defaults to **`False`** ([`iris_ma_env6_test_cfg.py:468`](../../../../iris_ma_env6_test_cfg.py#L468)), and the A1 experiments — including the default `a1_with_aoi` — do not override it ([`experiment_registry.py:119-124`](../../../../experiments/experiment_registry.py#L119-L124)).

So under the as-shipped config: `_triangulation_result_obs is None` for every step of every eval episode → `tri_valid_obs` is all-False → `tracker._trace_valid_count` never increments → all triangulation metrics zero out.

**Defect 2 (compounding):** Even when `enable_triangulation=True` is set, [`_reset_idx`](../../../../iris_ma_env6_test.py#L2579-L2581) clears both triangulation results to `None` unconditionally each reset:

```python
# Clear triangulation results (will be recomputed on first step)
self._triangulation_result_gt = None
self._triangulation_result_obs = None
```

For `_triangulation_result_gt` this happens **after** `_get_rewards` (which populated it at [line 1496](../../../../iris_ma_env6_test.py#L1496)) and **before** the next step's `_get_rewards`. With 1024 envs and ~500-step episodes, an env terminates/truncates almost every step (~1024/500 ≈ 2 envs per step on average), so `_triangulation_result_gt` is `None` at the time `_collect_step_metrics` runs most of the time.

For `_triangulation_result_obs`, it's repopulated in `_get_observations` after reset, so it survives — but only when `enable_triangulation=True`.

This second defect doesn't trigger the all-zeros symptom by itself, but it is the reason `_collect_step_metrics`'s fallback for `tri_gt is None` (a no-op that masks scene-validity) ever fires.

### Mathematical proof that GT triangulation works

The GT triangulation path runs **unconditionally** in `_get_rewards`:

```python
# iris_ma_env6_test.py:1495-1498
# Level 1 (FIM): covariance at GT target position
self._triangulation_result_gt = self._compute_triangulation(
    states=reward_states, use_gt_target=True
)
```

This is what the training-time `triangulation` reward component reads from. fix2 reached `triangulation` reward of 24.6 at 380k (per the 2026-05-15 experiment doc §3.2), so the GT triangulation pipeline is correct — only the eval-side metric gating is wrong.

### Reproduction

```bash
source /home/usrg/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/evaluate.py \
    --experiment a1_with_aoi \
    --task Isaac-Iris-MA6-Direct-Test-v0 \
    --checkpoint <any ckpt from a known-good training run> \
    --step 100000 \
    --num_envs 1024 --num_episodes 1 \
    --no-record-trajectory --no-record-action-trace --no-timeseries \
    --headless --output /tmp/eval_repro.json
python3 -c "import json; d=json.load(open('/tmp/eval_repro.json')); \
  print('tri_valid_ratio_mean:', d['tri_valid_ratio_mean']); \
  print('task_success_rate   :', d['task_success_rate']); \
  print('visibility_mean     :', d['visibility_mean'])"
```

Expected (current bug): `tri_valid_ratio_mean = 0.0`, `task_success_rate = 0.0`, `visibility_mean > 0.5`.

### Desired behavior

After the fix, every eval cell should produce non-zero triangulation metrics consistent with the checkpoint's training-time `triangulation` reward component. Concretely:

- For `agent_400000.pt` from the fix2 run evaluated at `--step 399000`: `tri_valid_ratio_mean` should be > 0.5 (corresponds to the training-time `pair_valid_rate ≈ 0.79`).
- For `agent_40000.pt` evaluated at `--step 39000`: `tri_valid_ratio_mean` should be > 0.5 as well (early-phase pair_valid_rate plateaus around 0.85 per the fix2 doc §3.2).
- For the ticket-034 regression cells, the 400k-ckpt vs reference-ckpt ratio on `tri_valid_ratio_mean` should match the pattern seen in `visibility_mean` (≪ 0.80 at low eval steps, ≈ 1.0 at 399k).

### Fix options

The fix has two reasonable shapes; either or both can apply.

**Option A (eval-side, minimal change):** in `_collect_step_metrics`, switch the validity source from `_triangulation_result_obs` to `_triangulation_result_gt` for the `tri_valid` gate, since GT triangulation always runs. Continue using `_triangulation_result_obs` for the *position* estimate (RMSE) **only when it's not None**; fall back to `_triangulation_result_gt.position` otherwise. This change is ~10 LOC in [`evaluate.py:186-212`](../../../../experiments/evaluate.py#L186-L212).

Trade-off: when `enable_triangulation=False`, `tri_pos` falls back to GT-derived position, which means RMSE measured against the GT target collapses to whatever bias the GT triangulator has — not the obs-path RMSE the policy is trying to minimize. Acceptable for the validity/visibility/task-success metrics; not acceptable for `triangulation_rmse_mean` headline.

**Option B (config-side, more accurate but heavier):** in `evaluate.py:main`, force `env_cfg.enable_triangulation = True` (or set it on each affected experiment in the registry). This makes the eval mirror what the policy *would* see if its observation pipeline included triangulation, and produces the obs-path RMSE directly. Trade-off: changes the observation shape the env produces — but `evaluate.py` is loading checkpoints, not training them, so the observation-shape change is consumed only by the policy's forward pass. **The policy's observation shape must still match its training-time observation shape**, so forcing `enable_triangulation=True` at eval time only works if the *training* run also had `enable_triangulation=True`. For the fix2 run this needs to be verified before applying Option B.

**Option C (env-side, defect 2 only):** drop the unconditional `_triangulation_result_gt = None` / `_triangulation_result_obs = None` lines in [`_reset_idx`](../../../../iris_ma_env6_test.py#L2579-L2581). They are described as "will be recomputed on first step" but they're computed in `_get_rewards`/`_get_observations` of the *current* step regardless of reset, so the clear is a no-op for envs that don't reset and a footgun for envs that do. Removing these two lines closes Defect 2 cleanly. Independent of A/B.

Recommendation: **A + C**, both are local, both are reviewable in one diff, and together they produce non-zero triangulation metrics for all existing experiments without changing observation shapes. Option B is preferred if training-time `enable_triangulation` is confirmed True (verifying this is part of the fix).

### Scope boundary

- Do not change `MetricTracker` semantics. The `task_success_rate`, `track_maintenance_rate`, `accuracy_rate` thresholds and definitions stay as is.
- Do not change reward computation. `_triangulation_result_gt` and its consumers in `_get_rewards` are correct and must not be touched.
- Do not change the registered experiment definitions (`a1_with_aoi`, etc.) unless Option B is chosen, in which case set `enable_triangulation` per experiment and verify training-time observation shape consistency.

### Acceptance criteria

1. Reproduction command above produces `tri_valid_ratio_mean > 0.5` and `task_success_rate > 0` for `agent_400000.pt` from the fix2 run at `--step 399000`.
2. Ticket 034's 7-cell eval suite re-runs cleanly; the visibility-based regression ratio table is reproduced (sanity check) and a new triangulation-based table is added with the same monotonic-recovery pattern.
3. No regression in the non-triangulation metrics (`visibility_mean`, `collision_rate_mean`, etc.).

### Related

- Ticket 034 — uses `visibility_mean` as the regression witness specifically because this defect makes the cleaner triangulation-based metrics unusable. Resolving 035 unblocks the triangulation-side validation in ticket 034's later stages.
- `MetricTracker.compute_final_metrics()` ([metrics/metric_tracker.py:280+](../../../../experiments/metrics/metric_tracker.py#L280)) — output schema is correct; the input it receives via `tracker.step(tri_valid=...)` is what's broken.
- `evaluate.py:178` "Uses the observation-path triangulation for position estimate (midpoint method) and the GT-path triangulation for covariance trace." — this comment documents the intended design; the implementation in lines 186-212 silently degrades to all-zeros when the obs path is disabled instead of falling back to the GT path.
