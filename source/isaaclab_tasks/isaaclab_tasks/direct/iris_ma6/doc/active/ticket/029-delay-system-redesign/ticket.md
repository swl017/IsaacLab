# Ticket 029 — Delay system redesign: bbox field-split + dual-cache for state fields

**Status**: Open. Replaces ticket 028 (the diagnostic flag is no longer needed once this lands).
**Parent**: post-mortem [doc/experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md](../../../experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md)
**Owner**: next session
**Time estimate**: 1.5–2 days for code + 0.5 day for tests + 1 training run for verification.

---

## 0. TL;DR for the next session

1. The delay system has a long-standing bug: when reward and observation calls share a `DelayPipelineV3` instance for the same field, the second call hits an idempotency guard and returns the first call's cached payload. Reward call always runs first (`_get_rewards` at `DirectRLEnv.step` L365 before `_get_observations` at L384). So the observation path silently returns clean reward-path data; storage-side noise is discarded.
2. Ticket 020's `detection_pipeline_split=True` (current default) works around this *for bbox only* by giving `bboxes_2d` a second pipeline. It does not fix the underlying issue, leaves position/velocity Gaussian noise still discarded, and gives reward and observation independent latency draws (wrong — should be the same physical detection).
3. **Fix it properly**:
   - **Bbox**: split into two fields (`bboxes_2d_raycaster` from raycaster, `bboxes_2d_replicator` from detector replicator). Each gets its own pipeline. Eliminates the shared resource entirely. The two sources are physically distinct sensors (instant geometric raycast vs. detector inference) — independent pipelines and independent latency cfgs are correct.
   - **All other fields**: keep one shared pipeline per field, but refactor `DelayPipelineV3.advance` to accept a `(raw, noisy)` pair, run all stateful stages **once** with **shared** staleness/latency/dropout draws, and cache **both** delayed payloads. `query(use_noise=…)` returns the matching cache. This is the only design that supports future noise propagation through derived fields (`body_combined_angular_velocity_w`, `gimbal_azimuth_world`, etc.) — the user explicitly requires that derived noisy values be computed from noisy primitives BEFORE delay, which forbids "noise after delay."
4. Delete `detection_pipeline_split` flag, `_DETECTION_SUFFIXES`, `_ego_detection_pipelines`, `_other_detection_pipelines`, and `set_dropout_rate_by_perspective` machinery added in ticket 020.
5. After landing, run training. The 120k cliff (see post-mortem §3 trace) should be re-evaluated with the corrected curriculum semantics.

**Do not split this work into multiple commits per file.** The intermediate states are not runnable. Land it as one commit covering all the changes in §3, with the test suite updated in the same commit.

---

## 1. Background — read this before touching code

### 1.1 The bug, line by line

`DirectRLEnv.step()` order ([direct_rl_env.py:363-384](../../../../../../../isaaclab/isaaclab/envs/direct_rl_env.py#L363-L384)):
```python
L363:  self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
L365:  self.reward_buf = self._get_rewards()       # ← runs FIRST
L384:  self.obs_buf = self._get_observations()    # ← runs SECOND
```

`_get_rewards()` calls (with default `use_noisy_rewards=False`):
```python
# iris_ma_env6_test.py:1151-1158
reward_states = self._delay_system.get_all_states_for_rewards(...)
# → multi_agent_wrapper._build_all_states(use_noise=False)
# → for each field: get_delayed(field, perspective, use_noise=False, allow_dropout=False)
```

`_get_observations()` calls:
```python
# iris_ma_env6_test.py:1496-1500
delayed_states = self._delay_system.get_all_states_for_observations(...)
# → multi_agent_wrapper._build_all_states(use_noise=True)
# → for each field: get_delayed(field, perspective, use_noise=True, allow_dropout=True)
```

`UnifiedDelaySystem.get_delayed` ([delay_system_v3.py:246-311](../../../delay_system_v3/delay_system_v3.py#L246-L311)) routes both calls to **the same pipeline instance** (`_ego_pipelines[field_name]` / `_other_pipelines[field_name]`) for any field that is not in `_DETECTION_SUFFIXES` AND is not under `detection_pipeline_split=True`.

`DelayPipelineV3.advance` ([delay_pipeline_v3.py:236-298](../../../delay_system_v3/delay_pipeline_v3.py#L236-L298)):
```python
L266:  time_advanced = (t_current - self._last_advance_time).abs() > 1e-6
L267:  if not time_advanced.any(): return                              # ← guard
L269:  self._last_advance_time = t_current.clone()
# … stages run on `data` argument …
L294:  self._cached_data_no_dropout   = data.clone()
L298:  self._cached_data_with_dropout = data.clone()
```

Trace at sim step k:
- Reward call: `data = storage.get_raw(f)` (clean); `advance(clean)` → guard passes (first call), runs, caches **clean** result into both `_cached_*` slots.
- Obs call: `data = storage.get_noisy(f)` (Gaussian-perturbed or detector-replicator output); `advance(noisy)` → guard fires (`t_current == _last_advance_time`), returns immediately. `query()` returns cached **clean** data. The noisy payload is silently discarded.

Visualized in [tests/detection_pipeline_split.png](../../../delay_system_v3/tests/detection_pipeline_split.png) (right column, `split=False`): obs (red) is identical to reward (blue).

### 1.2 What ticket 020 did and why it is not the right fix

Ticket 020 added:
- `_ego_detection_pipelines: Dict[str, DelayPipelineV3]` and `_other_detection_pipelines` ([delay_system_v3.py:90-91](../../../delay_system_v3/delay_system_v3.py#L90-L91)).
- `_DETECTION_SUFFIXES = ("bboxes_2d",)` ([delay_system_v3.py:53](../../../delay_system_v3/delay_system_v3.py#L53)).
- A registration branch ([delay_system_v3.py:185-206](../../../delay_system_v3/delay_system_v3.py#L185-L206)) that creates a second pipeline for `bboxes_2d` fields when `detection_pipeline_split=True`.
- Routing in `get_delayed` ([delay_system_v3.py:285-294](../../../delay_system_v3/delay_system_v3.py#L285-L294)) that picks the detection pipeline for `use_noise=True` queries.
- `set_dropout_rate_by_perspective` ([multi_agent_wrapper.py](../../../delay_system_v3/multi_agent_wrapper.py)) so burst dropout (which only applies to "other" perspective) doesn't zero ego dropout.
- A `detection_pipeline_split: bool = True` flag on `UnifiedDelayCfgV3` and `MultiAgentDelaySystemV3Cfg`.

Problems with this approach:
1. **Bbox-only**: only fields in `_DETECTION_SUFFIXES` get the fix. Position/velocity Gaussian noise is still discarded — the curriculum's `_noise_scale` ramp is a no-op for those fields.
2. **Independent latency draws**: with two separate pipelines, the reward path and observation path each sample their own latency — but they're modeling the same physical detection event (one camera capture, one image, one bbox). Reward "saw" the detection at a different sim time than the policy "saw" it. Wrong.
3. **Independent dropout**: burst dropout applies to "other" perspective; the workaround `set_dropout_rate_by_perspective` exists only because the per-pipeline dropout mask diverged.
4. **Conceptual debt**: the `detection_pipeline_split` flag has no physical meaning — it's a dial for "how broken the simulation is." That's not a config; that's a bug.

### 1.3 The user's hard constraint on noise propagation

For state fields (position, velocity, derived quantities), the system is designed so that derived noisy fields can be computed as `f(noisy primitives)` rather than `f(clean primitives) + independent_gaussian`. The current code uses the latter (independent Gaussian per derived field at storage time) — see [multi_agent_wrapper.py:309-353](../../../delay_system_v3/multi_agent_wrapper.py#L309-L353) and `_add_noise_to_states` at L820+. This is a known approximation.

If we ever want to upgrade to proper propagation, the dataflow must be:
```
clean primitives ──┐
                   ├──▶ noisy primitives (sampled in env)
noise samples ─────┘            │
                                ├──▶ noisy derived (computed in env from noisy primitives)
                                ▼
              storage._noisy[field] = noisy primitive OR noisy derived
                                ▼
              pipeline.advance(raw, noisy, …) — both go through delay together
                                ▼
              query(use_noise=True) returns delayed noisy
```

Critically: noise is in the **storage layer** (the `_noisy[f]` payload), and the noisy payload **must reach the pipeline cache**. This rules out "noise after delay" (Option B from prior conversation) — that strategy can't reconstruct `f(noisy primitives)` from a delayed clean primitive.

It also rules out "two independent pipelines per state field" (Option C). Independent pipelines = independent latency/dropout/staleness draws. The sample at `t-Δ_reward` and `t-Δ_obs` would be different time slices of the same physical signal. We need shared delay realization with two payloads.

The only design that satisfies this is dual-cache shared-pipeline (this ticket).

---

## 2. Target design

### 2.1 Two regimes

| Field class | Examples | Pipeline structure | Storage |
|---|---|---|---|
| **Detection** (multiple physical sources) | `bboxes_2d_raycaster`, `bboxes_2d_replicator` | one pipeline per field — independent realizations | clean only per field; replicator writes `bboxes_2d_replicator` directly |
| **State** (one source, two views) | `body_position_w`, `body_linear_velocity_w`, `body_combined_angular_velocity_w`, `gimbal_azimuth_world`, all others | one pipeline per field — dual cache, shared draws | `_raw` clean + `_noisy` perturbed, both stamped same |

### 2.2 `DelayPipelineV3` — new advance signature

```python
def advance(
    self,
    raw_data: torch.Tensor,            # required
    noisy_data: torch.Tensor | None,   # optional — when None, behaves like clean-only
    timestamp: torch.Tensor,
    t_current: torch.Tensor,
    burst_dropout_mask: torch.Tensor | None = None,
) -> None:
    """Advance pipeline with shared draws applied to both payloads.

    Single advance per sim step. Idempotent across multiple `process()` calls
    at the same t_current. When noisy_data is None (e.g. for bbox _gt/_det fields
    with no separate noisy payload), only the raw path is processed and cached;
    the noisy cache mirrors raw so query(use_noise=True) is well-defined.
    """
```

Internal restructure:

- One `_last_advance_time` (unchanged).
- One staleness state `(_staleness_held_data, _staleness_held_timestamp)` becomes a pair: `(_staleness_held_raw, _staleness_held_noisy, _staleness_held_timestamp)`.
- One latency state: `_data_buffer` becomes `_raw_data_buffer` + `_noisy_data_buffer`. `_timestamp_buffer` unchanged. All three buffers indexed by the same `time_lags` from the single `_latency_sampler`. Append guard (`_last_append_time`) unchanged.
- Dropout state pair: `(_dropout_held_raw, _dropout_held_noisy, _dropout_held_timestamp)`. Single `drop_mask` sampled (or supplied via `burst_dropout_mask`) and applied to both payloads.
- FOL state pair: `_lag_output_raw`, `_lag_output_noisy`. Same `alpha`, applied independently.
- Cache slots: `_cached_raw_no_dropout`, `_cached_noisy_no_dropout`, `_cached_raw_with_dropout`, `_cached_noisy_with_dropout`, `_cached_ts_no_dropout`, `_cached_ts_with_dropout`.

Reads:
```python
def query(self, use_noise: bool, allow_dropout: bool) -> Tuple[Tensor, Tensor]:
    if use_noise:
        cache_data = self._cached_noisy_with_dropout if allow_dropout else self._cached_noisy_no_dropout
    else:
        cache_data = self._cached_raw_with_dropout   if allow_dropout else self._cached_raw_no_dropout
    cache_ts   = self._cached_ts_with_dropout       if allow_dropout else self._cached_ts_no_dropout
    return cache_data.clone(), cache_ts.clone()
```

`process()` becomes:
```python
def process(
    self, raw_data, noisy_data, timestamp, t_current,
    use_noise: bool, allow_dropout: bool,
    burst_dropout_mask: torch.Tensor | None = None,
) -> Tuple[Tensor, Tensor]:
    self.advance(raw_data, noisy_data, timestamp, t_current, burst_dropout_mask)
    return self.query(use_noise=use_noise, allow_dropout=allow_dropout)
```

### 2.3 `UnifiedDelaySystem.get_delayed` — read both from storage

```python
def get_delayed(self, field_name, perspective, use_noise, allow_dropout,
                burst_dropout_mask=None) -> Tuple[Tensor, Tensor]:
    if field_name not in self._field_shapes:
        raise KeyError(...)

    # Determine pipeline (one per field, no detection split)
    pipelines = self._ego_pipelines if perspective == "ego" else self._other_pipelines
    pipeline = pipelines[field_name]
    cfg = self._cfg.ego if perspective == "ego" else self._cfg.other

    # Read both payloads from storage. Storage guarantees same timestamp.
    raw_data, ts = self._storage.get_raw(field_name)
    if self._storage.has_noisy(field_name):
        noisy_data, _ = self._storage.get_noisy(field_name)
    else:
        noisy_data = None  # bbox _gt / _det fields — clean only, no separate noisy

    apply_dropout = allow_dropout and (
        (use_noise and cfg.dropout_for_observations)
        or (not use_noise and cfg.dropout_for_rewards)
    )
    effective_burst_mask = burst_dropout_mask if apply_dropout else None

    return pipeline.process(
        raw_data, noisy_data, ts, self._t_current,
        use_noise=use_noise, allow_dropout=apply_dropout,
        burst_dropout_mask=effective_burst_mask,
    )
```

Note that `get_aoi`, `set_delay_mode`, `set_dropout_rate`, `set_field_delay_mode`, `reset`, `step` should all simplify — they no longer need to iterate detection-pipeline dicts.

### 2.4 `FieldStorage` — minor extension

Add `has_noisy(field_name) -> bool` (returns `field_name in self._noisy`) so `get_delayed` can decide whether to pass `noisy_data`. This is a one-line addition.

For the bbox fields (`bboxes_2d_raycaster`, `bboxes_2d_replicator`), `store()` is called WITHOUT `noisy_data` and WITHOUT `noise_std > 0` — so the `_noisy[field_name]` entry is just `data.clone()`. To keep the "no noisy payload" semantic clean, modify `store()` to NOT populate `_noisy[field_name]` when `noisy_data is None and noise_std == 0`. Then `has_noisy(field_name)` returns False for those fields, and `get_delayed` knows to pass `noisy_data=None` to the pipeline.

```python
# field_storage.py:104-111 — replace
if noisy_data is not None:
    self._noisy[field_name] = noisy_data.to(self._device).clone()
elif noise_std > 0:
    noise = torch.randn_like(data) * noise_std
    self._noisy[field_name] = data + noise
# else: don't populate _noisy — has_noisy(field_name) returns False
```

Backward-compat watch: any call site that does `storage.get_noisy(f)` for a clean-only field will now KeyError. The only call site is `UnifiedDelaySystem.get_delayed`, which is being rewritten in this ticket to gate on `has_noisy`. Grep for other call sites (`grep -rn "get_noisy" delay_system_v3/`) and confirm.

### 2.5 `MultiAgentDelaySystemV3` (multi_agent_wrapper) — bbox split + remove split-aware code

Bbox `store()` currently ([multi_agent_wrapper.py:383-401](../../../delay_system_v3/multi_agent_wrapper.py#L383-L401)):
```python
if replicated_bboxes is not None:
    self._delay_system.store(
        prefix + "bboxes_2d", data.bboxes_2d,
        timestamp=ts_detection, noise_std=0.0,
        noisy_data=replicated_bboxes,
    )
else:
    bbox_noise = self._get_noise_std("bbox", agent_id)
    self._delay_system.store(
        prefix + "bboxes_2d", data.bboxes_2d,
        timestamp=ts_detection, noise_std=bbox_noise,
    )
```

Replaced with two unconditional store calls:
```python
# GT bbox — raycaster output, used by reward path
self._delay_system.store(
    prefix + "bboxes_2d_raycaster", data.bboxes_2d,
    timestamp=ts_detection,
)

# Detection bbox — replicator output if available, else fall back to GT
# (the fallback is what 2be3 effectively had: clean obs because no replicator)
det_payload = replicated_bboxes if replicated_bboxes is not None else data.bboxes_2d
self._delay_system.store(
    prefix + "bboxes_2d_replicator", det_payload,
    timestamp=ts_detection,
)
```

Field reads in `_build_all_states` ([multi_agent_wrapper.py:548-554](../../../delay_system_v3/multi_agent_wrapper.py#L548-L554)):
```python
# Currently:
bbox, _ = self._delay_system.get_delayed(
    prefix + "bboxes_2d", perspective, use_noise, allow_dropout
)
data.bboxes_2d = bbox

# Becomes:
bbox_field = prefix + ("bboxes_2d_replicator" if use_noise else "bboxes_2d_raycaster")
bbox, _ = self._delay_system.get_delayed(
    bbox_field, perspective, use_noise=False, allow_dropout=allow_dropout,
    burst_dropout_mask=burst_dropout_mask,
)
data.bboxes_2d = bbox
```

`use_noise=False` on the call is intentional: each bbox field has a single payload (clean, from its own source), so the dual-cache distinction does not apply here.

`set_dropout_rate_by_perspective`: keep the API for now but adjust internally. Burst dropout still applies only to "other" perspective. This needs auditing — see §3.5.

### 2.6 Pipeline configuration for bbox split

[delay_cfg_v3.py:611](../../../delay_system_v3/delay_cfg_v3.py#L611) currently has `detection_pipeline_split: bool = True`. Delete it.

The `field_overrides` dict in `PerspectiveCfg` lets you set per-suffix latency configs. Today there's likely one entry for `"bboxes_2d"`. After the rename, you want potentially-different configs for `"bboxes_2d_raycaster"` and `"bboxes_2d_replicator"`. Default behavior:
- `bboxes_2d_raycaster`: short or zero latency cfg (raycaster is instant — but we may model camera capture latency).
- `bboxes_2d_replicator`: existing detection latency cfg (models camera + YOLO inference).

Initial migration: copy whatever existing `bboxes_2d` override exists into both `bboxes_2d_raycaster` and `bboxes_2d_replicator` so behavior is preserved on day one. Tune later.

### 2.7 What gets deleted

After this ticket lands, the following code/fields should be removed:
- `UnifiedDelayCfgV3.detection_pipeline_split` field, MultiAgentDelaySystemV3Cfg `detection_pipeline_split`, all params plumbing for it.
- `UnifiedDelaySystem._DETECTION_SUFFIXES`.
- `UnifiedDelaySystem._ego_detection_pipelines`, `_other_detection_pipelines`.
- All `if self._detection_pipeline_split:` branches (the registration branch, the `_all_pipelines` helper variants, the bulk-op branches).
- The diagnostic `tests/plot_detection_pipeline_split.py` — replace its purpose with a regression test (see §4).
- Ticket 028 (delete `doc/active/ticket/028-detection-pipeline-split-flag/`) — the diagnostic it implements is no longer needed.

---

## 3. Step-by-step migration plan

Do the steps in this order. Each step should leave the code in a state where the test suite at least runs (even if some tests temporarily fail). Land everything in a single commit.

### Step 1 — `FieldStorage.has_noisy` + conditional `_noisy` population

File: [delay_system_v3/field_storage.py](../../../delay_system_v3/field_storage.py)

- Add `def has_noisy(self, field_name: str) -> bool: return field_name in self._noisy`.
- Update `store()` body so `_noisy[field_name]` is only written when `noisy_data is not None or noise_std > 0`. Otherwise leave `_noisy` alone.
- Update `get_noisy()` docstring to note that it raises if the field has no noisy payload.
- Update `clear_field` / `clear_all` / `reset` — already only touch `_noisy` if the entry exists.

### Step 2 — `DelayPipelineV3` dual-cache rewrite

File: [delay_system_v3/delay_pipeline_v3.py](../../../delay_system_v3/delay_pipeline_v3.py)

This is the biggest single change. Recommended approach: write a new `advance` as `_advance_pair`, keep the old `advance` as a thin wrapper that calls `_advance_pair(data, None, …)`. Once the `UnifiedDelaySystem` is migrated, delete the old wrapper.

Exact changes:

- `__init__` — split state buffers:
  - `_staleness_held_data` → `_staleness_held_raw` + `_staleness_held_noisy`
  - `_dropout_held_data` → `_dropout_held_raw` + `_dropout_held_noisy`
  - `_lag_output` → `_lag_output_raw` + `_lag_output_noisy`
  - `_data_buffer = CircularBuffer(max_steps, num_envs, device)` → `_raw_data_buffer` + `_noisy_data_buffer` (each a `CircularBuffer`). `_timestamp_buffer` unchanged.
  - `_cached_data_with_dropout` / `_cached_data_no_dropout` → split into raw/noisy variants (4 cache slots for data + 2 for timestamp).
- `advance` — new signature `(raw_data, noisy_data, timestamp, t_current, burst_dropout_mask=None)`:
  - `if noisy_data is None: noisy_data = raw_data` at top of the function — handles the bbox case where there's no separate noisy payload. The dual-cache slots will hold identical clean data; `query(use_noise=True)` returns the same as `query(use_noise=False)`. This is the correct semantic for bbox `_gt` / `_det` fields, each of which has only one source.
  - Idempotency guard: unchanged.
  - `_apply_staleness_pair(raw, noisy, ts, t_current)` — sample staleness mask once, apply to both raw and noisy. Returns `(raw_held, noisy_held, ts_held)`.
  - `_apply_latency_pair(raw, noisy, ts, t_current)` — single `_last_append_time` guard, append both buffers under the same mask, retrieve both with the same `time_lags` from the single `_latency_sampler`.
  - `_apply_first_order_lag` — call separately for raw and noisy, each with its own `_lag_output_*` state. `dt`/`tau` shared.
  - Cache pre-dropout: `_cached_raw_no_dropout = raw`, `_cached_noisy_no_dropout = noisy`, `_cached_ts_no_dropout = ts`.
  - `_apply_dropout_pair(raw, noisy, ts, mask)` — sample mask once (or use `burst_dropout_mask`), apply to both. Returns `(raw_dropped, noisy_dropped, ts_dropped)`.
  - Cache post-dropout: 3 fields.
- `query(use_noise, allow_dropout)` — replaces the existing `query(allow_dropout)`. Returns the matching cache slot.
- `process(raw_data, noisy_data, timestamp, t_current, use_noise, allow_dropout, burst_dropout_mask=None)` — calls advance then query.
- `reset()` — zero all the new cache/state buffers, including the two `*_buffer` instances and the dual lag/staleness/dropout held tensors.

Pseudo-code for the staleness pair (the others are analogous):

```python
def _apply_staleness_pair(self, raw, noisy, ts, t_current):
    if not self._staleness_initialized:
        self._staleness_held_raw   = raw.clone()
        self._staleness_held_noisy = noisy.clone()
        self._staleness_held_timestamp = ts.clone()
        self._last_detection_time = t_current.clone()
        self._staleness_initialized = True
        return raw.clone(), noisy.clone(), ts.clone()

    time_since_last  = t_current - self._last_detection_time
    detection_period = self._staleness_sampler.period_values
    should_update    = time_since_last >= detection_period

    sud_raw = should_update
    for _ in range(len(raw.shape) - 1):    sud_raw   = sud_raw.unsqueeze(-1)
    sud_noisy = should_update
    for _ in range(len(noisy.shape) - 1):  sud_noisy = sud_noisy.unsqueeze(-1)

    self._staleness_held_raw       = torch.where(sud_raw,   raw,   self._staleness_held_raw)
    self._staleness_held_noisy     = torch.where(sud_noisy, noisy, self._staleness_held_noisy)
    self._staleness_held_timestamp = torch.where(should_update, ts, self._staleness_held_timestamp)
    self._last_detection_time      = torch.where(should_update, t_current, self._last_detection_time)

    return (self._staleness_held_raw.clone(),
            self._staleness_held_noisy.clone(),
            self._staleness_held_timestamp.clone())
```

The pair functions assume `raw.shape == noisy.shape`. Add an assertion at the top of `advance` to catch shape mismatches early (helps debugging).

### Step 3 — `UnifiedDelaySystem` simplification

File: [delay_system_v3/delay_system_v3.py](../../../delay_system_v3/delay_system_v3.py)

- Delete `_DETECTION_SUFFIXES`, `_ego_detection_pipelines`, `_other_detection_pipelines`, `_detection_pipeline_split`, the `_all_pipelines()` helper, the `if not self._detection_pipeline_split:` branches in `set_delay_mode`, `set_dropout_rate`, `set_field_delay_mode`, `set_field_dropout_rate`, `reset`, `step`.
- `register_field`: drop the suffix-aware branch ([L185-206](../../../delay_system_v3/delay_system_v3.py#L185-L206)). Single per-field pipeline registration.
- `get_delayed`: rewrite per §2.3.
- `get_aoi`: still works through `get_delayed`; should be unchanged in spirit but verify after the signature change.
- `store`: signature unchanged (still accepts `noise_std` and `noisy_data`); just trust `FieldStorage`'s new conditional `_noisy` population.

### Step 4 — Bbox field split

File: [delay_system_v3/multi_agent_wrapper.py](../../../delay_system_v3/multi_agent_wrapper.py)

- Update the `_FIELDS` registration list ([L36-67](../../../delay_system_v3/multi_agent_wrapper.py#L36-L67)): replace `("bboxes_2d", None)` with `("bboxes_2d_raycaster", None)` and `("bboxes_2d_replicator", None)`.
- Update `update_ground_truth` ([L383-401](../../../delay_system_v3/multi_agent_wrapper.py#L383-L401)) per §2.5.
- Update `_build_all_states` bbox read ([L548-554](../../../delay_system_v3/multi_agent_wrapper.py#L548-L554)) per §2.5.
- Audit other read sites: any consumer reading `bboxes_2d` directly from delay system needs to choose `_gt` vs `_det`. Grep `grep -n "bboxes_2d" delay_system_v3/multi_agent_wrapper.py` and update each. Probable additional sites: triangulation reads in env, AoI computation.
- `_get_noise_std("bbox", …)` — bbox no longer goes through Gaussian noise injection; the detector replicator handles that. Either remove the `"bbox"` case from `_get_noise_std` or leave it as dead code (not called any more).

### Step 5 — Env-side updates

File: [iris_ma_env6_test.py](../../../iris_ma_env6_test.py)

- The `update_ground_truth(agent_id, gt_states, replicated_bboxes=…)` call ([L1060-1062](../../../iris_ma_env6_test.py#L1060-L1062)) — signature unchanged.
- Any direct `_delay_system.get_delayed(prefix + "bboxes_2d", …)` calls — there shouldn't be any in env code (everything goes through the wrapper), but grep to confirm: `grep -n "bboxes_2d" iris_ma_env6_test.py`.
- AoI computations using `data.timestamp_detection` ([L1722, L1760](../../../iris_ma_env6_test.py)) — unchanged. The `_gt` and `_det` fields share `ts_detection` so AoI is the same regardless of which the consumer reads from.

### Step 6 — Config cleanup

Files: [delay_system_v3/delay_cfg_v3.py](../../../delay_system_v3/delay_cfg_v3.py), [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py)

- Delete `detection_pipeline_split` from `UnifiedDelayCfgV3` ([L378](../../../delay_system_v3/delay_cfg_v3.py#L378)) and `MultiAgentDelaySystemV3Cfg` ([L611](../../../delay_system_v3/delay_cfg_v3.py#L611)).
- Delete the params plumbing at [delay_cfg_v3.py:739](../../../delay_system_v3/delay_cfg_v3.py#L739).
- `field_overrides` for `bboxes_2d` (if any) → split into `bboxes_2d_raycaster` and `bboxes_2d_replicator`. Initial values: copy. Tune later.
- Surface `bboxes_2d_raycaster` / `bboxes_2d_replicator` in any env-cfg layer that exposes per-field overrides.

### Step 7 — Tests

See §4 for the test plan. Update at least:
- `tests/test_per_agent.py`
- `tests/test_reward_modes.py`
- `tests/test_pipeline_external_mask.py`
- `tests/run_tests.py`

Delete `tests/plot_detection_pipeline_split.py` (its purpose is now a regression test, see §4.4).

### Step 8 — Ticket 028 cleanup

Delete `doc/active/ticket/028-detection-pipeline-split-flag/`. The diagnostic it specifies is obsolete.

### Step 9 — Update the post-mortem

Append a "Resolved by ticket 029" section to [doc/experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md](../../../experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md). Note that the corrected curriculum semantics (real noise + delay reaching observations on every state field) require curriculum re-tuning before the next training run.

### Step 10 — Update CONTEXT.md / module docs

- [delay_system_v3/CONTEXT.md](../../../delay_system_v3/CONTEXT.md) — describe the dual-cache pipeline contract and the bbox split.
- [delay_system_v3/doc/README.md](../../../delay_system_v3/doc/README.md) — same.

### Step 11 — Update progress.txt and feature_list.json per the iris_ma6 session protocol.

---

## 4. Test plan

### 4.1 Unit tests (delay_system_v3/tests)

**`test_pipeline_dual_cache.py`** (new). Cover:
- `advance(raw, None, …)` populates both raw and noisy caches with raw payload (clean-only fallback for bbox-style usage). `query(use_noise=True)` and `query(use_noise=False)` return identical data.
- `advance(raw, raw + N(0, σ), …)` populates raw cache with raw and noisy cache with raw + N. Both have the same delay realization (same staleness mask, same latency draw, same dropout mask). Verify by checking that `(noisy_cache - raw_cache)` is the noise after delay (no time misalignment).
- Idempotency: calling `advance` twice at the same `t_current` does not corrupt cached data. Calling with different `(raw, noisy)` second time returns first-call's caches.
- Reset clears all 4 data caches + 2 timestamp caches.
- Burst dropout mask supplied externally is applied to both raw and noisy paths (not just noisy).

**`test_reward_modes.py`** (update existing). The test fixtures should now exercise the dual-cache path. Key invariants to assert:
- Reward path (`use_noise=False`) returns delayed clean data.
- Obs path (`use_noise=True`) returns delayed noisy data with the same timestamp as the reward path's data for the same field/perspective at the same `t_current`. **This is the key regression test for the bug** — write it explicitly:

```python
# At each sim step k:
reward_data, reward_ts = system.get_delayed(f, "ego", use_noise=False, allow_dropout=False)
obs_data,    obs_ts    = system.get_delayed(f, "ego", use_noise=True,  allow_dropout=False)
assert torch.allclose(reward_ts, obs_ts), "delay realizations diverged"
# obs_data should NOT equal reward_data when noise is enabled:
assert not torch.allclose(reward_data, obs_data), "noise was discarded (idempotency bug)"
```

**`test_per_agent.py`** (update existing). Verify the bbox split:
- `update_ground_truth(agent_id, states, replicated_bboxes=…)` populates both `bboxes_2d_raycaster` and `bboxes_2d_replicator`.
- `update_ground_truth(agent_id, states, replicated_bboxes=None)` populates `bboxes_2d_raycaster` with raycaster GT and `bboxes_2d_replicator` with the same GT (fallback).
- Reward-path read returns `bboxes_2d_raycaster` data; obs-path read returns `bboxes_2d_replicator` data.
- The two pipelines have independent `_last_advance_time` (advancing one does not gate the other).

**`test_pipeline_external_mask.py`** (update existing). Burst dropout mask interaction with the new pair-aware dropout stage.

### 4.2 Regression test for the post-mortem bug

Replace `plot_detection_pipeline_split.py` with `tests/test_noise_reaches_obs.py`. The test sets up a single-field shared pipeline, stores `(raw, raw + large_noise)` per step for 100 steps, queries reward and obs paths in the env-step order (reward first), and asserts:

```python
noise_rms_obs_minus_reward = ((obs_history - reward_history) ** 2).mean().sqrt()
assert noise_rms_obs_minus_reward > expected_min, (
    "Observation path returned reward path data — idempotency bug regression"
)
```

This is the explicit regression for the bug fixed by this ticket. It MUST pass after the rewrite.

### 4.3 Test runner

Update `tests/run_tests.py` to include the new test files. Use the AppLauncher template from [/home/usrg/IsaacPX4/IsaacLab/CLAUDE.md](../../../../../../../CLAUDE.md) "Generating Tests for Functional Modules" section — but note that `delay_system_v3` tests are pure-torch (no Isaac Sim required), so they currently run without AppLauncher. Keep that property; the new tests should also be import-only.

### 4.4 Iterative test workflow

Per the "Iterative Test-Fix Workflow" in [CLAUDE.md](../../../../../../../CLAUDE.md):

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/run_tests.py \
    > source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/test_result.txt 2>&1
```

Document each iteration in `tests/error_log.txt` per the standard format. Maximum 10 iterations before pausing for design review.

### 4.5 End-to-end smoke

After unit tests pass:
```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_iris_ma6.py \
    --num_envs 2 --enable_camera
```
With a curriculum-overridden noise scale (manually call `_delay_system.set_noise_scale(1.0)` early in the script), the yellow "delayed bbox" overlay should:
- Be delayed relative to green GT (latency stage works).
- Show offset from GT (detector replicator output reaches observations).
- Blink occasionally (FN / dropout reaches observations).

If any of those properties is missing, the bug is regressing.

### 4.6 Training validation

Once smoke passes, run training. The 120k cliff in `2695ffe1e3_revert_to_2be3` was the symptom of the bug. After this ticket, training should still hit a difficulty inflection at 120k (fixed-delay phase) — but it should be a curriculum challenge the policy can adapt to (gradient signal present), not the abrupt cliff seen in `025af4d022` / `82364f333c` / `2695ffe1e3`. If the cliff persists, the issue is curriculum design, not the delay system — proceed to the curriculum re-tune work that is currently blocked on this ticket.

Save checkpoints at 100k / 120k / 140k / 160k for post-hoc analysis.

---

## 5. Risks and "do not do this"

1. **Do NOT keep `detection_pipeline_split` as a "diagnostic" flag.** The whole point of this ticket is that the bug is structurally impossible after the rewrite — there's nothing to diagnose. Delete it.

2. **Do NOT introduce per-call `noise_after_delay` for state fields.** That violates the user's noise-propagation requirement. Read §1.3 again if tempted.

3. **Do NOT split state fields into separate `_clean` / `_noisy` field names** (the bbox approach generalized). It would force two pipelines per state field with independent latency draws — wrong physical model. State fields share one physical sample observed two ways; bbox fields are two physical sensors. Different requirements, different solutions.

4. **Do NOT change `_get_rewards` / `_get_observations` ordering** to "fix" the bug. That kicks the can; the underlying race-condition design is still wrong.

5. **Do NOT leave `_DETECTION_SUFFIXES` as a comment or dead code.** Delete it cleanly. Future maintainers should not see vestigial detection-routing logic.

6. **Avoid widening `advance`'s required signature changes if possible.** The `noisy_data: Optional[Tensor]` argument with `None`-fallback to clean preserves the call contract for any external user of `DelayPipelineV3` directly. Confirm there are no such users (`grep -rn "DelayPipelineV3(" source/`) — there shouldn't be any outside `UnifiedDelaySystem.register_field`.

7. **Performance**: dual cache doubles the pipeline state memory and roughly doubles the per-stage `torch.where` ops. At the system scale (1024 envs × ~25 fields × 6 agents) this is small compared to physics + sim overhead. Don't pre-optimize. Measure step time before and after with `torch.cuda.Event` if you want a number.

8. **The `_apply_first_order_lag` for raw vs noisy is independent**. The lag filter has internal state per signal — applying the same `alpha` to two different signals gives two different filtered outputs. This is correct (each signal smooths to itself).

9. **Test plot artifacts**: do NOT regenerate `detection_pipeline_split.png` after the change — that file is documentation of the bug the post-mortem references. Leave it on disk; just stop linking to it from active code/tests.

---

## 6. Success criteria

- All `delay_system_v3/tests` pass.
- The regression test in §4.2 passes.
- `detection_pipeline_split` flag is deleted from cfg + system + plumbing. `git grep detection_pipeline_split` returns no hits in code (post-mortem and ticket docs may still reference it historically).
- `_DETECTION_SUFFIXES`, `_ego_detection_pipelines`, `_other_detection_pipelines` are deleted. `git grep _ego_detection_pipelines` returns no hits.
- `bboxes_2d_raycaster` and `bboxes_2d_replicator` are first-class fields in `multi_agent_wrapper._FIELDS`. The single `bboxes_2d` field name no longer exists in the storage / pipeline layer (it may still exist on the in-memory `AgentStates.data.bboxes_2d` attribute that consumers read — that is a per-agent runtime variable, not a storage key).
- Teleop smoke (§4.5) shows yellow != green when noise is forced to 1.0.
- A 20k headless training run finishes without errors and reaches `pair_valid_rate > 0.85` by step 16k (matches 2be3 / 928b / 492f early bootstrap, since `_noise_scale` is 0 pre-100k and the curriculum doesn't change in this range).

---

## 7. After this ticket

- Re-tune the observation-corruption curriculum. The lesson from 5ce56e3ae8 is critical: noise/delay/dropout/FP-FN belong AFTER the base tracking skill is bootstrapped, not at step 0. With observations now actually carrying corruption (instead of silently clean), the previous timing assumptions are wrong. Proposed tickets:
  - 030: curriculum re-tune for corrected observation pipeline.
  - 031: model proper noise propagation through derived fields (compute `body_combined_angular_velocity_w` etc. from noisy primitives in env, store noisy version directly via `noisy_data=…`).
  - 032: independent latency configurations for `bboxes_2d_raycaster` (raycaster — fast / instant) vs `bboxes_2d_replicator` (replicator — YOLO-latency model).

- The sigma stagnation issue (post-bundle policies plateau at ~0.43) is orthogonal to the delay system bug and remains open. Do not expect this ticket alone to recover 928b's sigma trajectory.

---

## 8. Reference appendix — file map

| Concept | File:line |
|---|---|
| `UnifiedDelaySystem.get_delayed` (current dispatch) | [delay_system_v3.py:246-311](../../../delay_system_v3/delay_system_v3.py#L246-L311) |
| Pipeline registration / split branch | [delay_system_v3.py:185-206](../../../delay_system_v3/delay_system_v3.py#L185-L206) |
| `_DETECTION_SUFFIXES` | [delay_system_v3.py:53](../../../delay_system_v3/delay_system_v3.py#L53) |
| `_detection_pipeline_split` storage | [delay_system_v3.py:77](../../../delay_system_v3/delay_system_v3.py#L77) |
| `advance` + idempotency guard | [delay_pipeline_v3.py:236-298](../../../delay_system_v3/delay_pipeline_v3.py#L236-L298) |
| `query` (cache reads) | [delay_pipeline_v3.py:316-323](../../../delay_system_v3/delay_pipeline_v3.py#L316-L323) |
| `_apply_staleness` | [delay_pipeline_v3.py:348-394](../../../delay_system_v3/delay_pipeline_v3.py#L348-L394) |
| `_apply_latency` (and append guard) | [delay_pipeline_v3.py:396-444](../../../delay_system_v3/delay_pipeline_v3.py#L396-L444) |
| `_apply_dropout` | [delay_pipeline_v3.py:446-493](../../../delay_system_v3/delay_pipeline_v3.py#L446-L493) |
| `_apply_first_order_lag` | [delay_pipeline_v3.py:495-511](../../../delay_system_v3/delay_pipeline_v3.py#L495-L511) |
| `FieldStorage.store` | [field_storage.py:78-117](../../../delay_system_v3/field_storage.py#L78-L117) |
| `FieldStorage.get_raw` / `get_noisy` | [field_storage.py:119-149](../../../delay_system_v3/field_storage.py#L119-L149) |
| Wrapper field registration list | [multi_agent_wrapper.py:36-67](../../../delay_system_v3/multi_agent_wrapper.py#L36-L67) |
| `update_ground_truth` (bbox store) | [multi_agent_wrapper.py:233-401](../../../delay_system_v3/multi_agent_wrapper.py#L233-L401) |
| `_build_all_states` (bbox read) | [multi_agent_wrapper.py:443-696](../../../delay_system_v3/multi_agent_wrapper.py#L443-L696) |
| `_add_noise_to_states` (legacy noise injection) | [multi_agent_wrapper.py:820+](../../../delay_system_v3/multi_agent_wrapper.py#L820) |
| `_get_noise_std` | [multi_agent_wrapper.py:335-369](../../../delay_system_v3/multi_agent_wrapper.py#L335-L369) |
| Env reward / obs ordering | [iris_ma_env6_test.py:1151-1158](../../../iris_ma_env6_test.py#L1151-L1158), [iris_ma_env6_test.py:1496-1500](../../../iris_ma_env6_test.py#L1496-L1500) |
| Env timestamp capture | [iris_ma_env6_test.py:1040-1047](../../../iris_ma_env6_test.py#L1040-L1047) |
| Env replicator hookup | [iris_ma_env6_test.py:1050-1062](../../../iris_ma_env6_test.py#L1050-L1062) |
| Detector replicator output | [bbox_raycaster_v2/detector_replicator.py:201-280](../../../bbox_raycaster_v2/detector_replicator.py#L201-L280) |
| `bboxes_replicated` data field | [bbox_raycaster_v2/bbox_raycaster_v2_data.py:87](../../../bbox_raycaster_v2/bbox_raycaster_v2_data.py#L87) |
| DirectRLEnv step order | [/home/usrg/IsaacPX4/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:363-384](../../../../../../../isaaclab/isaaclab/envs/direct_rl_env.py#L363-L384) |
| Bug visualization (read-only) | [delay_system_v3/tests/detection_pipeline_split.png](../../../delay_system_v3/tests/detection_pipeline_split.png) |
| Bug-reproducer script (delete after rewrite) | [delay_system_v3/tests/plot_detection_pipeline_split.py](../../../delay_system_v3/tests/plot_detection_pipeline_split.py) |

---

## 9. Conversation history that produced this ticket

The full design rationale is in the post-mortem [doc/experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md](../../../experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md). Key prior decisions captured there:

1. The 120k cliff in every post-`3461991973` run is attributed to ticket 020 unmasking observation corruption that 2be3 silently dropped.
2. Ticket 028 was originally drafted as a one-flag diagnostic to confirm the attribution. This ticket supersedes it.
3. The user's design call: bbox = field-split (two physical sources); state fields = single shared pipeline with dual cache (one physical source, two views).
4. The user's noise-propagation requirement (forbids noise-after-delay): see §1.3.

**Resolved (final, after follow-up)**:
- **Field name**: single field `bboxes_2d` per agent. The initial design
  split it into `bboxes_2d_raycaster` / `bboxes_2d_replicator`; that was
  reverted after we noticed separate pipelines produce independent latency
  draws under `random` curriculum mode (physically wrong — both views model
  one camera capture event).
- **Payload layout**: dual-payload via storage's `_raw` / `_noisy` slots.
  - raw   = raycaster geometric GT (read by reward path, `use_noise=False`).
  - noisy = detector-replicator output (read by obs path, `use_noise=True`).
  - When the replicator is disabled, the noisy payload explicitly mirrors
    raw (`noisy_data=data.bboxes_2d`). Explicit mirror chosen over
    `noisy_data=None` so the intent is visible at the call site.
- **Latency at storage time**: zero. All delay is owned by the pipeline
  and configured via `PerspectiveCfg.field_overrides["bboxes_2d"]`. The
  replicator is NOT a latency source; it is a bbox content transform
  (FP/FN, calibrated noise, occlusion model).
- **Hookup**: both payloads flow through one `update_ground_truth` call at
  [iris_ma_env6_test.py:1060-1062](../../../iris_ma_env6_test.py#L1060-L1062).
  The wrapper performs ONE `store("bboxes_2d", raycaster, timestamp=ts_detection,
  noisy_data=replicator_or_raycaster_fallback)` call per agent.
- **Shared delay invariant**: the pipeline's `_apply_*_pair` stages run
  once per sim step with one staleness mask, one latency draw, and one
  dropout mask, applied to both payloads in lockstep. Reward and obs
  queries at the same step therefore share a timestamp and a delay
  realization — they differ only in content.
