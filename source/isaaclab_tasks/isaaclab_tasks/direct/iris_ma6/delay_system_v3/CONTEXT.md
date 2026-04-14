# Delay System V3 Module

## Purpose
Unified multi-agent communication latency simulation with staleness tracking,
dropout modeling, and ringbuffer-based state storage. Simulates realistic
sensor/communication delays for sim-to-real transfer.

## Inputs
- Ground-truth agent states per timestep:
  - Position `(N, A, 3)`, velocity `(N, A, 3)`, orientation `(N, A, 4)` (wxyz)
  - Angular velocity `(N, A, 3)` (body frame), linear acceleration `(N, A, 3)` (body frame)
  - Gimbal joint angles, world-frame gimbal azimuth/elevation/rates, camera intrinsics
  - Bounding box detections

## Outputs
- Delayed `AgentStates` with configurable latency per field
- `is_valid` mask indicating data availability (NaN where unavailable)
- Staleness timestamps per field

## Dependencies
None (standalone module).

## Key Files
- `delay_pipeline_v3.py` - Core delay pipeline with ringbuffer storage
- `multi_agent_wrapper.py` - Multi-agent orchestration layer
- `delay_system_v3.py` - Top-level system interface
- `agent_states.py` - AgentStates dataclass definition
- `delay_cfg_v3.py` - Delay configuration (latency distributions, dropout rates, noise: position/velocity/orientation/angular_velocity/acceleration/bbox)
- `sampling_strategies.py` - Latency sampling (uniform, gaussian, constant)
- `field_storage.py` - Per-field ringbuffer storage
- `derived_field_computers.py` - Computes derived fields from stored state

## Calling Contract

### MultiAgentDelaySystemV3 (wrapper)
- `update_ground_truth()`: **WRITE**. Call exactly once per step from `_update_state_cache()`.
  Pushes GT snapshots to field storage. Multiple calls create duplicate noise — avoid.
- `get_all_states_for_observations()`: **READ**. Safe to call multiple times per step.
- `get_all_states_for_rewards()`: **READ**. Safe to call multiple times per step.
- `set_delay_mode()`, `set_noise_scale()`, `set_dropout_rate()`: **CONFIG**. Call from `_get_rewards()` for curriculum.
- `reset()`: Call from `_reset_idx()`.

### DelayPipelineV3 (per-field pipeline)
- `advance(raw_data, noisy_data, timestamp, t_current, burst_dropout_mask=None)`:
  **WRITE**. Runs all stateful stages (staleness, latency buffer append, FOL
  filter, dropout mask sample) exactly once. Idempotent within a sim step via
  `_last_advance_time` guard.
  - Both payloads share ONE delay realization: one staleness mask, one latency
    draw, one dropout mask. Raw and noisy cannot diverge in delay.
  - `noisy_data=None` is the clean-only case (e.g. `bboxes_2d_raycaster` /
    `bboxes_2d_replicator` — each has a single source). Both cache slots are
    filled with the raw payload.
- `query(use_noise, allow_dropout)`: **READ**. Returns the matching cache slot
  of the last advance(). Four data slots (raw/noisy × with/no dropout), two
  shared timestamp slots. Safe to call any number of times.
- `process(raw_data, noisy_data, timestamp, t_current, use_noise, allow_dropout,
  burst_dropout_mask=None)`: Convenience wrapper — advance() then query().
- `process_single(data, timestamp, t_current, allow_dropout=True, ...)`:
  Clean-only shortcut equivalent to `process(data, None, ...)`. Used by tests
  and single-source teleop / diagnostic code paths.

**Pipeline stage order**: staleness → latency → FOL → dropout.
FOL is placed before dropout so the sensor filter operates on the channel signal
before packet-loss masking (physically correct: filter runs on received data,
dropout holds previously-filtered value on dropped steps). FOL maintains
independent filter state per payload (`_lag_output_raw` / `_lag_output_noisy`)
so each signal smooths to itself; staleness / latency / dropout use shared
masks and buffers.

**Dual-cache invariant (ticket 029)**: reward and observation queries at the
same sim step share ONE delay realization. The first `advance()` call populates
both caches from the `(raw, noisy)` pair. Subsequent `advance()` calls at the
same `t_current` are no-ops (idempotency guard). This is the structural fix
for the shared-pipeline bug that ticket 020 worked around with a separate
detection pipeline. Regression test: `tests/test_dual_cache.py`.

### Bbox as a dual-payload field (ticket 029)
Bounding boxes model one physical detection event with two views — the
raycaster geometric GT and the detector-replicator output. Both views are
stored in a SINGLE field `bboxes_2d` using the dual-payload storage:
- raw   = raycaster GT (read by reward path, `use_noise=False`).
- noisy = detector-replicator output (read by observation path, `use_noise=True`).
  When the replicator is disabled, the noisy payload explicitly mirrors the
  raw payload so the obs cache slot stays populated.
One timestamp (`ts_detection`) and one pipeline instance per perspective apply
a single latency / staleness / dropout realization to both payloads — they
cannot diverge in delay by construction. Storage-side latency is zero; all
delay is owned by the pipeline and configured via
`PerspectiveCfg.field_overrides["bboxes_2d"]`.

## Spec
None (self-documented via docstrings and `doc/` directory).
