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
- `advance(data, timestamp, t_current)`: **WRITE**. Runs all stateful stages
  (staleness, latency buffer append, FOL filter, dropout mask sample) exactly once.
  Idempotent within a sim step via `_last_advance_time` guard.
- `query(allow_dropout)`: **READ**. Returns cached output from last advance().
  `allow_dropout=False` returns pre-dropout data (for rewards).
  Safe to call any number of times.
- `process(data, timestamp, t_current, allow_dropout)`: Convenience wrapper —
  calls advance() then query(). Backward compatible. Safe for multi-call patterns.

**Pipeline stage order**: staleness → latency → FOL → dropout.
FOL is placed before dropout so the sensor filter operates on the channel signal
before packet-loss masking (physically correct: filter runs on received data,
dropout holds previously-filtered value on dropped steps).

## Spec
None (self-documented via docstrings and `doc/` directory).
