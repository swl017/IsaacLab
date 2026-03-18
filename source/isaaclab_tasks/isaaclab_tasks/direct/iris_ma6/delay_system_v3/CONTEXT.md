# Delay System V3 Module

## Purpose
Unified multi-agent communication latency simulation with staleness tracking,
dropout modeling, and ringbuffer-based state storage. Simulates realistic
sensor/communication delays for sim-to-real transfer.

## Inputs
- Ground-truth agent states per timestep:
  - Position `(N, A, 3)`, velocity `(N, A, 3)`
  - Gimbal angles, camera intrinsics
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
- `delay_cfg_v3.py` - Delay configuration (latency distributions, dropout rates)
- `sampling_strategies.py` - Latency sampling (uniform, gaussian, constant)
- `field_storage.py` - Per-field ringbuffer storage
- `derived_field_computers.py` - Computes derived fields from stored state

## Spec
None (self-documented via docstrings and `doc/` directory).
