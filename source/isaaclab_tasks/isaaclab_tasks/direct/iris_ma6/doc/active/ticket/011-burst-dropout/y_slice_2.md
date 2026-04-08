## Slice 2 Complete

### Files changed
- `delay_system_v3/delay_pipeline_v3.py`: advance(), _apply_dropout(), process() accept burst_dropout_mask
- `delay_system_v3/sampling_strategies.py`: Deleted PerAgentDropoutSampler
- `delay_system_v3/__init__.py`: Removed PerAgentDropoutSampler import/export
- `delay_system_v3/tests/test_per_agent.py`: Removed deleted-class tests
- `delay_system_v3/tests/test_pipeline_external_mask.py`: New — 5 tests

### Test checkpoint
Pipeline external mask: 5/5 pass. Per-agent tests: pass (exit 0).
