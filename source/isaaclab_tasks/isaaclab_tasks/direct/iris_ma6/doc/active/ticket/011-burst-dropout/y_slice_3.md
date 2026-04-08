## Slice 3 Complete

### Files changed
- `delay_system_v3/delay_cfg_v3.py`: Added burst_dropout field to MultiAgentDelayCfgV3
- `delay_system_v3/delay_system_v3.py`: get_delayed() accepts burst_dropout_mask
- `delay_system_v3/multi_agent_wrapper.py`: Burst sampler lifecycle — init, advance in set_time(), per-channel masks in _build_agent_states(), set_dropout_rate() disables i.i.d. when burst active, set_burst_params(), reset(), randomize_per_agent_params()

### Test checkpoint
All existing tests pass. Burst: 11/11. Pipeline: 5/5. Per-agent: exit 0.
