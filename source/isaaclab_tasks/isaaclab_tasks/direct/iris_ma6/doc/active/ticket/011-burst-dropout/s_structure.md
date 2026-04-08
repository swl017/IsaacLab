## Structure Outline — Ticket 011: Burst Dropout

### New files
- `delay_system_v3/burst_dropout.py`: BurstDropoutCfg, BurstDropoutSampler

### Modified files
- `delay_system_v3/delay_pipeline_v3.py`: advance(), _apply_dropout(), process() accept burst_dropout_mask
- `delay_system_v3/delay_system_v3.py`: get_delayed() accepts burst_dropout_mask
- `delay_system_v3/delay_cfg_v3.py`: burst_dropout on MultiAgentDelayCfgV3, burst params on DelaySystemKeyParams
- `delay_system_v3/multi_agent_wrapper.py`: burst sampler lifecycle (init, advance, reset, randomize, curriculum)
- `delay_system_v3/__init__.py`: exports
- `delay_system_v3/sampling_strategies.py`: delete PerAgentDropoutSampler
- `curriculum/curriculum_cfg.py`: burst phase fields + getter
- `iris_ma_env6_test.py`: curriculum wiring
