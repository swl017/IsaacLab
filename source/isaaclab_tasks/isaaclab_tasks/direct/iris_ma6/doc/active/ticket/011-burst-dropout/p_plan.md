## Implementation Plan

### Slice 1: BurstDropoutSampler core + config
- Create burst_dropout.py with BurstDropoutCfg + BurstDropoutSampler
- Implement Gilbert-Elliott Markov chain, advance(), sample_mask(), set_burst_params(), reset(), randomize_params()
- Add exports to __init__.py
- Test: verify burst statistics match 1/p_recovery over 1000+ steps

### Slice 2: Pipeline integration
- Modify advance(), _apply_dropout(), process() to accept burst_dropout_mask
- Delete PerAgentDropoutSampler
- Test: pipeline respects external mask (all-True/all-False), backward compat

### Slice 3: Multi-agent wrapper integration
- Add BurstDropoutCfg to MultiAgentDelayCfgV3
- Instantiate BurstDropoutSampler in wrapper __init__
- Advance in set_time(), pass masks in _build_agent_states()
- Disable per-pipeline i.i.d. when burst active
- Reset/randomize burst state
- Test: independent channels, asymmetry, burst_enabled=False identical

### Slice 4: Curriculum + environment wiring
- Add burst phase to CurriculumCfg (200k-220k)
- Add burst params to DelaySystemKeyParams + create_delay_cfg_from_params()
- Wire curriculum in iris_ma_env6_test.py
- Test: config builder, backward compat
