## Slice 4 Complete

### Files changed
- `curriculum/curriculum_cfg.py`: burst_dropout_start_step (200k), burst_dropout_end_step (220k), get_burst_dropout_progress()
- `delay_system_v3/delay_cfg_v3.py`: Burst params on DelaySystemKeyParams (5 fields), wired in create_delay_cfg_from_params()
- `iris_ma_env6_test.py`: Burst dropout curriculum ramp in _update_curriculum()

### Test checkpoint
Config integration: 5/5 pass. burst_enabled=False produces identical behavior — no sampler created.
