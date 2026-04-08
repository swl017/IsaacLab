## Slice 1 Complete

### Files changed
- `delay_system_v3/burst_dropout.py`: New — BurstDropoutCfg + BurstDropoutSampler (Gilbert-Elliott)
- `delay_system_v3/__init__.py`: Added exports
- `delay_system_v3/tests/test_burst_dropout.py`: New — 11 tests

### Test checkpoint
All 11 tests pass. Mean burst length 10.1 (expected 10.0). Mean good-run 49.2 (expected 50.0). Channel correlation -0.017 (independent).
