# Delay System V3 Test Suite

This test suite validates the V3 delay system implementation, focusing on the critical invariant that **timestamp always travels with its associated data** through all pipeline stages.

## Test Files

- **run_tests.py**: Main test runner containing all test suites
  - Configuration tests (5 tests)
  - Sampling strategy tests (9 tests)
  - Field storage tests (5 tests)
  - Pipeline tests (5 tests)
  - Timestamp synchronization tests (4 tests)
  - Curriculum mode tests (5 tests)

## Running Tests

### Run All Tests
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/run_tests.py --headless
```

### Run with Verbose Output
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/delay_system_v3/tests/run_tests.py --headless --test-verbose
```

## Test Categories

### 1. Configuration Tests
Validates dataclass configurations:
- DistributionCfg defaults (constant, normal, uniform)
- SamplingCfg defaults (per_step, per_episode, per_env_reset)
- LatencyCfg with min_steps
- Preset configurations (create_no_delay_cfg, create_fixed_delay_cfg, create_random_delay_cfg)

### 2. Sampling Strategy Tests
Validates parameter sampling:
- Constant distribution sampling
- Normal distribution with clamping
- Uniform distribution range
- Per-episode sampling (values constant within episode)
- Per-step sampling (values change each step)
- Latency step conversion (seconds to steps)
- min_steps enforcement
- Staleness FPS to period conversion
- Dropout probability sampling

### 3. Field Storage Tests
Validates raw/noisy data storage:
- Basic field storage and retrieval
- Noise injection (raw vs noisy differ)
- Timestamp storage and retrieval
- Custom timestamp support
- Multi-field storage with agents

### 4. Pipeline Tests (Critical)
Validates the core delay pipeline:
- Pipeline initialization
- Latency-only delay (2-step delay works correctly)
- Warmup handling (first step returns valid data, not zeros)
- **Dropout holds data AND timestamp together**
- **Staleness holds data AND timestamp together**

### 5. Timestamp Synchronization Tests (Critical)
Validates the fundamental timestamp-data invariant:
- **AoI = t_current - timestamp is correct** with latency
- AoI is always non-negative (no future timestamps)
- **Timestamp matches data through entire pipeline**
- Combined staleness and latency produce correct AoI

### 6. Curriculum Mode Tests
Validates curriculum learning support:
- Mode 'none' produces minimal delay (pass-through)
- Mode 'fixed' has deterministic (low variance) delay
- Mode 'random' enables staleness
- Progress [0,1] scales delay magnitude
- Dropout rate can be set independently

## Expected Results

All 33 tests should pass:
```
================================================================================
TEST SUMMARY
================================================================================
Total Tests: 33
Passed:      33 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Key Invariants Tested

1. **Timestamp-Data Coupling**: The returned timestamp is ALWAYS the capture time of the returned data, regardless of which pipeline stages are enabled.

2. **Warmup Handling**: CircularBuffer is properly initialized on first data, avoiding zeros output.

3. **Hold Semantics**: During staleness or dropout holds, BOTH data and timestamp are held together - ensuring AoI correctly reflects data age.

4. **Curriculum Progression**: Delay can be gradually introduced through mode and progress settings.

## Output Files

Test results are written to:
- `tests/test_result.txt`: Detailed test output with pass/fail status

## Common Issues

### Import Errors
Ensure the module is properly installed:
```bash
./isaaclab.sh -i
```

### CUDA Out of Memory
Reduce `num_envs` in test fixtures if needed (default: 8).

### Test Result Not Visible
Results are written to `test_result.txt` in the tests directory. Check this file if console output is truncated.
