# Delay System Test Suite

Comprehensive tests for the stochastic sample-and-hold delay system used in multi-agent RL environments for realistic sim-to-real transfer.

## Overview

This test suite validates all components of the delay system:
- Stochastic samplers with various distribution types
- Specialized samplers (first-order lag, quaternion SLERP, passthrough)
- Timestamp tracking and staleness calculation
- Full multi-agent delay system integration
- Ego vs. other agent perspective handling

## Test Files

- **run_tests.py**: Standalone test runner for all delay system tests
  - Stochastic sampler tests (constant, uniform, normal distributions)
  - Latency and dropout handling
  - First-order lag filtering and convergence
  - Quaternion SLERP normalization
  - Timestamp management (staleness, latency calculation)
  - Multi-agent integration (perspective switching, state queries)
  - Reset and state management

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended for Isaac Sim modules)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delay_system/tests/run_tests.py

# With verbose output (detailed error messages and debug info)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delay_system/tests/run_tests.py --test-verbose

# With headless rendering (default)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delay_system/tests/run_tests.py --headless

# With GUI (for debugging)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delay_system/tests/run_tests.py --enable_cameras
```

### Run on Specific Device

```bash
# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delay_system/tests/run_tests.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delay_system/tests/run_tests.py
```

## Test Organization

Tests are organized into four main suites:

### 1. Stochastic Sampler Tests
- **Constant period sampler**: Validates deterministic sampling at fixed rate
- **Uniform distribution**: Tests variable sampling rates with uniform random periods
- **Normal distribution**: Tests Gaussian-distributed sampling periods
- **Latency handling**: Validates delay buffer and sample release timing
- **Dropout detection**: Confirms stochastic packet loss behavior
- **Reset functionality**: Tests environment reset handling

### 2. Specialized Sampler Tests
- **FirstOrderLagSampler initialization**: Validates filter setup
- **FirstOrderLagSampler filtering**: Tests convergence to target values
- **QuaternionFirstOrderLagSampler initialization**: Validates identity quaternion start
- **QuaternionFirstOrderLagSampler normalization**: Confirms unit quaternion output
- **PassthroughSampler**: Tests instant, unmodified data passthrough
- **Reset functionality**: Validates per-environment state reset

### 3. Timestamp Tracking Tests
- **FieldTimestamp initialization**: Tests timestamp structure creation
- **Staleness calculation**: Validates `t_current - t_captured` computation
- **Latency calculation**: Tests `t_available - t_captured` computation
- **AgentStatesTimestamps**: Tests multi-group timestamp management
- **TimestampManager**: Validates multi-agent timestamp coordination

### 4. Delay System Integration Tests
- **DelaySystem initialization**: Tests full system setup with multi-agent config
- **Update and query ego states**: Validates fast ego processing pipeline
- **Multi-agent perspective**: Tests ego vs. other agent delay differences
- **Step and reset**: Validates time advancement and environment reset

## Expected Results

### Device Compatibility
- ✅ Tests run on both CUDA and CPU devices
- ✅ Automatic device detection and fallback

### Numerical Accuracy
- First-order lag convergence within 50 timesteps
- Quaternion normalization to unit length (tolerance: 1e-5)
- Timestamp calculations match expected values exactly

### Edge Case Handling
- Reset clears internal state for specified environments only
- Dropout rates approximate configured probabilities over many samples
- Latency buffers correctly delay samples across timesteps

### State Management
- Environment-specific resets preserve other environment states
- Multi-agent perspectives provide independent delayed views
- Timestamp tracking accurately reflects processing stages

## Common Issues

### CUDA Out of Memory
**Symptom**: Runtime error during sampler updates

**Solution**: Reduce `num_envs` in test fixtures
```python
num_envs = 4  # Reduced from 8 or 16
```

### Import Errors
**Symptom**: `ModuleNotFoundError` for delay_system components

**Solution**: Ensure module is installed in Isaac Lab environment
```bash
./isaaclab.sh -i
```

### AppLauncher Initialization Error
**Symptom**: Isaac Sim fails to start or crashes on import

**Solution**: Ensure AppLauncher is initialized before other imports. The test runner already follows this pattern - do not modify import order.

### Numerical Precision Issues
**Symptom**: Tests fail with small floating-point differences

**Solution**: Adjust tolerance in assertions (already set to 1e-5 for most tests)

## Test Best Practices

The delay system tests follow Isaac Lab testing conventions:

1. **AppLauncher Template**: All tests use proper Isaac Sim initialization
2. **Device Agnostic**: Tests work on both CUDA and CPU
3. **Parameterized Environments**: Easy to adjust `num_envs` for memory constraints
4. **Clear Test Names**: Descriptive names explaining what is being tested
5. **Test Independence**: Each test creates fresh instances
6. **Numerical Tolerances**: Appropriate floating-point comparison tolerances

## Performance Benchmarks

Expected test execution times (on RTX 3080):
- Stochastic sampler tests: ~2 seconds
- Specialized sampler tests: ~3 seconds
- Timestamp tracking tests: ~1 second
- Integration tests: ~5 seconds
- **Total**: ~11 seconds

CPU-only execution may take 2-3x longer.

## Contributing

When adding new delay system features:

1. **Add corresponding tests** to the appropriate test suite
2. **Ensure all existing tests pass** before committing
3. **Update this README** if adding new test categories
4. **Verify tests on both CPU and GPU**
5. **Follow AppLauncher template** for any new test files
6. **Use TestResults class** for consistent reporting

### Adding a New Test

```python
def test_new_feature(results: TestResults, device: torch.device):
    """Test new delay system feature."""
    print("\n" + "="*80)
    print("Testing New Feature")
    print("="*80)

    try:
        # Your test code here
        assert condition, "Failure message"
        results.add_pass("New feature test name")
    except Exception as e:
        results.add_fail("New feature test name", traceback.format_exc())
```

Then add to `main()` in `run_tests.py`:
```python
try:
    test_new_feature(results, device)
except Exception as e:
    results.add_error("New Feature suite", traceback.format_exc())
```

## References

- [Delay System Design Pattern](../DESIGN_PATTERN.md) - Complete architectural documentation
- [Delay System README](../README.md) - User guide and API reference
- [Usage Example](../usage_example.py) - Working code example

## Citation

If you use this delay system in your research, please cite:

```bibtex
@software{delay_system_2025,
  title={Delay System for Multi-Agent RL},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```
