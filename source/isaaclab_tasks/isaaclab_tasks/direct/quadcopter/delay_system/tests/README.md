# Delay System Test Suite

This directory contains tests for the delay system module, which provides configurable action and observation latency simulation for reinforcement learning environments.

## Test Files

- **run_tests.py**: Standalone test runner (Isaac Sim compatible)
  - DelayCfg tests: Configuration validation and initialization
  - DelaySystemCfg tests: System configuration validation
  - DelaySystem tests: Core functionality and delay computation
  - Integration tests: Real-world usage patterns

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended for Isaac Sim modules)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py

# With verbose output for debugging
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py --test-verbose
```

### Run with Specific Device

```bash
# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py

# GPU 0 only
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/delay_system/tests/run_tests.py
```

## Test Organization

### DelayCfg Tests
- Default initialization
- Custom initialization
- Validation (negative min_delay)
- Validation (max_delay < min_delay)

### DelaySystemCfg Tests
- Default initialization
- Action delay only configuration
- Both delays configured

### DelaySystem Tests
- Initialization with delays disabled
- Initialization with action delay enabled
- Initialization with both delays enabled
- Compute delayed action (no delay buffer)
- Compute delayed observation (no delay buffer)
- Compute delayed action with fixed delay
- Reset functionality
- Delay randomization
- Get delay info
- Manual delay setting

### Integration Tests
- Simulated training loop (100 steps)
- Verify delay timing (3-step fixed delay)
- Large batch size (4096 envs)

## Expected Results

- GPU/CPU compatibility: Tests run on both devices
- Configuration validation: Invalid configs raise appropriate errors
- Delay accuracy: Fixed delays produce correct temporal offset
- Randomization: Random delays within specified range
- Reset: Buffer state properly cleared and delays re-randomized

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test fixtures. Large batch test uses 4096 envs which may be too much for some GPUs.

### Import Errors
Ensure the module is properly installed:
```bash
./isaaclab.sh -i
```

### Delay Buffer Not Found
Ensure Isaac Lab is properly installed with the buffer utilities:
```bash
pip install -e source/isaaclab
```

## Contributing

When adding new features to the delay system:
1. Add corresponding unit tests in `run_tests.py`
2. Ensure all existing tests pass
3. Update this README for new test categories
4. Verify tests on both CPU and GPU
