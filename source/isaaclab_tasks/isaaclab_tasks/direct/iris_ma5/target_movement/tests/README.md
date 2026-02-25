# Target Movement Test Suite

Test suite for the target movement module that provides randomized target motion for the iris_ma5 multi-agent drone environment.

## Test Files

- **run_tests.py**: Standalone test runner with comprehensive test coverage
  - Initialization and configuration
  - Linear mode velocity tracking
  - Circular mode orbit tracking
  - Geofencing boundary enforcement
  - Altitude constraint enforcement
  - Reset functionality
  - Full integration cycle

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/target_movement/tests/run_tests.py

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/target_movement/tests/run_tests.py --test-verbose
```

### Run with Specific Device

```bash
# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/target_movement/tests/run_tests.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/target_movement/tests/run_tests.py
```

## Test Organization

### Initialization Tests
- Default configuration loading
- Custom configuration parameters
- Buffer allocation on correct device

### Linear Mode Tests
- Mode assignment with linear_weight=1.0
- Velocity tracking with acceleration control
- Velocity clamping to max_speed

### Circular Mode Tests
- Mode assignment with linear_weight=0.0
- Radius constraints (min/max)
- Orbit tracking behavior

### Geofencing Tests
- Curriculum-based geofence scaling formula
- X boundary bounce behavior
- Y boundary bounce behavior

### Altitude Tests
- Altitude bounce when below minimum
- No bounce when above minimum altitude

### Reset Tests
- Full reset of all environments
- Partial reset of subset of environments
- Mode reinitialization on reset

### Integration Tests
- Full 200-step simulation cycle
- Curriculum progress scaling (0 to 1)
- Mode switching over multiple steps

## Expected Results

- All tests should pass on both CUDA and CPU devices
- No NaN or Inf values in velocity or position buffers
- Geofencing correctly bounces targets off boundaries
- Altitude constraint prevents targets from going below minimum
- Curriculum scaling correctly scales output velocity

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test fixtures (currently set to 16-64).

### Import Errors
Ensure the module is properly installed:
```bash
./isaaclab.sh -i
```

### AppLauncher Errors
Tests require Isaac Sim to be properly configured. Run from the IsaacLab root directory.

## Contributing

When adding new features to the target movement module:
1. Add corresponding unit tests to `run_tests.py`
2. Ensure all existing tests pass
3. Verify tests work on both CPU and GPU
4. Update this README if adding new test categories
