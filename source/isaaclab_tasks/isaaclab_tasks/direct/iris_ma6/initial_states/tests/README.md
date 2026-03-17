# Initial States Test Suite

This test suite validates the initial_states module for the iris_ma6 environment.

## Test Files

- **run_tests.py**: Standalone test runner (AppLauncher-compatible)
  - Configuration tests (default and custom values)
  - Generator tests (initialization, shapes, bounds)
  - Designated observer tests (selection modes, gimbal pointing)
  - Gimbal curriculum tests (gradual randomization)
  - Velocity tests (curriculum scaling, bounds)
  - Zoom tests (range validation)
  - Integration tests (wrapper class, partial env_ids)

## Running Tests

### Run All Tests
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py
```

### Run with Verbose Output
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py --test-verbose
```

### Run with CPU Only
```bash
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/initial_states/tests/run_tests.py
```

## Test Organization

### Configuration Tests
- Default configuration values
- Custom configuration values

### Generator Tests
- Generator initialization
- Generate at progress=0.0 (easy)
- Generate at progress=1.0 (hard, no NaN/inf)
- Agent positions within cylinder bounds
- Agent clearance maintained
- Target distance bounds at progress=0
- Target distance bounds at progress=1

### Designated Observer Tests
- Fixed observer mode (always agent 0)
- Random observer mode (varies across envs)
- Observer gimbal points at target (< 1 deg error)

### Gimbal Curriculum Tests
- At progress=0, all agents point at target (>95%)
- At progress=1, agents randomized (<50% pointing)

### Velocity Tests
- Agent velocity at progress=0 (near zero)
- Agent velocity at progress=1 (within bounds)
- Target velocity at progress=0 (near zero)
- Target velocity at progress=1 (within bounds)

### Zoom Tests
- Zoom levels within configured range
- Zoom levels have variation

### Integration Tests
- Wrapper initialization
- Generate via wrapper
- Partial env_ids generation
- Config update

## Expected Results

All tests should pass with the following characteristics:
- Agent positions within cylinder (diameter/2 from center)
- Minimum 10m clearance between agents
- Target distance scales with curriculum progress
- Designated observer gimbal error < 1 degree
- Velocities scale from 0 at progress=0 to max at progress=1
- No NaN or inf values in any output

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test fixtures (default: 32).

### Import Errors
Ensure module is installed:
```bash
./isaaclab.sh -i
```

### AppLauncher Errors
Tests require Isaac Sim. Run via `./isaaclab.sh -p` not directly with Python.

## Contributing

When adding new features to initial_states:
1. Add corresponding unit tests
2. Ensure all existing tests pass
3. Update this README for new test files
4. Verify tests on both CPU and GPU
