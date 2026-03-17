# Target Controller Test Suite

This directory contains tests for the physics-based target controller module.

## Overview

The target controller provides physics-realistic target movement using the same
DroneController architecture as agents. These tests validate:

- Configuration loading and parameter handling
- Behavior FSM state management and transitions
- Velocity generation for all modes (linear, circular, approach, evade)
- Curriculum scaling functionality
- Constraint application (geofencing, altitude)

## Test Files

- **run_tests.py**: Standalone test runner (recommended for Isaac Sim modules)
  - Configuration tests
  - FSM initialization and state management
  - Velocity generator output validation
  - Curriculum scaling verification

## Running Tests

### Run All Tests (Recommended)

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/target_controller/tests/run_tests.py
```

### Run with Verbose Output

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/target_controller/tests/run_tests.py --test-verbose
```

### Run on CPU Only

```bash
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/target_controller/tests/run_tests.py
```

## Test Organization

### Configuration Tests

- Configuration instantiation
- Behavior profile definitions
- Parameter validation

### FSM Tests

- FSM initialization with correct shapes
- Reset behavior (all alive, APPROACH state)
- Alive indices retrieval
- State-based filtering

### Velocity Generator Tests

- **Linear Mode**: Random direction generation, timer-based updates
- **Circular Mode**: Orbit tracking, phase updates
- **Approach Mode**: Goal-directed velocity with path variants
- **Evade Mode**: Reactive evasion direction computation

### Curriculum Tests

- Max speed scaling (3 -> 12 m/s)
- Update interval scaling (longer -> shorter)
- Geofence size scaling (50 -> 200 m)

## Expected Results

All tests should pass with output like:

```
================================================================================
TARGET CONTROLLER TEST SUITE
================================================================================
Device:        cuda
Torch version: 2.x.x
...

================================================================================
Testing Configuration
================================================================================
  ✓ Configuration instantiation
  ✓ Behavior profiles defined

================================================================================
Testing Behavior FSM
================================================================================
  ✓ FSM initialization
  ✓ FSM reset
  ✓ Get alive indices
  ✓ Get state indices

================================================================================
Testing Velocity Generators
================================================================================
  ✓ Linear mode generator
  ✓ Circular mode generator
  ✓ Approach mode generator
  ✓ Evade mode generator
  ✓ Linear mode direction changes

================================================================================
Testing Curriculum Scaling
================================================================================
  ✓ Max speed scaling
  ✓ Update interval scaling
  ✓ Geofence scaling

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 13
Passed:      13 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Common Issues

### Import Errors

If you see import errors, ensure the module is properly installed:

```bash
./isaaclab.sh -i
```

### CUDA Out of Memory

Reduce `num_envs` in test fixtures if you encounter memory issues.

### DroneController Import Errors

The target controller depends on the iris_ma6 controller module. Ensure
it exists at `iris_ma6/controller/`.

## Contributing

When adding new features to the target controller:

1. Add corresponding tests to `run_tests.py`
2. Ensure all existing tests pass
3. Update this README for new test categories
4. Verify tests on both CPU and GPU

## Reference

- [target_movement_spec.md](../../doc/target_movement_spec.md): Full specification
- [CLAUDE.md](../../../../../CLAUDE.md): Test writing guidelines
