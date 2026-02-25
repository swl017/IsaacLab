# Safety Module Test Suite

This directory contains comprehensive unit and integration tests for the safety module.

## Test Files

- **test_collision_detector.py**: Tests for CollisionDetector class
  - Distance computation (basic, symmetry, diagonal, zero distance)
  - Collision detection (no collision, with collision, at threshold)
  - Collision penalties (various scenarios)
  - Velocity-based prediction
  - Reset functionality

- **test_ttc_computer.py**: Tests for TTCComputer class
  - TTC computation (various bbox scenarios)
  - Zoom invariance property
  - EMA smoothing behavior
  - Staleness decay
  - Zoom gate functionality
  - Derivative clipping
  - Reset functionality

- **test_safety_manager.py**: Tests for SafetyManager class
  - Integration between CollisionDetector and TTCComputer
  - Unified penalty computation
  - Feature enable/disable flags
  - State access methods
  - Reset coordination
  - Multi-step simulation scenarios

## Running Tests

### Run All Tests

```bash
# From repository root
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/ -v

# Or with python directly
python -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/ -v
```

### Run Specific Test File

```bash
# Test only collision detector
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/test_collision_detector.py -v

# Test only TTC computer
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/test_ttc_computer.py -v

# Test only safety manager
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/test_safety_manager.py -v
```

### Run Specific Test Class

```bash
# Test only distance computation
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/test_collision_detector.py::TestComputeDistances -v

# Test only TTC zoom invariance
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/test_ttc_computer.py::TestComputeTTC -v
```

### Run with Coverage

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/ --cov=isaaclab_tasks.direct.iris_ma3.safety --cov-report=html
```

### Run on CPU Only

```bash
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/tests/ -v
```

## Test Organization

Tests are organized into classes by functionality:

### CollisionDetector Tests
- `TestInitialization`: Basic setup and configuration
- `TestComputeDistances`: Distance matrix computation
- `TestDetectCollisions`: Collision detection logic
- `TestComputeCollisionPenalties`: Penalty computation
- `TestVelocityBasedPrediction`: Future collision prediction
- `TestClosestAgentDistance`: Minimum distance queries
- `TestReset`: State reset functionality

### TTCComputer Tests
- `TestInitialization`: Basic setup
- `TestComputeTTC`: Core TTC computation
- `TestZoomInvariance`: Zoom-invariant property
- `TestEMASmoothing`: Smoothing behavior
- `TestStalenessDecay`: Invalid detection handling
- `TestZoomGate`: Zoom motion gating
- `TestDerivativeClipping`: Numerical stability
- `TestPenaltyRange`: Output constraints
- `TestReset`: State reset functionality

### SafetyManager Tests
- `TestInitialization`: Configuration options
- `TestCollisionPenalties`: Collision penalty integration
- `TestTTCPenalties`: TTC penalty integration
- `TestUnifiedPenalties`: Combined penalty computation
- `TestStateAccess`: Matrix access methods
- `TestReset`: Coordinated reset
- `TestIntegration`: Multi-step simulation scenarios

## Expected Results

All tests should pass with the following characteristics:

- **GPU/CPU compatibility**: Tests run on both CUDA and CPU devices
- **Numerical accuracy**: Tolerances set to 1e-5 for floating-point comparisons
- **Edge case handling**: Tests verify behavior at boundaries and with invalid data
- **State management**: Reset functionality properly clears internal state
- **Integration**: Multi-step simulations produce expected behavior

## Common Issues

### CUDA Out of Memory

If you encounter CUDA OOM errors, reduce the number of environments in fixtures:

```python
@pytest.fixture
def num_envs():
    return 8  # Reduced from 16
```

### Import Errors

Ensure the safety module is properly installed:

```bash
./isaaclab.sh -i
```

### Test Failures

If tests fail, check:
1. Device availability (CUDA vs CPU)
2. Torch version compatibility
3. Numerical precision settings
4. Test isolation (state leakage between tests)

## Contributing

When adding new features to the safety module:

1. Add corresponding unit tests
2. Ensure all existing tests still pass
3. Update this README if new test files are added
4. Verify tests pass on both CPU and GPU
