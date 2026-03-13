# CBF Safety Filter Module Test Suite

This directory contains tests for the CBF Safety Filter Module components.

## Test Files

- **run_tests.py**: Standalone test runner (AppLauncher-based)
  - CBFDiagnostics tests
  - CPARewardShaper tests
  - RobustDeploymentFilter tests
  - CBFManager integration tests

## Running Tests

### Run All Tests

```bash
# Recommended: standalone runner
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/run_tests.py

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/run_tests.py --test-verbose

# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/run_tests.py
```

## Test Categories

### CBFDiagnostics Tests
- Initialization
- Pairwise distance computation
- Metrics computation (collision/no collision scenarios)

### CPARewardShaper Tests
- Initialization with config parameters
- Stationary drones (no penalty expected)
- Approaching drones (penalty expected)
- Parallel flight (minimal penalty expected)
- CPA info debugging method
- Reset functionality

### RobustDeploymentFilter Tests
- Initialization with D_deploy computation
- No constraint when drones far apart
- Constraint active when drones close
- Conservative behavior with unknown neighbor velocities
- Barrier values computation
- Reset functionality

### CBFManager Tests
- Training mode initialization
- Deployment mode initialization
- Training penalty computation
- Collision detection (GT-based)
- Diagnostics integration
- Filter actions (deployment mode)
- Property accessors
- Reset functionality

## Expected Results

All tests should pass with:
- GPU/CPU compatibility
- Correct penalty behavior (positive for approach, minimal for parallel flight)
- Proper collision detection
- Filter activation when constraints violated
- D_deploy ≈ 9.5m (with default config)

## Common Issues

### Import Errors
Ensure module is installed:
```bash
./isaaclab.sh -i
```

### CUDA Out of Memory
Tests use 16 environments by default. Reduce if needed.

### Tests Timing Out
Increase timeout or run with fewer environments.
