# Domain Randomization Test Suite

Comprehensive test suite for the domain randomization module used in iris_ma6.

## Overview

This test suite validates the domain randomization components:
- **CameraProcessor**: GPU-accelerated crop/resize pipeline for computational FOV/resolution simulation
- **PhysicsRandomizer**: Mass and material property randomization
- **GimbalRandomizer**: Gimbal joint offset and dynamics randomization
- **DomainRandomizer**: Main orchestrator coordinating all components

## Test Files

- **run_tests.py**: Standalone test runner with all test categories
  - Configuration instantiation and defaults
  - Camera processing (crop grids, image processing, intrinsic matrices)
  - Physics randomization (mass scales, material parameters)
  - Gimbal randomization (joint offsets, dynamics)
  - Integration tests (full orchestrator workflow)

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended - no Isaac Sim required)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/domain_randomization/tests/run_tests.py
```

### Run with Verbose Output

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/domain_randomization/tests/run_tests.py --test-verbose
```

### Run on CPU Only

```bash
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/domain_randomization/tests/run_tests.py
```

## Test Organization

### Configuration Tests
- Verify default configuration values
- Test nested configuration structures
- Validate configuration inheritance

### Camera Processing Tests
- **Initialization**: CameraProcessor creation with various configs
- **Randomization**: FOV scale and focal length sampling within ranges
- **Crop Grid Computation**: Symmetric crop grid building for centered principal points
- **Image Processing**: Batched grid_sample crop+resize pipeline
- **Intrinsic Matrices**: Correct f_x, f_y, c_x, c_y computation after crop/resize

### Physics Randomization Tests
- **Initialization**: PhysicsRandomizer creation
- **Mass Sampling**: Body mass scale, additions, and payload within configured ranges
- **Material Sampling**: Friction and restitution coefficient sampling
- **Distribution Types**: Uniform, log-uniform, and Gaussian distributions

### Gimbal Randomization Tests
- **Initialization**: GimbalRandomizer creation
- **Joint Offset Sampling**: Yaw, pitch, roll offsets within ranges
- **Dynamics Sampling**: Stiffness and damping scale factors
- **Per-Environment Storage**: Correct tensor shapes (num_envs, num_agents, 3)

### Integration Tests
- **Orchestrator Creation**: DomainRandomizer initializes all components
- **Full Randomization**: randomize_all() calls all component randomizers
- **State Summary**: get_current_state_summary() returns valid statistics

## Expected Results

- **GPU/CPU Compatibility**: Tests run on both CUDA and CPU devices
- **Numerical Accuracy**: Floating-point tolerances (atol=1e-5 for most comparisons)
- **Range Validation**: Sampled values within configured min/max ranges
- **Shape Correctness**: Tensor shapes match expected dimensions
- **Centered Principal Points**: c_x = W/2, c_y = H/2 after symmetric crops

## Test Parameters

Default test configuration:
- `num_envs`: 16 parallel environments
- `num_agents`: 3 agents per environment
- `device`: CUDA if available, else CPU

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test fixtures if GPU memory is limited.

### Import Errors
Ensure the module is installed:
```bash
./isaaclab.sh -i
```

### Floating-Point Tolerances
Some tests use relaxed tolerances for GPU computations. If tests fail due to small numerical differences, check tolerance values.

## Contributing

When adding new features to the domain randomization module:
1. Add corresponding unit tests to `run_tests.py`
2. Ensure all existing tests pass
3. Update this README for new test categories
4. Verify tests on both CPU and GPU
5. Document any new configuration options

## File Structure

```
domain_randomization/
├── __init__.py
├── domain_randomization_cfg.py
├── camera_processor.py
├── physics_randomizer.py
├── gimbal_randomizer.py
├── domain_randomizer.py
└── tests/
    ├── __init__.py
    ├── run_tests.py
    └── README.md
```
