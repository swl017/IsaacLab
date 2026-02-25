# Delay System V2 Test Suite

Test suite for the `MultiAgentDelaySystemV2` wrapper which adapts the quadcopter's field-based `DelaySystemV2` architecture for multi-agent environments.

## Overview

The delay system V2 provides:
- Field-based delay pipeline with DataBus architecture
- Dual pipeline (clean for rewards, noisy for observations)
- Perspective-aware delays (fast ego, slow inter-agent)
- Scalability to arbitrary number of agents
- API compatibility with iris_ma3 delay system

## Test Files

- **run_tests.py**: Standalone test runner (Isaac Sim compatible)
  - Initialization tests
  - API compatibility tests
  - Dual pipeline tests
  - Perspective-aware delay tests
  - Reset functionality tests
  - Derived field computation tests

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended for Isaac Sim modules)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/delay_system_v2/tests/run_tests.py

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/delay_system_v2/tests/run_tests.py --test-verbose
```

### Run with Specific Device

```bash
# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/delay_system_v2/tests/run_tests.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/delay_system_v2/tests/run_tests.py
```

## Test Organization

### Initialization Tests
- Basic initialization
- Custom configuration parameters
- Camera config setup

### API Compatibility Tests
- `update_time()` method
- `update_gt_states()` method
- `update_detections()` method
- `get_all_states_for_rewards()` method
- `get_all_states_for_observations()` method
- `gt_states` property
- `current_time` property
- `set_noise_progress_scale()` method

### Dual Pipeline Tests
- Clean pipeline produces noise-free data
- Noisy pipeline produces valid data

### Perspective Tests
- All agents visible from each perspective
- Perspective separation (ego vs other agents)

### Reset Tests
- Reset without initial states
- Reset with initial states
- Time reset on environment reset

### Derived Field Tests
- Camera position computation
- Camera orientation computation (with normalization)
- Ray direction computation
- Camera intrinsics availability

## Expected Results

When all tests pass:
```
================================================================================
TEST SUMMARY
================================================================================
Total Tests: XX
Passed:      XX (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Common Issues

### Import Errors
Ensure the module is installed:
```bash
./isaaclab.sh -i
```

### CUDA Out of Memory
Reduce `num_envs` in test fixtures (default: 16).

### Missing Camera Config
Tests require camera configuration to be set before accessing derived fields like camera position/orientation.

## Contributing

When adding new features to the delay system:
1. Add corresponding unit tests
2. Ensure all existing tests pass
3. Update this README for new test categories
4. Verify tests on both CPU and GPU
