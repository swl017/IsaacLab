# Distance-Based Formation Tests

Test suite for the distance-based formation generation system.

## Overview

These tests verify that the formation generator correctly:
1. Places agents and targets at curriculum-controlled distances
2. Maintains agent separation constraints
3. Places target height near mean agent height
4. Generates correct tensor shapes

## Running Tests

```bash
# Run all tests
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/randomization/tests/run_tests.py

# Run with verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/randomization/tests/run_tests.py --test-verbose

# Run on CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/randomization/tests/run_tests.py
```

## Test Categories

### Configuration Tests
- Default configuration values
- Custom configuration values

### Generator Tests
- Generator initialization
- Basic generation
- Distance constraints at scale_factor=0 (close)
- Distance constraints at scale_factor=1 (far)
- All agents within distance bounds
- Formation types (planar, grid, line)
- Target height near mean agent height
- Agent separation maintained

### Randomizer Tests
- Randomizer initialization (default and custom config)
- generate_formation_and_target method
- Partial env_ids subset generation

## Expected Output

```
================================================================================
DISTANCE-BASED FORMATION TEST SUITE
================================================================================
Device:        cuda
...

================================================================================
Testing DistanceBasedFormationCfg
================================================================================
  ✓ Default configuration values
  ✓ Custom configuration values

================================================================================
Testing DistanceBasedFormationGenerator
================================================================================
Configuration: 16 environments, 2 agents
  ✓ Generator initialization
  ✓ Basic generation (scale_factor=0.5)
  ✓ Distance constraints at scale=0 (mean=XX.Xm)
  ✓ Distance constraints at scale=1 (mean=XX.Xm)
  ✓ All agents within distance bounds (100 envs, 5 scales)
  ✓ Formation types (planar, grid, line)
  ✓ Target height near mean agent height (max diff=X.Xm)
  ✓ Agent separation maintained (min=X.Xm)

================================================================================
Testing Randomizer
================================================================================
  ✓ Randomizer initialization (default config)
  ✓ Randomizer initialization (custom config)
  ✓ generate_formation_and_target returns correct shapes
  ✓ Partial env_ids subset generation

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 14
Passed:      14 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Troubleshooting

### Import Errors
Ensure module is installed:
```bash
./isaaclab.sh -i
```

### CUDA Out of Memory
The tests use small numbers of environments. If still failing, try:
```bash
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p ...
```
