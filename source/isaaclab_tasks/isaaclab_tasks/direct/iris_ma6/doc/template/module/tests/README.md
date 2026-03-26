# {{ModuleName}} Test Suite

Tests for the `{{MODULE_NAME}}` module.

## Test Files

- **run_tests.py**: Standalone test runner
  - Initialization and configuration
  - Core computation (simple values, then realistic values)
  - Edge cases and boundary conditions
  - Reset and state management

## Running Tests

```bash
# Run all tests
./isaaclab.sh -p path/to/{{MODULE_NAME}}/tests/run_tests.py

# Verbose output
./isaaclab.sh -p path/to/{{MODULE_NAME}}/tests/run_tests.py --test-verbose

# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p path/to/{{MODULE_NAME}}/tests/run_tests.py
```

## Expected Results

- GPU/CPU compatibility: Tests run on both devices
- Numerical accuracy: Appropriate tolerances for floating-point operations
- State management: Reset clears internal state correctly

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test functions to lower memory usage.

### Import Errors
Ensure Isaac Lab is installed: `./isaaclab.sh -i`
