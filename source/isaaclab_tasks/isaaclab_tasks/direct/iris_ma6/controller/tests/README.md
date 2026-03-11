# iris_ma6 Controller Test Suite

This test suite validates the iris_ma6 controller module, which provides a realistic quadcopter control system with rotor-level physics, cascaded control, and configurable aerodynamic effects.

## Test Files

- **run_tests.py**: Standalone test runner that executes all controller unit tests
  - MixerMatrix tests (8 tests)
  - MotorDynamics tests (7 tests)
  - ZoomController tests (4 tests)
  - GimbalController tests (4 tests)
  - AttitudeController tests (4 tests)
  - VelocityController tests (4 tests)
  - AerodynamicEffects tests (4 tests)
  - DroneController integration tests (4 tests)

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py --headless

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py --headless --test-verbose

# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/run_tests.py --headless
```

### Test Output

Test results are displayed in the console with clear pass/fail indicators:

```
================================================================================
Testing MixerMatrix
================================================================================
  ✓ Initialization
  ✓ Aggregate - equal thrusts
  ✓ Aggregate - roll moment generation
  ...

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 39
Passed:      39 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Test Organization

### MixerMatrix Tests
- Initialization and matrix construction
- Thrust aggregation (equal thrusts, roll/pitch/yaw moments)
- Thrust allocation
- Round-trip verification (allocate → aggregate)
- Thrust-to-omega conversion

### MotorDynamics Tests
- Initialization
- First-order step response (63% at t=tau with tau=0.02s)
- Saturation limits (omega_min, omega_max)
- Thrust/torque computation (T = k_f * omega^2)
- Body wrench aggregation
- Reset functionality

### ZoomController Tests
- Initialization at 1x zoom
- First-order response (tau=0.1s)
- Range limits ([1x, 30x])
- FOV computation (FOV = base_fov / zoom)

### GimbalController Tests
- Initialization
- Rate integration
- Joint limits enforcement
- First-order dynamics (tau=0.05s)

### AttitudeController Tests
- Initialization with mixer
- Identity quaternion (zero error)
- Roll error generates corrective moment
- Quaternion attitude error computation

### VelocityController Tests
- Initialization with mass and gravity
- Hover (zero velocity → identity attitude + hover thrust)
- Forward velocity command generates pitch
- Integral anti-windup

### AerodynamicEffects Tests
- Level 0 (disabled) returns zero forces
- Level 1 (basic drag) opposes motion
- Fidelity level switching
- Reset clears gust state

### DroneController Integration Tests
- Full system initialization
- Hover step (outputs correct force/torque)
- Inner loop timing (control_dt substeps)
- Reset propagates to all components

## Test Criteria

| Component | Test | Criteria |
|-----------|------|----------|
| Motor | Step response | 63% in 0.02s (tau_motor) |
| Gimbal | Step response | 63% in 0.05s (tau_gimbal) |
| Zoom | Step response | 63% in 0.1s (tau_zoom) |
| Zoom | Range | Clamped to [1x, 30x] |
| Attitude | Identity | Near-zero moment output |
| Velocity | Hover | Identity attitude, m*g thrust |
| Integration | Hover | Positive Z force |

## Expected Results

- All tests should pass on both CUDA and CPU
- First-order systems should reach 63.2% at t=tau (using exact discretization)
- Saturation limits should be enforced for motors, gimbal, and zoom
- Aerodynamic effects should be configurable via fidelity levels

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test functions (default is 16).

### Import Errors
Ensure the module is properly installed:
```bash
./isaaclab.sh -i
```

### Numerical Precision
Tests use appropriate tolerances (typically `atol=1e-5` for floating-point comparisons).

## Contributing

When adding new features to the controller module:
1. Add corresponding unit tests to `run_tests.py`
2. Follow the existing test pattern (try/except with TestResults)
3. Include both simple and realistic test cases
4. Verify tests pass on both CPU and GPU
5. Update this README with new test descriptions
