# Triangulation Module Test Suite

Test suite for the iris_ma6 triangulation and uncertainty estimation module.

## Test Files

- **run_tests.py**: Comprehensive test suite
  - Utility functions (skew, quaternion conversion, camera transforms)
  - Ray direction computation from bounding boxes
  - Triangulation position accuracy
  - Validity-based returns (NaN + is_valid mask)
  - Condition number and geometry checks
  - Covariance computation
  - Full pipeline integration

## Running Tests

### Run All Tests
```bash
# Using standalone runner (recommended)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/run_tests.py

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/run_tests.py --test-verbose
```

### Run with CPU Only
```bash
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/run_tests.py
```

## Test Categories

### 1. Utility Functions
- Skew symmetric matrix construction
- Quaternion to rotation matrix conversion
- Camera transform building (fusing ego pose + gimbal)

### 2. Ray Direction Computation
- Center pixel gives correct forward ray
- Off-center pixels give different directions
- Gimbal rotation affects ray direction

### 3. Triangulation Position Accuracy
- Orthogonal cameras (midpoint triangulation)
- Known target position triangulation
- Multiple targets triangulation

### 4. Validity-Based Returns
- Single camera returns is_valid=False
- Behind camera detection
- Mixed validity per target
- Zero valid cameras returns invalid

### 5. Condition Number and Geometry
- Good geometry (90° baseline) gives low condition number
- Poor geometry (parallel rays) detection
- Configurable condition threshold

### 6. Covariance Computation
- Basic covariance computation
- Pose uncertainty increases covariance
- Quality metric options (trace, det, max_eig, sqrt_trace)
- Invalid covariance returns NaN

### 7. Full Pipeline Integration
- Full pipeline with bbox input
- Full pipeline with GT positions

## Expected Results

- All triangulation positions should be accurate within 1e-3 tolerance
- Covariance matrices should be symmetric and positive definite
- Invalid cases should return NaN with is_valid=False
- Quality metrics should be positive for valid entries

## Common Issues

### CUDA Out of Memory
Tests use small batch sizes by default. If OOM occurs, run on CPU.

### Import Errors
Ensure the triangulation module is in the Python path:
```bash
./isaaclab.sh -i
```

## Contributing

When adding new features to the triangulation module:
1. Add corresponding tests to `run_tests.py`
2. Ensure all existing tests pass
3. Update this README for new test categories
4. Verify tests on both CPU and GPU
