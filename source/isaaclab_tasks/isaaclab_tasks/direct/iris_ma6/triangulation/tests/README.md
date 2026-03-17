# Triangulation Module Test Suite

Test suite for the iris_ma6 triangulation and uncertainty estimation module.

## Test Files

- **run_tests.py**: Comprehensive unit test suite
  - Utility functions (skew, quaternion conversion, camera transforms)
  - Ray direction computation from bounding boxes
  - Triangulation position accuracy
  - Validity-based returns (NaN + is_valid mask)
  - Condition number and geometry checks
  - Covariance computation (including gimbal roll)
  - Full pipeline integration

- **monte_carlo_validation.py**: Monte Carlo validation for covariance estimation
  - Validates analytical covariance matches empirical statistics
  - Tests multiple camera configurations (2-3 cameras)
  - Validates gimbal roll contribution to uncertainty
  - Coverage statistics (1σ, 2σ, 3σ ellipsoid coverage)
  - Tests scenarios: orthogonal cameras, triangle formation, body tilt, far/close targets

## Running Tests

### Run Unit Tests
```bash
# Using standalone runner (recommended)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/run_tests.py

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/run_tests.py --test-verbose
```

### Run Monte Carlo Validation
```bash
# Default validation mode (10000 samples)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py

# With more samples for better statistics
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py --n-samples 20000

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py --test-verbose

# Run angle sweep experiment
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py --mode angle_sweep

# Run all experiments with plot saving
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py --mode all --save-plots

# Custom target distance for angle sweep
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/triangulation/tests/monte_carlo_validation.py --mode angle_sweep --target-distance 50.0 --save-plots
```

**Monte Carlo Modes:**
- `--mode validation`: Standard validation scenarios (default)
- `--mode angle_sweep`: Sweep viewing angle and plot uncertainty
- `--mode all`: Run all experiments

**Options:**
- `--n-samples N`: Number of MC samples (default: 10000)
- `--save-plots`: Save validation plots to file
- `--save-dir PATH`: Custom directory for plots
- `--target-distance DIST`: Target distance in meters for angle sweep (default: 20)

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

### 8. Monte Carlo Validation (monte_carlo_validation.py)
Validates that analytical covariance computation matches empirical statistics.

**Critical Assumption - Nominal vs Perturbed Parameters:**

The MC simulation models a realistic scenario where:
1. **True camera state is PERTURBED**: Position, orientation, and gimbal angles have random noise
2. **Triangulator uses NOMINAL parameters**: The system doesn't know the exact perturbations

This is the key insight: the analytical covariance formula assumes the triangulator uses
its best estimate (nominal parameters) while the true state is uncertain. The covariance
captures how triangulation error varies as the true state varies around the nominal.

```
TRUE STATE (perturbed)          SYSTEM BELIEF (nominal)
├── Position + noise      →     ├── Nominal position
├── Orientation + noise   →     ├── Nominal orientation
└── Gimbal + noise        →     └── Nominal gimbal angles
         ↓                               ↓
    Project target              Triangulate from pixels
    to image plane              using nominal params
         ↓                               ↓
      Pixel + noise       →     Triangulation estimate
                                (has error due to noise)
```

In practice, this matches real-world scenarios:
- State estimation (GPS, IMU) provides noisy estimates
- The triangulator uses these estimates without knowing exact errors
- The covariance predicts how triangulation accuracy degrades with state uncertainty

**Implementation:**
- Uses **batched computation** for efficiency: all N samples processed in parallel
- Samples ALL uncertainty sources:
  - Position noise (3D)
  - Body orientation (full 3D: roll, pitch, yaw)
  - Gimbal angles (yaw, roll, pitch)
  - Pixel detection noise (2D)
- Projects target using PERTURBED camera state (true world)
- Triangulates using NOMINAL parameters (system belief)
- Batched projection and triangulation on GPU
- Displays full 3x3 covariance matrices for detailed comparison

**Scenarios tested:**
- 2 cameras, orthogonal (with/without roll)
- 3 cameras, triangle formation
- 3 cameras with body tilt (roll stabilization active)
- Far target (large uncertainty)
- Close target (small uncertainty)

**Validation metrics:**
- Trace ratio (analytical/empirical): Expected 0.95-1.05 (close to 1.0)
- Effective dimensionality: Diagnostic metric measuring covariance anisotropy (1.0 = 1D, 3.0 = isotropic)
- Coverage statistics: % of samples within 1σ, 2σ, 3σ ellipsoids
- Expected coverage: Always based on chi²(3) for 3D covariance

**Coverage Statistics:**

For a 3D multivariate Gaussian, the squared Mahalanobis distance follows a chi-squared
distribution with DOF = 3, **regardless of the eigenvalue anisotropy**. This means:

| k-sigma | Expected Coverage |
|---------|-------------------|
| 1σ      | 19.9%             |
| 2σ      | 73.9%             |
| 3σ      | 97.1%             |

These values are computed using P(χ²(3) ≤ k²).

**Effective Dimensionality (Diagnostic Only):**

The **effective dimensionality** is computed using the participation ratio:
```
eff_dim = (Σ λᵢ)² / Σ λᵢ²
```
- eff_dim ≈ 1.0: Highly anisotropic (uncertainty mainly along one axis)
- eff_dim ≈ 3.0: Isotropic (equal uncertainty in all directions)

This measures the *shape* of the covariance ellipsoid but does NOT affect the
chi-squared DOF for coverage computation. It's a useful diagnostic to understand
the geometry of the triangulation uncertainty.

**Note:** With the correct nominal/perturbed separation, analytical and empirical
covariances match closely (trace ratio within ±5%). Small deviations may occur due to:
- First-order Jacobian approximation (small for small noise levels)
- Finite sample effects in Monte Carlo
- Numerical precision in projection/triangulation

### 9. Angle Sweep Experiment
Sweeps viewing angle between cameras and plots uncertainty vs angle.

**Purpose:** Verify that uncertainty decreases with wider viewing angles
(better triangulation geometry) and that analytical predictions track MC results.

**Expected behavior:**
- Minimum uncertainty at **90° viewing angle** (optimal triangulation geometry)
- Uncertainty increases as viewing angle decreases (rays become more parallel)
- Analytical and MC results should track closely (deviation < ±5%)

**Configuration:**
- Two cameras placed symmetrically around target
- Viewing angles: 10°, 20°, 30°, 45°, 60°, 75°, 90°
- Configurable target distance (default: 20m)

**Output:**
- Plot showing MC and analytical uncertainty vs viewing angle
- Deviation percentage between MC and analytical
- Summary table with all results
- Minimum uncertainty marked at 90° (correct geometry)

### 10. Validation Plots
When `--save-plots` is specified, generates:

- **validation_summary.png**: Bar charts showing:
  - Trace ratio (analytical/empirical) for each scenario
  - 2σ coverage percentage with pass/fail coloring
  - Reference lines for expected values and thresholds

- **angle_sweep.png**: Line plot showing:
  - MC uncertainty vs viewing angle (left axis)
  - Analytical uncertainty vs viewing angle (left axis)
  - Deviation percentage bars (right axis)

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

### MC Validation Shows 2-3x Trace Ratio
If the MC validation shows analytical covariance 2-3x larger than empirical:
- **Root cause**: MC is using perturbed params for both projection AND triangulation
- **Expected behavior**: Use perturbed params for projection (true state), nominal params for triangulation (system belief)
- The analytical formula assumes the triangulator doesn't know the true perturbations
- Check that `run_monte_carlo_batched()` separates nominal and perturbed parameters

### Minimum Uncertainty at Wrong Angle
If angle sweep shows minimum uncertainty at 45° instead of 90°:
- This is a symptom of the nominal/perturbed parameter issue above
- With correct separation, minimum should be at 90° (optimal geometry)

## Contributing

When adding new features to the triangulation module:
1. Add corresponding tests to `run_tests.py`
2. Ensure all existing tests pass
3. Update this README for new test categories
4. Verify tests on both CPU and GPU
