# CBF Safety Filter Module Test Suite

This directory contains tests and visualization tools for the CBF Safety Filter Module.

## Files

- **run_tests.py**: Standalone test runner (AppLauncher-based)
  - CBFDiagnostics tests
  - CPARewardShaper tests
  - RobustDeploymentFilter tests
  - CBFManager integration tests

- **plot_cbf_behavior.py**: Static visualization script for understanding CBF behavior
  - Head-on approach analysis
  - Parallel flight comparison
  - Crossing trajectories
  - 2D penalty landscape heatmaps
  - Time evolution during approach
  - Gamma (decay rate) effect
  - Distance-based vs CPA-based comparison

- **plot_cbf_animated.py**: Animated visualization showing CPA dynamics over time
  - Head-on approach with real-time CPA prediction
  - Evasive maneuver showing penalty reduction when turning
  - Step-by-step penalty formula explanation
  - Static explanation of why penalty converges to ~0.32

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

## Visualization Script

The `plot_cbf_behavior.py` script generates plots to build intuition about the CPA barrier.

### Run Visualization

```bash
# Generate plots to default directory (./cbf_behavior_plots)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_behavior.py

# Specify output directory
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_behavior.py --output-dir /path/to/plots

# Show plots interactively (requires display)
./isaaclab.sh -p ... --show
```

### Generated Plots

| Plot | Description |
|------|-------------|
| `01_head_on_approach.png` | Penalty vs distance/speed for head-on collision course |
| `02_parallel_flight.png` | Low penalty for parallel flight even at close range |
| `03_crossing_trajectories.png` | Penalty variation with crossing angle |
| `04_penalty_landscape.png` | 2D heatmaps showing penalty across relative positions |
| `05_time_evolution.png` | Penalty, distance, and barrier over time during approach |
| `06_gamma_effect.png` | Effect of CBF decay rate parameter |
| `07_distance_vs_cpa.png` | Key advantage of CPA: velocity awareness |

### Key Insights from Plots

1. **Velocity Awareness**: CPA barrier distinguishes approach direction
   - Parallel flight at 2.5m: ~0 penalty
   - Approaching at 2.5m: high penalty

2. **Predictive**: Catches approaching threats early via look-ahead horizon T

3. **Less Conservative**: Allows close parallel operation / formation flying

4. **Directional Gradient**: Provides clear signal for evasive maneuvers

## Animated Visualization Script

The `plot_cbf_animated.py` script generates MP4 animations showing CPA dynamics over time.

### Run Animated Visualization

```bash
# Generate animations to default directory (./cbf_animations)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_animated.py

# Specify output directory
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_animated.py --output-dir /path/to/animations

# Adjust animation settings
./isaaclab.sh -p ... --fps 30 --duration 6.0
```

### Generated Animations

| File | Description |
|------|-------------|
| `anim_01_head_on_approach.mp4` | Two drones approaching head-on, showing CPA point prediction |
| `anim_02_evasive_maneuver.mp4` | One drone turns to avoid collision, penalty drops |
| `anim_03_penalty_explained.mp4` | Step-by-step penalty computation with formula |
| `static_penalty_explanation.png` | Static diagram explaining the 0.32 convergence |
| `penalty_explanation.txt` | Text explanation of the math |

### Why Penalty Converges to ~0.32

For **head-on approach**, the CPA predicts collision (d_CPA = 0) regardless of current distance:

```
Given: D_s = 2m, γ = 2.0, Δt = 0.04s

1. CPA distance: d_CPA = 0 (head-on collision predicted)
2. Barrier value: h = d_CPA² - D_s² = 0 - 4 = -4 (negative = unsafe)
3. CBF condition:
   violation = max(0, (1 - γΔt)·h_current - h_next)
             = max(0, 0.92 × (-4) - (-4))
             = max(0, -3.68 + 4)
             = 0.32
```

The ~0.32 penalty represents the **rate** at which the barrier is becoming more negative.
This equals `γ·Δt·|h| = 2 × 0.04 × 4 = 0.32` when h is stable at -4.
