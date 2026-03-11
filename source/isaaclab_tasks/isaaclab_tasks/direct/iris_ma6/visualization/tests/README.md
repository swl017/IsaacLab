# Visualization Module Test Suite

This directory contains tests for the iris_ma6 visualization module.

## Test Files

- **run_tests.py**: Standalone test runner for all visualization components
  - CameraFrustum initialization and frustum computation
  - Zoom-aware FOV scaling
  - DetectionIndicator color logic (green/yellow)
  - CustomVisualization warmup period handling
  - Per-agent visualizer creation
  - Multi-target bbox_empty handling

## Running Tests

### Run All Tests

```bash
# Using standalone runner (recommended)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/visualization/tests/run_tests.py

# With verbose output
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/visualization/tests/run_tests.py --test-verbose
```

### Run with Specific Device

```bash
# CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/visualization/tests/run_tests.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/visualization/tests/run_tests.py
```

## Test Organization

### CameraFrustum Tests
- Initialization and debug draw interface acquisition
- Camera config tensor creation
- Frustum corner computation
- Zoom-dependent FOV scaling (higher zoom = narrower FOV)
- Clear method functionality

### DetectionIndicator Tests
- Initialization
- Color selection (green for detected, yellow for not detected)
- Mixed detection state handling
- Clear method functionality

### CustomVisualization Tests
- Wrapper initialization with per-agent visualizers
- Warmup period enforcement (10 frames before visualization)
- Update method safety (no crash before warmup)
- Multi-target bbox_empty dimension handling

## Expected Results

All tests should pass with output similar to:

```
================================================================================
VISUALIZATION MODULE TEST SUITE
================================================================================
Device:        cuda
Torch version: 2.x.x

================================================================================
Testing CameraFrustum
================================================================================
  ✓ CameraFrustum initialization
  ✓ create_camera_cfg_tensor
  ✓ Frustum computation (zoom=1.0)
  ✓ Frustum zoom scaling
  ✓ CameraFrustum clear method

================================================================================
Testing DetectionIndicator
================================================================================
  ✓ DetectionIndicator initialization
  ✓ Color logic - all detected (green)
  ✓ Color logic - none detected (yellow)
  ✓ Color logic - mixed detection
  ✓ DetectionIndicator clear method

================================================================================
Testing CustomVisualization
================================================================================
  ✓ CustomVisualization initialization
  ✓ Warmup period logic
  ✓ Per-agent visualizer creation
  ✓ Update before warmup (no crash)
  ✓ Update after warmup
  ✓ Multi-target bbox_empty handling

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 16
Passed:      16 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

## Visual Verification

For visual verification (requires GUI mode), run the test environment:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/test_env.py --num_envs 4
```

You should see:
- White wireframe frustums extending from each drone's camera
- Colored lines from each camera to the target:
  - **Green**: Target is detected (visible in camera)
  - **Yellow**: Target is not detected (out of view or occluded)

## Common Issues

### Isaac Sim Import Errors
Ensure the module is installed:
```bash
./isaaclab.sh -i
```

### GPU Crashes
If you experience GPU crashes:
1. Reduce `num_envs` in tests
2. Ensure warmup period is respected (MIN_FRAMES_BEFORE_VISUALIZATION = 10)
3. Run in headless mode for testing

### Debug Draw Not Appearing
- Debug draw requires GUI mode (not headless)
- Ensure `debug_vis: true` in environment config
