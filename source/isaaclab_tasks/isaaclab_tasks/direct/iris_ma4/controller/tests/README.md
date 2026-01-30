# Controller Tests

This directory contains test scripts for the `iris_ma3` controller module (point mass controller).

## Available Tests

### test_acceleration_sensor.py

Tests the accuracy of linear acceleration measurements from the robot sensors.

**Purpose:**
- Verify that `robot.data.body_lin_acc_w` reports correct acceleration values
- Check that applied forces produce expected accelerations (F = ma)
- Validate post-reset acceleration readings

**Usage:**
```bash
# Run with default settings (1 env, 200 steps)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_acceleration_sensor.py

# Run with custom settings
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_acceleration_sensor.py \
    --num_envs 4 \
    --num_steps 300 \
    --headless
```

**Expected Output:**
- Measured acceleration should match F/m (within ~5%)
- For 10N force on 1.619kg robot: expected ~6.17 m/s²
- Velocity should increase linearly over time
- Post-reset values should be correct

**Test Configuration:**
The test uses the quadcopter environment with `TEST_ACCELERATION_SENSOR = True` mode enabled,
which applies a fixed 10N force in the X-axis and logs acceleration measurements every 10 steps.

---

### test_gimbal_actuation.py

Tests gimbal joint actuation and servo response.

**Purpose:**
- Verify gimbal joints (yaw, pitch, roll) respond correctly to position commands
- Measure tracking accuracy between commanded and actual joint positions
- Validate gimbal stabilizer roll computation (horizon leveling)

**Usage:**
```bash
# Run with default settings (1 env, 500 steps)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_gimbal_actuation.py

# Run with custom settings
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_gimbal_actuation.py \
    --num_envs 1 \
    --num_steps 500 \
    --headless
```

**Expected Output:**
- Yaw/pitch tracking error: < 5.7° mean error
- Roll stabilization: < 11.5° mean deviation from zero
- Gimbal should smoothly follow sinusoidal commands

**Test Configuration:**
Applies sinusoidal position commands to gimbal joints (±90° yaw, ±45° pitch) and measures tracking performance.
Requires robot with `yaw_joint`, `pitch_joint`, and `roll_joint` (e.g., IRIS_GIMBAL2_CFG).

---

### test_drone_stabilization_with_gimbal.py

Tests base body stabilization under gimbal actuation.

**Purpose:**
- Verify base body controller remains stable during gimbal motion
- Check for adverse coupling between gimbal actuation and base control
- Validate hybrid control approach (PhysX API + Articulation API)

**Usage:**
```bash
# Run with default settings (1 env, 500 steps)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_drone_stabilization_with_gimbal.py

# Run with custom settings
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_drone_stabilization_with_gimbal.py \
    --num_envs 1 \
    --num_steps 500 \
    --headless
```

**Expected Output:**
- Attitude stability: roll/pitch < 5.7° mean error
- Velocity tracking: < 0.3 m/s mean error for 1 m/s forward command
- Angular velocity: < 1.0 rad/s mean (stable flight)
- Gimbal motion should NOT disturb base body control

**Test Configuration:**
Commands constant 1 m/s forward velocity while simultaneously actuating gimbal with aggressive sinusoidal motion.
This validates that the hybrid control approach (direct PhysX forces for base, actuator API for gimbal) works without interference.

---

### test_camera_frustum_visualization.py

Tests camera frustum visualization during gimbal motion.

**Purpose:**
- Verify camera frustum is drawn correctly when GUI is enabled
- Ensure frustum follows gimbal yaw/pitch motion
- Validate camera pose computation from robot state and gimbal angles
- Check for errors during visualization updates

**Usage:**
```bash
# Must run with GUI enabled (visualization requires display)
timeout 30 python3 source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/controller/tests/test_camera_frustum_visualization.py
```

**Expected Output:**
- White frustum wireframe visible in viewport
- Frustum smoothly follows gimbal motion
- Update success rate ≥ 95%
- No rendering errors

**Visual Verification:**
When running, you should observe:
- Camera frustum rendered as white wireframe in the 3D viewport
- Frustum orientation changes as gimbal sweeps through yaw/pitch angles
- Frustum remains attached to robot (follows drone position)

**Test Configuration:**
Commands hovering (zero velocity) while actuating gimbal with sinusoidal sweep (2 full cycles).
Camera frustum visualization uses Isaac Sim debug draw interface to render frustum geometry.

**Note:** This test requires GUI mode and cannot run in headless mode.

---

## Test Structure

Tests follow the Isaac Lab convention:
- Each test is a standalone script with `AppLauncher` setup
- Command-line arguments for configuration
- Clear pass/fail criteria with console output
- Can run in headless mode for CI/CD integration

## Adding New Tests

To add a new test:

1. Create `test_<feature_name>.py` in this directory
2. Follow the template from existing tests:
   - Import `AppLauncher` and parse args
   - Create test environment/controller
   - Run test procedure with clear logging
   - Report results with pass/fail criteria
3. Update this README with test description and usage

## Related Documentation

- [API_REFERENCE.md](../API_REFERENCE.md) - Complete controller API
- [USAGE_GUIDE.md](../USAGE_GUIDE.md) - Integration examples and troubleshooting
- [point_mass.py](../point_mass.py) - Controller implementation
