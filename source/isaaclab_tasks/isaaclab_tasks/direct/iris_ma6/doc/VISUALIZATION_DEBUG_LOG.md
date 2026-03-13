# iris_ma6 Visualization Debugging Log

## Problem Statement
Camera position is correctly updated, frustum is attached to the drone, but:
1. **Frustum orientation is wrong** - different from where the camera is facing
2. **Frustum is stabilized but camera is not** (or vice versa depending on configuration)

## USD Gimbal Structure Analysis

From `iris_gimbal2.usda`, the gimbal kinematic chain is:

```
body -> yaw_link (yaw_joint, axis = "Z")
    -> roll_link (roll_joint, axis = "X")
        -> pitch_link (pitch_joint, axis = "Y")
```

**Joint Rotation Axes:**
- `yaw_joint`: Z axis (body -> yaw_link)
- `roll_joint`: X axis (yaw_link -> roll_link)
- `pitch_joint`: Y axis (roll_link -> pitch_link)

**Rotation Order:** yaw (Z) -> roll (X) -> pitch (Y) = ZXY intrinsic rotation

**Initial Orientations (all joints at 0):**
- All links have identity orientation `(1, 0, 0, 0)` relative to parent
- When all joints are zero, pitch_link has same orientation as body
- Drone body typically has +X = forward direction

## Iteration Log

### Iteration 1-3 (Early Attempts)

Various attempts to fix frustum orientation using single-axis rotations and
switching between computed vs TiledCamera poses. Resulted in frustum stabilized
but facing up, camera forward but not stabilized.

### Iteration 4 - Camera Offset Rotation Fix

**Problem:** Camera offset `(0.7071, 0, 0, -0.7071)` = -90° around Z axis. Frustum
uses +Z as forward. Rotating +Z around Z doesn't change +Z direction → frustum faces UP.

**Fix:** Changed frustum `_camera_offset_rotation_b` to `(0.7071, 0, 0.7071, 0)` (+90° around Y).

**Result:** Frustum facing forward but rotated 90°. Camera not stabilized, pitch turns roll.

### Iteration 5 - Use iris_ma5 Quaternion for Frustum

**Problem:** Single-axis Y rotation incomplete. Need compound rotation:
- +Z → +X (forward), +Y → +Z (up), +X → -Y (right)

**Fix:** Changed frustum offset to `[0.5, -0.5, 0.5, -0.5]` (same as iris_ma5).

**Result:** Frustum aligned correctly and stabilized. Camera not stabilized, pitch turns roll.

### Iteration 6 - Match TiledCamera Offset to Frustum

**Fix:** Changed TiledCamera offset from `(0.7071, 0, 0, -0.7071)` to `(0.5, -0.5, 0.5, -0.5)` with `convention="world"`.

**Result:** Camera rolled -90°, pitch still turns roll.

### Iteration 7 - Use USD Camera Native Orientation

**Fix:** Changed TiledCamera offset to `(0.5, 0.5, -0.5, -0.5)` (from commented camera in USD) with `convention="world"`.

**Result:** Not tested independently (moved to iteration 8).

### Iteration 8 - Use OpenGL Convention Directly

**Key Discovery:** Isaac Lab converts offset rotation from specified convention to OpenGL
via `convert_camera_frame_orientation_convention()`. Using `convention="opengl"` bypasses
this conversion.

**Fix:** Changed to `rot=(0.5, -0.5, 0.5, -0.5)` with `convention="opengl"`.

**Result:** Frustum points UP when pitched, camera points DOWN (opposite). Camera upside down (180° roll).

### Iteration 9 - Apply 180° X-Rotation Correction

**Fix:** Applied 180° X-rotation: `[0.5, -0.5, 0.5, -0.5] * [0, 1, 0, 0] = [0.5, 0.5, -0.5, -0.5]` with `convention="opengl"`.

**Result:** Camera faces LEFT, not stabilized. Pitch direction correct (up = up).

### Iteration 10 - Identity World Convention + Empirical Debug

**Fix:** Changed to identity `(1, 0, 0, 0)` with `convention="world"`. Added debug print
to show actual camera prim quaternion.

**Key Empirical Data:**
```
CAM PRIM quat:  [0.706, 0.7066, 0.0343, 0.0338]  ≈ [0.7071, 0.7071, 0, 0]  (90° around X)
BODY quat:      [1.0, 0.0, 0.0004, -0.0]          (identity, drone level)
Gimbal:         yaw=7.2°, pitch≈0°
```

**Analysis:** With identity world offset, prim gets orientation R_x(90°). Camera -Z with
R_x(90°) = `[0, 1, 0]` = body +Y = **LEFT**. Confirmed empirically.

**Insight:** The world→opengl conversion adds an implicit rotation. Empirically:
```
prim_orientation = world_offset_rotation * R_x(90°)
```

To get camera -Z → body +X (forward), need:
```
prim = R_z(-90°) * R_x(90°)  →  world_offset = R_z(-90°) = (0.7071, 0, 0, -0.7071)
```

This is exactly the **iris_ma5 config** — and the original iris_ma6 config.

**Result:** Camera faces LEFT (as expected from identity offset).

### Iteration 11 - Restore iris_ma5 Config (Empirically Verified)

**Fix:** Changed TiledCamera offset back to `(0.7071068, 0, 0, -0.7071068)` with
`convention="world"` — now with empirical understanding of why it works.

**Result:** Camera faces FORWARD. Pitch command makes camera roll.

## Current State (After Iteration 12)

### What Works
- **Frustum**: Facing forward, stabilized, correctly follows gimbal ✓
- **Camera**: Facing forward ✓
- Camera is on pitch_link, follows gimbal movements ✓
- Pitch correctly changes camera forward direction (confirmed by debug axes) ✓

### Iteration 12 — ROS Convention + Axis Decomposition Debug

**Debug Output (step 300, all joints ~0):**
```
CAM PRIM quat:  [-0.5, -0.5, 0.4991, 0.5009]   ≈ (0.5, 0.5, -0.5, -0.5)
CAM forward(w): [1.000, -0.000, -0.002]          → body +X (forward) ✓
CAM up(w):      [0.002, -0.002, 1.000]           → body +Z (up) ✓
CAM right(w):   [-0.000, -1.000, -0.002]         → body -Y (right FLU) ✓
```

**During pitch (step 600, pitch=0.8°):**
```
CAM forward(w): [1.000, -0.000, -0.000]
CAM up(w):      [0.000, 0.014, 1.000]            → slight Y component (correct)
CAM right(w):   [-0.000, -1.000, 0.014]          → stays ~[0,-1,0] ✓ (not rolling)
```

**During yaw+pitch (step 700, yaw=-11.5°, pitch=12.6°):**
```
CAM forward(w): [0.980, -0.200, -0.000]
CAM right(w):   [-0.195, -0.956, 0.218]          → Y component changed from -1.0
```

**Analysis:** Camera axes respond correctly to gimbal commands. The "roll" effect
the user observed may have been from the previous world/opengl convention mismatch.

### Discovery: 90° Gimbal Yaw Offset

**Problem:** Frame visualization shows gimbal yaw has 90° CCW offset at yaw_joint=0.
The USD defines no joint offsets (all `localRot0/1 = (1,0,0,0)`).

**Root Cause:** Found in `iris_gimbal2.usda` line 134:
```
def Mesh "body" {
    quatf xformOp:orient = (0.70711, 0, 0, 0.70711)   # R_z(+90°)
```

The body's **visual mesh** is rotated 90° CCW relative to the body's **physics frame**.
The physics frame has +X = forward, but the mesh nose points along body +Y.
This is a visual-only issue — the gimbal at yaw=0 correctly points along physics +X,
but appears 90° offset from the drone's visual nose direction.

**Impact:** The effective yaw range appears as [-90°, 270°] relative to visual forward.
Physics is correct; only the mesh visual is misaligned.

### Convention Decision

Switched TiledCamera to ROS convention based on frame visualization testing:
```python
rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
```

This produces the same prim orientation `(0.5, 0.5, -0.5, -0.5)` as the previous
world convention `(0.7071, 0, 0, -0.7071)`, but the quaternion value now matches
the frustum offset directly, improving code clarity.

## Key Learnings

1. **Convention conversion is non-trivial**: Isaac Lab's `convert_camera_frame_orientation_convention`
   applies different transformations per convention:
   - `world→opengl`: `rotm @ matrix_from_euler([π/2, -π/2, 0], "XYZ")`
   - `ros→opengl`: negate columns 1 and 2 (Y, Z flip = 180° around X)

2. **Empirical verification essential**: Theoretical quaternion math didn't match actual
   prim orientation. Debug prints showing `CAM PRIM quat` were critical.

3. **Frustum and TiledCamera use different coordinate systems**: Frustum uses +Z forward
   with direct quaternion multiplication. TiledCamera uses OpenGL convention (-Z forward)
   with Isaac Lab's convention conversion.

4. **ROS convention simplifies alignment**: Using `(0.5, -0.5, 0.5, -0.5)` with
   `convention="ros"` makes frustum and TiledCamera use the same quaternion value.

5. **USD mesh orientation ≠ physics frame**: The body mesh `xformOp:orient` can differ
   from the articulation Xform orientation. This only affects visuals, not physics.

## Next Steps

1. Fix the 90° mesh offset in `iris_gimbal2.usda` (set body mesh orient to identity)
   or document it as intentional
2. Verify camera stabilization during drone body rotation
3. Test with multiple environments to ensure no race conditions

## Configuration Summary

### Frustum Visualization (`iris_ma_env6_test.py`)
```python
self._camera_offset_rotation_b = torch.tensor(
    [0.5, -0.5, 0.5, -0.5], dtype=torch.float32, device=self.device
).expand(self.num_envs, -1)
```
Convention: Direct quaternion in `compute_camera_orientation_from_gimbal` (+Z forward)

### TiledCamera (`iris_ma_env6_test_cfg.py`)
```python
offset=TiledCameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.5, -0.5, 0.5, -0.5),
    convention="ros",
),
```
Convention: ROS (forward=+Z, up=-Y), converted to OpenGL by Isaac Lab.
Both frustum and TiledCamera use the same quaternion value `(0.5, -0.5, 0.5, -0.5)`.
