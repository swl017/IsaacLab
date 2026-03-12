# Frame and Coordinate Conventions

**Inherited from**: iris_ma5 (V5)
**Applies to**: iris_ma6 (V6)

This document defines all coordinate frames, transformations, and sign conventions used in the multi-agent drone observation/interception environment.

---

## 1) World Frame (Global Reference)

**Convention: ENU (East-North-Up)**

| Axis | Direction | Description |
|------|-----------|-------------|
| **X** | East | Right |
| **Y** | North | Forward |
| **Z** | Up | Opposite to gravity |

**Properties:**
- Isaac Sim default global frame
- Gravity acts in **-Z** direction
- All world-frame quantities use this convention
- Origin: Environment-specific (typically facility center in V6)

```
World Frame (ENU):

        Z (Up)
        ^
        |
        |
        |
        +-------> Y (North)
       /
      /
     v
    X (East)

    Gravity: -Z direction
```

---

## 2) Body Frame (Drone/Aircraft Reference)

**Base Convention: FLU (Forward-Left-Up)**

| Axis | Direction | Description |
|------|-----------|-------------|
| **X** | Forward | Roll axis |
| **Y** | Left | Pitch axis |
| **Z** | Up | Yaw axis |

```
Body Frame (FLU):

              Z (Up)
              ^
              |
              |
    Y (Left) <+-------> X (Forward)

    Top-down view (looking from above):

              X (Forward)
              ^
              |
              |
    Y (Left) <+

    Note: Rotations follow right-hand rule around each axis
          Roll: rotate around X (forward)
          Pitch: rotate around Y (left)
          Yaw: rotate around Z (up)
```

### 2.1 IRIS USD Frame Flip (CRITICAL)

The IRIS USD model has a **flipped body frame** — 180° roll relative to standard FLU convention.

```python
# From point_mass.py (lines 366-370)
# "Upright" configuration = 180° roll, not 0°
desired_quat_w = quat_from_euler_xyz(
    torch.full(..., math.pi, ...),  # roll = 180°
    torch.zeros(...),               # pitch = 0
    torch.zeros(...)                # yaw = 0
)
```

**Implication:** The physical representation has inverted X and Y axes relative to standard FLU. The control code treats 180° roll as the "level" configuration.

### 2.2 Body Mesh Visual Offset (CRITICAL)

The IRIS body **mesh** (not the Xform/physics frame) has a 90° CCW rotation around Z:

```
USD: body/mesh → xformOp:orient = (0.70711, 0, 0, 0.70711) = R_z(90°)
```

This is **visual-only** — the physics frame remains identity. The result:

| Direction | Physics Frame | Visual (Mesh) |
|-----------|--------------|---------------|
| **Forward** | Body +X | Body +Y |
| **Right** | Body -Y | Body +X |

```
Top-down view:

    Physics frame:          Visual (mesh):

         +X (fwd)                +Y (visual fwd)
          ^                       ^
          |                       |
    +Y <--●                 +X <--●
    (left)                   (visual right → physics fwd)
```

All gimbal and camera offsets must account for this 90° difference between
physics forward (+X) and visual forward (+Y). See §4.5 and §5.3.

### 2.3 Transformations

| Transformation | Operation |
|----------------|-----------|
| World → Body | Multiply vector by `q_body^{-1}` (inverse quaternion) |
| Body → World | Multiply vector by `q_body` |

---

## 3) Quaternion Convention

**Format: wxyz (scalar-first)**

$$q = (w, x, y, z) = \left(\cos\frac{\theta}{2},\ \sin\frac{\theta}{2} \cdot \hat{n}\right)$$

| Component | Description |
|-----------|-------------|
| w | Scalar part: $\cos(\theta/2)$ |
| x, y, z | Vector part: $\sin(\theta/2) \cdot \text{axis}$ |

**Used consistently throughout:**
- All `curr_quat_w` function signatures
- Isaac Lab utilities (`quat_mul`, `quat_rotate`, `quat_rotate_inverse`)
- Isaac Sim internal representation

### 3.1 Key Operations

```python
# Rotate vector v by quaternion q
v_rotated = quat_rotate(q, v)           # q * v * q^{-1}

# Inverse rotation (frame transformation)
v_rotated_inv = quat_rotate_inverse(q, v)  # q^{-1} * v * q
```

### 3.2 Double Cover Handling

When computing orientation error, ensure shortest-path rotation:

```python
# From point_mass.py (lines 380-382)
q_error = quat_mul(q_desired, quat_conjugate(q_current))
if q_error[..., 0] < 0:  # w component negative
    q_error = -q_error   # flip to shortest path
```

---

## 4) Gimbal Frame and Angles

### 4.1 Gimbal Base Frame

- **Coordinate system**: Same as body frame (FLU)
- **Mounted on**: Drone body (fixed attachment point)
- **Rotation order**: **Yaw → Roll → Pitch** (ZXY intrinsic)

### 4.2 Gimbal Angle Definitions

#### Gimbal Yaw (α)

| Property | Value |
|----------|-------|
| Rotation axis | **Z-axis (up)** |
| Range | [-200°, +200°] = [-3.49, +3.49] rad |
| Sign convention | **CCW positive** (viewed from above) |
| Computation | `atan2(y_gimbal, x_gimbal)` |
| Effect | Left/right camera pan |

#### Gimbal Pitch (β)

| Property | Value |
|----------|-------|
| Rotation axis | **Y-axis (left)** after yaw and roll |
| Range | [-70°, +70°] = [-1.22, +1.22] rad |
| **Sign convention** | **DOWN is POSITIVE** (pitch > 0 looks down) |
| Computation | `atan2(-z_gimbal, horizontal_dist)` |

**Critical:** This sign convention is inverted from some other systems. Always verify:
- pitch = 0° → horizontal view
- pitch = +30° → looking 30° downward
- pitch = -45° → looking 45° upward

```
Gimbal Pitch Sign Convention:

              Up (Z+)
               |
               |  Target (above)
               | /
               |/ pitch < 0 (looking up)
    Agent ----●---- Horizontal (pitch = 0)
               |\
               | \ pitch > 0 (looking down)
               |  \
               |   Target (below)
```

#### Gimbal Roll (ρ)

| Property | Value |
|----------|-------|
| Rotation axis | **X-axis (forward)** after yaw |
| Range | [-45°, +45°] = [-0.785, +0.785] rad |
| Computation | Auto-computed for horizon stabilization |

**Stabilization formula** (from `gimbal_stabilizer.py`, line 273):
```python
stabilizing_roll = atan2(-up_y_yawed, up_z_yawed / (cos_pitch + ε))
```

### 4.3 Gimbal Rotation Composition

```python
# ZXY intrinsic rotation order
# From derived_field_computers.py

q_gimbal = q_yaw * q_roll * q_pitch

# Final camera orientation in world frame
q_camera_world = q_body * q_gimbal * q_camera_offset
```

### 4.4 Joint Position Storage

Joint positions are stored as `[pitch, yaw, roll]` (indices 0, 1, 2).

### 4.5 Yaw Joint Offset (Software-Only)

Because the body mesh is rotated 90° CCW (see §2.2), gimbal yaw=0 must align
with visual forward (body +Y), not physics forward (body +X).

**Approach:** The gimbal controller works internally in body +X forward convention.
The offset is applied **only in env code** when setting joint targets:

```python
# gimbal_controller.py — exported constant
YAW_JOINT_OFFSET = -math.pi / 2   # -90°

# iris_ma_env6_test.py — single place offset is applied
gimbal_yaw_joint = gimbal_yaw + YAW_JOINT_OFFSET
robot.set_joint_position_target(
    target=torch.stack([gimbal_pitch, gimbal_yaw_joint, gimbal_roll], dim=-1),
    joint_ids=[pitch_idx, yaw_idx, roll_idx],
)
```

**Data flow:**

```
Controller output (body +X fwd)     Env code adds offset        Physics joint
   gimbal_yaw = 0            →    yaw_joint = -π/2        →   joint settles to -π/2
   gimbal_yaw = +π/2 (left)  →    yaw_joint = 0           →   joint settles to 0
```

**Consequence for derived fields:** When reading `robot.data.joint_pos`, the yaw
value already includes the offset. `compute_camera_orientation_from_gimbal()`
uses joint positions as-is — no additional offset needed.

---

## 5) Camera Frame

**Convention: OpenGL Standard**

| Axis | Direction | Description |
|------|-----------|-------------|
| **X** | Right | In image plane |
| **Y** | Down | In image plane (opposite to typical image coords) |
| **Z** | Forward | Into scene (depth) |

```
Camera Frame (OpenGL) - looking INTO the camera (from scene toward lens):

    +-------> X (Right)
    |
    |
    |
    v
    Y (Down)

    Z (Forward/Depth) points INTO the scene (away from viewer)

    Image plane (what the camera sees):
    +-------------------+
    | (0,0)       (W,0) |  ← top-left origin
    |                   |
    |    u increases →  |
    |    v increases ↓  |
    |                   |
    | (0,H)       (W,H) |  ← bottom-right
    +-------------------+

    Camera optical axis = +Z direction (looking forward into scene)
```

### 5.1 Camera Position in World Frame

$$p_{\text{cam}}^w = p_{\text{body}}^w + R(q_{\text{body}}) \cdot p_{\text{cam\_offset}}^b$$

Where:
- $p_{\text{body}}^w$: Body position in world frame
- $q_{\text{body}}$: Body orientation (wxyz)
- $p_{\text{cam\_offset}}^b$: Camera offset in body frame

### 5.2 Camera Orientation in World Frame

$$q_{\text{cam}}^w = q_{\text{body}} \otimes q_{\text{gimbal}}(\alpha, \beta, \rho) \otimes q_{\text{cam\_offset}}$$

### 5.3 TiledCamera Offset

The TiledCamera is mounted on the gimbal's `pitch_link`. Its offset rotation
maps the camera sensor frame to the gimbal link frame:

```python
# iris_ma_env6_test_cfg.py
offset=TiledCameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.5, -0.5, 0.5, -0.5),   # wxyz
    convention="ros",
)
```

**ROS convention:** Camera forward = +Z, camera up = -Y.
Isaac Lab internally converts this to OpenGL convention for rendering.

This quaternion was determined empirically via frame visualization to ensure the
TiledCamera image matches the expected view direction (body +Y = visual forward).

### 5.4 Frustum Visualization Offset

The frustum visualization uses a **separate** offset rotation (`_camera_offset_rotation_b`)
that is composed with the gimbal quaternion in `compute_camera_orientation_from_gimbal()`.

The frustum geometry uses +Z as forward. Combined with the gimbal yaw joint
(which includes `YAW_JOINT_OFFSET = -π/2`), the offset must map frustum +Z
to body +Y (visual forward) at gimbal-zero:

```python
# iris_ma_env6_test.py
self._camera_offset_rotation_b = torch.tensor(
    [0.7071, 0.0, -0.7071, 0.0],  # R_y(-90°), wxyz
    ...
)
```

**Derivation:**

```
At gimbal-zero, physics yaw joint = YAW_JOINT_OFFSET = -π/2

Chain: gimbal_quat * camera_offset
     = R_z(-90°) * R_y(-90°)

Verification:
  R_y(-90°) maps  +Z → -X
  R_z(-90°) maps  -X → +Y  ✓  (visual forward)
```

**Why different from TiledCamera offset?** The TiledCamera uses ROS convention
(Isaac Lab applies additional internal transforms). The frustum offset is used
directly as a quaternion in `quat_mul()` without convention conversion.

---

## 6) Image/Pixel Coordinate System

### 6.1 2D Bounding Box Format

**Format: (cx, cy, w, h)** — center x, center y, width, height

| Coordinate | Range | Description |
|------------|-------|-------------|
| cx, cy | [0, 1] | Normalized center coordinates |
| w, h | [0, 1] | Normalized width and height |

**Image origin**: Top-left corner
- x increases rightward
- y increases downward

### 6.2 Intrinsics Matrix (K)

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

| Parameter | Description |
|-----------|-------------|
| $f_x, f_y$ | Focal lengths in pixels (zoom-dependent) |
| $c_x, c_y$ | Principal point (image center, typically W/2, H/2) |

**Zoom scaling:**
$$f_{\text{zoomed}} = f_{\text{base}} \times \text{zoom\_level}$$

Principal point $(c_x, c_y)$ is unchanged by zoom.

### 6.3 Projection Formula

$$u = f_x \cdot \frac{X_c}{Z_c} + c_x$$
$$v = f_y \cdot \frac{Y_c}{Z_c} + c_y$$

Where $(X_c, Y_c, Z_c)$ is the 3D point in camera frame.

### 6.4 Unprojection (Ray from 2D Point)

```python
# Normalized ray in camera frame
x_n = (u - cx) / fx
y_n = (v - cy) / fy
d_c = [x_n, y_n, 1.0]
d_c_normalized = d_c / ||d_c||

# Transform to world frame
d_w = R_camera @ d_c_normalized
```

---

## 7) Triangulation Geometry

### 7.1 Ray Representation

| Component | Frame | Description |
|-----------|-------|-------------|
| Origin | World | Camera position $p_{\text{cam}}^w$ |
| Direction | World | Normalized 3D unit vector |

### 7.2 Ray from Bounding Box

1. Unproject 2D bbox center → normalized camera-frame ray
2. Normalize to unit vector
3. Rotate to world frame using camera orientation
4. Set to zero if bbox invalid

### 7.3 Triangulation Method

**Midpoint / Least Squares:**

$$\min_X \sum_i \|P_i (X - p_i)\|^2$$

Where $P_i = I - d_i d_i^T$ is the projection matrix perpendicular to ray $i$.

**Requirements:**
- ≥2 valid cameras
- Rank-3 geometry (non-degenerate)

**Output:** Triangulated position $X^w$ in world frame

---

## 8) Transformation Chain Summary

### 8.1 Visual Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     COORDINATE FRAME TRANSFORMATION CHAIN                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   WORLD (ENU)            BODY (FLU)             GIMBAL (FLU)                 │
│   ┌───────────┐          ┌───────────┐          ┌───────────┐               │
│   │    Z(Up)  │          │    Z(Up)  │          │    Z(Up)  │               │
│   │     ^     │  q_body  │     ^     │ q_gimbal │     ^     │               │
│   │     |     │ ───────> │     |     │ ───────> │     |     │               │
│   │ Y<--+     │          │ Y<--+-->X │          │ Y<--+-->X │               │
│   │ (N) |     │          │ (L) (Fwd) │          │ (L) (Fwd) │               │
│   │     v X(E)│          └───────────┘          └───────────┘               │
│   └───────────┘                                        │                     │
│    (East,North,Up)        (Fwd,Left,Up)                │ q_cam_offset        │
│                                                        v                     │
│                                                                               │
│   IMAGE (pixels)         CAMERA (OpenGL)                                     │
│   ┌───────────┐          ┌───────────┐                                       │
│   │ (0,0)     │          │ +-->X (R) │                                       │
│   │  +-->u    │    K     │ |         │                                       │
│   │  |        │ <─────── │ v Y (D)   │                                       │
│   │  v        │          │           │                                       │
│   │   v       │          │ Z into pg │                                       │
│   └───────────┘          └───────────┘                                       │
│   (right, down)          (Right,Down,Forward)                                │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 World → Image (Forward)

```
World Frame (ENU)
    ↓ [q_body^{-1}]
Body Frame (FLU, 180° flipped in USD)
    ↓ [Gimbal: yaw → roll → pitch]
Gimbal Frame (FLU)
    ↓ [q_camera_offset]
Camera Frame (OpenGL: X=right, Y=down, Z=forward)
    ↓ [K with zoom]
Image Plane (pixels: u=horizontal, v=vertical)
```

### 8.3 Image → World (Inverse)

```
Bbox Center (pixels)
    ↓ [K^{-1}, unprojection]
Normalized Camera Ray [x_n, y_n, 1]
    ↓ [normalize]
Unit Ray in Camera Frame
    ↓ [R_camera]
Unit Ray in World Frame
    ↓ [+ camera position]
3D Ray in World Frame (origin + λ·direction)
```

---

## 9) Angular Velocity Composition

### 9.1 Combined Angular Velocity

From `derived_field_computers.py` (lines 295-354):

```python
# Gimbal contribution in body frame
gimbal_angular_velocity_b = [
    0,           # X (roll): no direct gimbal contribution
    pitch_rate,  # Y: pitch axis aligned with body Y
    yaw_rate     # Z: yaw axis fixed to body Z
]

# Combined angular velocity
combined_angular_velocity_b = body_angular_velocity_b + gimbal_angular_velocity_b

# Transform to world frame
combined_angular_velocity_w = quat_rotate(q_body, combined_angular_velocity_b)
```

**Note:** Large gimbal angles cause axis misalignment — this approximation is valid for small-to-moderate gimbal deflections.

---

## 10) Delay System Coordinate Frames

### 10.1 Perspective-Aware Delays

| Path | Description | Latency |
|------|-------------|---------|
| `agent_i.field.ego` | Self-state (fast) | Low |
| `agent_i.field.other` | Communicated state (slow) | High |

### 10.2 Derived Field Recomputation

After delay, derived fields are recomputed from raw delayed states:
- Camera position: $p_c^w = p_b^w + R(q_b) \cdot p_{c,\text{offset}}^b$
- Camera orientation: $q_c^w = q_b \otimes q_{\text{gimbal}} \otimes q_{c,\text{offset}}$
- Ray directions: From bbox via unprojection and rotation

---

## 11) Special Conventions and Notes

### 11.1 Isaac Sim vs Real-World

| Aspect | Isaac Sim | Note |
|--------|-----------|------|
| World frame | ENU | Standard |
| Gravity | -Z direction | Standard |
| IRIS body frame | 180° roll flipped | Modeled explicitly in controller |

### 11.2 Critical Sign Conventions

| Angle | Positive Direction | Mnemonic |
|-------|-------------------|----------|
| Yaw | CCW (viewed from above) | "Turn left" |
| Pitch | Down | "Nose down" |
| Roll | Right wing down | "Bank right" |
| Gimbal yaw | CCW (viewed from above) | "Pan left" |
| **Gimbal pitch** | **Down** | **"Tilt down"** |
| Gimbal roll | CW (viewed from behind) | "Horizon correction" |

### 11.3 Zoom Behavior

- Focal length scaled: $f_{\text{zoom}} = f_{\text{base}} \times \text{zoom\_level}$
- Principal point unchanged
- Effective zoom rate = $(max\_zoom\_rate)^2$ due to double scaling in implementation

### 11.4 Gimbal Feasibility Constraints

From GIMBAL_GEOMETRY.md:

At horizontal distance $d_h$, feasible target heights are:
$$z_{\min}(d_h) = z_a - d_h \cdot \tan(45°) = z_a - d_h$$
$$z_{\max}(d_h) = z_a - d_h \cdot \tan(10°) \approx z_a - 0.176 \cdot d_h$$

**Asymmetric:** Can look up 45° but only ~10° down (due to gimbal limits).

**Minimum distance constraint:**
$$d_{h,\min} \approx 1.21 \times \Delta z_{\text{agents}}$$

#### Single Agent Feasible Zone (Vertical Cross-Section)

```
        Z
        ^
        |           Pitch = -45° boundary (looking up)
        |          /
     +  |         /
        |        /  Feasible
     z_a|-------● Agent     Zone
        |        \
        |         \
     -  |          \ Pitch = +10° boundary (looking down)
        |           \
        +------------------------> X (horizontal distance)
                d_h
```

#### 3D Feasible Zone (Truncated Cone)

The 2D cross-section above, when rotated 360° around the vertical Z-axis, forms a 3D truncated cone:

```
    Side view (same as above):          Top-down view (looking from above):

            Z                                     . _ ___ _ .
            ^    /← pitch=-45°                 ,'    /|\    ',
            |   /   (looking up)              /    /  |  \    \
            |  /                             |   /   |   \   |
            | /                              |  /    ●    \  |  Agent at center
    Agent→  ●--------                        |  \  Agent  /  |
            |\       ← pitch=0°               \   \   |   /   /
            | \      (horizontal)              ', _\__|__/_ ,'
            |  \                                  ' - - - '
            |   \← pitch=+10°
            v    (looking down)              Feasible region is the
                                             DONUT-shaped area between
         d_h →                               inner and outer circles

    3D visualization (isometric):

                    ___----''''----___       ← Upper boundary (pitch = +10°)
                 ,''        |        '',        (shallow cone, looking slightly down)
               ,'           |           ',
              /             |             \
             /              |              \
            |               ● Agent         |   ← Agent hovers at center
             \              |              /
              \             |             /
               ',           |           ,'
                 ''--___    |    ___--''     ← Lower boundary (pitch = -45°)
                        ''''|''''              (steep cone, looking up)
                            |
                            v Z (down from agent)

    Key insight: Targets must be BELOW the agent (positive pitch = looking down)
                 but the gimbal can look UP much further than DOWN
                 Result: Asymmetric cone, wider above agent level
```

#### Multi-Agent Feasible Zone Intersection

```
        Z
        ^
        |
     25 |     ● Agent 2 (z = 25m)
        |    /|\ Cone 2
        |   / | \
     20 | ● Agent 1 (z = 20m)
        |/|\  |  \
        / | \ |   \
       /  |  \|    \
      /   |  ⊗------\  Intersection Zone (feasible for both)
     /    | /  \     \
    /_____|/____\_____\________> X (horizontal distance)
                d_h

    ⊗ = Region visible to ALL agents (triangulation feasible)
```

---

## 12) Source File References

### Frame Definitions
- `point_mass.py` (lines 366-370): Body frame convention (180° roll flip)
- `gimbal_stabilizer.py` (lines 6-11, 92-102): Gimbal frame and angle conventions
- `gimbal_stabilizer_cfg.py`: Angle limits with sign conventions
- `iris_gimbal2.usda` (body mesh): `xformOp:orient = (0.70711, 0, 0, 0.70711)` — 90° visual offset

### Offsets and Mounting
- `controller/gimbal_controller.py`: `YAW_JOINT_OFFSET = -π/2` constant definition
- `iris_ma_env6_test.py`: Joint offset application, TiledCamera offset, frustum offset
- `iris_ma_env6_test_cfg.py`: TiledCamera config with ROS convention offset

### Transformations
- `derived_field_computers.py`: Camera position/orientation, ray directions
- `point_mass.py`: Body frame velocity/acceleration control

### Documentation
- `IRIS_MA5_TECHNICAL_REPORT.md`: Comprehensive mathematical formulation
- `GIMBAL_GEOMETRY.md`: Gimbal feasibility and pitch angle constraints
- `bbox_raycaster/USAGE_GUIDE.md`: Quaternion convention note

---

## 13) Quick Reference Table

| Frame | Convention | Axes (X, Y, Z) | Origin |
|-------|------------|----------------|--------|
| World | ENU | East, North, Up | Environment center |
| Body | FLU (180° flipped) | Forward, Left, Up | Drone CoM |
| Gimbal | FLU | Forward, Left, Up | Gimbal mount |
| Camera | OpenGL | Right, Down, Forward | Lens center |
| Image | Pixel | Right, Down, — | Top-left corner |

| Quantity | Format | Range |
|----------|--------|-------|
| Quaternion | wxyz | Unit quaternion |
| Gimbal yaw (α) | radians | [-3.49, +3.49] |
| Gimbal pitch (β) | radians | [-1.22, +1.22] |
| Gimbal roll (ρ) | radians | [-0.785, +0.785] |
| Bbox | (cx, cy, w, h) | [0, 1] normalized |
