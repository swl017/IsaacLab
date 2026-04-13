# Observation Redesign Specification

**Status**: Proposed
**Date**: 2026-04-10
**Motivation**: Analysis of the Aerial_To_Aerial_Interception codebase (TU Delft PATS-X) showed that ego-centric body-frame observations improved learning speed by 47% over world-frame. This spec proposes a heading-frame redesign for iris_ma6, informed by that finding and adapted to the multi-agent triangulation task.

---

## 1. Frame Definitions

### 1.1 Heading Frame (Vehicle-1 Frame)

The **heading frame** (also called the vehicle-1 frame) applies only the yaw component of the drone's orientation to the world frame:

```
R_v1 = Rz(ψ)

     ┌ cos ψ   -sin ψ   0 ┐
R  = │ sin ψ    cos ψ   0 │
     └ 0        0        1 ┘
```

**Properties:**
- X-axis points in the drone's heading direction (projected onto horizontal)
- Z-axis points up (gravity-aligned, identical to world frame)
- Invariant to roll and pitch (body tilt does not rotate the frame)

**Rationale for choosing heading frame over full body frame:**

| Criterion | World | Heading | Full Body |
|-----------|-------|---------|-----------|
| Ego-centric yaw | No | Yes | Yes |
| Stable under tilt | Yes | Yes | No |
| Matches action semantics | No | Approximately | No |
| Uncertainty rotation cost | N/A | Cheap (Rz only, σ_z exact) | Expensive (full 3x3) |
| Triangulation reasoning | Hard (absolute coords) | Natural (relative, stable) | Unstable (tilt-coupled) |

### 1.2 Conventions

| Frame | Abbreviation | Usage |
|-------|-------------|-------|
| World (ENU) | `_w` | Critic observations, combined angular velocity |
| Heading (vehicle-1) | `_v1` | Actor ego state, inter-agent relative state, triangulation tail |
| Body | `_b` | Angular velocity (gyro), linear acceleration (accel) |
| Joint | `_j` | Gimbal joint angles (scalar angles along each joint's rotation axis) |
| Image | `_img` | Bounding box (normalized pixel coordinates) |

---

## 2. Ego Observation (Actor) — 31D

| # | Feature | Symbol | Frame | Dims | Source | Notes |
|---|---------|--------|-------|------|--------|-------|
| 1 | Linear velocity | `vel_v1` | Heading | 3 | `Rz(ψ)^T @ vel_w` | Matches dynamics: pitch → forward accel |
| 2 | Roll | `φ` | Heading | 1 | `euler_xyz_from_quat()` | Raw roll = heading-frame roll |
| 3 | Pitch | `θ` | Heading | 1 | `euler_xyz_from_quat()` | Raw pitch = heading-frame pitch |
| 4 | World yaw | `[cos ψ, sin ψ]` | World | 2 | From quaternion | Continuous, no wrapping. Needed for multi-agent coordination geometry |
| 5 | Angular velocity | `ω_b` | Body | 3 | Gyroscope (IMU native) | Body frame is correct for gyro; transforming to heading would mix in yaw rate noise |
| 6 | Linear acceleration | `a_b` | Body | 3 | Accelerometer (IMU native) | Same rationale as angular velocity |
| 7 | Unproject ray | `ray_v1` | Heading | 3 | `Rz(ψ)^T @ ray_w` | Ray from camera through bbox center (unprojected from 2D detection). LOS pointing for spatial reasoning |
| 8 | Gimbal yaw joint | `θ_yaw_j` | Joint | 1 | `joint_positions[:, 1]` | Raw joint angle. Policy-commanded axis; subject to saturation at mechanical limits |
| 9 | Gimbal pitch joint | `θ_pitch_j` | Joint | 1 | `joint_positions[:, 0]` | Raw joint angle. Needed for pitch limit/saturation awareness (body tilt eats into pitch range) |
| 10 | Gimbal roll joint | `θ_roll_j` | Joint | 1 | `joint_positions[:, 2]` | Raw joint angle. Needed for roll limit awareness (auto-stabilized, but saturates under aggressive banking) |
| 11 | Combined angular velocity | `ω_cam_w` | World | 3 | Camera sweep rate | Inertial LOS rate; determines image blur. World frame is natural (blur is frame-independent) |
| 12 | Motion age-of-information | `aoi_motion` | — | 1 | `sim_time - timestamp_motion` | Staleness of ego motion state (e.g. delayed state estimation). Scalar, frame-independent |
| 13 | Bbox age-of-information | `aoi_bbox` | — | 1 | `sim_time - timestamp_detection` | Staleness of detection. Scalar, frame-independent |
| 14 | Zoom level | `z` | — | 1 | Current zoom state | Scalar |
| 15 | Effective HFOV | `hfov` | — | 1 | Zoom-adjusted FOV | Scalar, radians |
| 16 | Bounding box | `[cx, cy, w, h]` | Image | 4 | Normalized pixel coords | [0,1] range |
| 17 | Bbox empty flag | `empty` | — | 1 | Detection validity | 1 if no detection, 0 if valid |
| | **Total** | | | **31** | | Was 31 (removed: position 3D, yaw from Euler triplet; added: cos/sin yaw 2D, gimbal roll 1D, motion AoI 1D; net 0) |

### 2.1 Removed Features (with rationale)

| Feature | Old Dims | Why Removed |
|---------|----------|-------------|
| Ego position (world) | 3 | Redundant for actor. Triangulation geometry and coordination are captured by inter-agent relative positions. Geofencing: add boundary-distance features if needed later |
| Yaw as Euler angle | 1 (in 3D triplet) | Wrapping discontinuity at ±π. Replaced by continuous `[cos ψ, sin ψ]` (2D) |

### 2.2 Added Features (with rationale)

| Feature | Dims | Why Added |
|---------|------|-----------|
| `[cos ψ, sin ψ]` | 2 | Continuous world yaw for multi-agent coordination. Without it, two agents cannot distinguish flying-in-formation vs. head-on approach |
| Gimbal roll joint | 1 | Roll is auto-stabilized by the gimbal controller (`_compute_stabilizing_roll`), not policy-controlled. But aggressive banking can saturate the roll joint limit. Policy needs visibility to anticipate this |

### 2.3 Frame Change Summary

| Feature | Old Frame | New Frame |
|---------|-----------|-----------|
| Velocity | World | **Heading** |
| Attitude | Euler [φ,θ,ψ] world (3D) | **[φ, θ] heading (2D) + [cos ψ, sin ψ] world (2D)** |
| Unproject ray | World | **Heading** |
| Angular velocity | Body | Body (unchanged) |
| Acceleration | Body | Body (unchanged) |
| Combined ang vel | World | World (unchanged) |

---

## 3. Inter-Agent Observation (Actor) — 19D per other agent

| # | Feature | Symbol | Frame | Dims | Notes |
|---|---------|--------|-------|------|-------|
| 1 | Relative position | `Δpos_v1` | Heading | 3 | `Rz(ψ_ego)^T @ (pos_other - pos_ego)`. "Other drone is 20m ahead-right" |
| 2 | Relative velocity | `Δvel_v1` | Heading | 3 | `Rz(ψ_ego)^T @ (vel_other - vel_ego)`. Closing rate in ego heading frame |
| 3 | Other's unproject ray | `ray_other_v1` | Heading (ego's) | 3 | `Rz(ψ_ego)^T @ ray_other_w`. "Other drone's LOS direction from my perspective" |
| 4 | Combined angular velocity | `ω_cam_other_w` | World | 3 | Other's camera sweep rate (blur is inertial) |
| 5 | Convergence angle | `α_conv` | — | 1 | `arccos(ray_ego_w · ray_other_w)`. Key triangulation quality metric: 0°=parallel (useless), 90°=optimal |
| 6 | Baseline magnitude | `‖Δpos‖` | — | 1 | `‖pos_other - pos_ego‖`. Longer baseline = better triangulation |
| 7 | Baseline-ray angle | `α_base` | — | 1 | `arccos(baseline_unit · ray_ego_w)`. Whether ego looks along or across the baseline |
| 8 | Zoom level | `z_other` | — | 1 | Other's zoom state |
| 9 | Bbox empty flag | `empty_other` | — | 1 | Other's detection validity |
| 10 | Data age | `age_data` | — | 1 | `sim_time - timestamp_motion`. Communication staleness |
| 11 | Bbox age | `age_bbox` | — | 1 | `sim_time - timestamp_detection`. Detection staleness |
| | **Total per other** | | | **19** | Was 16 (added: 3 geometry features; changed: absolute pos/vel → relative heading-frame) |

### 3.1 Key Changes from Current

| Change | Old | New | Rationale |
|--------|-----|-----|-----------|
| Position | Absolute world (3D) | **Relative heading** (3D) | Ego-centric. "Partner is 25m ahead-left" is directly actionable |
| Velocity | Absolute world (3D) | **Relative heading** (3D) | Closing/separating rate. Removes need to mentally subtract ego velocity |
| Ray direction | Absolute world (3D) | **Ego heading frame** (3D) | "Partner is looking 30° left of the baseline from my perspective" |
| Geometry features | None | **+3D** (convergence, baseline, baseline-ray) | Pre-computed features that the FIM reward already optimizes for. Closes the observation-reward loop |

### 3.2 Geometry Feature Computation

```python
# Convergence angle (0 = parallel rays, π/2 = perpendicular = optimal)
cos_conv = (ray_ego_w * ray_other_w).sum(dim=-1)
alpha_conv = torch.acos(torch.clamp(cos_conv, -1.0 + 1e-6, 1.0 - 1e-6))  # [N, 1]

# Baseline vector and magnitude
baseline = pos_other_w - pos_ego_w                    # [N, 3]
baseline_mag = baseline.norm(dim=-1, keepdim=True)    # [N, 1]
baseline_unit = baseline / (baseline_mag + 1e-6)      # [N, 3]

# Baseline-ray angle (how ego looks relative to baseline direction)
cos_base = (baseline_unit * ray_ego_w).sum(dim=-1)
alpha_base = torch.acos(torch.clamp(cos_base, -1.0 + 1e-6, 1.0 - 1e-6))  # [N, 1]
```

**Note:** These features are rotation-invariant scalars — they do not depend on any frame choice.

### 3.3 Scaling to N > 2 Agents

For N agents, each agent observes N-1 inter-agent blocks. With 3 agents:
- Agent A sees: [ego 31D] + [B relative 19D] + [C relative 19D] + [tri tail 4D] = 73D
- Geometry features are computed per-pair: A-B convergence, A-C convergence, etc.

---

## 4. Triangulation Tail — 4D (Actor) / 6D (Critic)

### 4.1 Actor Tail

| # | Feature | Frame | Dims | Notes |
|---|---------|-------|------|-------|
| 1 | Triangulated position (relative) | Heading (ego's) | 3 | `Rz(ψ_ego)^T @ (tri_pos_w - pos_ego_w)`. "Target is 15m ahead, 5m up" |
| 2 | Uncertainty (scalar) | — | 1 | `sqrt(σ_x² + σ_y² + σ_z²)`. Rotation-invariant confidence. -1 if invalid |
| | **Total** | | **4** | Was 6 |

**Rationale for scalar uncertainty**: The full 3D std `[σ_x, σ_y, σ_z]` in world frame is not meaningful to the actor (axis-aligned decomposition of an anisotropic ellipsoid in an arbitrary frame). The actor mainly needs "how confident is this estimate" (scalar), not directional uncertainty. The scalar trace is rotation-invariant — no need to worry about frame mismatch.

**Invalid fallback**: When triangulation is invalid, position → `[0, 0, 0]` and uncertainty → `-1.0` (existing convention).

### 4.2 Critic Tail

| # | Feature | Frame | Dims | Notes |
|---|---------|-------|------|-------|
| 1 | Triangulated position | World | 3 | Absolute position for value estimation |
| 2 | Uncertainty std | World | 3 | Full directional uncertainty for critic |
| | **Total** | | **6** | Unchanged |

---

## 5. Critic Observation (Shared Critic for MAPPO)

The MAPPO critic receives a **global state** for value estimation. Recommendations:

| Component | Frame | Rationale |
|-----------|-------|-----------|
| All agent positions | World | Absolute for global geometry |
| All agent velocities | World | Consistent cross-agent comparison |
| All agent yaws | World `[cos ψ, sin ψ]` | Formation geometry |
| All unproject rays | World | Shared frame for triangulation quality |
| Triangulation tail | World (6D) | Absolute target estimate |
| Combined angular velocities | World | Image stability (all agents) |
| All bbox empty flags | — | Detection state |
| All AoI/age features | — | Staleness |

The critic does NOT need heading-frame transforms (no ego-centric reasoning needed for value estimation). World frame gives the critic a consistent, comparable view across all agents.

**Critic dimension** (2 agents): ~48D. Exact layout TBD during implementation.

---

## 6. Dimension Summary

### Per-Agent Actor Observation

| Block | Current Dims | New Dims | Change |
|-------|-------------|----------|--------|
| Ego | 31 | 31 | 0 |
| Inter-agent (×1 for 2 agents) | 16 | 19 | +3 |
| Triangulation tail | 6 | 4 | -2 |
| **Total (2 agents)** | **53** | **54** | **+1** |
| **Total (3 agents)** | **69** | **73** | **+4** |

Net dimension is roughly unchanged, but information density per dimension is significantly higher due to frame-appropriate representations and pre-computed geometry features.

---

## 7. Computation Cost

| Operation | Count per step | Cost |
|-----------|---------------|------|
| `Rz(ψ)^T @` (2D rotation) | ~7 per agent (vel, ray, inter-agent pos/vel/ray, tri_pos, + 1 spare) | Cheap: 1 sin/cos + 4 multiply-adds per 3-vector |
| `arccos` for geometry features | 2 per agent-pair (convergence + baseline-ray) | Moderate: clamp + arccos |
| `norm` for baseline magnitude | 1 per agent-pair | Cheap |
| Euler extraction | 1 per agent (unchanged) | Unchanged |

Total added cost: ~30 FLOPs per environment per step. Negligible compared to physics simulation and network forward pass.

---

## 8. Migration Plan

### 8.1 Implementation Order

1. **Add heading-frame utility function** to env or a shared utils module:
   ```python
   def rotate_to_heading_frame(vectors_w: Tensor, yaw: Tensor) -> Tensor:
       """Rotate world-frame vectors into heading (vehicle-1) frame.
       Args:
           vectors_w: [N, 3] or [N, K, 3] world-frame vectors
           yaw: [N] yaw angles in radians
       Returns:
           vectors_v1: same shape, in heading frame
       """
       cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
       # Rz(ψ)^T = Rz(-ψ)
       x = cos_y * vectors_w[..., 0] + sin_y * vectors_w[..., 1]
       y = -sin_y * vectors_w[..., 0] + cos_y * vectors_w[..., 1]
       z = vectors_w[..., 2]
       return torch.stack([x, y, z], dim=-1)
   ```

2. **Modify `_get_observations()`**:
   - Extract yaw from quaternion
   - Apply heading-frame rotation to: ego velocity, ego unproject ray, inter-agent relative pos/vel/ray, triangulation position
   - Replace Euler triplet with [φ, θ, cos ψ, sin ψ]
   - Add gimbal roll joint (keep gimbal yaw joint — policy-commanded, subject to saturation)
   - Add geometry features to inter-agent block
   - Replace tri tail with heading-frame relative + scalar uncertainty

3. **Update observation docstring and dimension constants**

4. **Update SKRL config** (`skrl_mappo_rnn_cfg.yaml`): Adjust observation dimensions

5. **Retrain and compare** learning curves against baseline

### 8.2 Backward Compatibility

This is a **breaking change** to the observation space. Existing trained models will not be compatible. This should be introduced as a clean experiment with fresh training.

### 8.3 Testing

- Verify heading-frame rotation is correct: `rotate_to_heading_frame(v_w, ψ=0)` should return `v_w`
- Verify geometry features: convergence angle = 0 when rays are parallel, π/2 when perpendicular
- Verify observation dimensions match spec
- Verify triangulation tail relative position: `tri_pos_v1 + ego_pos` (rotated back) ≈ `tri_pos_w`

---

## 9. References

- Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019 — Motivates 6D rotation matrix (considered, but [φ, θ] + [cos ψ, sin ψ] is sufficient for heading-frame decomposition since roll/pitch are small-angle and continuous)
- Aerial_To_Aerial_Interception (TU Delft PATS-X) — Body-frame relative observations showed 47% learning speedup. Analysis in `doc/active/ticket/011-sim2real-measurement-checklist/sim2real_priority.md`
- Vehicle-1 frame: Standard Euler angle intermediate frame from aerospace convention (Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics", §1.5)