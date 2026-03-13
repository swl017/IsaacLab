# Triangulation Uncertainty Estimation Module Specification

**Project:** `iris_ma6` — Multi-Drone Active Triangulation
**Scope:** Multi-camera triangulation and uncertainty quantification
**Status:** Draft v1
**Last Updated:** 2026-03-13

**Related Documents:**
- [iris_ma6_env_spec.md](iris_ma6_env_spec.md) — Environment overview
- [reward_spec.md](reward_spec.md) — Reward function design (GT vs. delayed states)
- [bbox_spec.md](bbox_spec.md) — Object detection module
- [frame_conventions.md](frame_conventions.md) — Coordinate systems and quaternion conventions

---

## 1. Overview

### 1.1 Purpose

The triangulation uncertainty estimation module serves three critical functions in the IRIS multi-agent observation system:

1. **Position Estimation**: Fuse bearing measurements from multiple cameras to estimate target 3D position
2. **Uncertainty Quantification**: Propagate measurement and pose uncertainties to output covariance
3. **Reward Signal**: Provide a scalar quality metric (trace of covariance) for RL training

### 1.2 Design Philosophy

The module follows the principle of **uncertainty-aware active perception**: the policy should not only estimate target position but also understand the quality of that estimate. This enables:

- **Geometry optimization**: Seek formations that minimize triangulation uncertainty
- **Observation-before-action**: Gate high-stakes decisions (e.g., interception) on localization confidence
- **Information-theoretic coordination**: Balance observation coverage across multiple targets

### 1.3 Module Responsibilities

| Responsibility | Input | Output |
|---------------|-------|--------|
| Ray generation | 2D bbox centers | 3D ray directions |
| Position triangulation | Camera poses, ray directions | Target position estimate |
| Covariance computation | Jacobians, uncertainty sources | 3×3 covariance matrix |
| Quality metric | Covariance matrix | Scalar (trace, determinant, etc.) |
| Validity determination | Geometry, detections | Boolean mask |

---

## 2. Mathematical Foundation

### 2.1 Triangulation Geometry

Given $C$ cameras observing target $T$, each camera $c$ produces a bearing ray:

$$\mathbf{r}_c = \mathbf{p}_c + t \cdot \mathbf{d}_c, \quad t \geq 0$$

where:
- $\mathbf{p}_c \in \mathbb{R}^3$: Camera position in world frame
- $\mathbf{d}_c \in \mathbb{R}^3$: Unit direction vector from camera toward target

The triangulated position minimizes the sum of squared perpendicular distances to all rays:

$$\hat{\mathbf{X}} = \arg\min_{\mathbf{X}} \sum_{c=1}^{C} \| (\mathbf{I} - \mathbf{d}_c \mathbf{d}_c^\top)(\mathbf{X} - \mathbf{p}_c) \|^2$$

**Closed-form solution (midpoint method):**

$$\hat{\mathbf{X}} = \mathbf{A}^{-1} \mathbf{b}$$

where:
$$\mathbf{A} = \sum_{c=1}^{C} (\mathbf{I} - \mathbf{d}_c \mathbf{d}_c^\top), \quad \mathbf{b} = \sum_{c=1}^{C} (\mathbf{I} - \mathbf{d}_c \mathbf{d}_c^\top) \mathbf{p}_c$$

### 2.2 Measurement Model

Each camera measures a 2D pixel location $\mathbf{u}_c = (u, v)^\top$ of the target's bounding box center. The projection function:

$$\mathbf{u}_c = \pi(\mathbf{K}_c, \mathbf{R}_{wc}, \mathbf{t}_{wc}, \mathbf{X})$$

where:
- $\mathbf{K}_c$: Camera intrinsic matrix (focal length, principal point)
- $\mathbf{R}_{wc}$: World-to-camera rotation
- $\mathbf{t}_{wc}$: Camera position in world frame
- $\mathbf{X}$: Target position in world frame

The explicit projection is:
$$\mathbf{X}_c = \mathbf{R}_{wc}^\top (\mathbf{X} - \mathbf{t}_{wc})$$
$$\mathbf{u}_c = \begin{pmatrix} f_x \frac{X_c}{Z_c} + c_x \\ f_y \frac{Y_c}{Z_c} + c_y \end{pmatrix}$$

### 2.3 Uncertainty Sources

The module models five independent uncertainty sources:

| Symbol | Description | Dimensionality | Default σ |
|--------|-------------|----------------|-----------|
| $\Sigma_{\text{pix}}$ | Pixel detection noise | 2×2 per camera | 7.0 px |
| $\Sigma_{t_{wb}}$ | Camera position uncertainty | 3×3 per camera | 0.1 m |
| $\Sigma_{\phi_{wb}}$ | Camera orientation uncertainty | 3×3 per camera | 0.001 rad |
| $\Sigma_\alpha$ | Gimbal yaw angle uncertainty | 1×1 per camera | 0.001 rad |
| $\Sigma_\beta$ | Gimbal pitch angle uncertainty | 1×1 per camera | 0.001 rad |
| $\Sigma_K$ | Intrinsic parameter uncertainty | 4×4 per camera | 10.0 (optional) |

### 2.4 Jacobian Computation

First-order uncertainty propagation linearizes the measurement function:

$$\delta \mathbf{u}_c \approx \mathbf{J}_X \delta \mathbf{X} + \mathbf{J}_\theta \delta \boldsymbol{\theta}$$

where $\boldsymbol{\theta}$ collects all nuisance parameters (pose, gimbal, intrinsics).

**Projection Jacobian w.r.t. camera-frame point:**
$$\frac{\partial \mathbf{u}}{\partial \mathbf{X}_c} = \begin{pmatrix} \frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\ 0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2} \end{pmatrix}$$

**Position Jacobian (world frame):**
$$\mathbf{J}_X = \frac{\partial \mathbf{u}}{\partial \mathbf{X}_c} \cdot \mathbf{R}_{wc}^\top$$

**Pose Jacobians:**
$$\mathbf{J}_t = -\frac{\partial \mathbf{u}}{\partial \mathbf{X}_c} \cdot \mathbf{R}_{wc}^\top$$
$$\mathbf{J}_\phi = -\frac{\partial \mathbf{u}}{\partial \mathbf{X}_c} \cdot [\mathbf{X}_c]_\times \cdot \mathbf{R}_{wc}^\top$$

**Gimbal Jacobians:**
$$\mathbf{J}_\alpha = -\frac{\partial \mathbf{u}}{\partial \mathbf{X}_c} \cdot [\mathbf{X}_c]_\times \cdot \mathbf{e}_\alpha$$
$$\mathbf{J}_\beta = -\frac{\partial \mathbf{u}}{\partial \mathbf{X}_c} \cdot [\mathbf{X}_c]_\times \cdot \mathbf{e}_\beta$$

where $[\cdot]_\times$ denotes the skew-symmetric matrix operator, and $\mathbf{e}_\alpha$, $\mathbf{e}_\beta$ are the gimbal rotation axes in camera frame.

### 2.5 Covariance Propagation Formula

The full covariance computation follows the weighted least squares formulation:

**Normal matrix (Fisher Information approximation):**
$$\mathbf{A} = \sum_{c=1}^{C} \mathbf{J}_{X,c}^\top \mathbf{W}_c \mathbf{J}_{X,c}$$

where $\mathbf{W}_c = \Sigma_{\text{pix},c}^{-1}$ is the weight matrix.

**Residual covariance (accounting for nuisance parameter uncertainty):**
$$\mathbf{S}_c = \Sigma_{\text{pix},c} + \mathbf{J}_{\theta,c} \Sigma_{\theta,c} \mathbf{J}_{\theta,c}^\top$$

**Final covariance:**
$$\Sigma_X = \mathbf{A}^{-1} \left( \sum_{c=1}^{C} \mathbf{J}_{X,c}^\top \mathbf{W}_c \mathbf{S}_c \mathbf{W}_c \mathbf{J}_{X,c} \right) \mathbf{A}^{-\top}$$

### 2.6 Quality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Trace (A-optimality)** | $\text{tr}(\Sigma_X)$ | Sum of variances (default) |
| **Determinant (D-optimality)** | $\det(\Sigma_X)$ | Volume of uncertainty ellipsoid |
| **Max eigenvalue (E-optimality)** | $\lambda_{\max}(\Sigma_X)$ | Worst-case variance |
| **Condition number** | $\lambda_{\max} / \lambda_{\min}$ | Anisotropy of uncertainty |

---

## 3. Current Implementation (iris_ma5)

### 3.1 Module Architecture

```
triangulation/
├── triang_cov_reward_torch.py    # Core implementation
│   ├── skew()                     # Skew-symmetric matrix utility
│   ├── rotation_matrix_from_euler()
│   ├── proj_jacobian_wrt_Xc()     # Projection Jacobian
│   ├── build_views_from_env_state() # Camera geometry builder
│   ├── jacs_for_view_batch()      # Jacobian computation
│   ├── triangulation_covariance_multi_camera()  # Full covariance
│   ├── triangulation_covariance_simple()  # Pixel-only version
│   ├── midpoint_method_batched()  # Ray triangulation
│   └── get_ray_dir_from_bbox()    # Bbox → ray conversion
└── tests/
    └── test_behind_camera.py      # Unit tests
```

### 3.2 Tensor Conventions

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X_w` | `[N, T, 3]` | Target positions (N envs, T targets) |
| `robot_positions` | `[N, C, 3]` | Camera positions (C cameras) |
| `robot_quats` | `[N, C, 4]` | Camera orientations (wxyz) |
| `gimbal_yaws` | `[N, C]` | Gimbal yaw angles |
| `gimbal_pitches` | `[N, C]` | Gimbal pitch angles |
| `camera_intrinsics` | `[N, C, 3, 3]` | Intrinsic matrices |
| `Sigma_pix` | `[N, C, 2, 2]` | Pixel noise covariance |
| `Sigma_X` | `[N, T, 3, 3]` | Output covariance |
| `trace_cov` | `[N, T]` | Trace metric |

### 3.3 Numerical Stability Measures

The current implementation includes:

1. **Division-by-zero protection**: Clamp Z-coordinate to ≥1e-12
2. **Pseudo-inverse fallback**: Use `torch.linalg.pinv()` when `inv()` fails
3. **NaN/Inf detection**: Post-computation validity checks

**Gaps identified:**
- No explicit condition number checking before inversion
- No regularization added to near-singular normal matrices
- Missing per-camera valid mask handling

### 3.4 Computational Complexity

For N environments, T targets, C cameras, M nuisance parameters:

| Operation | Complexity | Bottleneck |
|-----------|------------|------------|
| Jacobian computation | O(N × T × C × 3 × M) | Matrix multiplications |
| Block diagonal assembly | O(N × T × C²) | Loop over cameras |
| Matrix inversion | O(N × T × 27) | 3×3 inversion |
| Residual covariance | O(N × T × 4C² × M) | Batch matmul |

**Performance note**: The block diagonal assembly uses Python loops over cameras (C), which could be vectorized.

---

## 4. Proposed Improvements for iris_ma6

### 4.1 Per-Camera Valid Mask Handling

**Problem**: Current implementation processes all cameras, even when some have invalid detections (bbox_empty=1).

**Solution**: Add `valid_mask: [N, C]` parameter to mask invalid cameras.

```python
def triangulation_covariance_multi_camera(
    X_w: torch.Tensor,
    robot_positions: torch.Tensor,
    ...,
    valid_mask: Optional[torch.Tensor] = None,  # NEW: [N, C] boolean
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        Sigma_X: [N, T, 3, 3]
        trace_cov: [N, T]
        is_valid: [N, T] - validity mask (≥2 cameras required)
    """
```

**Implementation approach**:
1. Zero out Jacobian contributions from invalid cameras
2. Track valid camera count per (env, target): `num_valid = valid_mask.sum(dim=1)`
3. Mark triangulation invalid if `num_valid < 2`

### 4.2 Observability-Aware Degeneracy Handling

**Problem**: Some geometric configurations produce ill-conditioned normal matrices:
- Collinear cameras (poor baseline)
- Cameras all at similar distances/angles
- Target directly between two cameras

**Solution**: Add explicit observability checks and graceful degradation.

```python
CONDITION_THRESHOLD = 1e6
REGULARIZATION_EPS = 1e-6

def compute_with_observability_check(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        A_inv: Regularized inverse
        is_observable: Boolean mask
    """
    # Check condition number via SVD
    svd = torch.linalg.svdvals(A)  # [N, T, 3]
    cond_num = svd[..., 0] / (svd[..., -1] + 1e-12)
    is_observable = cond_num < CONDITION_THRESHOLD

    # Regularize ill-conditioned matrices
    A_reg = A + REGULARIZATION_EPS * torch.eye(3, device=A.device)

    # Invert with fallback
    try:
        A_inv = torch.linalg.inv(A_reg)
    except:
        A_inv = torch.linalg.pinv(A_reg)

    return A_inv, is_observable
```

**Additional observability metrics to expose**:
- Minimum baseline angle: $\theta_{\min} = \min_{i,j} \angle(\mathbf{d}_i, \mathbf{d}_j)$
- Geometric dilution of precision (GDOP): Related to trace of inverse Fisher information

### 4.3 Multi-Target Batched Computation

**Problem**: iris_ma6 with T targets requires efficient multi-target processing.

**Current approach** (iris_ma6_env_spec.md §5.1):
```python
for t in range(T_max):
    if alive[t]:
        Sigma_X[t] = triangulation_covariance_multi_camera(...)
```

**Proposed improvement**: The current `triang_cov_reward_torch.py` already supports T targets natively via the `[N, T, 3]` target position tensor. The environment-level loop is unnecessary.

**Recommendation**: Use single batched call with alive mask:
```python
Sigma_X, trace_cov, is_valid = triangulation_covariance_multi_camera(
    X_w=target_positions,  # [N, T_max, 3]
    ...,
    target_alive_mask=alive,  # [N, T_max] - NEW parameter
)
```

### 4.4 Extended Quality Metrics

**Current**: Only trace (A-optimality) is computed.

**Proposed**: Add configurable quality metric selection.

```python
class QualityMetric(Enum):
    TRACE = "trace"           # A-optimality: tr(Σ)
    DETERMINANT = "det"       # D-optimality: det(Σ)^(1/3)
    MAX_EIGENVALUE = "eig"    # E-optimality: λ_max(Σ)
    CONDITION = "cond"        # Anisotropy: λ_max / λ_min

def compute_quality_metric(
    Sigma_X: torch.Tensor,
    metric: QualityMetric = QualityMetric.TRACE
) -> torch.Tensor:
    """Compute scalar quality metric from covariance."""
```

**Research question**: Which metric produces better learned policies? (See §7 Ablation)

### 4.5 Temporal Covariance Filtering (Optional Extension)

**Motivation**: Current implementation treats each timestep independently. Temporal filtering could:
- Smooth noisy covariance estimates
- Provide covariance prediction during detection dropouts
- Enable AoI-weighted uncertainty fusion

**Proposed approach**: Simple exponential moving average (EMA) with validity gating.

```python
class TemporalCovarianceFilter:
    def __init__(self, alpha: float = 0.1, growth_rate: float = 1.05):
        """
        Args:
            alpha: EMA smoothing factor (0 = keep old, 1 = use new)
            growth_rate: Covariance inflation per timestep without observation
        """
        self.alpha = alpha
        self.growth_rate = growth_rate
        self.Sigma_filtered = None

    def update(
        self,
        Sigma_new: torch.Tensor,  # [N, T, 3, 3]
        is_valid: torch.Tensor    # [N, T]
    ) -> torch.Tensor:
        if self.Sigma_filtered is None:
            self.Sigma_filtered = Sigma_new.clone()
            return self.Sigma_filtered

        # Inflate old covariance (uncertainty grows without observation)
        Sigma_inflated = self.Sigma_filtered * self.growth_rate

        # EMA update where valid
        self.Sigma_filtered = torch.where(
            is_valid.unsqueeze(-1).unsqueeze(-1),
            self.alpha * Sigma_new + (1 - self.alpha) * self.Sigma_filtered,
            Sigma_inflated
        )

        return self.Sigma_filtered
```

**User feedback**: Covariance comutation itself should be improved to be AoI-aware. Improved math will be added later. Having an optinal temporal filtering (EMA) that can be toggled on and off sounds reasonable.

**Configuration option**: Enable via `cfg.use_temporal_covariance_filter = True`

### 4.6 Behind-Camera Detection Enhancement

**Current**: `midpoint_method_batched()` checks if triangulated point is behind all cameras and falls back to mean position.

**Proposed enhancement**: Per-camera behind-camera filtering.

```python
def filter_behind_camera_observations(
    pts: torch.Tensor,      # [N, C, 3] camera positions
    dirs: torch.Tensor,     # [N, C, T, 3] ray directions
    X_tri: torch.Tensor,    # [N, T, 3] triangulated position
) -> torch.Tensor:
    """
    Returns:
        is_in_front: [N, C, T] mask of cameras where target is in front
    """
    # Vector from camera to triangulated point
    cam_to_point = X_tri.unsqueeze(1) - pts.unsqueeze(2)  # [N, C, T, 3]

    # Dot product with ray direction (positive = in front)
    dot_product = (cam_to_point * dirs).sum(dim=-1)  # [N, C, T]

    return dot_product > 0
```

**Use case**: Weight Jacobian contributions by in-front mask, or exclude behind-camera observations from triangulation.

### 4.7 Validity-Based Return Pattern (No Fallback Values)

**Problem with current iris_ma5 approach**: When triangulation fails (insufficient cameras, ill-conditioned, behind-camera), the code returns "fallback" values:
- Position: mean of camera positions
- Covariance: negative identity matrix (`-I`)
- Trace: `-1.0` sentinel

This pollutes downstream computations and requires remembering magic sentinel values.

**Proposed approach**: Return undefined values + explicit validity flags. No fallback values.

#### Position Triangulation

```python
def midpoint_method_batched(
    pts: torch.Tensor,       # [N, C, 3]
    dirs: torch.Tensor,      # [N, C, T, 3]
    valid_mask: torch.Tensor # [N, C] camera validity
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X_tri: [N, T, 3] - triangulated position (UNDEFINED where is_valid=False)
        is_valid: [N, T] - validity mask

    Note: Do NOT use X_tri values where is_valid=False. They contain NaN.
    """
    # Initialize with NaN (undefined)
    X_tri = torch.full((N, T, 3), float('nan'), device=device)

    # Compute validity conditions
    num_valid_cameras = valid_mask.sum(dim=1)  # [N]
    is_solvable = num_valid_cameras >= 2
    is_well_conditioned = condition_number < THRESHOLD
    is_in_front = ~all_behind_camera

    is_valid = is_solvable & is_well_conditioned & is_in_front

    # Only write to valid entries
    if is_valid.any():
        X_tri[is_valid] = solve_triangulation(...)

    return X_tri, is_valid
```

#### Covariance Computation

```python
def triangulation_covariance_multi_camera(
    X_w: torch.Tensor,           # [N, T, 3]
    ...,
    camera_valid_mask: torch.Tensor,  # [N, C]
    target_valid_mask: torch.Tensor,  # [N, T] from triangulation
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        Sigma_X: [N, T, 3, 3] - covariance (UNDEFINED where is_valid=False)
        trace_cov: [N, T] - trace metric (UNDEFINED where is_valid=False)
        is_valid: [N, T] - combined validity mask

    Note: Sigma_X and trace_cov contain NaN where is_valid=False.
    """
    # Initialize with NaN
    Sigma_X = torch.full((N, T, 3, 3), float('nan'), device=device)
    trace_cov = torch.full((N, T), float('nan'), device=device)

    # Inherit validity from triangulation + add covariance-specific checks
    is_valid = target_valid_mask & (condition_number < THRESHOLD)

    # Only compute for valid entries
    if is_valid.any():
        Sigma_X[is_valid], trace_cov[is_valid] = compute_covariance(...)

    return Sigma_X, trace_cov, is_valid
```

#### Validity Conditions Summary

| Condition | Check | Failure Mode |
|-----------|-------|--------------|
| Minimum cameras | `num_valid >= 2` | Insufficient observations |
| Well-conditioned | `cond_num < 1e6` | Degenerate geometry (collinear) |
| In-front | `any(dot > 0)` | Target behind all cameras |
| Finite result | `isfinite(X_tri)` | Numerical failure |
| Positive definite | `diag(Sigma) > 0` | Invalid covariance |

#### Benefits Over Sentinel Values

| Aspect | Sentinel (`-1`, `-I`) | Validity Flag + NaN |
|--------|----------------------|---------------------|
| Magic values to remember | Yes | No |
| Silent misuse | Possible (uses sentinel as real value) | Crashes with NaN |
| Downstream handling | Check for specific sentinel | Check `is_valid` boolean |
| Debugging | Hard to spot | NaN propagation obvious |
| Code clarity | `if trace < 0` | `if not is_valid` |

---

## 5. Integration Points

### 5.1 BBoxRaycaster → Triangulation Pipeline

**Critical: Ray computation fuses ego pose and gimbal orientation.**

The ray direction in world frame depends on:
1. **Ego pose**: Robot position `p_wb` and orientation quaternion `q_wb`
2. **Gimbal angles**: Yaw `α` and pitch `β`
3. **Bbox center**: Pixel coordinates `(u, v)` from detection

The transformation chain (see Appendix A):
$$\mathbf{R}_{wc} = \mathbf{R}_{wb}(q_{wb}) \cdot \mathbf{R}_{bg}(\alpha, \beta) \cdot \mathbf{R}_{gc}$$

Ray direction in world frame:
$$\mathbf{d}_w = \mathbf{R}_{wc} \cdot \mathbf{K}^{-1} \cdot \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}$$

**Implication for uncertainty**: Errors in ego pose OR gimbal angles both propagate to ray direction uncertainty, which is why the covariance computation includes Jacobians for both sources (§2.4).

```
BBoxRaycaster Output:
├── bbox_2d: [N, C, T, 4]      # xywh format
├── bbox_valid: [N, C, T]      # detection validity
└── bbox_depth: [N, C, T]      # (optional) depth ordering

      ↓

get_ray_dir_from_bbox(bbox_2d, K, R_wc)
  - R_wc = R_wb @ R_bg @ R_gc  ← Fuses ego pose + gimbal
      ↓

Ray Directions: [N, C, T, 3]   # In WORLD frame (fused pose+gimbal)

      ↓

midpoint_method_batched(valid_mask=bbox_valid.any(dim=2))
      ↓

X_tri: [N, T, 3]           # NaN where invalid
is_tri_valid: [N, T]       # Validity mask

      ↓

triangulation_covariance_multi_camera(target_valid_mask=is_tri_valid)
      ↓

Sigma_X: [N, T, 3, 3]      # NaN where invalid
trace_cov: [N, T]          # NaN where invalid
is_cov_valid: [N, T]       # Combined validity
```

### 5.2 Triangulation → Reward Computation

From [reward_spec.md](reward_spec.md):

| Reward Mode | State Source | Triangulation Input |
|-------------|--------------|---------------------|
| FIM (geometric proxy) | GT drone positions | GT target position |
| GT-anchored estimation | GT drone positions | Triangulated position |
| End-to-end | Delayed drone positions | Triangulated position |

**Proposed validity-aware reward computation**:
```python
# Get triangulation results with validity
X_tri, Sigma_X, trace_cov, is_valid = self._compute_triangulation(...)

# Compute quality metric ONLY for valid entries
# Invalid entries get zero reward (not penalty)
quality = torch.where(
    is_valid,
    torch.sqrt(10.0 / trace_cov.clamp(min=1e-6)).clamp(0, 100),
    torch.zeros_like(trace_cov)
)

# Apply with curriculum gating
r_triangulation = quality * scale * progress_coord

# Optional: Penalty for losing valid triangulation (encourages maintaining visibility)
# r_visibility_loss = -penalty * (was_valid_last_step & ~is_valid).float()
```

**Key difference from iris_ma5**: No need to check for sentinel values (`trace < 0`). The `is_valid` mask handles all failure modes uniformly.

### 5.3 Observation Space Features

From [iris_ma6_env_spec.md](iris_ma6_env_spec.md) §3.2.3:

| Feature | Dims | Source |
|---------|------|--------|
| localization std | 3 | $\sqrt{\text{diag}(\Sigma_X^{(t)})}$ |
| tri_valid | 1 | `is_valid` flag (binary) |

**Validity-aware computation**:
```python
# Per-target localization uncertainty in observation
# Zero-fill for invalid (policy sees "no information available")
raw_std = torch.sqrt(torch.diagonal(Sigma_X, dim1=-2, dim2=-1))  # [N, T, 3]

loc_std = torch.where(
    is_valid.unsqueeze(-1).expand_as(raw_std),
    raw_std,
    torch.zeros_like(raw_std)  # Invalid → zero std (unknown)
)

# Alternatively: use large value to indicate high uncertainty
# loc_std_alt = torch.where(is_valid.unsqueeze(-1), raw_std, torch.full_like(raw_std, 100.0))
```

**Design choice**: Zero vs large value for invalid uncertainty:
- **Zero**: "No information" — policy learns that zero means unavailable
- **Large value**: "Very uncertain" — more intuitive but may confuse with actual high uncertainty

Recommendation: Use zero + separate `tri_valid` flag in observation.

---

## 6. Configuration

### 6.1 Configuration Dataclass

```python
@configclass
class TriangulationCfg:
    """Triangulation module configuration."""

    # Uncertainty source standard deviations
    pix_std: float = 7.0           # Pixel detection noise (pixels)
    pos_std: float = 0.1           # Position uncertainty (meters)
    ori_std: float = 0.001         # Orientation uncertainty (radians)
    gimbal_std: float = 0.001      # Gimbal angle uncertainty (radians)
    intrinsics_std: float = 10.0   # Intrinsics uncertainty (optional)

    # Uncertainty source inclusion flags
    include_pose_uncertainty: bool = True
    include_gimbal_uncertainty: bool = True
    include_intrinsics_uncertainty: bool = False

    # Numerical stability
    regularization_eps: float = 1e-6
    condition_threshold: float = 1e6

    # Quality metric selection
    quality_metric: str = "trace"  # "trace", "det", "eig", "cond"

    # Temporal filtering (optional)
    use_temporal_filter: bool = False
    temporal_alpha: float = 0.1
    covariance_growth_rate: float = 1.05

    # Validation thresholds
    min_cameras_required: int = 2
    max_trace_threshold: float = 1000.0  # Invalid if trace exceeds
```

### 6.2 Default Covariance Matrices

```python
# Pixel noise (isotropic)
Sigma_pix = torch.eye(2) * (cfg.pix_std ** 2)  # [2, 2]

# Position uncertainty (diagonal)
Sigma_twb = torch.eye(3) * (cfg.pos_std ** 2)  # [3, 3]

# Orientation uncertainty (diagonal)
Sigma_phiwb = torch.eye(3) * (cfg.ori_std ** 2)  # [3, 3]

# Gimbal uncertainties (scalar)
Sigma_alpha = torch.tensor([[cfg.gimbal_std ** 2]])  # [1, 1]
Sigma_beta = torch.tensor([[cfg.gimbal_std ** 2]])   # [1, 1]

# Intrinsics uncertainty (fx, fy, cx, cy)
Sigma_K = torch.eye(4) * (cfg.intrinsics_std ** 2)  # [4, 4]
```

---

## 7. Ablation Design

### 7.1 Uncertainty Source Ablations

| Experiment | Pixel | Pose | Gimbal | Intrinsics | Purpose |
|------------|-------|------|--------|------------|---------|
| T1 (baseline) | ✓ | ✓ | ✓ | ✗ | Current iris_ma5 default |
| T2 (pixel-only) | ✓ | ✗ | ✗ | ✗ | Fast computation baseline |
| T3 (full) | ✓ | ✓ | ✓ | ✓ | Maximum uncertainty modeling |
| T4 (no-gimbal) | ✓ | ✓ | ✗ | ✗ | Gimbal contribution isolated |
| T5 (no-pose) | ✓ | ✗ | ✓ | ✗ | Pose contribution isolated |

**Evaluation metrics**:
- Training convergence (reward curve)
- Final triangulation RMSE
- Computational throughput (envs/sec)

### 7.2 Quality Metric Ablations

| Experiment | Metric | Description |
|------------|--------|-------------|
| M1 | trace | Sum of variances (default) |
| M2 | det | Uncertainty volume |
| M3 | max_eig | Worst-case variance |
| M4 | sqrt_trace | Standard deviation sum |

**Research question**: Does the choice of metric affect learned formation geometry?

### 7.3 Temporal Filtering Ablations

| Experiment | Filter | Growth Rate | Purpose |
|------------|--------|-------------|---------|
| F1 (none) | Off | — | Baseline (current) |
| F2 (ema) | EMA α=0.1 | 1.05 | Smoothing + uncertainty growth |
| F3 (aggressive) | EMA α=0.3 | 1.10 | Faster adaptation |

**Evaluation focus**: Behavior under detection dropouts.

---

## 8. Validation & Testing

### 8.1 Unit Test Scenarios

| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Two orthogonal cameras | 90° baseline | Low trace, well-conditioned |
| Two parallel cameras | 0° baseline | High trace, ill-conditioned, is_valid may be False |
| Target behind camera | Behind all cameras | is_valid=False, X_tri=NaN |
| Single camera | C=1 | is_valid=False, X_tri=NaN |
| Mixed valid/invalid | 2 valid, 1 invalid | Use only valid cameras |
| Large pixel noise | σ_pix=50 | Proportionally larger covariance |
| Zero pixel noise | σ_pix=0 | Numerical stability (no division by zero) |

### 8.2 Numerical Accuracy Benchmarks

Compare against analytical solutions for known geometries:

1. **Two cameras, perpendicular baseline**: Analytical covariance derivable
2. **Equilateral camera configuration**: Isotropic covariance expected
3. **Varying baseline distance**: Covariance should scale as d²

### 8.3 Test File Structure

```
triangulation/
└── tests/
    ├── __init__.py
    ├── run_tests.py              # Standalone test runner
    ├── test_triangulation.py     # Core triangulation tests
    ├── test_covariance.py        # Covariance computation tests
    ├── test_jacobians.py         # Jacobian accuracy tests
    ├── test_edge_cases.py        # Degeneracy handling
    └── README.md
```

---

## 9. Implementation Roadmap

### 9.1 Phase 1: Core Improvements (Current Scope)

| Task | Priority | Complexity |
|------|----------|------------|
| **Validity-based returns (no fallbacks)** | **High** | **Medium** |
| Add valid_mask parameter | High | Low |
| Add observability checking | High | Medium |
| Expose condition number | Medium | Low |
| Add alternative quality metrics | Medium | Low |
| Documentation and tests | High | Medium |

### 9.2 Phase 2: Multi-Target Optimization

| Task | Priority | Complexity |
|------|----------|------------|
| Vectorize camera loops | Medium | Medium |
| Per-target valid mask | High | Medium |
| Multi-target occlusion handling | Medium | High |

### 9.3 Phase 3: Advanced Features (Future)

| Task | Priority | Complexity |
|------|----------|------------|
| Temporal covariance filtering | Low | Medium |
| Kalman-style fusion | Low | High |
| Information-theoretic metrics | Low | Medium |

---

## 10. Summary of Improvements from iris_ma5

| Aspect | iris_ma5 | iris_ma6 (Proposed) |
|--------|----------|---------------------|
| **Invalid handling** | Sentinel values (`-1`, `-I`, mean position) | NaN + explicit `is_valid` mask |
| Valid mask handling | None | Per-camera, per-target masks |
| Degeneracy detection | Basic (fallback on failure) | Condition number + regularization |
| Quality metrics | Trace only | Configurable (trace/det/eig) |
| Multi-target | Sequential loop | Batched computation |
| Temporal filtering | None | Optional EMA filter (AoI-aware math TBD) |
| Observability metrics | None | Baseline angle, GDOP |
| Configuration | Hardcoded | Dataclass with all parameters |

---

## Appendix A: Frame Conventions Reference

From [frame_conventions.md](frame_conventions.md):

- **World frame**: ENU (East-North-Up)
- **Body frame**: FLU (Forward-Left-Up)
- **Camera frame**: RDF (Right-Down-Forward) — OpenCV convention
- **Quaternion**: wxyz (scalar-first)
- **Gimbal order**: Yaw (Z) then Pitch (Y)

**Transformation chain**:
$$\mathbf{R}_{wc} = \mathbf{R}_{wb} \cdot \mathbf{R}_{bg} \cdot \mathbf{R}_{gc}$$

where:
- $\mathbf{R}_{wb}$: World-to-body (from quaternion)
- $\mathbf{R}_{bg}$: Body-to-gimbal (from yaw, pitch)
- $\mathbf{R}_{gc}$: Gimbal-to-camera (fixed ENU→RDF)

---

## Appendix B: Related Literature

1. **Triangulation theory**: Hartley & Zisserman, "Multiple View Geometry" (2003)
2. **Fisher Information for active perception**: Bajcsy et al., "Active Perception" (1988)
3. **Multi-camera covariance**: Snavely et al., "Skeletal Graphs for Scene Reconstruction" (2008)
4. **Uncertainty-aware MARL**: Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem" (2022)
