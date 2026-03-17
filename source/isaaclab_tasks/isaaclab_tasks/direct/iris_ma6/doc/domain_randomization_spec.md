# Domain Randomization Specification

This document specifies the domain randomization strategy for the iris_ma6 environment, covering physics properties, computational camera simulation, and gimbal mount variations.

**Scope Distinction:**
- **Domain Randomization** (this document): *What* the system is (mass, friction, camera FOV, sensor characteristics)
- **Initial States** (separate module): *Where* the system starts (position, velocity, orientation)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Randomization Hierarchy](#2-randomization-hierarchy)
3. [Physics Property Randomization](#3-physics-property-randomization)
4. [Camera Parameter Randomization](#4-camera-parameter-randomization)
5. [Gimbal Mount Randomization](#5-gimbal-mount-randomization)
6. [Implementation Architecture](#6-implementation-architecture)
7. [Configuration Schema](#7-configuration-schema)

---

## 1. Overview

### 1.1 Goals

- **Sim-to-real transfer**: Randomize physics and sensor parameters to bridge the reality gap
- **Robust policies**: Train agents that generalize across hardware variations
- **Computational efficiency**: Use GPU-accelerated transformations where possible

### 1.2 Scope

| Category | Randomization Method | Runtime Modifiable |
|----------|---------------------|-------------------|
| Mass, Inertia | USD physics API | Yes (reset) |
| Scale | USD transform | No (prestartup only) |
| Friction, Restitution | USD material API | Yes (reset) |
| Camera Intrinsics | Computational projection | Yes (per-step) |
| Camera Resolution | Computational crop/resize | Yes (per-step) |
| Gimbal Joints | Articulation API | Yes (per-step) |
| Gimbal Mount Offset | USD transform | No (prestartup only) |

### 1.3 Constraints

- **Centered principal points only**: No principal point offset randomization
- **Square pixels only**: f_x = f_y assumed
- **Fixed render resolution**: Computational processing simulates resolution variations

---

## 2. Randomization Hierarchy

Randomization is applied at multiple levels:

```
┌─────────────────────────────────────────────────────────┐
│ Level 1: Pre-startup (USD-level, once per session)     │
│   - Scale randomization                                │
│   - Gimbal mount offset                                │
│   - Visual texture/color                               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Level 2: Per-environment (at env creation)             │
│   - Base mass/inertia variations                       │
│   - Base friction coefficients                         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Level 3: Per-episode (at reset)                        │
│   - Mass additive noise                                │
│   - Friction scaling                                   │
│   - Camera focal length                                │
│   - Gimbal joint offsets                               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Level 4: Per-agent (within episode)                    │
│   - Agent-specific parameter variations                │
│   - Heterogeneous swarm simulation                     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Level 5: Per-step (every simulation step)              │
│   - Computational camera transforms                    │
│   - Sensor noise injection                             │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Physics Property Randomization

### 3.1 Mass Randomization

**Target**: Drone body, payload, gimbal components

```python
@dataclass
class MassRandomizationCfg:
    """Configuration for mass randomization."""

    # Body mass variation
    body_mass_scale_range: tuple[float, float] = (0.9, 1.1)  # ±10%
    body_mass_add_range: tuple[float, float] = (-0.05, 0.05)  # ±50g

    # Payload mass (simulates different payloads)
    payload_mass_range: tuple[float, float] = (0.0, 0.2)  # 0-200g

    # Gimbal mass variation
    gimbal_mass_scale_range: tuple[float, float] = (0.95, 1.05)

    # Distribution type
    distribution: str = "uniform"  # "uniform", "log_uniform", "gaussian"

    # Automatic inertia recomputation
    recompute_inertia: bool = True
```

**Implementation**:
```python
def randomize_drone_mass(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    cfg: MassRandomizationCfg,
):
    """Apply mass randomization to drone bodies."""
    robot = env.scene["robot"]

    # Scale base mass
    mass_scale = torch.empty(len(env_ids)).uniform_(*cfg.body_mass_scale_range)

    # Add payload mass
    payload = torch.empty(len(env_ids)).uniform_(*cfg.payload_mass_range)

    # Apply to simulation
    # ... implementation details
```

### 3.2 Scale Randomization

**Constraint**: Only at prestartup (before simulation begins)

```python
@dataclass
class ScaleRandomizationCfg:
    """Configuration for geometric scale randomization."""

    # Uniform scale (same for all axes)
    uniform_scale_range: tuple[float, float] | None = (0.95, 1.05)

    # Per-axis scale (independent axes)
    per_axis_scale: dict[str, tuple[float, float]] | None = None
    # Example: {"x": (0.9, 1.1), "y": (0.9, 1.1), "z": (0.95, 1.05)}

    # Target bodies (regex supported)
    body_names: list[str] = field(default_factory=lambda: ["base_link"])
```

**Note**: Requires `replicate_physics=False` in scene config.

### 3.3 Friction and Restitution

```python
@dataclass
class MaterialRandomizationCfg:
    """Configuration for physics material randomization."""

    # Friction coefficients
    static_friction_range: tuple[float, float] = (0.7, 1.3)
    dynamic_friction_range: tuple[float, float] = (0.5, 1.0)

    # Restitution (bounciness)
    restitution_range: tuple[float, float] = (0.0, 0.3)

    # Number of material buckets (for efficiency)
    num_buckets: int = 64

    # Ensure dynamic <= static friction
    make_consistent: bool = True
```

---

## 4. Camera Parameter Randomization

### 4.1 Computational Resolution Simulation

**Strategy**: Render at maximum resolution, apply computational transformations to simulate various camera configurations.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RENDERING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Render    │    │    Crop     │    │   Resize    │         │
│  │ (1920x1080) │ -> │  (ROI)      │ -> │  (target)   │         │
│  │  Full HD    │    │  FOV sim    │    │  res sim    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  Intrinsic Matrix Transformation:                               │
│                                                                 │
│  K_render    ->    K_crop    ->    K_final                     │
│  [f, 0, cx]       [f, 0, cx']      [f', 0, cx'']               │
│  [0, f, cy]       [0, f, cy']      [0, f', cy'']               │
│  [0, 0, 1 ]       [0, 0, 1  ]      [0, 0, 1   ]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Mathematical Formulation

#### 4.2.1 Render Resolution (Fixed)

```
W_render = 1920  (render width - Full HD)
H_render = 1080  (render height - Full HD)
f_render = focal length in pixels at render resolution
cx_render = W_render / 2 = 960   (centered principal point)
cy_render = H_render / 2 = 540   (centered principal point)

K_render = [[f_render,    0,     cx_render],
            [   0,     f_render, cy_render],
            [   0,        0,        1     ]]
```

#### 4.2.2 Crop Transformation (FOV Simulation)

Cropping simulates narrower field of view (telephoto effect):

```
crop_x = horizontal crop start (pixels from left)
crop_y = vertical crop start (pixels from top)
crop_w = crop width
crop_h = crop height

# New principal point after crop (still centered within crop region)
cx_crop = cx_render - crop_x = crop_w / 2  (for centered crops)
cy_crop = cy_render - crop_y = crop_h / 2  (for centered crops)

# Focal length unchanged by crop
f_crop = f_render

K_crop = [[f_crop,   0,    cx_crop],
          [  0,   f_crop,  cy_crop],
          [  0,      0,       1   ]]
```

**Constraint**: For centered principal points, crops must be symmetric:
```
crop_x = (W_render - crop_w) / 2
crop_y = (H_render - crop_h) / 2
```

#### 4.2.3 Resize Transformation (Resolution Simulation)

Resizing to target resolution scales focal length and principal point:

```
W_target = target width
H_target = target height

scale_x = W_target / crop_w
scale_y = H_target / crop_h

# For square pixels: scale_x == scale_y
scale = scale_x  (assuming square pixels)

f_final = f_crop * scale
cx_final = cx_crop * scale = W_target / 2  (for centered)
cy_final = cy_crop * scale = H_target / 2  (for centered)

K_final = [[f_final,    0,    cx_final],
           [   0,    f_final, cy_final],
           [   0,       0,       1    ]]
```

#### 4.2.4 Combined Transformation

For a given target resolution and effective FOV:

```python
def compute_camera_transform(
    render_size: tuple[int, int],      # (W_render, H_render)
    target_size: tuple[int, int],      # (W_target, H_target)
    fov_scale: float,                  # 1.0 = full FOV, 0.5 = half FOV (2x zoom)
    f_render: float,                   # Focal length at render resolution
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute crop region and final intrinsic matrix.

    Args:
        render_size: Rendered image size (width, height)
        target_size: Target output size (width, height)
        fov_scale: FOV scaling factor (1.0 = no zoom, <1.0 = zoom in)
        f_render: Focal length in pixels at render resolution

    Returns:
        crop_params: (crop_x, crop_y, crop_w, crop_h)
        K_final: 3x3 intrinsic matrix for target resolution
    """
    W_render, H_render = render_size
    W_target, H_target = target_size

    # Compute crop region (symmetric for centered principal point)
    crop_w = int(W_render * fov_scale)
    crop_h = int(H_render * fov_scale)
    crop_x = (W_render - crop_w) // 2
    crop_y = (H_render - crop_h) // 2

    # Compute scale factor
    scale = W_target / crop_w  # Assumes square pixels: W_target/crop_w == H_target/crop_h

    # Compute final focal length
    f_final = f_render * scale

    # Principal point (centered)
    cx_final = W_target / 2.0
    cy_final = H_target / 2.0

    # Build intrinsic matrix
    K_final = torch.tensor([
        [f_final,    0.0,   cx_final],
        [   0.0, f_final,   cy_final],
        [   0.0,    0.0,       1.0 ]
    ])

    crop_params = (crop_x, crop_y, crop_w, crop_h)

    return crop_params, K_final
```

### 4.3 Camera Randomization Configuration

```python
@dataclass
class CameraRandomizationCfg:
    """Configuration for computational camera randomization."""

    # Render resolution (fixed, maximum quality - Full HD)
    render_width: int = 1920
    render_height: int = 1080

    # Target resolution range (simulated via resize)
    target_width_range: tuple[int, int] = (640, 1920)
    target_height_range: tuple[int, int] = (360, 1080)

    # Discrete resolution options (if None, continuous sampling)
    # Common 16:9 resolutions
    discrete_resolutions: list[tuple[int, int]] | None = [
        (1920, 1080),  # Full HD (1080p)
        (1280, 720),   # HD (720p)
        (640, 360),    # nHD (360p)
    ]

    # FOV scale range (1.0 = full FOV, smaller = zoom in)
    fov_scale_range: tuple[float, float] = (0.5, 1.0)

    # Focal length randomization (at render resolution)
    focal_length_range: tuple[float, float] = (800.0, 1200.0)  # pixels (scaled for 1080p)

    # Randomization frequency
    randomize_per_step: bool = False   # Per-step randomization
    randomize_per_episode: bool = True # Per-episode randomization
    randomize_per_agent: bool = False  # Per-agent variations
```

### 4.4 GPU-Accelerated Image Processing

```python
class CameraProcessor:
    """GPU-accelerated camera image processing for resolution simulation."""

    def __init__(
        self,
        cfg: CameraRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Pre-allocate crop parameters per env/agent
        self.crop_params = torch.zeros(num_envs, num_agents, 4, device=device)
        self.intrinsic_matrices = torch.zeros(num_envs, num_agents, 3, 3, device=device)
        self.target_sizes = torch.zeros(num_envs, num_agents, 2, dtype=torch.int, device=device)

    def randomize(self, env_ids: torch.Tensor | None = None):
        """Sample new camera parameters for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        num_samples = len(env_ids) * self.num_agents

        # Sample FOV scales
        fov_scales = torch.empty(num_samples, device=self.device).uniform_(
            *self.cfg.fov_scale_range
        )

        # Sample focal lengths
        focal_lengths = torch.empty(num_samples, device=self.device).uniform_(
            *self.cfg.focal_length_range
        )

        # Sample target resolutions
        if self.cfg.discrete_resolutions:
            indices = torch.randint(
                len(self.cfg.discrete_resolutions),
                (num_samples,),
                device=self.device
            )
            resolutions = torch.tensor(
                self.cfg.discrete_resolutions,
                device=self.device
            )[indices]
        else:
            widths = torch.randint(
                self.cfg.target_width_range[0],
                self.cfg.target_width_range[1] + 1,
                (num_samples,),
                device=self.device
            )
            heights = torch.randint(
                self.cfg.target_height_range[0],
                self.cfg.target_height_range[1] + 1,
                (num_samples,),
                device=self.device
            )
            resolutions = torch.stack([widths, heights], dim=-1)

        # Compute crop and intrinsic parameters
        self._compute_transforms(env_ids, fov_scales, focal_lengths, resolutions)

    def _compute_transforms(
        self,
        env_ids: torch.Tensor,
        fov_scales: torch.Tensor,
        focal_lengths: torch.Tensor,
        resolutions: torch.Tensor,
    ):
        """Compute crop regions and intrinsic matrices."""
        W_render = self.cfg.render_width
        H_render = self.cfg.render_height

        # Reshape to (num_envs, num_agents)
        fov_scales = fov_scales.view(len(env_ids), self.num_agents)
        focal_lengths = focal_lengths.view(len(env_ids), self.num_agents)
        resolutions = resolutions.view(len(env_ids), self.num_agents, 2)

        # Compute crop dimensions
        crop_w = (W_render * fov_scales).int()
        crop_h = (H_render * fov_scales).int()
        crop_x = ((W_render - crop_w) // 2).int()
        crop_y = ((H_render - crop_h) // 2).int()

        # Store crop parameters
        self.crop_params[env_ids, :, 0] = crop_x
        self.crop_params[env_ids, :, 1] = crop_y
        self.crop_params[env_ids, :, 2] = crop_w
        self.crop_params[env_ids, :, 3] = crop_h

        # Store target sizes
        self.target_sizes[env_ids] = resolutions

        # Compute scale factors
        W_target = resolutions[:, :, 0].float()
        scale = W_target / crop_w.float()

        # Compute final focal lengths
        f_final = focal_lengths * scale

        # Build intrinsic matrices (centered principal points)
        cx = W_target / 2.0
        cy = resolutions[:, :, 1].float() / 2.0

        self.intrinsic_matrices[env_ids, :, 0, 0] = f_final
        self.intrinsic_matrices[env_ids, :, 1, 1] = f_final
        self.intrinsic_matrices[env_ids, :, 0, 2] = cx
        self.intrinsic_matrices[env_ids, :, 1, 2] = cy
        self.intrinsic_matrices[env_ids, :, 2, 2] = 1.0

    def process_images(
        self,
        images: torch.Tensor,  # (num_envs, num_agents, H_render, W_render, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply crop and resize to rendered images.

        Returns:
            processed_images: List of processed images (variable sizes)
            intrinsic_matrices: (num_envs, num_agents, 3, 3)
        """
        # For training with variable resolutions, we have two options:
        # 1. Pad to max size (simpler, may waste compute)
        # 2. Use nested tensors or process in groups (more complex)

        # Option 1: Process and pad to maximum target size
        max_w = self.cfg.target_width_range[1]
        max_h = self.cfg.target_height_range[1]

        processed = torch.zeros(
            self.num_envs, self.num_agents, max_h, max_w, images.shape[-1],
            device=self.device, dtype=images.dtype
        )

        # Apply per-environment/agent crops and resizes
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                crop_x, crop_y, crop_w, crop_h = self.crop_params[env_idx, agent_idx].int()
                target_w, target_h = self.target_sizes[env_idx, agent_idx]

                # Crop
                cropped = images[env_idx, agent_idx,
                                crop_y:crop_y+crop_h,
                                crop_x:crop_x+crop_w]

                # Resize (using torch interpolation)
                cropped = cropped.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                resized = torch.nn.functional.interpolate(
                    cropped,
                    size=(target_h.item(), target_w.item()),
                    mode='bilinear',
                    align_corners=False
                )
                resized = resized.squeeze(0).permute(1, 2, 0)  # (H, W, C)

                # Place in output (top-left aligned, rest is padding)
                processed[env_idx, agent_idx, :target_h, :target_w] = resized

        return processed, self.intrinsic_matrices

    def process_images_batched(
        self,
        images: torch.Tensor,  # (num_envs * num_agents, H_render, W_render, C)
        output_size: tuple[int, int],  # Fixed output size for batched processing
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched processing with fixed output size (more efficient for training).

        All images are cropped according to their FOV scale and resized to
        the same output size. The intrinsic matrix is adjusted accordingly.
        """
        N = images.shape[0]
        H_render, W_render = images.shape[1:3]
        H_out, W_out = output_size

        # Flatten crop params
        crop_params = self.crop_params.view(N, 4)

        # Use grid_sample for differentiable crop+resize
        # Build sampling grid for each image
        grids = self._build_crop_grids(crop_params, (H_render, W_render), output_size)

        # Rearrange for grid_sample: (N, C, H, W)
        images_nchw = images.permute(0, 3, 1, 2)

        # Apply grid sample
        processed = torch.nn.functional.grid_sample(
            images_nchw,
            grids,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        # Rearrange back: (N, H, W, C)
        processed = processed.permute(0, 2, 3, 1)

        # Recompute intrinsics for fixed output size
        intrinsics = self._compute_intrinsics_for_output(output_size)

        return processed, intrinsics

    def _build_crop_grids(
        self,
        crop_params: torch.Tensor,  # (N, 4): crop_x, crop_y, crop_w, crop_h
        input_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Build sampling grids for grid_sample."""
        N = crop_params.shape[0]
        H_in, W_in = input_size
        H_out, W_out = output_size

        # Normalize crop coordinates to [-1, 1]
        crop_x = crop_params[:, 0]
        crop_y = crop_params[:, 1]
        crop_w = crop_params[:, 2]
        crop_h = crop_params[:, 3]

        # Create output grid coordinates
        y_out = torch.linspace(-1, 1, H_out, device=self.device)
        x_out = torch.linspace(-1, 1, W_out, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_out, x_out, indexing='ij')

        # Map to input coordinates for each image
        # grid_sample uses [-1, 1] where -1 is left/top, 1 is right/bottom
        grids = torch.zeros(N, H_out, W_out, 2, device=self.device)

        for i in range(N):
            # Map output grid to crop region in input
            x_start = 2.0 * crop_x[i] / W_in - 1.0
            x_end = 2.0 * (crop_x[i] + crop_w[i]) / W_in - 1.0
            y_start = 2.0 * crop_y[i] / H_in - 1.0
            y_end = 2.0 * (crop_y[i] + crop_h[i]) / H_in - 1.0

            grids[i, :, :, 0] = (grid_x + 1) / 2 * (x_end - x_start) + x_start
            grids[i, :, :, 1] = (grid_y + 1) / 2 * (y_end - y_start) + y_start

        return grids
```

### 4.5 Integration with Observation Pipeline

```python
class CameraObservationProcessor:
    """Integrates camera randomization with the observation pipeline."""

    def __init__(
        self,
        camera_cfg: CameraRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        self.processor = CameraProcessor(camera_cfg, num_envs, num_agents, device)
        self.fixed_output_size = (360, 640)  # Fixed size for policy input (nHD)

    def reset(self, env_ids: torch.Tensor):
        """Randomize camera parameters on environment reset."""
        self.processor.randomize(env_ids)

    def process(
        self,
        raw_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Process raw camera images for policy input.

        Returns:
            dict with:
                - "images": Processed images (N, H, W, C)
                - "intrinsics": Camera intrinsic matrices (N, 3, 3)
                - "fov_scale": FOV scale factors (N,) - for auxiliary loss
        """
        images, intrinsics = self.processor.process_images_batched(
            raw_images,
            self.fixed_output_size
        )

        return {
            "images": images,
            "intrinsics": intrinsics,
            "crop_params": self.processor.crop_params,
        }
```

---

## 5. Gimbal Mount Randomization

### 5.1 Joint Position Randomization

```python
@dataclass
class GimbalRandomizationCfg:
    """Configuration for gimbal joint randomization."""

    # Joint position offsets (radians)
    yaw_offset_range: tuple[float, float] = (-0.1, 0.1)    # ±5.7°
    pitch_offset_range: tuple[float, float] = (-0.05, 0.05) # ±2.9°
    roll_offset_range: tuple[float, float] = (-0.02, 0.02)  # ±1.1°

    # Joint dynamics randomization
    stiffness_scale_range: tuple[float, float] = (0.8, 1.2)
    damping_scale_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.0, 0.1)

    # Randomization mode
    mode: str = "reset"  # "reset", "startup", "prestartup"
```

### 5.2 Mount Offset Randomization (USD-level)

**Note**: Only available at prestartup.

```python
@dataclass
class MountOffsetRandomizationCfg:
    """Configuration for gimbal mount offset randomization (prestartup only)."""

    # Position offset from nominal mount point (meters)
    position_offset_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.01, 0.01),  # ±1cm forward/back
            "y": (-0.01, 0.01),  # ±1cm left/right
            "z": (-0.005, 0.005), # ±5mm up/down
        }
    )

    # Rotation offset (radians)
    rotation_offset_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "roll": (-0.02, 0.02),   # ±1.1°
            "pitch": (-0.02, 0.02),  # ±1.1°
            "yaw": (-0.02, 0.02),    # ±1.1°
        }
    )
```

---

## 6. Implementation Architecture

### 6.1 Randomization Manager

```python
class DomainRandomizationManager:
    """Centralized manager for all domain randomization."""

    def __init__(
        self,
        cfg: DomainRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        # Initialize sub-managers
        self.physics_randomizer = PhysicsRandomizer(cfg.physics, num_envs, device)
        self.camera_processor = CameraProcessor(cfg.camera, num_envs, num_agents, device)
        self.gimbal_randomizer = GimbalRandomizer(cfg.gimbal, num_envs, num_agents, device)

    def prestartup(self, scene):
        """Apply USD-level randomization before simulation."""
        if self.cfg.physics.scale_enabled:
            self.physics_randomizer.randomize_scale(scene)
        if self.cfg.gimbal.mount_offset_enabled:
            self.gimbal_randomizer.randomize_mount_offset(scene)

    def startup(self, env):
        """Apply post-simulation initialization."""
        self.physics_randomizer.randomize_mass(env)
        self.physics_randomizer.randomize_materials(env)

    def reset(self, env, env_ids: torch.Tensor):
        """Apply per-episode randomization."""
        self.physics_randomizer.randomize_mass_noise(env, env_ids)
        self.camera_processor.randomize(env_ids)
        self.gimbal_randomizer.randomize_joint_offsets(env, env_ids)

    def step(self, env):
        """Apply per-step randomization (if enabled)."""
        if self.cfg.camera.randomize_per_step:
            self.camera_processor.randomize()
```

### 6.2 Integration with DirectRLEnv

```python
class IrisMA6Env(DirectRLEnv):
    """Environment with integrated domain randomization."""

    def __init__(self, cfg: IrisMA6EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Initialize randomization manager
        self.randomization = DomainRandomizationManager(
            cfg.randomization,
            self.num_envs,
            self.cfg.num_agents,
            self.device,
        )

    def _setup_scene(self):
        super()._setup_scene()
        # Apply prestartup randomization
        self.randomization.prestartup(self.scene)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # Apply per-episode randomization
        self.randomization.reset(self, env_ids)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()

        # Process camera images through randomization pipeline
        if "images" in obs:
            processed = self.randomization.camera_processor.process(obs["images"])
            obs["images"] = processed["images"]
            obs["camera_intrinsics"] = processed["intrinsics"]

        return obs
```

---

## 7. Configuration Schema

### 7.1 Complete Configuration

```python
@dataclass
class DomainRandomizationCfg:
    """Complete domain randomization configuration."""

    # Enable/disable randomization
    enabled: bool = True

    # Physics randomization
    physics: PhysicsRandomizationCfg = field(default_factory=PhysicsRandomizationCfg)

    # Camera randomization
    camera: CameraRandomizationCfg = field(default_factory=CameraRandomizationCfg)

    # Gimbal randomization
    gimbal: GimbalRandomizationCfg = field(default_factory=GimbalRandomizationCfg)

    # Seed for reproducibility
    seed: int | None = None


@dataclass
class PhysicsRandomizationCfg:
    """Physics property randomization configuration."""

    # Mass
    mass: MassRandomizationCfg = field(default_factory=MassRandomizationCfg)

    # Scale (prestartup only)
    scale: ScaleRandomizationCfg = field(default_factory=ScaleRandomizationCfg)
    scale_enabled: bool = False  # Disabled by default (requires replicate_physics=False)

    # Materials
    material: MaterialRandomizationCfg = field(default_factory=MaterialRandomizationCfg)
```

### 7.2 Example Configuration

```python
# Full randomization for sim-to-real
sim2real_randomization = DomainRandomizationCfg(
    enabled=True,
    physics=PhysicsRandomizationCfg(
        mass=MassRandomizationCfg(
            body_mass_scale_range=(0.85, 1.15),
            payload_mass_range=(0.0, 0.3),
        ),
        material=MaterialRandomizationCfg(
            static_friction_range=(0.5, 1.5),
            dynamic_friction_range=(0.3, 1.0),
        ),
    ),
    camera=CameraRandomizationCfg(
        render_width=1920,
        render_height=1080,
        discrete_resolutions=[(1920, 1080), (1280, 720), (640, 360)],
        fov_scale_range=(0.6, 1.0),
        focal_length_range=(800.0, 1200.0),
        randomize_per_episode=True,
    ),
    gimbal=GimbalRandomizationCfg(
        yaw_offset_range=(-0.15, 0.15),
        pitch_offset_range=(-0.1, 0.1),
        stiffness_scale_range=(0.7, 1.3),
    ),
)

# Minimal randomization for debugging
debug_randomization = DomainRandomizationCfg(
    enabled=False,
)
```

---

## Appendix A: Intrinsic Matrix Reference

### A.1 OpenCV/Isaac Sim Convention

```
K = [[f_x,  0,  c_x],      f_x, f_y: Focal length in pixels
     [ 0,  f_y, c_y],      c_x, c_y: Principal point in pixels
     [ 0,   0,   1 ]]
```

### A.2 Conversion Formulas

```python
# Isaac Sim aperture to focal length (pixels)
f_pixels = focal_length_cm * image_width / horizontal_aperture_cm

# Focal length to aperture
horizontal_aperture_cm = focal_length_cm * image_width / f_pixels

# FOV from focal length
fov_horizontal = 2 * atan(image_width / (2 * f_pixels))
fov_vertical = 2 * atan(image_height / (2 * f_pixels))

# Focal length from FOV
f_pixels = image_width / (2 * tan(fov_horizontal / 2))
```

### A.3 Default Values

| Parameter | Value | Unit |
|-----------|-------|------|
| Focal length (Isaac Sim default) | 24.0 | cm |
| Horizontal aperture (35mm format) | 20.955 | cm |
| Vertical aperture (4:3 aspect) | 15.2909 | cm |
| Resulting horizontal FOV | ~47° | degrees |

---

## Appendix B: Performance Considerations

### B.1 Memory Usage

- Rendered images: `num_envs × num_agents × H × W × C × dtype_size`
- Crop parameters: `num_envs × num_agents × 4 × 4 bytes` (negligible)
- Processing buffer: Same as rendered images

### B.2 Computational Cost

| Operation | Relative Cost | GPU Accelerated |
|-----------|--------------|-----------------|
| Render | High | Yes |
| Crop | Low | Yes (via indexing) |
| Resize | Medium | Yes (grid_sample) |
| Intrinsic computation | Negligible | Yes |

### B.3 Optimization Tips

1. **Use batched processing** with fixed output size for training efficiency
2. **Pre-compute grids** when crop parameters don't change per-step
3. **Use half precision** (fp16) for image processing when possible
4. **Minimize CPU-GPU transfers** by keeping all processing on GPU

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-17 | Initial specification |
