# Domain Randomization Module

Comprehensive domain randomization for sim-to-real transfer in the iris_ma6 multi-agent drone tracking environment.

## Overview

This module provides randomization of system properties to bridge the sim-to-real gap:

| Category | What it randomizes | Purpose |
|----------|-------------------|---------|
| **Camera** | FOV, resolution, focal length | Simulate different camera hardware |
| **Physics** | Mass, friction, restitution | Account for manufacturing variations |
| **Gimbal** | Joint offsets, dynamics | Model calibration errors |

### Scope Distinction

- **Domain Randomization** (this module): *What* the system is (mass, friction, camera FOV)
- **Initial States** (separate module): *Where* the system starts (position, velocity)

## Architecture

```
DomainRandomizer (Orchestrator)
├── CameraProcessor      # GPU-accelerated crop/resize pipeline
├── PhysicsRandomizer    # Mass and material randomization
└── GimbalRandomizer     # Joint offset and dynamics randomization
```

## Quick Start

```python
from isaaclab_tasks.direct.iris_ma6.domain_randomization import (
    DomainRandomizationCfg,
    DomainRandomizer,
)

# Create with default configuration
cfg = DomainRandomizationCfg()
randomizer = DomainRandomizer(
    cfg=cfg,
    num_envs=256,
    num_agents=3,
    device="cuda",
)

# Randomize at episode reset
env_ids = torch.arange(128, device="cuda")
randomizer.randomize_all(env_ids)

# Process camera images through crop+resize pipeline
processed_images, intrinsics = randomizer.process_images(
    images,  # (N, C, H, W) rendered images
    output_size=(360, 640),  # target resolution
)

# Get gimbal offsets to apply to position targets
offsets = randomizer.get_gimbal_offsets()  # (num_envs, num_agents, 3)
```

## Configuration

### Top-Level Configuration

```python
@configclass
class DomainRandomizationCfg:
    enabled: bool = True                    # Master switch
    physics: PhysicsRandomizationCfg        # Mass, friction, scale
    camera: CameraRandomizationCfg          # FOV, resolution, focal length
    gimbal: GimbalRandomizationCfg          # Joint offsets, dynamics
    mount_offset: MountOffsetRandomizationCfg  # Prestartup only
    seed: int | None = None                 # For reproducibility
```

### Camera Configuration

```python
@configclass
class CameraRandomizationCfg:
    enabled: bool = True

    # Render resolution (fixed at max quality)
    render_width: int = 1920
    render_height: int = 1080

    # FOV simulation via crop
    fov_scale_range: tuple[float, float] = (0.5, 1.0)
    # 1.0 = full FOV, 0.5 = crop to 50% (2x zoom)

    # Focal length in pixels
    focal_length_range: tuple[float, float] = (800.0, 1200.0)

    # Discrete output resolutions
    discrete_resolutions: list[tuple[int, int]] = [
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (640, 360),    # nHD
    ]

    # Randomization frequency
    randomize_per_episode: bool = True
    randomize_per_step: bool = False
    randomize_per_agent: bool = False
```

### Physics Configuration

```python
@configclass
class PhysicsRandomizationCfg:
    mass: MassRandomizationCfg
    scale: ScaleRandomizationCfg      # Prestartup only
    material: MaterialRandomizationCfg

@configclass
class MassRandomizationCfg:
    enabled: bool = True
    body_mass_scale_range: tuple[float, float] = (0.9, 1.1)   # ±10%
    body_mass_add_range: tuple[float, float] = (-0.05, 0.05)  # ±50g
    payload_mass_range: tuple[float, float] = (0.0, 0.2)      # 0-200g
    distribution: str = "uniform"  # or "log_uniform", "gaussian"
    recompute_inertia: bool = True

@configclass
class MaterialRandomizationCfg:
    enabled: bool = True
    static_friction_range: tuple[float, float] = (0.7, 1.3)
    dynamic_friction_range: tuple[float, float] = (0.5, 1.0)
    restitution_range: tuple[float, float] = (0.0, 0.3)
    make_consistent: bool = True  # Ensure dynamic <= static
```

### Gimbal Configuration

```python
@configclass
class GimbalRandomizationCfg:
    enabled: bool = True

    # Joint position offsets (radians)
    yaw_offset_range: tuple[float, float] = (-0.1, 0.1)    # ±5.7°
    pitch_offset_range: tuple[float, float] = (-0.05, 0.05) # ±2.9°
    roll_offset_range: tuple[float, float] = (-0.02, 0.02)  # ±1.1°

    # Joint dynamics
    stiffness_scale_range: tuple[float, float] = (0.8, 1.2)
    damping_scale_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.0, 0.1)

    # Randomization frequency
    randomize_offsets_per_episode: bool = True
    randomize_dynamics_per_episode: bool = False
```

## Camera Processing Pipeline

The camera processor simulates different camera configurations through computational operations on rendered images.

### Pipeline Flow

```
Render (1920×1080) → Crop (FOV simulation) → Resize (resolution simulation)
```

### FOV Simulation via Crop

```
fov_scale = 0.5 means:
- Crop to 960×540 pixels (center region)
- Effectively 2× zoom (narrower FOV)

fov_scale = 1.0 means:
- No crop (full image)
- Original FOV
```

### Intrinsic Matrix Transformations

The pipeline maintains correct intrinsic matrices through all transformations:

```
After crop:
  f_x' = f_x (unchanged)
  c_x' = c_x - crop_x

After resize (scale s = output_w / crop_w):
  f_x'' = f_x' × s
  c_x'' = c_x' × s

For centered crops (symmetric):
  c_x_final = output_w / 2
  c_y_final = output_h / 2
```

### Implementation

Uses `torch.nn.functional.grid_sample` for GPU-accelerated, differentiable crop+resize:

```python
# Build sampling grids for all images
grids = self._build_crop_grids(crop_params, input_size, output_size)

# Apply crop+resize in single operation
processed = F.grid_sample(
    images, grids,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False
)
```

## API Reference

### DomainRandomizer

Main orchestrator class coordinating all randomization.

```python
class DomainRandomizer:
    def __init__(
        self,
        cfg: DomainRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
    ): ...

    # Main randomization methods
    def randomize_all(self, env_ids: torch.Tensor | None = None): ...
    def randomize_camera(self, env_ids: torch.Tensor | None = None): ...
    def randomize_physics(self, env_ids: torch.Tensor | None = None): ...
    def randomize_gimbal(self, env_ids: torch.Tensor | None = None): ...

    # Application methods (require asset references)
    def apply_physics_randomization(
        self, asset: Articulation | RigidObject, env_ids=None, body_ids=None
    ): ...
    def apply_gimbal_dynamics(
        self, articulation: Articulation, env_ids=None, gimbal_joint_ids=None
    ): ...

    # Camera processing
    def process_images(
        self, images: torch.Tensor, output_size: tuple[int, int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    # Getters
    def get_gimbal_offsets(self, env_ids=None) -> torch.Tensor: ...
    def get_intrinsic_matrices(self) -> torch.Tensor: ...
    def get_fov_scales(self) -> torch.Tensor: ...
    def get_mass_scales(self, env_ids=None) -> torch.Tensor: ...
    def get_payload_masses(self, env_ids=None) -> torch.Tensor: ...

    # Component access
    def get_camera_processor(self) -> CameraProcessor: ...
    def get_physics_randomizer(self) -> PhysicsRandomizer: ...
    def get_gimbal_randomizer(self) -> GimbalRandomizer: ...

    # Logging
    def get_current_state_summary(self) -> dict: ...
```

### CameraProcessor

GPU-accelerated camera image processing.

```python
class CameraProcessor:
    def __init__(
        self,
        cfg: CameraRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
    ): ...

    def randomize(self, env_ids: torch.Tensor | None = None): ...

    def process_images_batched(
        self,
        images: torch.Tensor,  # (N, C, H, W) or (N, H, W, C)
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def get_intrinsic_matrices(self) -> torch.Tensor: ...
    def get_fov_scales(self) -> torch.Tensor: ...
    def get_crop_params(self) -> torch.Tensor: ...
```

### PhysicsRandomizer

Mass and material property randomization.

```python
class PhysicsRandomizer:
    def __init__(
        self,
        cfg: PhysicsRandomizationCfg,
        num_envs: int,
        device: torch.device | str,
    ): ...

    # Sampling (stores values, doesn't apply)
    def sample_mass_parameters(self, env_ids=None): ...
    def sample_material_parameters(self, env_ids=None): ...

    # Application (requires asset reference)
    def apply_mass_randomization(self, asset, env_ids=None, body_ids=None): ...
    def apply_material_randomization(self, asset, env_ids=None): ...

    # Getters
    def get_mass_scales(self, env_ids=None) -> torch.Tensor: ...
    def get_payload_masses(self, env_ids=None) -> torch.Tensor: ...
    def get_friction_coefficients(self, env_ids=None) -> tuple: ...
    def get_restitution(self, env_ids=None) -> torch.Tensor: ...
```

### GimbalRandomizer

Gimbal joint offset and dynamics randomization.

```python
class GimbalRandomizer:
    def __init__(
        self,
        cfg: GimbalRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
    ): ...

    # Sampling
    def randomize_joint_offsets(self, env_ids=None): ...
    def randomize_dynamics(self, env_ids=None): ...

    # Application
    def apply_dynamics_randomization(
        self, articulation, env_ids=None, gimbal_joint_ids=None
    ): ...

    # Getters
    def get_joint_offsets(self, env_ids=None) -> torch.Tensor: ...  # (N, A, 3)
    def get_yaw_offsets(self, env_ids=None) -> torch.Tensor: ...
    def get_pitch_offsets(self, env_ids=None) -> torch.Tensor: ...
    def get_roll_offsets(self, env_ids=None) -> torch.Tensor: ...
    def get_stiffness_scales(self, env_ids=None) -> torch.Tensor: ...
    def get_damping_scales(self, env_ids=None) -> torch.Tensor: ...
```

## Environment Integration

### Episode Reset

```python
def _reset_idx(self, env_ids: torch.Tensor):
    # Randomize domain parameters
    self.domain_randomizer.randomize_all(env_ids)

    # Apply physics randomization to assets
    for agent_idx in range(self.num_agents):
        self.domain_randomizer.apply_physics_randomization(
            self.robots[agent_idx], env_ids
        )

    # Reset other state...
```

### Gimbal Control

```python
def _apply_action(self, actions: torch.Tensor):
    # Get gimbal offsets
    offsets = self.domain_randomizer.get_gimbal_offsets()

    # Apply offsets to gimbal position targets
    gimbal_targets = self._compute_gimbal_targets(actions)
    gimbal_targets_corrected = gimbal_targets + offsets

    # Send to actuators
    self._set_gimbal_positions(gimbal_targets_corrected)
```

### Observation Processing

```python
def _get_observations(self) -> dict:
    # Get raw camera images
    images = self._get_camera_images()  # (num_envs * num_agents, C, H, W)

    # Process through domain randomization pipeline
    processed, intrinsics = self.domain_randomizer.process_images(
        images, output_size=(360, 640)
    )

    return {
        "images": processed,
        "intrinsics": intrinsics,
        # ... other observations
    }
```

## Testing

Run the test suite:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/domain_randomization/tests/run_tests.py
```

Test categories:
- Configuration instantiation and defaults
- Camera processing (crop/resize/intrinsics)
- Physics randomization (mass/material sampling)
- Gimbal randomization (joint offsets/dynamics)
- Integration (orchestrator workflow)

## File Structure

```
domain_randomization/
├── __init__.py                  # Module exports
├── domain_randomization_cfg.py  # Configuration dataclasses
├── camera_processor.py          # GPU crop/resize pipeline
├── physics_randomizer.py        # Mass/material randomization
├── gimbal_randomizer.py         # Joint offset/dynamics
├── domain_randomizer.py         # Main orchestrator
├── doc/
│   └── README.md               # This file
└── tests/
    ├── __init__.py
    ├── run_tests.py            # Test suite
    └── README.md               # Test documentation
```

## Design Decisions

### Why Computational Camera Randomization?

Isaac Sim camera resolution cannot be changed at runtime. Instead:
1. Render at maximum resolution (1920×1080)
2. Crop to simulate different FOVs
3. Resize to simulate different resolutions

This is computationally efficient (single grid_sample) and maintains correct intrinsic matrices.

### Why Symmetric Crops Only?

We constrain to centered principal points (c_x = W/2, c_y = H/2) because:
1. Simplifies intrinsic matrix computation
2. Most real cameras have near-centered principal points
3. Off-center principal points add complexity without significant benefit

### Why Separate Sampling and Application?

Physics/gimbal randomizers separate sampling from application:
1. **Sampling**: Generates new random values (no asset reference needed)
2. **Application**: Writes values to simulation (requires asset reference)

This allows:
- Testing without simulation
- Logging sampled values before application
- Batch application across multiple assets

## References

- [Domain Randomization Spec](../doc/domain_randomization_spec.md)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Domain Randomization for Sim-to-Real Transfer](https://arxiv.org/abs/1703.06907)
