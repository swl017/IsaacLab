# BBoxRayCasterV2 Documentation

This directory contains documentation for the BBoxRayCasterV2 module - a GPU-accelerated batched bounding box raycaster for multi-agent reinforcement learning environments.

## Document Index

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, data flow, and component overview |
| [CONFIGURATION.md](CONFIGURATION.md) | Complete configuration reference with examples |
| [OCCLUSION_DETECTION.md](OCCLUSION_DETECTION.md) | Occlusion detection pipeline and algorithms |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Integration guide with code examples |
| [ISAAC_SIM_BBOX_COMPARISON.md](ISAAC_SIM_BBOX_COMPARISON.md) | Comparison with Isaac Sim built-in bbox annotators |

## Quick Overview

**BBoxRayCasterV2** extracts 2D bounding boxes from 3D targets across multiple cameras and parallel environments using batched GPU operations. Key features:

- **Fully Batched Operations**: All computations are vectorized for maximum GPU efficiency
- **Multi-Environment Support**: Process thousands of environments in parallel
- **Occlusion Detection**: Raycasting-based detection for static obstacles and dynamic agents
- **Self-Occlusion**: Detects when an agent's own body blocks camera view
- **Temporal Smoothing**: EMA filter prevents oscillation at occlusion boundaries
- **Configurable Validation**: Size constraints, partial detection, FOV checking

## Tensor Dimensions

Throughout this module, tensors follow a consistent naming convention:

| Symbol | Meaning | Description |
|--------|---------|-------------|
| N | num_envs | Number of parallel environments |
| C | num_cameras | Cameras per environment (= num_agents) |
| T | num_targets | Targets per environment |
| K | num_test_points | Occlusion test points per target (1, 4, or 9) |

## Module Structure

```
bbox_raycaster_v2/
├── __init__.py                  # Module exports
├── bbox_raycaster_v2.py         # Main raycaster class
├── bbox_raycaster_v2_cfg.py     # Configuration dataclass
├── bbox_raycaster_v2_data.py    # Output data container
├── utils/
│   ├── __init__.py
│   ├── projection.py            # Camera projection utilities
│   ├── bbox_ops.py              # Bounding box operations
│   ├── mesh_utils.py            # USD mesh loading
│   ├── occlusion.py             # Basic occlusion checking
│   ├── occlusion_fully_batched.py  # Optimized batched occlusion
│   └── inter_target_occlusion.py   # Image-space IoU occlusion
├── tests/
│   ├── run_tests.py             # Test runner
│   └── test_bbox_raycaster_v2.py
└── doc/                         # This documentation
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-03 | Full mesh loading (body + propellers), temporal smoothing, shape bug fixes |
| 1.0.0 | 2026-01 | Initial implementation with basic occlusion |
