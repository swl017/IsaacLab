# file: __init__.py

"""Self-contained batched bounding box raycasting V2 for multi-agent RL environments.

This module provides efficient GPU-accelerated 2D bounding box extraction from 3D targets
across multiple cameras and parallel environments. It includes occlusion detection via
raycasting and comprehensive validation.

Key Features:
    - Fully batched operations for maximum GPU efficiency
    - Occlusion detection using Warp raycasting
    - Configurable validation (partial detection, size constraints)
    - Memory-efficient with pre-allocated buffers
    - Numerically stable with edge case handling
    - Debug visualization support

Example:
    >>> from bbox_raycaster_v2 import BBoxRayCasterV2, BBoxRayCasterV2Cfg
    >>> 
    >>> cfg = BBoxRayCasterV2Cfg(
    ...     target_prim_paths=["/World/envs/env_.*/target"],
    ...     mesh_prim_paths=["/World/ground"],
    ...     num_cameras_per_env=2,
    ...     enable_occlusion_check=True,
    ... )
    >>> 
    >>> raycaster = BBoxRayCasterV2(cfg, num_envs=4096, num_targets_per_env=1, device="cuda:0")
    >>> raycaster.update(camera_poses, camera_intrinsics, target_poses)
    >>> 
    >>> # Access results
    >>> bboxes = raycaster.data.bboxes  # (N, C, T, 4)
    >>> bbox_empty = raycaster.data.bbox_empty  # (N, C, T)
"""

from .bbox_raycaster_v2 import BBoxRayCasterV2
from .bbox_raycaster_v2_cfg import BBoxRayCasterV2Cfg
from .bbox_raycaster_v2_data import BBoxRayCasterV2Data
from .detector_replicator import DetectorReplicator, DetectorReplicatorCfg, NoiseModelParams
from . import utils

__all__ = [
    "BBoxRayCasterV2",
    "BBoxRayCasterV2Cfg",
    "BBoxRayCasterV2Data",
    "DetectorReplicator",
    "DetectorReplicatorCfg",
    "NoiseModelParams",
    "utils",
]

__version__ = "2.0.0"
