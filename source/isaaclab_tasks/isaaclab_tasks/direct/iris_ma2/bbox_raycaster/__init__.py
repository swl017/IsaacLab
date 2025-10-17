# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched bounding box raycasting module for multi-agent RL environments.

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
    >>> from bbox_raycaster import BBoxRayCaster, BBoxRayCasterCfg
    >>> 
    >>> cfg = BBoxRayCasterCfg(
    ...     target_prim_paths=["/World/envs/env_.*/target"],
    ...     mesh_prim_paths=["/World/ground"],
    ...     num_cameras_per_env=2,
    ...     enable_occlusion_check=True,
    ... )
    >>> 
    >>> raycaster = BBoxRayCaster(cfg, num_envs=4096, num_targets_per_env=1, device="cuda:0")
    >>> raycaster.update(camera_poses, camera_intrinsics, target_poses)
    >>> 
    >>> # Access results
    >>> bboxes = raycaster.data.bboxes  # (N, C, T, 4)
    >>> valid = raycaster.data.valid_mask  # (N, C, T)
"""

from .bbox_raycaster import BBoxRayCaster
from .bbox_raycaster_cfg import BBoxRayCasterCfg
from .bbox_raycaster_data import BBoxRayCasterData

__all__ = ["BBoxRayCaster", "BBoxRayCasterCfg", "BBoxRayCasterData"]

__version__ = "1.0.0"