# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the batched bounding box raycaster."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass


@configclass
class BBoxRayCasterCfg:
    """Configuration for the batched bounding box raycaster.
    
    This configuration defines parameters for extracting 2D bounding boxes from 3D targets
    with occlusion detection and validation in a batched multi-agent environment.
    """

    # Target configuration
    target_prim_paths: list[str] = MISSING
    """List of primitive paths for targets to detect.
    
    Example: ["/World/envs/env_.*/target"]
    The bounding boxes of these targets will be computed.
    """

    # Mesh configuration for raycasting
    mesh_prim_paths: list[str] = MISSING
    """List of mesh primitive paths to use for occlusion raycasting.
    
    Example: ["/World/ground", "/World/envs/env_.*/obstacles"]
    These meshes are used to detect occlusions between cameras and targets.
    """

    # Camera configuration
    num_cameras_per_env: int = MISSING
    """Number of cameras per environment.
    
    This should match the number of agents with cameras in your multi-agent setup.
    """

    # Validation thresholds
    min_bbox_size: tuple[float, float] = (0.01, 0.01)
    """Minimum bounding box size as fraction of image dimensions. Defaults to (0.01, 0.01).
    
    Format is (width_fraction, height_fraction). Bboxes smaller than this are marked invalid.
    For example, (0.01, 0.01) means minimum 1% of image width and height.
    """

    max_bbox_size: tuple[float, float] = (0.95, 0.95)
    """Maximum bounding box size as fraction of image dimensions. Defaults to (0.95, 0.95).
    
    Format is (width_fraction, height_fraction). Bboxes larger than this are marked invalid.
    For example, (0.95, 0.95) means maximum 95% of image width and height.
    """

    partial_detection_allowed: bool = False
    """Whether to allow partial detections (some corners out of FOV). Defaults to False.
    
    If False, all 8 corners of the 3D bounding box must be visible in the camera FOV.
    If True, bboxes are computed even if some corners are outside the FOV.
    """

    min_bbox_area_pixels: float = 4.0
    """Minimum bounding box area in pixels to be considered valid. Defaults to 4.0.
    
    This prevents degenerate bounding boxes (e.g., 2x2 pixels or smaller) from being marked valid.
    """

    # Occlusion detection
    enable_occlusion_check: bool = True
    """Whether to perform occlusion checking using raycasting. Defaults to True.
    
    If enabled, rays are cast from camera to target to detect occlusions.
    """

    occlusion_ray_pattern: Literal["9point", "corners_only", "center_only"] = "center_only"
    """Pattern for occlusion test points. Defaults to "center_only".
    
    - "9point": Test 4 corners + 4 edge midpoints + 1 center (most accurate)
    - "corners_only": Test only 4 corners (faster)
    - "center_only": Test only center point (fastest, least accurate)
    """

    occlusion_visibility_threshold: float = 0.5
    """Fraction of test points that must be visible to mark target as visible. Defaults to 0.5.
    
    For example, with "9point" pattern and threshold 0.5, at least 5 out of 9 points must be visible.
    """

    occlusion_ray_tolerance: float = 1.1
    """Tolerance multiplier for ray-target intersection checking. Defaults to 1.1.
    
    Multiplies the target's bbox diagonal to account for floating-point errors.
    Larger values are more permissive but may allow false positives.
    """

    # Performance
    max_distance: float = 100.0
    """Maximum raycasting distance in meters. Defaults to 100.0.
    
    Targets beyond this distance are not processed.
    """

    # Numerical stability
    projection_epsilon: float = 1e-6
    """Epsilon for preventing division by zero in projections. Defaults to 1e-6."""

    quat_normalize_epsilon: float = 1e-8
    """Epsilon for quaternion normalization. Defaults to 1e-8."""

    gimbal_lock_threshold: float = 0.99
    """Threshold for detecting gimbal lock (z-axis verticality). Defaults to 0.99."""

    warn_gimbal_lock: bool = False
    """Whether to warn when gimbal lock is detected. Defaults to False."""

    # Debug and visualization
    debug_vis: bool = False
    """Enable debug visualization. Defaults to False."""

    debug_vis_corners: bool = False
    """Visualize projected bbox corners. Defaults to False. Only used if debug_vis is True."""

    debug_vis_occlusion_rays: bool = False
    """Visualize occlusion rays. Defaults to False. Only used if debug_vis is True."""

    debug_memory: bool = False
    """Track and print memory usage. Defaults to False."""

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/BBoxRayCaster"
    )
    """Configuration for visualization markers. Defaults to RAY_CASTER_MARKER_CFG.
    
    Note:
        This attribute is only used when debug visualization is enabled.
    """