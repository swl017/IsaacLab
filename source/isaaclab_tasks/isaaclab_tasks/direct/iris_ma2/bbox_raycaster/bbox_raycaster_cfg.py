# file: bbox_raycaster_cfg.py
"""Configuration for the batched bounding box raycaster with body-local occlusion."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass


@configclass
class BBoxRayCasterCfg:
    """Configuration for the batched bounding box raycaster.
    
    This configuration defines parameters for extracting 2D bounding boxes from 3D targets
    with body-local occlusion detection and validation in a batched multi-agent environment.
    """

    # Target configuration
    target_prim_paths: list[str] = MISSING
    """List of primitive paths for targets to detect.
    
    Example: ["/World/envs/env_.*/target"]
    The bounding boxes of these targets will be computed.
    """

    # Mesh configuration for raycasting
    mesh_prim_paths: list[str] = MISSING
    """List of mesh primitive paths for static environment occlusion.
    
    Example: ["/World/ground", "/World/envs/env_.*/obstacles"]
    These meshes are used to detect occlusions from static environment (ground, walls, etc.).
    """

    # Agent mesh configuration (NEW)
    load_agent_meshes: bool = True
    """Whether to load agent meshes for drone-to-drone occlusion detection. Defaults to True.
    
    When enabled, the raycaster will load each agent's mesh in body-local coordinates and
    test for occlusions between agents (e.g., one drone blocking another's view).
    """

    agent_mesh_simplification: float = 1.0
    """Mesh simplification factor for agent meshes. Defaults to 1.0 (no simplification).
    
    Range: 0.0 to 1.0, where:
    - 1.0: Keep all triangles (most accurate, slowest)
    - 0.5: Keep ~50% of triangles (balanced)
    - 0.1: Keep ~10% of triangles (fastest, least accurate)
    
    Lower values speed up raycasting but may miss fine details.
    """

    use_collision_proxy: bool = False
    """Use simple box collision proxies instead of full meshes. Defaults to False.
    
    When True, uses axis-aligned bounding boxes for agent collision instead of detailed meshes.
    This is much faster but less accurate. Useful for debugging or when mesh loading fails.
    """

    # Camera configuration
    num_cameras_per_env: int = MISSING
    """Number of cameras per environment.
    
    This should match the number of agents with cameras in your multi-agent setup.
    """
    
    num_cameras_per_agent: int = 1
    """Number of cameras per agent. Defaults to 1.
    
    This should match the number of cameras assigned to each agent.
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
    
    If enabled, rays are cast from camera to target to detect occlusions from:
    1. Static environment (ground, obstacles)
    2. Dynamic agents (other drones, if load_agent_meshes is True)
    """

    occlusion_ray_pattern: Literal["9point", "corners_only", "center_only"] = "center_only"
    """Pattern for occlusion test points. Defaults to "center_only".
    
    - "center_only": Test only center point (fastest, least accurate)
    - "corners_only": Test only 4 bottom corners (faster)
    - "9point": Test 4 corners + 4 edge midpoints + 1 center (most accurate, slowest)
    
    More test points increase accuracy but slow down raycasting.
    """

    occlusion_visibility_threshold: float = 0.5
    """Fraction of test points that must be visible to mark target as visible. Defaults to 0.5.
    
    For example, with "9point" pattern and threshold 0.5, at least 5 out of 9 points must be visible.
    With "center_only", this threshold has no effect (single point is binary).
    """

    occlusion_ray_tolerance: float = 1.1
    """Tolerance multiplier for ray-target intersection checking. Defaults to 1.1.
    
    Multiplies the target's bbox radius to account for floating-point errors and provide
    some margin. Larger values are more permissive but may allow false positives.
    """

    # Performance
    max_distance: float = 100.0
    """Maximum raycasting distance in meters. Defaults to 100.0.
    
    Targets beyond this distance are not processed for occlusion. Set this based on your
    environment size to improve performance.
    """

    # Numerical stability
    projection_epsilon: float = 1e-6
    """Epsilon for preventing division by zero in projections. Defaults to 1e-6."""

    quat_normalize_epsilon: float = 1e-8
    """Epsilon for quaternion normalization. Defaults to 1e-8."""

    gimbal_lock_threshold: float = 0.99
    """Threshold for detecting gimbal lock (z-axis verticality). Defaults to 0.99.
    
    When camera z-axis is nearly vertical (dot product > threshold), a warning may be issued.
    """

    warn_gimbal_lock: bool = False
    """Whether to warn when gimbal lock is detected. Defaults to False.
    
    Enable this during debugging to identify problematic camera orientations.
    """

    # Debug and visualization
    debug_vis: bool = False
    """Enable debug visualization. Defaults to False.
    
    When enabled, visualizes bounding boxes, rays, and other debug information in the viewport.
    """

    debug_vis_corners: bool = False
    """Visualize projected bbox corners. Defaults to False.
    
    Only used if debug_vis is True. Shows the 3D corners of bounding boxes in world frame.
    """

    debug_vis_occlusion_rays: bool = False
    """Visualize occlusion rays. Defaults to False.
    
    Only used if debug_vis is True. Shows rays cast from cameras to targets and their hit points.
    """

    debug_memory: bool = False
    """Track and print memory usage. Defaults to False.
    
    When enabled, prints GPU memory usage at various stages of the update cycle. Useful for
    profiling and optimization.
    """

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/BBoxRayCaster"
    )
    """Configuration for visualization markers. Defaults to RAY_CASTER_MARKER_CFG.
    
    Note:
        This attribute is only used when debug visualization is enabled.
    """