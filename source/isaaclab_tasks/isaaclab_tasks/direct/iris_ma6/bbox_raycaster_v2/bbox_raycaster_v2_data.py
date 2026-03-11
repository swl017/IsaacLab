# file: bbox_raycaster_v2_data.py
import torch
from dataclasses import dataclass


@dataclass
class BBoxRayCasterV2Data:
    """Data container for the batched bounding box raycaster V2.
    
    This container holds all data related to bounding box detection across multiple
    environments, cameras, and targets in a batched format for efficient GPU operations.
    """

    # Camera poses in world frame
    camera_pos_w: torch.Tensor = None
    """Position of camera origins in world frame.
    
    Shape is (N, C, 3), where N is the number of environments and C is the number of cameras per environment.
    """

    camera_quat_w: torch.Tensor = None
    """Orientation of camera origins in quaternion (w, x, y, z) in world frame.
    
    Shape is (N, C, 4), where N is the number of environments and C is the number of cameras per environment.
    """

    # Target poses in world frame
    target_pos_w: torch.Tensor = None
    """Position of targets in world frame.
    
    Shape is (N, T, 3), where N is the number of environments and T is the number of targets per environment.
    """

    target_quat_w: torch.Tensor = None
    """Orientation of targets in quaternion (w, x, y, z) in world frame.
    
    Shape is (N, T, 4), where N is the number of environments and T is the number of targets per environment.
    """

    # Camera intrinsics
    intrinsic_matrices: torch.Tensor = None
    """Camera intrinsic matrices.
    
    Shape is (N, C, 3, 3), where N is the number of environments and C is the number of cameras per environment.
    Each 3x3 matrix contains [fx, 0, cx; 0, fy, cy; 0, 0, 1].
    """

    image_shapes: torch.Tensor = None
    """Image dimensions for each camera.
    
    Shape is (N, C, 2), where the last dimension contains (height, width) in pixels.
    """

    # Output: Bounding boxes
    bboxes: torch.Tensor = None
    """2D bounding boxes in pixel coordinates.
    
    Shape is (N, C, T, 4), where the last dimension contains (center_x, center_y, width, height) in pixels.
    """

    bboxes_xyxy: torch.Tensor = None
    """2D bounding boxes in pixel coordinates.

    Shape is (N, C, T, 4), where the last dimension contains (x_min, y_min, x_max, y_max) in pixels.
    """

    bboxes_normalized: torch.Tensor = None
    """2D bounding boxes normalized by image dimensions.
    
    Shape is (N, C, T, 4), where values are in [0, 1] range relative to image dimensions.
    """

    bbox_empty: torch.Tensor = None
    """Mask indicating empty bounding box slots.

    Shape is (N, C, T). A value of True means bbox data is empty and zero-filled.
    """

    bbox_compute_empty_mask: torch.Tensor = None
    bbox_visibility_empty_mask: torch.Tensor = None
    bbox_corners_empty_mask: torch.Tensor = None
    bbox_size_empty_mask: torch.Tensor = None
    bbox_confidence: torch.Tensor = None
    """Confidence for each bbox. Shape is (N, C, T)."""

    occluded: torch.Tensor = None
    """Inter-target occlusion mask. Shape is (N, C, T)."""

    self_occlusion_mask: torch.Tensor = None
    """Self-occlusion mask from ego body mesh. Shape is (N, C, T)."""

    # Debug/visualization data (optional)
    projected_corners_2d: torch.Tensor = None
    """Projected 2D corner positions for debugging.
    
    Shape is (N, C, T, 8, 2), containing pixel coordinates of 8 bbox corners.
    Only populated when debug visualization is enabled.
    """

    corners_valid_mask: torch.Tensor = None
    """Mask indicating which corners are valid (within FOV and in front of camera).
    
    Shape is (N, C, T, 8). Only populated when debug visualization is enabled.
    """

    occlusion_test_points_w: torch.Tensor = None
    """Test points in world frame used for occlusion detection.
    
    Shape is (N, T, K, 3), where K is the number of test points per target.
    Only populated when debug visualization is enabled.
    """

    occlusion_ray_hits_w: torch.Tensor = None
    """Ray hit positions in world frame for occlusion rays.
    
    Shape is (N, C, T, K, 3), where K is the number of occlusion test rays per target.
    Only populated when debug visualization is enabled.
    """

    occlusion_visibility_ratio: torch.Tensor = None
    """Ratio of visible test points for each target.
    
    Shape is (N, C, T). Values range from 0.0 (fully occluded) to 1.0 (fully visible).
    Only populated when occlusion checking is enabled.
    """

    ray_dir_w: torch.Tensor = None
    """Ray directions from camera to target in world frame.

    Shape is (N, C, T, 3), where N is the number of environments, C is the number of cameras per environment,
    and T is the number of targets per environment.
    """
