# file: utils/__init__.py

"""Utilities for batched bounding box raycasting."""

from .bbox_ops import (
    bbox_xyxy_to_xywh,
    check_all_corners_visible,
    compute_2d_bbox_from_corners,
    get_bbox_corners_local,
    normalize_bboxes,
    validate_bbox_sizes,
)
from .occlusion import (
    batch_check_occlusion_body_local,
    compute_bbox_diagonal,
    compute_bbox_size,
    generate_occlusion_test_points,
)
from .projection import (
    batch_project_to_image_plane,
    batch_transform_points,
    batch_transform_to_camera_frame,
    create_intrinsic_matrix_tensor,
    check_gimbal_lock,
    check_points_in_fov,
    validate_projections,
)

__all__ = [
    # bbox_ops
    "bbox_xyxy_to_xywh",
    "check_all_corners_visible",
    "compute_2d_bbox_from_corners",
    "get_bbox_corners_local",
    "normalize_bboxes",
    "validate_bbox_sizes",
    # occlusion
    "batch_check_occlusion_body_local",
    "compute_bbox_diagonal",
    "compute_bbox_size",
    "generate_occlusion_test_points",
    # projection
    "batch_project_to_image_plane",
    "batch_transform_points",
    "batch_transform_to_camera_frame",
    "create_intrinsic_matrix_tensor",
    "check_gimbal_lock",
    "check_points_in_fov",
    "validate_projections",
]