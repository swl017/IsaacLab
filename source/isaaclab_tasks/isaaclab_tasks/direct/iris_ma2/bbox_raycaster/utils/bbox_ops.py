# file: utils/bbox_ops.py

"""Bounding box operations for batched bbox raycasting."""

import torch
import numpy as np
from typing import Optional

import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom


def get_bbox_corners_local(
    prim_path: str,
    device: str
) -> torch.Tensor:
    """Extract 8 corners of axis-aligned bounding box from a prim.
    
    Uses Isaac Sim's create_bbox_cache to get the bounding box, then computes
    the 8 corners in the local frame of the object.
    
    Args:
        prim_path: Path to the primitive.
        device: Device to create tensor on.
        
    Returns:
        8 corners of the bounding box in local frame. Shape (8, 3).
        Order: [min_x_min_y_min_z, max_x_min_y_min_z, min_x_max_y_min_z, max_x_max_y_min_z,
                min_x_min_y_max_z, max_x_min_y_max_z, min_x_max_y_max_z, max_x_max_y_max_z]
    """
    # Get the first matching prim (accept any valid prim)
    prim = prim_utils.get_first_matching_child_prim(
        prim_path, predicate=lambda x: "target" in x
    )

    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for bbox extraction: {prim_path}")
    
    # Create bbox cache and compute bounds
    bbox_cache = bounds_utils.create_bbox_cache()
    bbox = bbox_cache.ComputeWorldBound(prim)
    
    # Get bounding box range
    bbox_range = bbox.GetRange()
    min_corner = np.array(bbox_range.GetMin())
    max_corner = np.array(bbox_range.GetMax())
    
    # Get prim's world transform to convert to local frame
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    world_transform_np = np.array(world_transform).T  # Transpose for correct format
    
    # Compute inverse transform to get local frame
    try:
        inv_transform = np.linalg.inv(world_transform_np)
    except np.linalg.LinAlgError:
        # If inverse fails, use identity (assume bbox is already in local frame)
        inv_transform = np.eye(4)
    
    # Generate 8 corners in world frame
    corners_world = np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
    ])
    
    # Transform to local frame
    corners_homogeneous = np.hstack([corners_world, np.ones((8, 1))])
    corners_local_homogeneous = (inv_transform @ corners_homogeneous.T).T
    corners_local = corners_local_homogeneous[:, :3]
    
    return torch.tensor(corners_local, dtype=torch.float32, device=device)


def compute_2d_bbox_from_corners(
    corners_2d: torch.Tensor,
    valid_corners: torch.Tensor,
    min_area_pixels: float = 4.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D bounding box from projected corners.
    
    Args:
        corners_2d: Projected 2D corner positions. Shape (N, C, T, 8, 2).
        valid_corners: Mask indicating valid corners. Shape (N, C, T, 8).
        min_area_pixels: Minimum bbox area in pixels to be considered valid.
        
    Returns:
        Tuple of:
            - bbox_xyxy: Bounding boxes in (min_x, min_y, max_x, max_y) format. Shape (N, C, T, 4).
            - valid_bbox: Boolean mask for valid bounding boxes. Shape (N, C, T).
    """
    # Use large values for invalid corners to not affect min/max
    INF = 1e6
    
    # Prepare corners for min computation
    corners_masked_min = torch.where(
        valid_corners.unsqueeze(-1),
        corners_2d,
        torch.full_like(corners_2d, INF)
    )
    
    # Prepare corners for max computation
    corners_masked_max = torch.where(
        valid_corners.unsqueeze(-1),
        corners_2d,
        torch.full_like(corners_2d, -INF)
    )
    
    # Compute min and max over corners dimension
    min_xy, _ = torch.min(corners_masked_min, dim=-2)  # (N, C, T, 2)
    max_xy, _ = torch.max(corners_masked_max, dim=-2)  # (N, C, T, 2)
    
    # Check if any valid corners exist
    has_valid_corners = valid_corners.any(dim=-1)  # (N, C, T)
    
    # Compute bbox dimensions
    width = max_xy[..., 0] - min_xy[..., 0]
    height = max_xy[..., 1] - min_xy[..., 1]
    
    # Clamp to prevent negative dimensions
    width = torch.clamp(width, min=0.0)
    height = torch.clamp(height, min=0.0)
    
    # Check for degenerate boxes
    area = width * height
    valid_area = area >= min_area_pixels
    valid_dimensions = (width > 0) & (height > 0)
    
    # Combine validity checks
    valid_bbox = has_valid_corners & valid_area & valid_dimensions
    
    # Stack into xyxy format
    bbox_xyxy = torch.stack([
        min_xy[..., 0],
        min_xy[..., 1],
        max_xy[..., 0],
        max_xy[..., 1]
    ], dim=-1)
    
    return bbox_xyxy, valid_bbox


def bbox_xyxy_to_xywh(bbox_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert bounding box from xyxy to xywh format.
    
    Args:
        bbox_xyxy: Bboxes in (min_x, min_y, max_x, max_y) format. Shape (N, C, T, 4).
        
    Returns:
        Bboxes in (center_x, center_y, width, height) format. Shape (N, C, T, 4).
    """
    min_x, min_y, max_x, max_y = bbox_xyxy.split(1, dim=-1)
    
    # Compute center and dimensions
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    w = max_x - min_x
    h = max_y - min_y
    
    return torch.cat([cx, cy, w, h], dim=-1)


def normalize_bboxes(
    bboxes: torch.Tensor,
    image_shapes: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """Normalize bounding boxes by image dimensions.
    
    Args:
        bboxes: Bboxes in (center_x, center_y, width, height) format. Shape (N, C, T, 4).
        image_shapes: Image dimensions (height, width). Shape (N, C, 2).
        epsilon: Epsilon to prevent division by zero.
        
    Returns:
        Normalized bboxes in [0, 1] range. Shape (N, C, T, 4).
    """
    N, C = image_shapes.shape[:2]
    
    # Extract dimensions
    img_h = image_shapes[:, :, 0].view(N, C, 1, 1)  # (N, C, 1, 1)
    img_w = image_shapes[:, :, 1].view(N, C, 1, 1)
    
    # Prevent division by zero
    img_w_safe = torch.clamp(img_w, min=epsilon)
    img_h_safe = torch.clamp(img_h, min=epsilon)
    
    # Split bbox components
    cx, cy, w, h = bboxes.split(1, dim=-1)
    
    # Normalize
    cx_norm = cx / img_w_safe
    cy_norm = cy / img_h_safe
    w_norm = w / img_w_safe
    h_norm = h / img_h_safe
    
    # Clamp to [0, 1] to handle numerical errors
    cx_norm = torch.clamp(cx_norm, 0.0, 1.0)
    cy_norm = torch.clamp(cy_norm, 0.0, 1.0)
    w_norm = torch.clamp(w_norm, 0.0, 1.0)
    h_norm = torch.clamp(h_norm, 0.0, 1.0)
    
    return torch.cat([cx_norm, cy_norm, w_norm, h_norm], dim=-1)


def validate_bbox_sizes(
    bboxes_norm: torch.Tensor,
    min_size: tuple[float, float],
    max_size: tuple[float, float]
) -> torch.Tensor:
    """Check if normalized bbox sizes are within valid range.
    
    Args:
        bboxes_norm: Normalized bboxes (cx, cy, w, h). Shape (N, C, T, 4).
        min_size: Minimum (width, height) as fraction of image.
        max_size: Maximum (width, height) as fraction of image.
        
    Returns:
        Boolean mask for valid bbox sizes. Shape (N, C, T).
    """
    # Extract width and height
    w = bboxes_norm[..., 2]
    h = bboxes_norm[..., 3]
    
    # Check size constraints
    valid_w = (w >= min_size[0]) & (w <= max_size[0])
    valid_h = (h >= min_size[1]) & (h <= max_size[1])
    
    return valid_w & valid_h


def check_all_corners_visible(
    valid_corners: torch.Tensor,
    allow_partial: bool = False
) -> torch.Tensor:
    """Check if required corners are visible.
    
    Args:
        valid_corners: Mask for valid corners. Shape (N, C, T, 8).
        allow_partial: Whether to allow partial detections.
        
    Returns:
        Boolean mask indicating sufficient corner visibility. Shape (N, C, T).
    """
    if allow_partial:
        # At least one corner must be valid
        return valid_corners.any(dim=-1)
    else:
        # All corners must be valid
        return valid_corners.all(dim=-1)