# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Optional

import isaacsim.core.utils.bounds as bounds_utils
from pxr import UsdGeom

import isaaclab.utils.math as math_utils


class BBoxGenerator:
    """Generates 2D bounding boxes for 3D targets in camera coordinates."""

    def __init__(
        self,
        camera_width: int = 640,
        camera_height: int = 480,
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
        device: str = "cuda",
    ):
        """Initialize the bounding box generator.

        Args:
            camera_width: Camera image width in pixels
            camera_height: Camera image height in pixels
            focal_length: Camera focal length in mm (base focal length)
            horizontal_aperture: Camera horizontal aperture in mm
            device: Device to run computations on
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.base_focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.device = device
        self._bbox_cache = bounds_utils.create_bbox_cache()

    def project_3d_to_2d(self, points_3d_camera: torch.Tensor, focal_length: torch.Tensor | None = None) -> torch.Tensor:
        """Project 3D points in camera frame to 2D image coordinates.

        Args:
            points_3d_camera: 3D points in camera frame [N, 3] or [N, M, 3]
            focal_length: Dynamic focal length tensor [N] (optional, uses base if None)

        Returns:
            2D image coordinates [N, 2] or [N, M, 2]
        """
        original_shape = points_3d_camera.shape
        if len(original_shape) == 3:
            # Reshape [N, M, 3] -> [N*M, 3]
            points_3d_camera = points_3d_camera.view(-1, 3)
            if focal_length is not None:
                focal_length = focal_length.repeat_interleave(original_shape[1])

        # Use dynamic focal length if provided, otherwise use base
        if focal_length is None:
            fx = (self.base_focal_length / self.horizontal_aperture) * self.camera_width
            fy = fx
            cx = self.camera_width / 2.0
            cy = self.camera_height / 2.0
        else:
            fx = (focal_length / self.horizontal_aperture) * self.camera_width
            fy = fx
            cx = self.camera_width / 2.0
            cy = self.camera_height / 2.0

            # Expand scalars to match points dimensions
            if fx.dim() == 1:
                fx = fx.unsqueeze(-1)
                fy = fy.unsqueeze(-1)

        # Avoid division by zero
        z = torch.clamp(points_3d_camera[:, 2:3], min=1e-6)

        # Project to normalized image coordinates
        x_norm = points_3d_camera[:, 0:1] / z
        y_norm = points_3d_camera[:, 1:2] / z

        # Convert to pixel coordinates
        u = fx * x_norm + cx
        v = fy * y_norm + cy

        points_2d = torch.cat([u, v], dim=1)

        if len(original_shape) == 3:
            # Reshape back to [N, M, 2]
            points_2d = points_2d.view(original_shape[0], original_shape[1], 2)

        return points_2d

    def generate_2d_bbox(
        self,
        prim_path: str,
        camera_view_transform: torch.Tensor,
        focal_length: torch.Tensor | None = None,
        min_bbox_size: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D bounding boxes for a 3D target prim.

        Args:
            prim_path: The path to the prim to generate the bounding box for.
            camera_view_transform: The view transform of the camera.
            focal_length: Dynamic focal length tensor [N] (optional, uses base if None)
            min_bbox_size: Minimum bounding box size in pixels

        Returns:
            Tuple of (bounding_boxes, valid_mask)
            - bounding_boxes: [N, 4] (x_min, y_min, x_max, y_max) in pixel coordinates
            - valid_mask: [N] boolean mask indicating if bbox is valid (target visible)
        """
        # Compute the AABB of the prim in world coordinates
        bbox3d_w = bounds_utils.compute_aabb(self._bbox_cache, prim_path)
        if bbox3d_w is None:
            return torch.zeros((1, 4), device=self.device), torch.zeros((1,), dtype=torch.bool, device=self.device)

        # aabb is a numpy array of shape (2, 3) -> convert to corners
        corners_w = torch.tensor(
            [
                [bbox3d_w[0, 0], bbox3d_w[0, 1], bbox3d_w[0, 2]],
                [bbox3d_w[1, 0], bbox3d_w[0, 1], bbox3d_w[0, 2]],
                [bbox3d_w[0, 0], bbox3d_w[1, 1], bbox3d_w[0, 2]],
                [bbox3d_w[0, 0], bbox3d_w[0, 1], bbox3d_w[1, 2]],
                [bbox3d_w[1, 0], bbox3d_w[1, 1], bbox3d_w[0, 2]],
                [bbox3d_w[1, 0], bbox3d_w[0, 1], bbox3d_w[1, 2]],
                [bbox3d_w[0, 0], bbox3d_w[1, 1], bbox3d_w[1, 2]],
                [bbox3d_w[1, 0], bbox3d_w[1, 1], bbox3d_w[1, 2]],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # Transform corners to camera frame
        corners_c = math_utils.transform_points(corners_w, camera_view_transform)

        # Project 3D corners to 2D
        corners_2d = self.project_3d_to_2d(corners_c, focal_length)

        # Calculate axis-aligned bounding box in image coordinates
        x_min = torch.min(corners_2d[:, 0])
        x_max = torch.max(corners_2d[:, 0])
        y_min = torch.min(corners_2d[:, 1])
        y_max = torch.max(corners_2d[:, 1])

        # Check if bounding box is within image bounds
        valid_bounds = (x_max > 0) and (x_min < self.camera_width) and (y_max > 0) and (y_min < self.camera_height)

        # Clamp to image boundaries
        x_min = torch.clamp(x_min, 0, self.camera_width - 1)
        x_max = torch.clamp(x_max, 0, self.camera_width - 1)
        y_min = torch.clamp(y_min, 0, self.camera_height - 1)
        y_max = torch.clamp(y_max, 0, self.camera_height - 1)

        # Check minimum bounding box size
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        valid_size = (bbox_width >= min_bbox_size) and (bbox_height >= min_bbox_size)

        # Combine all validity checks
        valid_mask = valid_bounds and valid_size

        # Stack bounding box coordinates
        bounding_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=0).unsqueeze(0)

        return bounding_boxes, torch.tensor([valid_mask], dtype=torch.bool, device=self.device)

    def get_bbox_center_and_size(self, bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert bounding boxes to center and size format.

        Args:
            bboxes: Bounding boxes [N, 4] (x_min, y_min, x_max, y_max)

        Returns:
            Tuple of (centers, sizes)
            - centers: [N, 2] (center_x, center_y)
            - sizes: [N, 2] (width, height)
        """
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        centers = torch.stack([center_x, center_y], dim=1)
        sizes = torch.stack([width, height], dim=1)

        return centers, sizes

    def generate_2d_bbox_from_pose(
        self,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
        camera_position: torch.Tensor,
        camera_orientation: torch.Tensor,
        target_size: torch.Tensor | None = None,
        focal_length: torch.Tensor | None = None,
        min_bbox_size: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D bounding boxes from target and camera poses.

        Args:
            target_position: Target position in world frame [N, 3]
            target_orientation: Target orientation quaternion in world frame [N, 4] (w, x, y, z)
            camera_position: Camera position in world frame [N, 3]
            camera_orientation: Camera orientation quaternion in world frame [N, 4] (w, x, y, z)
            target_size: Size of the target [N, 3] or [3] (default: 0.2m cube)
            focal_length: Dynamic focal length tensor [N] (optional, uses base if None)
            min_bbox_size: Minimum bounding box size in pixels

        Returns:
            Tuple of (bounding_boxes, valid_mask)
            - bounding_boxes: [N, 4] (x_min, y_min, x_max, y_max) in pixel coordinates
            - valid_mask: [N] boolean mask indicating if bbox is valid (target visible)
        """
        batch_size = target_position.shape[0]
        
        # Default target size if not provided (0.2m cube)
        if target_size is None:
            target_size = torch.tensor([0.2, 0.2, 0.2], device=self.device).expand(batch_size, 3)
        elif target_size.dim() == 1:
            target_size = target_size.expand(batch_size, 3)
            
        # Generate 8 corners of a box centered at origin
        half_size = target_size / 2.0
        corners_local = torch.stack([
            torch.stack([-half_size[:, 0], -half_size[:, 1], -half_size[:, 2]], dim=1),
            torch.stack([+half_size[:, 0], -half_size[:, 1], -half_size[:, 2]], dim=1),
            torch.stack([-half_size[:, 0], +half_size[:, 1], -half_size[:, 2]], dim=1),
            torch.stack([+half_size[:, 0], +half_size[:, 1], -half_size[:, 2]], dim=1),
            torch.stack([-half_size[:, 0], -half_size[:, 1], +half_size[:, 2]], dim=1),
            torch.stack([+half_size[:, 0], -half_size[:, 1], +half_size[:, 2]], dim=1),
            torch.stack([-half_size[:, 0], +half_size[:, 1], +half_size[:, 2]], dim=1),
            torch.stack([+half_size[:, 0], +half_size[:, 1], +half_size[:, 2]], dim=1),
        ], dim=1)  # [N, 8, 3]
        
        # Transform corners to world frame
        corners_world = math_utils.quat_rotate(target_orientation.unsqueeze(1), corners_local) + target_position.unsqueeze(1)
        
        # Transform to camera frame
        # First translate to camera origin
        corners_cam_translated = corners_world - camera_position.unsqueeze(1)
        # Then rotate to camera frame (inverse rotation)
        camera_quat_inv = math_utils.quat_conjugate(camera_orientation)
        corners_camera = math_utils.quat_rotate(camera_quat_inv.unsqueeze(1), corners_cam_translated)
        
        # Project to 2D
        points_2d = self.project_3d_to_2d(corners_camera, focal_length)  # [N, 8, 2]
        
        # Calculate bounding boxes and validity
        valid_mask = corners_camera[:, :, 2] > 0.01  # Z > 0.01m (in front of camera)
        valid_per_env = valid_mask.any(dim=1)  # At least one corner visible
        
        bounding_boxes = torch.zeros((batch_size, 4), device=self.device)
        
        for i in range(batch_size):
            if valid_per_env[i]:
                valid_points = points_2d[i][valid_mask[i]]
                if len(valid_points) > 0:
                    x_min = torch.clamp(valid_points[:, 0].min(), 0, self.camera_width - 1)
                    y_min = torch.clamp(valid_points[:, 1].min(), 0, self.camera_height - 1)
                    x_max = torch.clamp(valid_points[:, 0].max(), 0, self.camera_width - 1)
                    y_max = torch.clamp(valid_points[:, 1].max(), 0, self.camera_height - 1)
                    
                    # Check minimum size requirement
                    if (x_max - x_min) >= min_bbox_size and (y_max - y_min) >= min_bbox_size:
                        bounding_boxes[i] = torch.tensor([x_min, y_min, x_max, y_max], device=self.device)
                    else:
                        valid_per_env[i] = False
                else:
                    valid_per_env[i] = False
        
        return bounding_boxes, valid_per_env

    def normalize_bbox(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Normalize bounding boxes to [0, 1] range.

        Args:
            bboxes: Bounding boxes [N, 4] (x_min, y_min, x_max, y_max)

        Returns:
            Normalized bounding boxes [N, 4]
        """
        normalized_bboxes = bboxes.clone()
        normalized_bboxes[:, [0, 2]] /= self.camera_width  # Normalize x coordinates
        normalized_bboxes[:, [1, 3]] /= self.camera_height  # Normalize y coordinates

        return normalized_bboxes