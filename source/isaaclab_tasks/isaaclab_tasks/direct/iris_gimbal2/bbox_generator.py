# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Optional

import isaaclab.utils.math as math_utils


class BBoxGenerator:
    """Generates 2D bounding boxes for 3D targets in camera coordinates."""
    
    def __init__(
        self,
        camera_width: int = 640,
        camera_height: int = 480,
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
        device: str = "cuda"
    ):
        """Initialize the bounding box generator.
        
        Args:
            camera_width: Camera image width in pixels
            camera_height: Camera image height in pixels  
            focal_length: Camera focal length in mm
            horizontal_aperture: Camera horizontal aperture in mm
            device: Device to run computations on
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.device = device
        
        # Calculate camera intrinsics
        self.fx = (focal_length / horizontal_aperture) * camera_width
        self.fy = self.fx  # Assuming square pixels
        self.cx = camera_width / 2.0
        self.cy = camera_height / 2.0
        
        # Camera intrinsic matrix
        self.K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy], 
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
    
    def world_to_camera_frame(
        self,
        target_pos_world: torch.Tensor,
        camera_pos_world: torch.Tensor,
        camera_quat_world: torch.Tensor
    ) -> torch.Tensor:
        """Transform target position from world frame to camera frame.
        
        Args:
            target_pos_world: Target position in world frame [N, 3]
            camera_pos_world: Camera position in world frame [N, 3] 
            camera_quat_world: Camera quaternion in world frame [N, 4] (w, x, y, z)
            
        Returns:
            Target position in camera frame [N, 3]
        """
        # Transform target to camera frame
        target_pos_camera, _ = math_utils.subtract_frame_transforms(
            camera_pos_world, camera_quat_world, target_pos_world
        )
        return target_pos_camera
    
    def project_3d_to_2d(self, points_3d_camera: torch.Tensor) -> torch.Tensor:
        """Project 3D points in camera frame to 2D image coordinates.
        
        Args:
            points_3d_camera: 3D points in camera frame [N, 3] or [N, M, 3]
            
        Returns:
            2D image coordinates [N, 2] or [N, M, 2]
        """
        original_shape = points_3d_camera.shape
        if len(original_shape) == 3:
            # Reshape [N, M, 3] -> [N*M, 3]
            points_3d_camera = points_3d_camera.view(-1, 3)
        
        # Avoid division by zero
        z = torch.clamp(points_3d_camera[:, 2:3], min=1e-6)
        
        # Project to normalized image coordinates
        x_norm = points_3d_camera[:, 0:1] / z
        y_norm = points_3d_camera[:, 1:2] / z
        
        # Convert to pixel coordinates
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        
        points_2d = torch.cat([u, v], dim=1)
        
        if len(original_shape) == 3:
            # Reshape back to [N, M, 2]
            points_2d = points_2d.view(original_shape[0], original_shape[1], 2)
        
        return points_2d
    
    def get_target_bbox_3d_corners(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        target_size: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    ) -> torch.Tensor:
        """Get 3D bounding box corners for a target.
        
        Args:
            target_pos: Target position [N, 3]
            target_quat: Target quaternion [N, 4] (w, x, y, z)
            target_size: Target dimensions (width, height, depth)
            
        Returns:
            3D bounding box corners [N, 8, 3]
        """
        num_envs = target_pos.shape[0]
        
        # Define unit cube corners
        half_w, half_h, half_d = target_size[0]/2, target_size[1]/2, target_size[2]/2
        unit_corners = torch.tensor([
            [-half_w, -half_h, -half_d],  # 0: bottom-back-left
            [+half_w, -half_h, -half_d],  # 1: bottom-back-right
            [+half_w, +half_h, -half_d],  # 2: bottom-front-right
            [-half_w, +half_h, -half_d],  # 3: bottom-front-left
            [-half_w, -half_h, +half_d],  # 4: top-back-left
            [+half_w, -half_h, +half_d],  # 5: top-back-right
            [+half_w, +half_h, +half_d],  # 6: top-front-right
            [-half_w, +half_h, +half_d],  # 7: top-front-left
        ], device=self.device, dtype=torch.float32)
        
        # Expand for all environments
        corners = unit_corners.unsqueeze(0).repeat(num_envs, 1, 1)  # [N, 8, 3]
        
        # Transform corners by target pose
        corners_world = math_utils.quat_rotate(target_quat.unsqueeze(1), corners) + target_pos.unsqueeze(1)
        
        return corners_world
    
    def generate_2d_bbox(
        self,
        target_pos_world: torch.Tensor,
        target_quat_world: torch.Tensor,
        camera_pos_world: torch.Tensor,
        camera_quat_world: torch.Tensor,
        target_size: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        min_bbox_size: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D bounding boxes for 3D targets.
        
        Args:
            target_pos_world: Target positions in world frame [N, 3]
            target_quat_world: Target quaternions in world frame [N, 4]
            camera_pos_world: Camera positions in world frame [N, 3]
            camera_quat_world: Camera quaternions in world frame [N, 4]
            target_size: Target dimensions (width, height, depth)
            min_bbox_size: Minimum bounding box size in pixels
            
        Returns:
            Tuple of (bounding_boxes, valid_mask)
            - bounding_boxes: [N, 4] (x_min, y_min, x_max, y_max) in pixel coordinates
            - valid_mask: [N] boolean mask indicating if bbox is valid (target visible)
        """
        num_envs = target_pos_world.shape[0]
        
        # Get 3D bounding box corners in world frame
        corners_world = self.get_target_bbox_3d_corners(target_pos_world, target_quat_world, target_size)
        
        # Transform corners to camera frame
        corners_camera = self.world_to_camera_frame(corners_world.view(-1, 3), 
                                                   camera_pos_world.repeat_interleave(8, dim=0),
                                                   camera_quat_world.repeat_interleave(8, dim=0))
        corners_camera = corners_camera.view(num_envs, 8, 3)
        
        # Check if target is in front of camera (positive Z)
        z_coords = corners_camera[:, :, 2]
        valid_depth = torch.all(z_coords > 0.1, dim=1)  # All corners must be in front
        
        # Project 3D corners to 2D
        corners_2d = self.project_3d_to_2d(corners_camera)  # [N, 8, 2]
        
        # Calculate axis-aligned bounding box in image coordinates
        x_coords = corners_2d[:, :, 0]
        y_coords = corners_2d[:, :, 1]
        
        x_min = torch.min(x_coords, dim=1)[0]
        x_max = torch.max(x_coords, dim=1)[0]
        y_min = torch.min(y_coords, dim=1)[0]
        y_max = torch.max(y_coords, dim=1)[0]
        
        # Check if bounding box is within image bounds
        valid_bounds = (
            (x_max > 0) & (x_min < self.camera_width) &
            (y_max > 0) & (y_min < self.camera_height)
        )
        
        # Clamp to image boundaries
        x_min = torch.clamp(x_min, 0, self.camera_width - 1)
        x_max = torch.clamp(x_max, 0, self.camera_width - 1)
        y_min = torch.clamp(y_min, 0, self.camera_height - 1)
        y_max = torch.clamp(y_max, 0, self.camera_height - 1)
        
        # Check minimum bounding box size
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        valid_size = (bbox_width >= min_bbox_size) & (bbox_height >= min_bbox_size)
        
        # Combine all validity checks
        valid_mask = valid_depth & valid_bounds & valid_size
        
        # Stack bounding box coordinates
        bounding_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        
        return bounding_boxes, valid_mask
    
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
    
    def normalize_bbox(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Normalize bounding boxes to [0, 1] range.
        
        Args:
            bboxes: Bounding boxes [N, 4] (x_min, y_min, x_max, y_max)
            
        Returns:
            Normalized bounding boxes [N, 4]
        """
        normalized_bboxes = bboxes.clone()
        normalized_bboxes[:, [0, 2]] /= self.camera_width   # Normalize x coordinates
        normalized_bboxes[:, [1, 3]] /= self.camera_height  # Normalize y coordinates
        
        return normalized_bboxes