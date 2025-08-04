# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Bounding box extraction utility for Isaac Lab camera sensors.

This module provides functions to extract bounding boxes from semantic and instance
segmentation data provided by Isaac Lab cameras.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2


class BBoxExtractor:
    """
    Utility class for extracting bounding boxes from segmentation data.
    
    Args:
        min_area: Minimum area threshold for valid bounding boxes (default: 100 pixels)
        class_filter: List of semantic class IDs to extract bboxes for (None = all classes)
    """
    
    def __init__(self, min_area: int = 100, class_filter: Optional[List[int]] = None):
        self.min_area = min_area
        self.class_filter = class_filter
    
    def extract_semantic_bboxes(
        self, 
        semantic_data: torch.Tensor,
        image_height: int,
        image_width: int
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Extract bounding boxes from semantic segmentation data.
        
        Args:
            semantic_data: Semantic segmentation tensor (H, W) with class IDs
            image_height: Height of the image
            image_width: Width of the image
            
        Returns:
            List of bounding box dictionaries with keys:
            - 'class_id': Semantic class ID
            - 'center_x': X coordinate of bbox center (normalized 0-1)
            - 'center_y': Y coordinate of bbox center (normalized 0-1) 
            - 'width': Width of bbox (normalized 0-1)
            - 'height': Height of bbox (normalized 0-1)
            - 'center_x_px': X coordinate of bbox center (pixels)
            - 'center_y_px': Y coordinate of bbox center (pixels)
            - 'width_px': Width of bbox (pixels)
            - 'height_px': Height of bbox (pixels)
            - 'area': Area of bbox in pixels
        """
        bboxes = []
        
        # Convert to numpy if tensor
        if isinstance(semantic_data, torch.Tensor):
            seg_data = semantic_data.detach().cpu().numpy()
        else:
            seg_data = semantic_data
        
        # Handle different data formats
        if seg_data.ndim == 3:
            # If RGB format, convert to single channel
            if seg_data.shape[2] == 3:
                seg_data = seg_data[:, :, 0]  # Take first channel
            elif seg_data.shape[2] == 1:
                seg_data = seg_data[:, :, 0]
        
        # Get unique class IDs (excluding background/0)
        unique_classes = np.unique(seg_data)
        unique_classes = unique_classes[unique_classes > 0]  # Remove background
        print(f"[INFO]: Found {len(unique_classes)} unique classes in segmentation data.")
        print(f"[INFO]: Unique classes: {unique_classes}")

        # Filter classes if specified
        if self.class_filter is not None:
            unique_classes = [c for c in unique_classes if c in self.class_filter]
        
        for class_id in unique_classes:
            # Create binary mask for this class
            mask = (seg_data == class_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by minimum area
                if area < self.min_area:
                    continue
                
                # Calculate center coordinates
                center_x_px = x + w // 2
                center_y_px = y + h // 2
                
                # Normalize to 0-1 range
                center_x = center_x_px / image_width
                center_y = center_y_px / image_height
                width_norm = w / image_width
                height_norm = h / image_height
                
                bbox = {
                    'class_id': int(class_id),
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'width': float(width_norm),
                    'height': float(height_norm),
                    'center_x_px': int(center_x_px),
                    'center_y_px': int(center_y_px),
                    'width_px': int(w),
                    'height_px': int(h),
                    'area': int(area)
                }
                
                bboxes.append(bbox)
        
        return bboxes
    
    def extract_instance_bboxes(
        self,
        instance_data: torch.Tensor,
        image_height: int,
        image_width: int
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Extract bounding boxes from instance segmentation data.
        
        Args:
            instance_data: Instance segmentation tensor (H, W) with instance IDs
            image_height: Height of the image
            image_width: Width of the image
            
        Returns:
            List of bounding box dictionaries with keys:
            - 'instance_id': Instance ID
            - 'center_x': X coordinate of bbox center (normalized 0-1)
            - 'center_y': Y coordinate of bbox center (normalized 0-1)
            - 'width': Width of bbox (normalized 0-1)
            - 'height': Height of bbox (normalized 0-1)
            - 'center_x_px': X coordinate of bbox center (pixels)
            - 'center_y_px': Y coordinate of bbox center (pixels)
            - 'width_px': Width of bbox (pixels)
            - 'height_px': Height of bbox (pixels)
            - 'area': Area of bbox in pixels
        """
        bboxes = []
        
        # Convert to numpy if tensor
        if isinstance(instance_data, torch.Tensor):
            seg_data = instance_data.detach().cpu().numpy()
        else:
            seg_data = instance_data
        
        # Handle different data formats
        if seg_data.ndim == 3:
            if seg_data.shape[2] == 3:
                seg_data = seg_data[:, :, 0]  # Take first channel
            elif seg_data.shape[2] == 1:
                seg_data = seg_data[:, :, 0]
        
        # Get unique instance IDs (excluding background/0)
        unique_instances = np.unique(seg_data)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        for instance_id in unique_instances:
            # Create binary mask for this instance
            mask = (seg_data == instance_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by minimum area
                if area < self.min_area:
                    continue
                
                # Calculate center coordinates
                center_x_px = x + w // 2
                center_y_px = y + h // 2
                
                # Normalize to 0-1 range
                center_x = center_x_px / image_width
                center_y = center_y_px / image_height
                width_norm = w / image_width
                height_norm = h / image_height
                
                bbox = {
                    'instance_id': int(instance_id),
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'width': float(width_norm),
                    'height': float(height_norm),
                    'center_x_px': int(center_x_px),
                    'center_y_px': int(center_y_px),
                    'width_px': int(w),
                    'height_px': int(h),
                    'area': int(area)
                }
                
                bboxes.append(bbox)
        
        return bboxes
    
    def visualize_bboxes(
        self,
        image: np.ndarray,
        bboxes: List[Dict[str, Union[int, float]]],
        bbox_type: str = "semantic"
    ) -> np.ndarray:
        """
        Visualize bounding boxes on an image.
        
        Args:
            image: RGB image (H, W, 3)
            bboxes: List of bounding box dictionaries
            bbox_type: Type of bboxes ("semantic" or "instance")
            
        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()
        
        for bbox in bboxes:
            # Get pixel coordinates
            x = bbox['center_x_px'] - bbox['width_px'] // 2
            y = bbox['center_y_px'] - bbox['height_px'] // 2
            w = bbox['width_px']
            h = bbox['height_px']
            
            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            if bbox_type == "semantic":
                label = f"Class {bbox['class_id']}"
            else:
                label = f"Inst {bbox['instance_id']}"
            
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image


def extract_bboxes_from_camera(
    camera_data: Dict[str, torch.Tensor],
    bbox_type: str = "semantic",
    min_area: int = 100,
    class_filter: Optional[List[int]] = None
) -> List[Dict[str, Union[int, float]]]:
    """
    Convenience function to extract bounding boxes from camera sensor data.
    
    Args:
        camera_data: Camera output data dictionary
        bbox_type: Type of segmentation to use ("semantic" or "instance")
        min_area: Minimum area threshold for valid bounding boxes
        class_filter: List of class IDs to filter (semantic only)
        
    Returns:
        List of bounding box dictionaries
    """
    extractor = BBoxExtractor(min_area=min_area, class_filter=class_filter)
    
    if bbox_type == "semantic":
        if "semantic_segmentation" not in camera_data:
            raise ValueError("Semantic segmentation data not available in camera output")
        
        seg_data = camera_data["semantic_segmentation"][0]  # Take first environment
        
    elif bbox_type == "instance":
        if "instance_segmentation_fast" not in camera_data:
            raise ValueError("Instance segmentation data not available in camera output")
        
        seg_data = camera_data["instance_segmentation_fast"][0]  # Take first environment
        
    else:
        raise ValueError(f"Unknown bbox_type: {bbox_type}")
    
    # Get image dimensions
    height, width = seg_data.shape[:2]
    
    if bbox_type == "semantic":
        return extractor.extract_semantic_bboxes(seg_data, height, width)
    else:
        return extractor.extract_instance_bboxes(seg_data, height, width)