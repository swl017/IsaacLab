# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom wrapper for recording videos with bounding box overlays and saving bbox data during training.
"""

import gymnasium as gym
import numpy as np
import os
import cv2
import json
from typing import Dict, Any, Optional, List
import torch

from bbox_extractor import BBoxExtractor, extract_bboxes_from_camera


class BBoxVideoWrapper(gym.Wrapper):
    """
    A wrapper that records videos with bounding box overlays and saves bbox data.
    
    Args:
        env: The environment to wrap
        video_folder: Directory to save videos and bbox data
        step_trigger: Function that determines when to start recording (default: every 2000 steps)
        video_length: Number of frames per video (default: 200)
        camera_name: Name of the camera sensor in the environment (default: "camera")
        bbox_type: Type of bounding boxes to extract ("semantic" or "instance", default: "semantic")
        min_area: Minimum area threshold for valid bounding boxes (default: 100)
        class_filter: List of class IDs to filter (semantic only, default: None)
        fps: Frames per second for video recording (default: 30)
        save_bbox_data: Whether to save bbox data to JSON files (default: True)
        overlay_bboxes: Whether to overlay bboxes on video (default: True)
        disable_logger: Whether to disable logging (default: True)
    """
    
    def __init__(
        self,
        env: gym.Env,
        video_folder: str | None = None,
        step_trigger: callable = lambda step: step % 2000 == 0,
        video_length: int = 200,
        camera_name: str = "camera",
        bbox_type: str = "semantic",
        min_area: int = 100,
        class_filter: Optional[List[int]] = None,
        fps: int = 30,
        save_bbox_data: bool = True,
        overlay_bboxes: bool = True,
        disable_logger: bool = True,
    ):
        super().__init__(env)
        
        self.video_folder = video_folder
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.camera_name = camera_name
        self.bbox_type = bbox_type
        self.min_area = min_area
        self.class_filter = class_filter
        self.fps = fps
        self.save_bbox_data = save_bbox_data
        self.overlay_bboxes = overlay_bboxes
        self.disable_logger = disable_logger
        
        # Create directories
        if self.video_folder is not None:
            os.makedirs(self.video_folder, exist_ok=True)
            if self.save_bbox_data:
                self.bbox_data_folder = os.path.join(self.video_folder, "bbox_data")
                os.makedirs(self.bbox_data_folder, exist_ok=True)
        
        # Recording state
        self.step_id = 0
        self.recording = False
        self.recorded_frames = 0
        self.video_id = 0
        self.video_writer = None
        self.current_bbox_data = []
        
        # BBox extractor
        self.bbox_extractor = BBoxExtractor(min_area=min_area, class_filter=class_filter)
    
    def _get_camera_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get camera data from the environment."""
        if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, self.camera_name):
            camera = getattr(self.env.unwrapped.scene, self.camera_name)
            
            if hasattr(camera, 'data') and hasattr(camera.data, 'output'):
                return camera.data.output
        return None
    
    def _extract_frame_with_bboxes(self) -> Optional[tuple[np.ndarray, List[Dict]]]:
        """Extract RGB frame and bounding boxes from camera sensor."""
        camera_data = self._get_camera_data()
        if camera_data is None:
            return None
        
        # Get RGB image
        if "rgb" not in camera_data:
            return None
        
        rgb_data = camera_data["rgb"][0]  # Take first environment
        if isinstance(rgb_data, torch.Tensor):
            rgb_data = rgb_data.detach().cpu().numpy()
        
        # Handle different RGB formats
        if rgb_data.shape[2] == 4:  # RGBA
            frame = (rgb_data[..., :3] * 255).astype(np.uint8)
        else:  # RGB
            frame = (rgb_data * 255).astype(np.uint8)
        
        # Extract bounding boxes
        try:
            bboxes = extract_bboxes_from_camera(
                camera_data, 
                bbox_type=self.bbox_type,
                min_area=self.min_area,
                class_filter=self.class_filter
            )
        except Exception as e:
            if not self.disable_logger:
                print(f"Warning: Could not extract bboxes: {e}")
            bboxes = []
        
        # Overlay bboxes on frame if requested
        if self.overlay_bboxes and bboxes:
            frame = self.bbox_extractor.visualize_bboxes(frame, bboxes, self.bbox_type)
        
        return frame, bboxes
    
    def _start_recording(self):
        """Start recording a new video."""
        pass
        # if self.recording:
        #     self._stop_recording()
        
        # self.recording = True
        # self.recorded_frames = 0
        # self.current_bbox_data = []
        
        # self.video_path = os.path.join(
        #     self.video_folder, 
        #     f"bbox_{self.bbox_type}_step_{self.step_id:06d}.mp4"
        # )
        
        # # Initialize video writer with a default frame size
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (640, 480))
        # self.video_writer_initialized = False
        
        # if not self.disable_logger:
        #     print(f"Started recording bbox video: {self.video_path}")
    
    def _stop_recording(self):
        """Stop recording the current video and save bbox data."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save bbox data to JSON
        if self.save_bbox_data and self.current_bbox_data:
            bbox_filename = f"bbox_{self.bbox_type}_step_{self.step_id:06d}.json"
            bbox_path = os.path.join(self.bbox_data_folder, bbox_filename)
            
            bbox_summary = {
                "video_path": self.video_path,
                "step_id": self.step_id,
                "bbox_type": self.bbox_type,
                "total_frames": len(self.current_bbox_data),
                "frames": self.current_bbox_data
            }
            
            with open(bbox_path, 'w') as f:
                json.dump(bbox_summary, f, indent=2)
            
            if not self.disable_logger:
                print(f"Saved bbox data: {bbox_path}")
        
        self.recording = False
        self.video_id += 1
        
        if not self.disable_logger:
            print(f"Stopped recording bbox video after {self.recorded_frames} frames")
    
    def _record_frame(self):
        """Record a single frame with bbox data if currently recording."""
        if not self.recording:
            return
        
        result = self._extract_frame_with_bboxes()
        if result is not None:
            frame, bboxes = result
            h, w = frame.shape[:2]
            
            # Initialize video writer with correct frame size if needed
            if not self.video_writer_initialized:
                self.video_writer.release()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
                self.video_writer_initialized = True
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
            
            # Store bbox data for this frame
            if self.save_bbox_data:
                frame_data = {
                    "frame_id": self.recorded_frames,
                    "step_id": self.step_id,
                    "bboxes": bboxes
                }
                self.current_bbox_data.append(frame_data)
            
            self.recorded_frames += 1
            
            # Stop recording if we've reached the target length
            if self.recorded_frames >= self.video_length:
                self._stop_recording()
    
    def step(self, action):
        """Step the environment and record frames if needed."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if we should start recording
        if self.step_trigger(self.step_id) and not self.recording:
            self._start_recording()
        
        # Record frame if currently recording
        self._record_frame()
        
        self.step_id += 1
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment."""
        # Stop any ongoing recording
        if self.recording:
            self._stop_recording()
        
        return self.env.reset(**kwargs)
    
    def close(self):
        """Close the wrapper and clean up resources."""
        if self.recording:
            self._stop_recording()
        super().close()
    
    def get_bbox_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded bounding boxes."""
        if not self.save_bbox_data or not os.path.exists(self.bbox_data_folder):
            return {}
        
        stats = {
            "total_videos": 0,
            "total_frames": 0,
            "total_bboxes": 0,
            "bbox_counts_per_frame": [],
            "class_distribution": {},
        }
        
        # Process all JSON files
        for filename in os.listdir(self.bbox_data_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(self.bbox_data_folder, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                stats["total_videos"] += 1
                stats["total_frames"] += data["total_frames"]
                
                for frame_data in data["frames"]:
                    frame_bbox_count = len(frame_data["bboxes"])
                    stats["total_bboxes"] += frame_bbox_count
                    stats["bbox_counts_per_frame"].append(frame_bbox_count)
                    
                    # Count classes/instances
                    for bbox in frame_data["bboxes"]:
                        if self.bbox_type == "semantic":
                            key = f"class_{bbox['class_id']}"
                        else:
                            key = f"instance_{bbox['instance_id']}"
                        
                        stats["class_distribution"][key] = stats["class_distribution"].get(key, 0) + 1
        
        # Calculate averages
        if stats["total_frames"] > 0:
            stats["avg_bboxes_per_frame"] = stats["total_bboxes"] / stats["total_frames"]
        else:
            stats["avg_bboxes_per_frame"] = 0
        
        return stats