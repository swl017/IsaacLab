# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom visualization wrapper for iris_ma6 environment.

This module provides a unified visualization manager that handles:
- Camera frustum visualization per agent
- Detection indicator visualization per agent
- Warmup period to prevent GPU crashes from Isaac Sim timing bugs
- Safe visualization with exception handling
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, TYPE_CHECKING

from .camera_frustum import CameraFrustum
from .covariance_ellipsoid import CovarianceEllipsoid
from .detection_indicator import DetectionIndicator
from .frame_visualizer import FrameVisualizer

if TYPE_CHECKING:
    from isaaclab.sensors import TiledCameraCfg


# Minimum frames to wait before allowing visualization.
# This prevents GPU crashes from Isaac Sim 4.5's timing bug where Vulkan
# tries to access USD prims before mesh data is fully populated.
MIN_FRAMES_BEFORE_VISUALIZATION = 10


class CustomVisualization:
    """Unified visualization manager for iris_ma6 multi-agent drone environment.

    Manages per-agent camera frustum and detection indicator visualizers.
    Implements a warmup period to avoid GPU crashes from Isaac Sim timing bugs.

    Usage:
        visualization = CustomVisualization(num_envs, possible_agents, camera_cfg, device)

        # Each simulation step:
        visualization.step()  # Update frame counter

        # Each observation step (when visualization data is available):
        visualization.update(
            camera_poses=camera_poses,
            target_pos=target_pos,
            bbox_empty=bbox_empty,
            zoom_levels=zoom_levels,
        )
    """

    def __init__(
        self,
        num_envs: int,
        possible_agents: list[str],
        camera_cfg: TiledCameraCfg,
        device: torch.device | str,
    ):
        """Initialize the visualization manager.

        Args:
            num_envs: Number of environments.
            possible_agents: List of agent identifiers.
            camera_cfg: Camera configuration for frustum visualization.
            device: Device to use for tensor operations.
        """
        self.num_envs = num_envs
        self.possible_agents = possible_agents
        self.camera_cfg = camera_cfg
        self.device = device

        # Frame counter for warmup period
        self._frame_count = 0
        self._visualization_enabled = False

        # Per-agent visualizers
        self.camera_frustum: Dict[str, CameraFrustum] = {
            agent_id: CameraFrustum() for agent_id in possible_agents
        }
        self.detection_indicator: Dict[str, DetectionIndicator] = {
            agent_id: DetectionIndicator(num_envs, device) for agent_id in possible_agents
        }

        # Frame axes on body and gimbal links
        self.frame_visualizer = FrameVisualizer(scale=0.15)

        # Covariance ellipsoid visualizers (per-agent, unique prim paths)
        self.covariance_ellipsoid: Dict[str, CovarianceEllipsoid] = {
            agent_id: CovarianceEllipsoid(agent_id) for agent_id in possible_agents
        }

        # Track if clear has been called this frame
        self._cleared_this_frame = False

    def step(self):
        """Increment frame counter for warmup tracking.

        Call this method each simulation step to track warmup progress.
        After MIN_FRAMES_BEFORE_VISUALIZATION frames, visualization will be enabled.
        """
        self._frame_count += 1
        if self._frame_count >= MIN_FRAMES_BEFORE_VISUALIZATION:
            self._visualization_enabled = True

        # Update covariance ellipsoid frame counters (separate warmup tracking)
        for agent_id in self.possible_agents:
            self.covariance_ellipsoid[agent_id].step()

        # Reset clear flag for new frame
        self._cleared_this_frame = False

    @property
    def is_ready(self) -> bool:
        """Check if visualization is ready (warmup period has passed)."""
        return self._visualization_enabled

    def _clear_all(self):
        """Clear all debug draw lines once per frame."""
        if self._cleared_this_frame:
            return

        # Clear from any one visualizer (they share the same debug draw interface)
        if self.possible_agents:
            first_agent = self.possible_agents[0]
            self.camera_frustum[first_agent].clear()
            # Note: detection_indicator shares the same draw_interface,
            # so clearing once clears all lines

        self._cleared_this_frame = True

    def update(
        self,
        camera_poses: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        target_pos: torch.Tensor,
        bbox_empty: Dict[str, torch.Tensor],
        zoom_levels: Dict[str, torch.Tensor],
    ):
        """Update all visualizations with current data.

        This should be called once per observation step (policy frequency).
        Lines are cleared and redrawn each frame.

        Args:
            camera_poses: Dictionary mapping agent_id to (position, orientation) tuples.
                - position: Camera position in world frame (N, 3)
                - orientation: Camera orientation as quaternion wxyz (N, 4)
            target_pos: Target position in world frame (N, 3).
            bbox_empty: Dictionary mapping agent_id to bbox_empty tensor (N,) or (N, T).
                - True means target is NOT detected (no valid bbox)
            zoom_levels: Dictionary mapping agent_id to zoom level tensor (N,).
        """
        if not self._visualization_enabled:
            return

        # Clear previous frame's lines
        self._clear_all()

        # Draw visualizations for each agent
        for agent_id in self.possible_agents:
            try:
                # Get camera pose for this agent
                camera_pos, camera_quat = camera_poses[agent_id]

                # Get zoom level
                zoom = zoom_levels[agent_id]
                if zoom.dim() == 0:
                    zoom = zoom.unsqueeze(0)

                # Draw camera frustum
                self.camera_frustum[agent_id].draw_frustum(
                    camera_position=camera_pos,
                    camera_orientation=camera_quat,
                    camera_cfg=self.camera_cfg,
                    zoom_level=zoom,
                    device=self.device,
                )

                # Get detection status (inverted from bbox_empty)
                agent_bbox_empty = bbox_empty[agent_id]
                # Handle multi-target case: if (N, T), take first target
                if agent_bbox_empty.dim() > 1:
                    agent_bbox_empty = agent_bbox_empty[:, 0]
                detected = ~agent_bbox_empty  # detected = NOT bbox_empty

                # Draw detection indicator
                self.detection_indicator[agent_id].draw_indicator(
                    start_points=camera_pos,
                    end_points=target_pos,
                    detected=detected,
                )
            except Exception:
                # Silently skip visualization on errors to prevent crashes
                # This can happen during early simulation steps
                pass

    def update_frames(
        self,
        link_poses: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    ):
        """Update frame axes visualization on body and gimbal links.

        Args:
            link_poses: Dict mapping link name (body, yaw_link, roll_link, pitch_link)
                to (positions, orientations) tuples.
                - positions: (num_envs * num_agents, 3)
                - orientations: (num_envs * num_agents, 4) wxyz
        """
        if not self._visualization_enabled:
            return

        try:
            self.frame_visualizer.update(link_poses)
        except Exception:
            pass

    def visualize_frustums(
        self,
        camera_poses: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        zoom_levels: Dict[str, torch.Tensor],
    ):
        """Draw only camera frustums (without detection indicators).

        Useful for debugging camera/gimbal orientation without detection data.

        Args:
            camera_poses: Dictionary mapping agent_id to (position, orientation) tuples.
            zoom_levels: Dictionary mapping agent_id to zoom level tensor (N,).
        """
        if not self._visualization_enabled:
            return

        self._clear_all()

        for agent_id in self.possible_agents:
            try:
                camera_pos, camera_quat = camera_poses[agent_id]
                zoom = zoom_levels[agent_id]
                if zoom.dim() == 0:
                    zoom = zoom.unsqueeze(0)

                self.camera_frustum[agent_id].draw_frustum(
                    camera_position=camera_pos,
                    camera_orientation=camera_quat,
                    camera_cfg=self.camera_cfg,
                    zoom_level=zoom,
                    device=self.device,
                )
            except Exception:
                pass

    def visualize_detection_indicators(
        self,
        camera_poses: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        target_pos: torch.Tensor,
        bbox_empty: Dict[str, torch.Tensor],
    ):
        """Draw only detection indicators (without frustums).

        Useful for debugging detection status without frustum clutter.

        Args:
            camera_poses: Dictionary mapping agent_id to (position, orientation) tuples.
            target_pos: Target position in world frame (N, 3).
            bbox_empty: Dictionary mapping agent_id to bbox_empty tensor.
        """
        if not self._visualization_enabled:
            return

        self._clear_all()

        for agent_id in self.possible_agents:
            try:
                camera_pos, _ = camera_poses[agent_id]

                agent_bbox_empty = bbox_empty[agent_id]
                if agent_bbox_empty.dim() > 1:
                    agent_bbox_empty = agent_bbox_empty[:, 0]
                detected = ~agent_bbox_empty

                self.detection_indicator[agent_id].draw_indicator(
                    start_points=camera_pos,
                    end_points=target_pos,
                    detected=detected,
                )
            except Exception:
                pass

    def update_covariance_ellipsoids(
        self,
        translations: torch.Tensor,
        covariance: torch.Tensor,
        is_valid: torch.Tensor,
    ):
        """Update covariance ellipsoid visualization for triangulation uncertainty.

        Draws ellipsoids centered at triangulated target positions with axes
        proportional to the standard deviations from the covariance matrix.
        All agents share the same triangulation result (cooperative observation).

        Args:
            translations: Triangulated target positions (N, 3) in world frame.
            covariance: Covariance matrices (N, 3, 3) from triangulation.
            is_valid: Validity mask (N,) indicating successful triangulation.
        """
        if not self._visualization_enabled:
            return

        # All agents visualize the same triangulation result
        # Use the first agent's visualizer (they share the target)
        first_agent = self.possible_agents[0]
        try:
            self.covariance_ellipsoid[first_agent].visualize(
                translations=translations,
                covariance=covariance,
                is_valid=is_valid,
            )
        except Exception:
            # Silently skip on errors to prevent crashes
            pass

    def update_covariance_ellipsoids_from_std(
        self,
        translations: torch.Tensor,
        std_dev: torch.Tensor,
        is_valid: torch.Tensor,
    ):
        """Update covariance ellipsoids from pre-computed standard deviations.

        Convenience method when standard deviations are already computed.

        Args:
            translations: Triangulated target positions (N, 3) in world frame.
            std_dev: Standard deviations per axis (N, 3).
            is_valid: Validity mask (N,) indicating successful triangulation.
        """
        if not self._visualization_enabled:
            return

        first_agent = self.possible_agents[0]
        try:
            self.covariance_ellipsoid[first_agent].visualize_from_std(
                translations=translations,
                std_dev=std_dev,
                is_valid=is_valid,
            )
        except Exception:
            pass

    def update_covariance_ellipsoids_obs(
        self,
        translations: torch.Tensor,
        covariance: torch.Tensor,
        is_valid: torch.Tensor,
    ):
        """Update covariance ellipsoid visualization for observed triangulation.

        Same as update_covariance_ellipsoids but draws in gray to distinguish
        from ground truth.

        Args:
            translations: Triangulated target positions (N, 3) in world frame.
            covariance: Covariance matrices (N, 3, 3) from triangulation.
            is_valid: Validity mask (N,) indicating successful triangulation.
        """
        if not self._visualization_enabled:
            return

        # Gray color for observed (delayed/noisy) triangulation result
        gray_color = (0.5, 0.5, 0.5, 1.0)

        # Use second agent's visualizer to avoid overlapping with GT
        # If only one agent, use the same visualizer (will overlap but different color)
        agent = self.possible_agents[1] if len(self.possible_agents) > 1 else self.possible_agents[0]
        try:
            self.covariance_ellipsoid[agent].visualize(
                translations=translations,
                covariance=covariance,
                is_valid=is_valid,
                color=gray_color,
            )
        except Exception:
            pass
