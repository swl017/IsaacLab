# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame axes visualization for drone body and gimbal links.

Uses Isaac Lab's VisualizationMarkers with FRAME_MARKER_CFG to render
XYZ coordinate frames on body, yaw_link, roll_link, and pitch_link.
"""

from __future__ import annotations

import torch
from typing import Dict

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Links to visualize with coordinate frames
FRAME_LINKS = ("body", "yaw_link", "roll_link", "pitch_link")


class FrameVisualizer:
    """Visualizes coordinate frames on drone body and gimbal links.

    Creates one VisualizationMarkers (PointInstancer) per link type.
    Each call to update() repositions the frame markers to match the
    current body/gimbal link poses across all agents and environments.
    """

    def __init__(self, scale: float = 0.15):
        """Initialize frame marker prims for each link type.

        Args:
            scale: Uniform scale for the frame markers.
        """
        self._markers: Dict[str, VisualizationMarkers] = {}
        for link_name in FRAME_LINKS:
            cfg = FRAME_MARKER_CFG.replace(
                prim_path=f"/Visuals/FrameVis/{link_name}",
            )
            # Override scale for drone-sized links
            for marker_key in cfg.markers:
                cfg.markers[marker_key].scale = (scale, scale, scale)
            self._markers[link_name] = VisualizationMarkers(cfg)

    def update(self, link_poses: Dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """Update frame marker transforms.

        Args:
            link_poses: Dict mapping link name to (positions, orientations).
                - positions: (M, 3) world-frame positions
                - orientations: (M, 4) quaternions in wxyz convention
        """
        for link_name, marker in self._markers.items():
            if link_name in link_poses:
                pos, quat = link_poses[link_name]
                marker.visualize(translations=pos, orientations=quat)

    def set_visibility(self, visible: bool):
        """Show or hide all frame markers."""
        for marker in self._markers.values():
            marker.set_visibility(visible)
