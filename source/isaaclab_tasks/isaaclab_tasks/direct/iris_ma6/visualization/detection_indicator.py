# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detection indicator visualization for iris_ma6 environment.

This module provides LOS (line-of-sight) line visualization from agents to targets.
Lines are color-coded based on detection status:
- Green: Target is detected (visible in camera)
- Yellow: Target is not detected (not visible or occluded)
"""

from __future__ import annotations

import torch


class DetectionIndicator:
    """Visualizes detection status with color-coded LOS lines.

    Draws lines from camera positions to target positions with colors
    indicating whether the target is currently detected.
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        """Initialize the detection indicator visualizer.

        Args:
            num_envs: Number of environments.
            device: Device to use for tensor operations.
        """
        self.num_envs = num_envs
        self.device = device

        # Try to get debug draw interface (may fail in headless mode)
        self.draw_interface = None
        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        except (ImportError, ModuleNotFoundError):
            try:
                from omni.isaac.debug_draw import _debug_draw as omni_debug_draw

                self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            except (ImportError, ModuleNotFoundError):
                pass  # Debug draw not available (headless mode)

        # Pre-allocate color tensors
        self._color_green = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0]], device=self.device
        )
        self._color_yellow = torch.tensor(
            [[1.0, 1.0, 0.0, 1.0]], device=self.device
        )

    def clear(self):
        """Clear all previously drawn indicator lines."""
        if self.draw_interface is not None:
            self.draw_interface.clear_lines()

    def get_line_colors(self, detected: torch.Tensor) -> torch.Tensor:
        """Compute line colors based on detection status.

        Args:
            detected: Detection status per environment (N,) boolean tensor.

        Returns:
            Line colors tensor (N, 4) with RGBA values.
        """
        line_colors_green = self._color_green.expand(detected.shape[0], -1)
        line_colors_yellow = self._color_yellow.expand(detected.shape[0], -1)

        return torch.where(
            detected.unsqueeze(-1),
            line_colors_green,
            line_colors_yellow,
        )

    def draw_indicator(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        detected: torch.Tensor,
    ):
        """Draw detection indicator lines from start to end points.

        Args:
            start_points: Line start positions (N, 3), typically camera positions.
            end_points: Line end positions (N, 3), typically target positions.
            detected: Detection status per environment (N,) boolean tensor.
        """
        # Skip drawing if debug draw is not available (headless mode)
        if self.draw_interface is None:
            return

        line_colors = self.get_line_colors(detected)
        line_thickness = [1.0] * detected.shape[0]

        self.draw_interface.draw_lines(
            start_points.tolist(),
            end_points.tolist(),
            line_colors.tolist(),
            line_thickness,
        )
