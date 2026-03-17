# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Covariance ellipsoid visualization for triangulation uncertainty.

This module provides wireframe ellipsoid visualization for representing triangulation
uncertainty from covariance matrices. Each ellipsoid is centered at the
triangulated position, with radii proportional to the standard deviations
along each axis.

Uses debug_draw lines instead of VisualizationMarkers to avoid GPU crashes
from Isaac Sim's mesh population timing issues.

Color: Cyan (3/255, 252/255, 248/255) - same as iris_ma5
"""

from __future__ import annotations

import math
import torch


# Color choice from iris_ma5: Cyan/turquoise (RGBA)
ELLIPSOID_COLOR = (3 / 255, 252 / 255, 248 / 255, 1.0)

# Scale multiplier applied to standard deviation for visualization
# This makes the ellipsoid more visible while still representing uncertainty
SCALE_MULTIPLIER = 2.5

# Number of segments for wireframe circles
NUM_CIRCLE_SEGMENTS = 16


class CovarianceEllipsoid:
    """Visualizes triangulation covariance as wireframe ellipsoids.

    Uses debug_draw lines to draw 3 orthogonal ellipse rings representing
    the uncertainty ellipsoid from covariance matrices. The ellipsoid axes
    are aligned with the world frame, with radii proportional to the
    standard deviations from the covariance diagonal.

    This approach avoids GPU crashes that occur with VisualizationMarkers
    due to Isaac Sim's mesh population timing issues.

    Usage:
        # Create visualizer
        visualizer = CovarianceEllipsoid("drone_0")

        # Each step:
        visualizer.visualize(
            translations=triangulated_positions,  # (N, 3)
            covariance=covariance_matrices,       # (N, 3, 3)
        )
    """

    def __init__(self, agent_id: str):
        """Initialize the covariance ellipsoid visualizer.

        Args:
            agent_id: Unique agent identifier (for future multi-agent support).
        """
        self.agent_id = agent_id
        self._frame_count = 0

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

        # Pre-compute unit circle points for efficient ellipse drawing
        self._circle_points = self._generate_circle_points(NUM_CIRCLE_SEGMENTS)

    def _generate_circle_points(self, num_segments: int) -> list[tuple[float, float]]:
        """Generate unit circle points for wireframe ellipse drawing.

        Args:
            num_segments: Number of line segments to approximate circle.

        Returns:
            List of (cos, sin) pairs for each segment endpoint.
        """
        points = []
        for i in range(num_segments + 1):
            angle = 2 * math.pi * i / num_segments
            points.append((math.cos(angle), math.sin(angle)))
        return points

    def step(self):
        """Increment frame counter for warmup tracking.

        Call this each simulation step before attempting visualization.
        """
        self._frame_count += 1

    @property
    def is_ready(self) -> bool:
        """Check if visualizer is ready for rendering.

        Debug draw is always ready after a short warmup.
        """
        # Small warmup to let simulation stabilize
        return self._frame_count > 5 and self.draw_interface is not None

    def visualize(
        self,
        translations: torch.Tensor,
        covariance: torch.Tensor,
        is_valid: torch.Tensor | None = None,
    ) -> bool:
        """Visualize covariance ellipsoids at triangulated positions.

        Draws 3 orthogonal wireframe ellipses (XY, XZ, YZ planes).

        Args:
            translations: Ellipsoid center positions (N, 3).
            covariance: Covariance matrices (N, 3, 3).
            is_valid: Optional validity mask (N,). If provided, invalid entries
                are skipped.

        Returns:
            True if visualization was performed, False if skipped.
        """
        if not self.is_ready:
            return False

        # Extract diagonal elements (variances) from covariance matrices
        # covariance: (N, 3, 3) -> diag: (N, 3)
        variances = torch.diagonal(covariance, dim1=-2, dim2=-1)

        # Handle NaN values - replace with small positive value
        variances = torch.where(
            torch.isnan(variances),
            torch.ones_like(variances) * 1e-6,
            variances,
        )

        # Clamp to avoid sqrt of negative values
        variances = torch.clamp(variances, min=1e-12)

        # Convert variance to standard deviation and apply scale
        std_dev = torch.sqrt(variances) * SCALE_MULTIPLIER

        return self._draw_ellipsoids(translations, std_dev, is_valid)

    def visualize_from_std(
        self,
        translations: torch.Tensor,
        std_dev: torch.Tensor,
        is_valid: torch.Tensor | None = None,
    ) -> bool:
        """Visualize covariance ellipsoids from pre-computed standard deviations.

        This is a convenience method when standard deviations are already computed.

        Args:
            translations: Ellipsoid center positions (N, 3).
            std_dev: Standard deviations per axis (N, 3).
            is_valid: Optional validity mask (N,). If provided, invalid entries
                are skipped.

        Returns:
            True if visualization was performed, False if skipped.
        """
        if not self.is_ready:
            return False

        # Handle NaN values
        std_safe = torch.where(
            torch.isnan(std_dev),
            torch.zeros_like(std_dev),
            std_dev,
        )

        # Apply scale multiplier
        scales = std_safe * SCALE_MULTIPLIER

        return self._draw_ellipsoids(translations, scales, is_valid)

    def _draw_ellipsoids(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        is_valid: torch.Tensor | None,
    ) -> bool:
        """Draw wireframe ellipsoids using debug_draw lines.

        Draws 3 orthogonal ellipse rings for each ellipsoid:
        - XY plane (horizontal)
        - XZ plane (vertical, front-facing)
        - YZ plane (vertical, side-facing)

        Args:
            centers: Ellipsoid centers (N, 3).
            radii: Radii along each axis (N, 3).
            is_valid: Validity mask (N,) or None.

        Returns:
            True if drawing succeeded.
        """
        if self.draw_interface is None:
            return False

        # Move tensors to CPU for drawing
        centers_cpu = centers.detach().cpu()
        radii_cpu = radii.detach().cpu()

        if is_valid is not None:
            valid_cpu = is_valid.detach().cpu()
        else:
            valid_cpu = torch.ones(centers.shape[0], dtype=torch.bool)

        # Collect all line segments
        start_points = []
        end_points = []

        for i in range(centers_cpu.shape[0]):
            if not valid_cpu[i]:
                continue

            cx, cy, cz = centers_cpu[i].tolist()
            rx, ry, rz = radii_cpu[i].tolist()

            # Skip if radii are too small
            if rx < 0.01 and ry < 0.01 and rz < 0.01:
                continue

            # Draw XY plane ellipse (horizontal)
            for j in range(len(self._circle_points) - 1):
                cos1, sin1 = self._circle_points[j]
                cos2, sin2 = self._circle_points[j + 1]
                start_points.append([cx + rx * cos1, cy + ry * sin1, cz])
                end_points.append([cx + rx * cos2, cy + ry * sin2, cz])

            # Draw XZ plane ellipse (vertical, front-facing)
            for j in range(len(self._circle_points) - 1):
                cos1, sin1 = self._circle_points[j]
                cos2, sin2 = self._circle_points[j + 1]
                start_points.append([cx + rx * cos1, cy, cz + rz * sin1])
                end_points.append([cx + rx * cos2, cy, cz + rz * sin2])

            # Draw YZ plane ellipse (vertical, side-facing)
            for j in range(len(self._circle_points) - 1):
                cos1, sin1 = self._circle_points[j]
                cos2, sin2 = self._circle_points[j + 1]
                start_points.append([cx, cy + ry * cos1, cz + rz * sin1])
                end_points.append([cx, cy + ry * cos2, cz + rz * sin2])

        if not start_points:
            return True  # Nothing to draw but not an error

        # Draw all lines at once
        colors = [list(ELLIPSOID_COLOR)] * len(start_points)
        thicknesses = [2.0] * len(start_points)

        try:
            self.draw_interface.draw_lines(
                start_points,
                end_points,
                colors,
                thicknesses,
            )
            return True
        except Exception:
            return False
