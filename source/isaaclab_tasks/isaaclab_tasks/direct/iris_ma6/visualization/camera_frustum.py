# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera frustum visualization for iris_ma6 environment.

This module provides zoom-aware camera frustum visualization using Isaac Sim's
debug draw interface. The frustum is computed based on camera intrinsics and
dynamically adjusts with zoom level changes.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate

if TYPE_CHECKING:
    from isaaclab.sensors import TiledCameraCfg


class CameraFrustum:
    """Visualizes camera field-of-view frustum using debug draw lines.

    The frustum is rendered as a wireframe pyramid from the camera position
    to the far plane corners. The frustum size dynamically adjusts with
    zoom level changes.
    """

    def __init__(self):
        """Initialize the camera frustum visualizer."""
        self.camera_position: torch.Tensor | None = None
        self.camera_orientation: torch.Tensor | None = None
        self.camera_intrinsics: torch.Tensor | None = None

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

    def clear(self):
        """Clear all previously drawn frustum lines."""
        if self.draw_interface is not None:
            self.draw_interface.clear_lines()

    def draw_frustum(
        self,
        camera_position: torch.Tensor,
        camera_orientation: torch.Tensor,
        camera_cfg: TiledCameraCfg,
        zoom_level: torch.Tensor,
        device: str | torch.device = "cpu",
    ):
        """Draw the camera frustum for all environments.

        Args:
            camera_position: Camera positions in world frame (N, 3).
            camera_orientation: Camera orientations as quaternions wxyz (N, 4).
            camera_cfg: Camera configuration containing intrinsics.
            zoom_level: Zoom level per environment (N,).
            device: Device to use for tensor operations.
        """
        # Skip drawing if debug draw is not available (headless mode)
        if self.draw_interface is None:
            return

        num_envs = camera_position.shape[0]
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation

        # Create camera config tensor
        camera_cfg_tensor = create_camera_cfg_tensor(camera_cfg, num_envs, device=device)

        # Compute frustum corners in camera frame
        frustum_data = self._compute_camera_frustum_batched(
            camera_cfg_tensor, zoom_level, device=device
        )
        frustum_b = frustum_data["far_corners"]  # (N, 4, 3)

        # Transform frustum corners to world frame
        frustum_w = frustum_b.clone()
        for i in range(frustum_b.shape[1]):
            frustum_w[:, i] = self.camera_position + quat_rotate(
                self.camera_orientation, frustum_b[:, i]
            )

        # Draw frustum lines
        line_colors = [[1.0, 1.0, 1.0, 0.3]] * num_envs
        line_thicknesses = [1.0] * num_envs

        # Draw lines from camera to each frustum corner
        for i in range(frustum_b.shape[1]):
            self.draw_interface.draw_lines(
                self.camera_position.tolist(),
                frustum_w[:, i, :].tolist(),
                line_colors,
                line_thicknesses,
            )
            # Draw edges connecting frustum corners
            if i < frustum_b.shape[1] - 1:
                self.draw_interface.draw_lines(
                    frustum_w[:, i, :].tolist(),
                    frustum_w[:, i + 1, :].tolist(),
                    line_colors,
                    line_thicknesses,
                )
            else:
                # Connect last corner back to first
                self.draw_interface.draw_lines(
                    frustum_w[:, i, :].tolist(),
                    frustum_w[:, 0, :].tolist(),
                    line_colors,
                    line_thicknesses,
                )

    def _compute_camera_frustum_batched(
        self,
        camera_cfg_tensor: torch.Tensor,
        zoom_level: torch.Tensor,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        """Compute camera frustum parameters for n environments using batched operations.

        Args:
            camera_cfg_tensor: Batched tensor containing camera parameters (N, 6).
                Expected order: [width, height, focal_length, horizontal_aperture, near_plane, far_plane]
            zoom_level: Zoom level per environment (N,).
            device: Device to use for tensor operations.
            dtype: Data type for tensors.

        Returns:
            Dictionary containing batched frustum parameters.
        """
        camera_cfg_tensor = camera_cfg_tensor.to(device=device, dtype=dtype)
        n = camera_cfg_tensor.shape[0]

        # Extract camera parameters
        width = camera_cfg_tensor[:, 0]
        height = camera_cfg_tensor[:, 1]
        focal_length = camera_cfg_tensor[:, 2] * zoom_level
        horizontal_aperture = camera_cfg_tensor[:, 3]

        # Clamp near/far planes
        near_plane_max = torch.ones_like(camera_cfg_tensor[:, 4]) * 0.5
        near_plane = torch.where(
            camera_cfg_tensor[:, 4] > near_plane_max,
            camera_cfg_tensor[:, 4],
            near_plane_max,
        )

        far_plane_min = torch.ones_like(camera_cfg_tensor[:, 5]) * 1.0
        far_plane_max = torch.ones_like(camera_cfg_tensor[:, 5]) * 2.0 * zoom_level
        far_plane = torch.where(
            camera_cfg_tensor[:, 5] < far_plane_min,
            far_plane_min,
            torch.where(
                camera_cfg_tensor[:, 5] > far_plane_max,
                far_plane_max,
                camera_cfg_tensor[:, 5],
            ),
        )

        # Calculate field of view
        horizontal_fov_rad = 2 * torch.atan(horizontal_aperture / (2 * focal_length))
        aspect_ratio = width / height
        vertical_aperture = horizontal_aperture / aspect_ratio
        vertical_fov_rad = 2 * torch.atan(vertical_aperture / (2 * focal_length))

        # Calculate frustum dimensions at far plane
        far_height = 2 * far_plane * torch.tan(vertical_fov_rad / 2)
        far_width = 2 * far_plane * torch.tan(horizontal_fov_rad / 2)

        # Frustum corners at far plane (in camera coordinates)
        # Shape: (N, 4, 3) - N environments, 4 corners, 3 coordinates (x, y, z)
        far_corners = torch.stack(
            [
                torch.stack(
                    [-far_width / 2, -far_height / 2, far_plane], dim=1
                ),  # bottom-left
                torch.stack(
                    [far_width / 2, -far_height / 2, far_plane], dim=1
                ),  # bottom-right
                torch.stack(
                    [far_width / 2, far_height / 2, far_plane], dim=1
                ),  # top-right
                torch.stack(
                    [-far_width / 2, far_height / 2, far_plane], dim=1
                ),  # top-left
            ],
            dim=1,
        )

        return {
            "n_environments": n,
            "far_corners": far_corners,
            "far_plane": far_plane,
            "horizontal_fov_rad": horizontal_fov_rad,
            "vertical_fov_rad": vertical_fov_rad,
        }


def create_camera_cfg_tensor(
    camera_cfg: TiledCameraCfg,
    num_envs: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create batched tensor from camera configuration.

    Args:
        camera_cfg: A TiledCameraCfg object.
        num_envs: Number of environments.
        device: Device to create tensors on.
        dtype: Data type for tensors.

    Returns:
        Batched tensor of shape (N, 6) containing camera parameters:
        [width, height, focal_length, horizontal_aperture, near_plane, far_plane]
    """
    camera_params = [
        (
            camera_cfg.width,
            camera_cfg.height,
            camera_cfg.spawn.focal_length,
            camera_cfg.spawn.horizontal_aperture,
            camera_cfg.spawn.clipping_range[0],
            camera_cfg.spawn.clipping_range[1],
        )
    ] * num_envs

    return torch.tensor(camera_params, device=device, dtype=dtype)
