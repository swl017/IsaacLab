# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smooth camera controller for video recording with interpolated camera motion."""

from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Literal, Sequence

from isaaclab.envs.ui import ViewportCameraController


@dataclass
class SmoothCameraCfg:
    """Configuration for smooth camera controller."""

    mode: Literal["follow_target", "follow_centroid", "orbit", "fixed"] = "follow_centroid"
    """Camera tracking mode."""

    smoothing_factor: float = 0.1
    """Smoothing factor for exponential moving average (0-1). Lower = smoother."""

    offset: tuple[float, float, float] = (0.0, 0.0, 30.0)
    """Camera offset from tracked point (x, y, z) in world frame."""

    look_down: bool = False
    """If True, camera looks straight down at tracked point."""

    orbit_radius: float = 30.0
    """Radius for orbit mode (meters)."""

    orbit_height: float = 15.0
    """Height for orbit mode (meters above target)."""

    orbit_speed: float = 0.02
    """Angular speed for orbit mode (radians per second)."""

    fixed_eye: tuple[float, float, float] = (50.0, 0.0, 20.0)
    """Fixed camera position for fixed mode."""

    fixed_lookat: tuple[float, float, float] = (0.0, 0.0, 10.0)
    """Fixed look-at point for fixed mode."""

    # Dynamic zoom settings
    dynamic_zoom: bool = True
    """Enable dynamic zoom based on agent spread."""

    zoom_margin: float = 1.5
    """Margin multiplier for zoom (1.5 = 50% extra space around agents)."""

    min_zoom_distance: float = 15.0
    """Minimum camera distance (maximum zoom in), in meters."""

    max_zoom_distance: float = 80.0
    """Maximum camera distance (maximum zoom out / initial state), in meters."""

    zoom_smoothing: float = 0.05
    """Smoothing factor for zoom transitions (lower = smoother zoom)."""

    fov_degrees: float = 60.0
    """Camera field of view in degrees (used for zoom calculation)."""


class SmoothCameraController:
    """Smooth interpolated camera controller for professional video recording.

    This controller wraps the ViewportCameraController and adds:
    - Smooth exponential interpolation between camera positions
    - Multiple tracking modes (follow_target, follow_centroid, orbit, fixed)
    - Configurable smoothing for professional-looking camera motion

    Usage:
        # In evaluation script (with viewport)
        smooth_camera = SmoothCameraController(
            viewport_controller=env.viewport_camera_controller,
            cfg=SmoothCameraCfg(mode="chase", smoothing_factor=0.08),
        )

        # Headless mode (no viewport controller)
        smooth_camera = SmoothCameraController(
            viewport_controller=None,
            cfg=SmoothCameraCfg(mode="chase", smoothing_factor=0.08),
        )

        # In main loop
        for step in range(num_steps):
            # ... simulation step ...
            smooth_camera.update(
                agents_pos=agent_positions,  # [num_agents, 3]
                target_pos=target_position,   # [3]
                dt=env.step_dt,
            )
            # For headless, get pose and pass to video recorder
            eye, lookat = smooth_camera.get_current_pose()
            video_recorder.set_camera_pose(eye, lookat)
    """

    def __init__(
        self,
        viewport_controller: ViewportCameraController | None,
        cfg: SmoothCameraCfg | None = None,
        env_index: int = 0,
    ):
        """Initialize the smooth camera controller.

        Args:
            viewport_controller: The Isaac Lab viewport camera controller.
                If None, operates in headless mode (camera pose only, no viewport update).
            cfg: Configuration for smooth camera. If None, uses defaults.
            env_index: Index of the environment to track.
        """
        self.vc = viewport_controller
        self.cfg = cfg if cfg is not None else SmoothCameraCfg()
        self.env_index = env_index

        # Current camera state (initialized on first update)
        self._current_eye: np.ndarray | None = None
        self._current_lookat: np.ndarray | None = None

        # Orbit state
        self._orbit_angle: float = 0.0

        # Dynamic zoom state - start at max zoom out (initial state)
        self._current_zoom_distance: float = self.cfg.max_zoom_distance

        # Update counter for debug output
        self._update_count: int = 0

        # Set initial view
        self._initialize_view()

    def _initialize_view(self):
        """Initialize camera view based on mode."""
        # Disable viewport controller's automatic tracking so we can control the camera
        if self.vc is not None:
            # Set to 'world' origin type to stop auto-tracking any asset
            self.vc.cfg.origin_type = "world"
            self.vc.viewer_origin = torch.zeros(3)

        if self.cfg.mode == "fixed":
            self._current_eye = np.array(self.cfg.fixed_eye)
            self._current_lookat = np.array(self.cfg.fixed_lookat)
            if self.vc is not None:
                self.vc.update_view_location(
                    eye=self._current_eye.tolist(),
                    lookat=self._current_lookat.tolist(),
                )

    def set_mode(self, mode: str, **kwargs):
        """Change camera mode at runtime.

        Args:
            mode: Camera mode ('follow_target', 'follow_centroid', 'orbit', 'fixed').
            **kwargs: Mode-specific parameters to update in cfg.
        """
        self.cfg.mode = mode

        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        # Reset state for orbit mode
        if mode == "orbit":
            self._orbit_angle = 0.0

    def set_smoothing(self, factor: float):
        """Update smoothing factor.

        Args:
            factor: Smoothing factor (0-1). Lower = smoother.
        """
        self.cfg.smoothing_factor = max(0.01, min(1.0, factor))

    def update(
        self,
        agents_pos: torch.Tensor | np.ndarray,
        target_pos: torch.Tensor | np.ndarray,
        dt: float,
    ):
        """Update camera position with smooth interpolation.

        This should be called after each simulation step.

        Args:
            agents_pos: Agent positions [num_agents, 3] or [3] for single agent.
            target_pos: Target position [3].
            dt: Time step in seconds.
        """
        # Convert to numpy if needed
        if isinstance(agents_pos, torch.Tensor):
            agents_pos = agents_pos.detach().cpu().numpy()
        if isinstance(target_pos, torch.Tensor):
            target_pos = target_pos.detach().cpu().numpy()

        # Ensure correct shapes
        agents_pos = np.atleast_2d(agents_pos)  # [num_agents, 3]
        target_pos = np.asarray(target_pos).flatten()[:3]  # [3]

        # Compute target camera pose based on mode
        target_eye, target_lookat = self._compute_target_pose(agents_pos, target_pos, dt)

        # Initialize current state if needed
        if self._current_eye is None:
            self._current_eye = target_eye.copy()
            self._current_lookat = target_lookat.copy()
        else:
            # Smooth interpolation (exponential moving average)
            alpha = self.cfg.smoothing_factor
            self._current_eye = self._current_eye + alpha * (target_eye - self._current_eye)
            self._current_lookat = self._current_lookat + alpha * (target_lookat - self._current_lookat)

        # Apply to viewport controller (if available - not in headless mode)
        if self.vc is not None:
            # Call set_camera_view directly on the simulation context
            # This bypasses the viewport controller's origin-based positioning
            self.vc._env.sim.set_camera_view(
                eye=self._current_eye.tolist(),
                target=self._current_lookat.tolist(),
            )
            # Debug output for first few frames to verify updates
            if self._update_count < 3:
                print(f"[SmoothCamera] Update {self._update_count}: eye={self._current_eye}, lookat={self._current_lookat}")

        self._update_count += 1

    def _compute_required_zoom_distance(
        self,
        agents_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> float:
        """Compute required camera distance to fit all agents in frame.

        Args:
            agents_pos: Agent positions [num_agents, 3].
            target_pos: Target position [3].

        Returns:
            Required camera distance in meters.
        """
        # Combine all points we need to see
        all_points = np.vstack([agents_pos, target_pos.reshape(1, 3)])

        # Calculate the maximum spread in XY plane (horizontal)
        min_xy = np.min(all_points[:, :2], axis=0)
        max_xy = np.max(all_points[:, :2], axis=0)
        spread_xy = np.linalg.norm(max_xy - min_xy)

        # Also consider Z spread for non-overhead cameras
        z_spread = np.max(all_points[:, 2]) - np.min(all_points[:, 2])

        # Use the larger of XY spread or Z spread
        max_spread = max(spread_xy, z_spread)

        # Apply margin multiplier
        required_spread = max_spread * self.cfg.zoom_margin

        # Calculate required distance based on FOV
        # For overhead camera: distance = spread / (2 * tan(fov/2))
        fov_rad = math.radians(self.cfg.fov_degrees)
        half_fov_tan = math.tan(fov_rad / 2)

        # Account for both horizontal and vertical FOV (assume 16:9 aspect)
        # Use the more constraining dimension
        required_distance = required_spread / (2 * half_fov_tan)

        # Clamp to min/max zoom limits
        required_distance = max(self.cfg.min_zoom_distance,
                               min(self.cfg.max_zoom_distance, required_distance))

        return required_distance

    def _compute_target_pose(
        self,
        agents_pos: np.ndarray,
        target_pos: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute target camera eye and lookat based on mode.

        Args:
            agents_pos: Agent positions [num_agents, 3].
            target_pos: Target position [3].
            dt: Time step for orbit animation.

        Returns:
            Tuple of (eye_position, lookat_position).
        """
        mode = self.cfg.mode
        offset = np.array(self.cfg.offset)

        # Compute dynamic zoom distance if enabled
        if self.cfg.dynamic_zoom and mode != "fixed":
            target_zoom_distance = self._compute_required_zoom_distance(agents_pos, target_pos)

            # Smooth zoom transition
            alpha = self.cfg.zoom_smoothing
            self._current_zoom_distance = (
                self._current_zoom_distance + alpha * (target_zoom_distance - self._current_zoom_distance)
            )

            # Scale the offset to achieve the zoom distance
            # The offset direction determines camera positioning, magnitude determines zoom
            offset_magnitude = np.linalg.norm(offset)
            if offset_magnitude > 0.001:
                # Scale offset to match zoom distance
                zoom_scale = self._current_zoom_distance / offset_magnitude
                dynamic_offset = offset * zoom_scale
            else:
                # For zero XY offset (overhead view), adjust Z height
                dynamic_offset = np.array([0.0, 0.0, self._current_zoom_distance])
        else:
            dynamic_offset = offset

        if mode == "follow_target":
            # Camera follows the target
            lookat = target_pos
            eye = target_pos + dynamic_offset

        elif mode == "follow_centroid":
            # Camera follows centroid of all agents and target
            all_points = np.vstack([agents_pos, target_pos.reshape(1, 3)])
            centroid = np.mean(all_points, axis=0)
            lookat = centroid
            eye = centroid + dynamic_offset

        elif mode == "orbit":
            # Camera orbits around the centroid
            all_points = np.vstack([agents_pos, target_pos.reshape(1, 3)])
            centroid = np.mean(all_points, axis=0)

            # Update orbit angle
            self._orbit_angle += self.cfg.orbit_speed * dt

            # Use dynamic zoom for orbit radius and height
            if self.cfg.dynamic_zoom:
                orbit_radius = self._current_zoom_distance * 0.8  # Slightly closer for orbit
                orbit_height = self._current_zoom_distance * 0.5
            else:
                orbit_radius = self.cfg.orbit_radius
                orbit_height = self.cfg.orbit_height

            # Compute camera position on orbit
            x = centroid[0] + orbit_radius * math.cos(self._orbit_angle)
            y = centroid[1] + orbit_radius * math.sin(self._orbit_angle)
            z = centroid[2] + orbit_height

            eye = np.array([x, y, z])
            lookat = centroid

        elif mode == "fixed":
            # Static camera position
            eye = np.array(self.cfg.fixed_eye)
            lookat = np.array(self.cfg.fixed_lookat)

        else:
            raise ValueError(f"Unknown camera mode: {mode}")

        # Override lookat for look_down mode
        if self.cfg.look_down and mode != "fixed":
            lookat = np.array([eye[0], eye[1], 0.0])

        return eye, lookat

    def set_env_index(self, env_index: int):
        """Change which environment to track.

        Args:
            env_index: New environment index.
        """
        self.env_index = env_index
        if self.vc is not None:
            self.vc.set_view_env_index(env_index)

    def get_current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current camera eye and lookat positions.

        Returns:
            Tuple of (eye_position, lookat_position).
        """
        eye = self._current_eye if self._current_eye is not None else np.array(self.cfg.fixed_eye)
        lookat = self._current_lookat if self._current_lookat is not None else np.array(self.cfg.fixed_lookat)
        return eye.copy(), lookat.copy()

    def jump_to(self, eye: Sequence[float], lookat: Sequence[float]):
        """Instantly move camera to specified position (no smoothing).

        Args:
            eye: Camera eye position [x, y, z].
            lookat: Camera look-at position [x, y, z].
        """
        self._current_eye = np.array(eye)
        self._current_lookat = np.array(lookat)
        if self.vc is not None:
            self.vc._env.sim.set_camera_view(
                eye=self._current_eye.tolist(),
                target=self._current_lookat.tolist(),
            )
