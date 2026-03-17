# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for domain randomization in iris_ma6.

This module defines configuration dataclasses for all domain randomization
parameters including physics properties, camera simulation, and gimbal dynamics.

Scope Distinction:
    - Domain Randomization (this module): *What* the system is
      (mass, friction, camera FOV, sensor characteristics)
    - Initial States (separate module): *Where* the system starts
      (position, velocity, orientation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from isaaclab.utils import configclass


# =============================================================================
# Mass Randomization Configuration
# =============================================================================


@configclass
class MassRandomizationCfg:
    """Configuration for rigid body mass randomization.

    Supports both scaling and additive mass variations to simulate
    manufacturing tolerances and payload variations.
    """

    # =========================================================================
    # Enable/Disable
    # =========================================================================

    enabled: bool = True
    """Enable mass randomization."""

    # =========================================================================
    # Body Mass Variation
    # =========================================================================

    body_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    """Scale factor range for body mass (multiplicative). Default ±10%."""

    body_mass_add_range: tuple[float, float] = (-0.05, 0.05)
    """Additive mass range in kg. Default ±50g."""

    # =========================================================================
    # Payload Mass
    # =========================================================================

    payload_mass_range: tuple[float, float] = (0.0, 0.2)
    """Additional payload mass range in kg. Simulates different payloads."""

    # =========================================================================
    # Gimbal Mass Variation
    # =========================================================================

    gimbal_mass_scale_range: tuple[float, float] = (0.95, 1.05)
    """Scale factor range for gimbal component masses."""

    # =========================================================================
    # Sampling Parameters
    # =========================================================================

    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    """Distribution type for sampling mass variations."""

    recompute_inertia: bool = True
    """Whether to automatically recompute inertia after mass changes."""


# =============================================================================
# Scale Randomization Configuration
# =============================================================================


@configclass
class ScaleRandomizationCfg:
    """Configuration for geometric scale randomization.

    Note:
        Scale randomization can only be applied at prestartup (before simulation).
        Requires `replicate_physics=False` in scene config.
    """

    enabled: bool = False
    """Enable scale randomization. Disabled by default (requires special setup)."""

    uniform_scale_range: tuple[float, float] | None = (0.95, 1.05)
    """Uniform scale range (same for all axes). Set to None for per-axis scaling."""

    per_axis_scale: dict[str, tuple[float, float]] | None = None
    """Per-axis scale ranges. Example: {"x": (0.9, 1.1), "y": (0.9, 1.1), "z": (0.95, 1.05)}"""

    body_names: list[str] = field(default_factory=lambda: ["base_link"])
    """Target body names for scale randomization (regex supported)."""


# =============================================================================
# Material Randomization Configuration
# =============================================================================


@configclass
class MaterialRandomizationCfg:
    """Configuration for physics material randomization (friction, restitution)."""

    enabled: bool = True
    """Enable material randomization."""

    # =========================================================================
    # Friction Coefficients
    # =========================================================================

    static_friction_range: tuple[float, float] = (0.7, 1.3)
    """Static friction coefficient range."""

    dynamic_friction_range: tuple[float, float] = (0.5, 1.0)
    """Dynamic friction coefficient range."""

    # =========================================================================
    # Restitution
    # =========================================================================

    restitution_range: tuple[float, float] = (0.0, 0.3)
    """Restitution (bounciness) coefficient range."""

    # =========================================================================
    # Optimization Parameters
    # =========================================================================

    num_buckets: int = 64
    """Number of material buckets for efficient randomization.

    PhysX has a limit of 64,000 unique materials. Using buckets avoids
    creating too many unique materials during reset.
    """

    make_consistent: bool = True
    """Ensure dynamic friction <= static friction for physical consistency."""


# =============================================================================
# Physics Randomization Configuration (Composite)
# =============================================================================


@configclass
class PhysicsRandomizationCfg:
    """Composite configuration for all physics randomization.

    Groups mass, scale, and material randomization settings.
    """

    mass: MassRandomizationCfg = field(default_factory=MassRandomizationCfg)
    """Mass randomization configuration."""

    scale: ScaleRandomizationCfg = field(default_factory=ScaleRandomizationCfg)
    """Scale randomization configuration (prestartup only)."""

    material: MaterialRandomizationCfg = field(default_factory=MaterialRandomizationCfg)
    """Material (friction/restitution) randomization configuration."""


# =============================================================================
# Camera Randomization Configuration
# =============================================================================


@configclass
class CameraRandomizationCfg:
    """Configuration for computational camera randomization.

    Camera resolution randomization is achieved through computational
    crop and resize operations on rendered images, allowing simulation
    of different camera configurations without changing the actual render.

    Pipeline:
        Render (1920x1080) -> Crop (FOV sim) -> Resize (resolution sim)

    Constraints:
        - Centered principal points only (symmetric crops)
        - Square pixels only (f_x = f_y)
    """

    enabled: bool = True
    """Enable camera randomization."""

    # =========================================================================
    # Render Resolution (Fixed)
    # =========================================================================

    render_width: int = 1920
    """Render width in pixels (Full HD). Fixed at maximum quality."""

    render_height: int = 1080
    """Render height in pixels (Full HD). Fixed at maximum quality."""

    # =========================================================================
    # Target Resolution
    # =========================================================================

    target_width_range: tuple[int, int] = (640, 1920)
    """Target output width range in pixels."""

    target_height_range: tuple[int, int] = (360, 1080)
    """Target output height range in pixels."""

    discrete_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1920, 1080),  # Full HD (1080p)
            (1280, 720),  # HD (720p)
            (640, 360),  # nHD (360p)
        ]
    )
    """Discrete resolution options. If None, samples continuously from ranges."""

    # =========================================================================
    # FOV Simulation
    # =========================================================================

    fov_scale_range: tuple[float, float] = (0.5, 1.0)
    """FOV scale range. 1.0 = full FOV, smaller = zoom in (narrower FOV).

    Example: 0.5 means crop to 50% of rendered image (2x zoom effect).
    """

    # =========================================================================
    # Focal Length
    # =========================================================================

    focal_length_range: tuple[float, float] = (800.0, 1200.0)
    """Focal length range in pixels at render resolution.

    Scaled appropriately for 1080p. Typical range corresponds to ~35-50mm
    equivalent on 35mm format.
    """

    # =========================================================================
    # Randomization Frequency
    # =========================================================================

    randomize_per_step: bool = False
    """Randomize camera parameters every simulation step."""

    randomize_per_episode: bool = True
    """Randomize camera parameters at episode reset."""

    randomize_per_agent: bool = False
    """Use different parameters for each agent (heterogeneous cameras)."""


# =============================================================================
# Gimbal Randomization Configuration
# =============================================================================


@configclass
class GimbalRandomizationCfg:
    """Configuration for gimbal joint and dynamics randomization.

    Simulates mechanical tolerances and variations in gimbal systems.
    """

    enabled: bool = True
    """Enable gimbal randomization."""

    # =========================================================================
    # Joint Position Offsets
    # =========================================================================

    yaw_offset_range: tuple[float, float] = (-0.1, 0.1)
    """Yaw joint offset range in radians. Default ±5.7°."""

    pitch_offset_range: tuple[float, float] = (-0.05, 0.05)
    """Pitch joint offset range in radians. Default ±2.9°."""

    roll_offset_range: tuple[float, float] = (-0.02, 0.02)
    """Roll joint offset range in radians. Default ±1.1°."""

    # =========================================================================
    # Joint Dynamics Randomization
    # =========================================================================

    stiffness_scale_range: tuple[float, float] = (0.8, 1.2)
    """Stiffness scale factor range for gimbal actuators."""

    damping_scale_range: tuple[float, float] = (0.8, 1.2)
    """Damping scale factor range for gimbal actuators."""

    friction_range: tuple[float, float] = (0.0, 0.1)
    """Joint friction coefficient range."""

    # =========================================================================
    # Randomization Mode
    # =========================================================================

    randomize_offsets_per_episode: bool = True
    """Randomize joint offsets at episode reset."""

    randomize_dynamics_per_episode: bool = False
    """Randomize joint dynamics (stiffness/damping) at episode reset."""


# =============================================================================
# Mount Offset Randomization Configuration
# =============================================================================


@configclass
class MountOffsetRandomizationCfg:
    """Configuration for gimbal mount offset randomization.

    Note:
        Mount offset randomization can only be applied at prestartup.
        Simulates mechanical mounting tolerances.
    """

    enabled: bool = False
    """Enable mount offset randomization. Disabled by default (prestartup only)."""

    # =========================================================================
    # Position Offset
    # =========================================================================

    position_x_range: tuple[float, float] = (-0.01, 0.01)
    """X position offset range in meters. Default ±1cm forward/back."""

    position_y_range: tuple[float, float] = (-0.01, 0.01)
    """Y position offset range in meters. Default ±1cm left/right."""

    position_z_range: tuple[float, float] = (-0.005, 0.005)
    """Z position offset range in meters. Default ±5mm up/down."""

    # =========================================================================
    # Rotation Offset
    # =========================================================================

    rotation_roll_range: tuple[float, float] = (-0.02, 0.02)
    """Roll rotation offset range in radians. Default ±1.1°."""

    rotation_pitch_range: tuple[float, float] = (-0.02, 0.02)
    """Pitch rotation offset range in radians. Default ±1.1°."""

    rotation_yaw_range: tuple[float, float] = (-0.02, 0.02)
    """Yaw rotation offset range in radians. Default ±1.1°."""


# =============================================================================
# Domain Randomization Configuration (Top-Level)
# =============================================================================


@configclass
class DomainRandomizationCfg:
    """Top-level configuration for all domain randomization.

    This is the main configuration class that composes all randomization
    settings for physics, camera, and gimbal systems.

    Example:
        ```python
        cfg = DomainRandomizationCfg(
            enabled=True,
            physics=PhysicsRandomizationCfg(
                mass=MassRandomizationCfg(
                    body_mass_scale_range=(0.85, 1.15),
                ),
            ),
            camera=CameraRandomizationCfg(
                fov_scale_range=(0.6, 1.0),
            ),
        )
        ```
    """

    enabled: bool = True
    """Master switch to enable/disable all domain randomization."""

    # =========================================================================
    # Sub-configurations
    # =========================================================================

    physics: PhysicsRandomizationCfg = field(default_factory=PhysicsRandomizationCfg)
    """Physics randomization configuration (mass, scale, materials)."""

    camera: CameraRandomizationCfg = field(default_factory=CameraRandomizationCfg)
    """Camera randomization configuration (resolution, FOV, focal length)."""

    gimbal: GimbalRandomizationCfg = field(default_factory=GimbalRandomizationCfg)
    """Gimbal randomization configuration (joint offsets, dynamics)."""

    mount_offset: MountOffsetRandomizationCfg = field(
        default_factory=MountOffsetRandomizationCfg
    )
    """Mount offset randomization configuration (prestartup only)."""

    # =========================================================================
    # Global Settings
    # =========================================================================

    seed: int | None = None
    """Random seed for reproducibility. If None, uses system random."""
