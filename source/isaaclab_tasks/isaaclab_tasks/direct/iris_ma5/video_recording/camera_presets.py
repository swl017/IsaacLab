# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-defined camera presets for different video recording scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .smooth_camera import SmoothCameraCfg


@dataclass
class CameraPreset:
    """A camera preset with descriptive metadata."""

    name: str
    """Preset name."""

    description: str
    """Human-readable description of when to use this preset."""

    config: SmoothCameraCfg
    """The camera configuration."""


# Pre-defined camera presets for common use cases
# All presets include dynamic zoom to keep agents in frame
CAMERA_PRESETS: dict[str, CameraPreset] = {
    # Bird's eye view - good for tactical overview
    "overhead": CameraPreset(
        name="overhead",
        description="Bird's eye view looking straight down. Good for formation overview.",
        config=SmoothCameraCfg(
            mode="follow_centroid",
            offset=(0.0, 0.0, 60.0),  # 60m above centroid
            look_down=True,
            smoothing_factor=0.08,
            dynamic_zoom=True,
            min_zoom_distance=20.0,
            max_zoom_distance=120.0,
            zoom_margin=2.0,  # Extra margin to ensure all agents/target visible
        ),
    ),
    # Chase camera - follows target from behind
    "chase": CameraPreset(
        name="chase",
        description="Chase camera following target from behind. Good for action shots.",
        config=SmoothCameraCfg(
            mode="follow_target",
            offset=(-25.0, 0.0, 12.0),  # 25m behind, 12m up
            look_down=False,
            smoothing_factor=0.06,
            dynamic_zoom=True,
            min_zoom_distance=20.0,
            max_zoom_distance=100.0,
            zoom_margin=1.8,  # Increased to fit all agents and target
        ),
    ),
    # Side view - lateral perspective
    "side": CameraPreset(
        name="side",
        description="Side view from a fixed lateral position. Good for consistent framing.",
        config=SmoothCameraCfg(
            mode="fixed",
            fixed_eye=(60.0, 0.0, 25.0),
            fixed_lookat=(0.0, 0.0, 15.0),
            smoothing_factor=0.1,
            dynamic_zoom=False,  # Fixed mode doesn't use dynamic zoom
        ),
    ),
    # Orbit - cinematic circular motion
    "orbit": CameraPreset(
        name="orbit",
        description="Orbiting camera around the scene. Cinematic establishing shots.",
        config=SmoothCameraCfg(
            mode="orbit",
            orbit_radius=40.0,
            orbit_height=20.0,
            orbit_speed=0.03,  # ~2 degrees per second
            smoothing_factor=0.1,
            dynamic_zoom=True,
            min_zoom_distance=25.0,
            max_zoom_distance=80.0,
        ),
    ),
    # Formation view - optimized for multi-agent observation
    "formation": CameraPreset(
        name="formation",
        description="Follows agent centroid with elevated perspective. Good for multi-agent.",
        config=SmoothCameraCfg(
            mode="follow_centroid",
            offset=(0.0, -30.0, 40.0),  # Behind and above
            look_down=False,
            smoothing_factor=0.05,
            dynamic_zoom=True,
            min_zoom_distance=25.0,
            max_zoom_distance=120.0,
            zoom_margin=1.9,  # Generous margin for multi-agent scenarios
        ),
    ),
    # Close-up - tight follow on target
    "closeup": CameraPreset(
        name="closeup",
        description="Close follow on target. Good for detail shots.",
        config=SmoothCameraCfg(
            mode="follow_target",
            offset=(-10.0, 5.0, 5.0),  # Close behind and slightly to side
            look_down=False,
            smoothing_factor=0.12,  # More responsive
            dynamic_zoom=True,
            min_zoom_distance=10.0,
            max_zoom_distance=40.0,  # Limited max zoom for closeup
            zoom_margin=1.3,  # Tighter framing
        ),
    ),
    # Wide shot - distant overview
    "wide": CameraPreset(
        name="wide",
        description="Wide establishing shot from distance. Shows full scene context.",
        config=SmoothCameraCfg(
            mode="follow_centroid",
            offset=(0.0, -80.0, 50.0),  # Far back and high
            look_down=False,
            smoothing_factor=0.03,  # Very smooth
            dynamic_zoom=True,
            min_zoom_distance=40.0,  # Stays wide
            max_zoom_distance=120.0,
            zoom_margin=2.0,  # Lots of margin
        ),
    ),
    # Isometric - classic game-style view
    "isometric": CameraPreset(
        name="isometric",
        description="Isometric-style view at 45 degree angle. Classic tactical game perspective.",
        config=SmoothCameraCfg(
            mode="follow_centroid",
            offset=(35.0, -35.0, 35.0),  # Equal offset in all axes
            look_down=False,
            smoothing_factor=0.07,
            dynamic_zoom=True,
            min_zoom_distance=30.0,
            max_zoom_distance=120.0,
            zoom_margin=1.8,  # Ensures all agents and target visible
        ),
    ),
}


def get_preset(name: str) -> SmoothCameraCfg:
    """Get a camera configuration by preset name.

    Args:
        name: Preset name (e.g., 'overhead', 'chase', 'orbit').

    Returns:
        SmoothCameraCfg for the requested preset.

    Raises:
        ValueError: If preset name is not found.
    """
    if name not in CAMERA_PRESETS:
        available = ", ".join(CAMERA_PRESETS.keys())
        raise ValueError(f"Unknown camera preset: '{name}'. Available: {available}")
    return CAMERA_PRESETS[name].config


def list_presets() -> list[str]:
    """List all available camera preset names.

    Returns:
        List of preset names.
    """
    return list(CAMERA_PRESETS.keys())


def describe_presets() -> str:
    """Get human-readable descriptions of all presets.

    Returns:
        Formatted string with preset names and descriptions.
    """
    lines = ["Available Camera Presets:", "-" * 40]
    for name, preset in CAMERA_PRESETS.items():
        lines.append(f"  {name:12s} - {preset.description}")
    return "\n".join(lines)
