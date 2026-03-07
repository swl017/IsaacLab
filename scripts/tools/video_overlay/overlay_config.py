# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for video overlay rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MetricDisplayConfig:
    """Configuration for displaying a single metric."""

    name: str
    """Display name for the metric."""

    key: str = ""
    """Key to look up in metrics dict (if different from name)."""

    unit: str = ""
    """Unit string (e.g., 'deg', 'm', '%')."""

    decimals: int = 2
    """Number of decimal places to show."""

    good_threshold: float = 0.0
    """Values below this are colored green (for metrics where lower is better)."""

    bad_threshold: float = 1.0
    """Values above this are colored red (for metrics where lower is better)."""

    invert_thresholds: bool = False
    """If True, higher values are good (green) and lower values are bad (red)."""

    show_in_panel: bool = True
    """Whether to show this metric in the info panel."""

    def __post_init__(self):
        if not self.key:
            self.key = self.name.lower().replace(" ", "_")


@dataclass
class PanelLayoutConfig:
    """Configuration for overlay panel layout."""

    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_right"
    """Panel position on screen."""

    width: int = 380
    """Panel width in pixels (wider to fit time-series graphs)."""

    padding: int = 15
    """Internal padding in pixels."""

    margin: int = 20
    """Margin from screen edge."""

    background_alpha: float = 0.7
    """Background transparency (0-1)."""

    font_scale: float = 0.6
    """OpenCV font scale for metric values."""

    title_font_scale: float = 0.75
    """OpenCV font scale for panel title."""

    line_height: int = 28
    """Pixels between lines."""

    border_width: int = 2
    """Border line width."""


@dataclass
class GraphConfig:
    """Configuration for animated metric graphs."""

    enabled: bool = False
    """Whether to show animated graph."""

    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "bottom_left"
    """Graph position on screen."""

    width: int = 400
    """Graph width in pixels."""

    height: int = 150
    """Graph height in pixels."""

    window_size: int = 100
    """Number of timesteps to show in graph."""

    metrics: list[str] = field(default_factory=lambda: ["viewing_angle", "triangulation_rmse"])
    """Metrics to plot in the graph."""

    background_alpha: float = 0.7
    """Background transparency."""


@dataclass
class ColorTheme:
    """Color theme for overlays (BGR format for OpenCV)."""

    background: tuple[int, int, int] = (30, 30, 30)
    """Panel background color."""

    text_primary: tuple[int, int, int] = (255, 255, 255)
    """Primary text color (white)."""

    text_secondary: tuple[int, int, int] = (180, 180, 180)
    """Secondary text color (gray)."""

    good: tuple[int, int, int] = (0, 200, 0)
    """Good value color (green)."""

    warning: tuple[int, int, int] = (0, 200, 255)
    """Warning value color (yellow)."""

    bad: tuple[int, int, int] = (0, 0, 255)
    """Bad value color (red)."""

    accent: tuple[int, int, int] = (255, 150, 50)
    """Accent color for borders and titles (orange)."""

    na_color: tuple[int, int, int] = (128, 128, 128)
    """Color for N/A values."""


# Default metric configurations
DEFAULT_METRICS = [
    MetricDisplayConfig(
        name="Viewing Angle",
        key="viewing_angle",
        unit="deg",
        decimals=1,
        good_threshold=30.0,
        bad_threshold=20.0,
        invert_thresholds=True,  # Higher angle is better
    ),
    MetricDisplayConfig(
        name="RMSE",
        key="triangulation_rmse",
        unit="m",
        decimals=2,
        good_threshold=2.0,
        bad_threshold=5.0,
        invert_thresholds=False,  # Lower RMSE is better
    ),
    MetricDisplayConfig(
        name="Visibility",
        key="visibility",
        unit="%",
        decimals=0,
        good_threshold=0.8,
        bad_threshold=0.5,
        invert_thresholds=True,  # Higher visibility is better
    ),
    MetricDisplayConfig(
        name="Tri Valid",
        key="tri_valid",
        unit="",
        decimals=0,
        good_threshold=0.5,
        bad_threshold=0.5,
        invert_thresholds=True,  # Valid (1) is better
    ),
]


@dataclass
class OverlayConfig:
    """Master configuration for video overlay tool."""

    # Input/Output paths
    input_video: str = ""
    """Path to input video file."""

    input_json: str = ""
    """Path to input JSON metrics file."""

    output_video: str = ""
    """Path to output video file."""

    # Data type
    data_type: Literal["timeseries", "trajectory"] = "timeseries"
    """Type of JSON data to read."""

    env_id: int = 0
    """Environment ID for trajectory data."""

    # Layout configuration
    panel: PanelLayoutConfig = field(default_factory=PanelLayoutConfig)
    """Info panel layout configuration."""

    graph: GraphConfig = field(default_factory=GraphConfig)
    """Animated graph configuration."""

    # Metrics to display
    metrics: list[MetricDisplayConfig] = field(default_factory=lambda: DEFAULT_METRICS.copy())
    """List of metrics to display."""

    # Theme
    theme: ColorTheme = field(default_factory=ColorTheme)
    """Color theme."""

    # Timestamp display
    show_timestamp: bool = True
    """Whether to show simulation time."""

    timestamp_position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "bottom_left"
    """Position of timestamp display."""

    # Output settings
    output_fps: float | None = None
    """Output video FPS (None = match input)."""

    codec: str = "mp4v"
    """OpenCV fourcc codec string."""
