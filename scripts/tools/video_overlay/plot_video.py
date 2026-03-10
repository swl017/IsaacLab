#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate a standalone animated metrics plot video for IROS paper demonstrations.

This creates a professional-quality video showing time-series plots of tracking metrics
over the entire episode duration. Unlike the overlay tool, this generates a dedicated
visualization video with:
- Full-width animated time-series graphs
- Professional styling with gridlines, legends, and annotations
- Smooth playhead animation showing current timestep
- Optional vertical sync line across all plots
- Configurable color schemes and layouts

Usage examples:

    # Basic plot video from timeseries data
    python plot_video.py \\
        --input-json results.json \\
        --output-video metrics_plot.mp4

    # Custom metrics and styling
    python plot_video.py \\
        --input-json results.json \\
        --output-video metrics_plot.mp4 \\
        --metrics viewing_angle triangulation_rmse visibility \\
        --style dark \\
        --fps 30

    # With trajectory data (specific environment)
    python plot_video.py \\
        --input-json trajectory.json \\
        --output-video metrics_plot.mp4 \\
        --data-type trajectory \\
        --env-id 0

    # Side-by-side with simulation video (same duration)
    python plot_video.py \\
        --input-json results.json \\
        --output-video metrics_plot.mp4 \\
        --sync-video simulation.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# Try to import matplotlib with Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# Color schemes
COLOR_SCHEMES = {
    "dark": {
        "background": "#1a1a2e",
        "panel": "#16213e",
        "grid": "#2a2a4a",
        "text": "#e8e8e8",
        "text_secondary": "#a0a0a0",
        "viewing_angle": "#00d4aa",  # Teal
        "triangulation_rmse": "#ff6b6b",  # Coral
        "visibility": "#4ecdc4",  # Cyan
        "tri_valid": "#ffe66d",  # Yellow
        "sqrt_trace_sigma": "#c792ea",  # Purple
        "ego_aoi": "#ff9f43",  # Orange - ego delay
        "other_aoi": "#ee5a24",  # Dark orange - comm delay
        "distance_to_target": "#a29bfe",  # Light purple
        "target_speed": "#fd79a8",  # Pink
        "playhead": "#ffffff",
        "good_zone": "#00d4aa33",
        "bad_zone": "#ff6b6b33",
    },
    "light": {
        "background": "#f5f5f5",
        "panel": "#ffffff",
        "grid": "#e0e0e0",
        "text": "#2a2a2a",
        "text_secondary": "#707070",
        "viewing_angle": "#00897b",
        "triangulation_rmse": "#e53935",
        "visibility": "#00acc1",
        "tri_valid": "#fdd835",
        "sqrt_trace_sigma": "#8e24aa",
        "ego_aoi": "#f57c00",  # Orange
        "other_aoi": "#d84315",  # Dark orange
        "distance_to_target": "#7e57c2",  # Purple
        "target_speed": "#ec407a",  # Pink
        "playhead": "#333333",
        "good_zone": "#00897b22",
        "bad_zone": "#e5393522",
    },
    "paper": {
        "background": "#ffffff",
        "panel": "#ffffff",
        "grid": "#cccccc",
        "text": "#000000",
        "text_secondary": "#555555",
        "viewing_angle": "#1f77b4",  # Matplotlib blue
        "triangulation_rmse": "#d62728",  # Matplotlib red
        "visibility": "#2ca02c",  # Matplotlib green
        "tri_valid": "#ff7f0e",  # Matplotlib orange
        "sqrt_trace_sigma": "#9467bd",  # Matplotlib purple
        "ego_aoi": "#e377c2",  # Matplotlib pink
        "other_aoi": "#bcbd22",  # Matplotlib olive
        "distance_to_target": "#17becf",  # Matplotlib cyan
        "target_speed": "#8c564b",  # Matplotlib brown
        "playhead": "#000000",
        "good_zone": "#2ca02c22",
        "bad_zone": "#d6272822",
    },
}

METRIC_CONFIG = {
    "viewing_angle": {
        "display_name": "Viewing Angle",
        "unit": "deg",
        "aux_lines": [90.0],  # Single auxiliary line at 90 degrees
        "aux_line_color": "text_secondary",  # Use secondary text color
        "invert": True,  # Higher is better
        "y_label": "Angle (°)",
    },
    "triangulation_rmse": {
        "display_name": "Triangulation RMSE",
        "unit": "m",
        "aux_lines": [2.0],  # Single auxiliary line at 2m
        "aux_line_color": "viewing_angle",  # Green color
        "invert": False,  # Lower is better
        "y_label": "RMSE (m)",
        "log_scale": True,  # Use log scale for better visualization
    },
    "visibility": {
        "display_name": "Visibility",
        "unit": "%",
        "aux_lines": [],  # No auxiliary lines
        "invert": True,
        "y_label": "Visibility",
        "scale": 100,  # Convert 0-1 to percentage
    },
    "tri_valid": {
        "display_name": "Triangulation Valid",
        "unit": "",
        "aux_lines": [],  # No auxiliary lines
        "invert": True,
        "y_label": "Valid",
    },
    "sqrt_trace_sigma": {
        "display_name": "Position Uncertainty",
        "unit": "m",
        "aux_lines": [5.0],  # Reference line at 5m
        "aux_line_color": "viewing_angle",
        "invert": False,
        "y_label": "Uncertainty (m)",
    },
    "ego_aoi": {
        "display_name": "Ego Detection Delay",
        "unit": "s",
        "aux_lines": [],
        "invert": False,  # Lower is better
        "y_label": "Delay (s)",
    },
    "other_aoi": {
        "display_name": "Comm Delay",
        "unit": "s",
        "aux_lines": [],
        "invert": False,  # Lower is better
        "y_label": "Delay (s)",
    },
    "distance_to_target": {
        "display_name": "Distance to Target",
        "unit": "m",
        "aux_lines": [],
        "invert": False,
        "y_label": "Distance (m)",
    },
    "target_speed": {
        "display_name": "Target Speed",
        "unit": "m/s",
        "aux_lines": [],
        "invert": False,
        "y_label": "Speed (m/s)",
    },
}


class PlotVideoGenerator:
    """Generate animated time-series plot video from metrics data."""

    def __init__(
        self,
        metrics_data: dict,
        step_dt: float,
        output_path: str,
        metrics_to_plot: list[str],
        style: str = "dark",
        fps: int = 30,
        resolution: tuple[int, int] = (1920, 1080),
        show_std: bool = True,
        show_thresholds: bool = True,
        title: str = "Performance Metrics",
        overlay_groups: list[list[str]] | None = None,
    ):
        """Initialize the plot video generator.

        Args:
            metrics_data: Dictionary containing timeseries or trajectory data.
            step_dt: Simulation step time in seconds.
            output_path: Output video file path.
            metrics_to_plot: List of metric keys to plot.
            style: Color scheme name ('dark', 'light', 'paper').
            fps: Output video frame rate.
            resolution: Video resolution (width, height).
            show_std: Show standard deviation bands (for timeseries data).
            show_thresholds: Show good/bad threshold regions.
            title: Video title text.
            overlay_groups: List of metric groups to overlay on same axes.
                Example: [["ego_aoi", "other_aoi"]] overlays both delays.
        """
        self.metrics_data = metrics_data
        self.step_dt = step_dt
        self.output_path = output_path
        self.metrics_to_plot = metrics_to_plot
        self.colors = COLOR_SCHEMES.get(style, COLOR_SCHEMES["dark"])
        self.fps = fps
        self.resolution = resolution
        self.show_std = show_std
        self.show_thresholds = show_thresholds
        self.title = title
        self.overlay_groups = overlay_groups or []

        # Determine number of timesteps
        self.num_timesteps = self._get_num_timesteps()
        self.total_time = self.num_timesteps * step_dt

        # Prepare time array
        self.time_array = np.arange(self.num_timesteps) * step_dt

        # Extract and prepare metric data
        self.metric_arrays = {}
        self.metric_stds = {}
        for metric in metrics_to_plot:
            data, std = self._extract_metric_data(metric)
            if data is not None:
                self.metric_arrays[metric] = data
                if std is not None:
                    self.metric_stds[metric] = std

        # Build plot groups (metrics that share the same axes)
        self._build_plot_groups()

        # Create matplotlib figure
        self._setup_figure()

    def _build_plot_groups(self):
        """Build groups of metrics that share the same axes."""
        # Start with overlay groups
        used_metrics = set()
        self.plot_groups = []

        for group in self.overlay_groups:
            valid_group = [m for m in group if m in self.metric_arrays]
            if valid_group:
                self.plot_groups.append(valid_group)
                used_metrics.update(valid_group)

        # Add remaining metrics as individual groups
        for metric in self.metric_arrays:
            if metric not in used_metrics:
                self.plot_groups.append([metric])

    def _get_num_timesteps(self) -> int:
        """Determine number of timesteps from data."""
        for key in self.metrics_to_plot:
            if key in self.metrics_data:
                data = self.metrics_data[key]
                if isinstance(data, dict) and "mean" in data:
                    return len(data["mean"])
                elif isinstance(data, list):
                    return len(data)
        return 0

    def _extract_metric_data(self, metric_key: str) -> tuple:
        """Extract metric array and optional std from data.

        Returns:
            Tuple of (values_array, std_array or None).
        """
        if metric_key not in self.metrics_data:
            return None, None

        data = self.metrics_data[metric_key]

        # Handle timeseries format with mean/std
        if isinstance(data, dict) and "mean" in data:
            values = np.array(data["mean"])
            std = np.array(data.get("std", [])) if "std" in data else None
            return values, std

        # Handle simple list format
        if isinstance(data, list):
            return np.array(data), None

        return None, None

    def _setup_figure(self):
        """Create and configure the matplotlib figure."""
        # Calculate figure size in inches (assuming 100 DPI for crisp rendering)
        dpi = 100
        fig_width = self.resolution[0] / dpi
        fig_height = self.resolution[1] / dpi

        # Create figure with dark background
        self.fig = Figure(figsize=(fig_width, fig_height), dpi=dpi,
                         facecolor=self.colors["background"])

        # Create subplots - one per plot group (grouped metrics share axes)
        num_plots = len(self.plot_groups)
        if num_plots == 0:
            num_plots = 1

        # Layout: vertically stacked plots with shared x-axis
        self.axes = []
        for i in range(num_plots):
            ax = self.fig.add_subplot(num_plots, 1, i + 1)
            ax.set_facecolor(self.colors["panel"])
            self.axes.append(ax)

        # Add title
        self.fig.suptitle(
            self.title,
            fontsize=20,
            fontweight='bold',
            color=self.colors["text"],
            y=0.98
        )

        # Adjust layout
        self.fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.94])

        # Create canvas for rendering
        self.canvas = FigureCanvasAgg(self.fig)

    def _draw_frame(self, current_timestep: int) -> np.ndarray:
        """Draw a single frame at the given timestep.

        Args:
            current_timestep: Current timestep index.

        Returns:
            Frame as numpy array (RGB).
        """
        current_time = current_timestep * self.step_dt

        # Clear and redraw each subplot (one per plot group)
        for ax_idx, metric_group in enumerate(self.plot_groups):
            ax = self.axes[ax_idx]
            ax.clear()
            ax.set_facecolor(self.colors["panel"])

            # Track y limits across all metrics in group
            group_y_min = float('inf')
            group_y_max = float('-inf')
            use_log_scale = False
            legend_handles = []
            legend_labels = []

            # Determine if any metric in group uses log scale
            for metric_key in metric_group:
                config = METRIC_CONFIG.get(metric_key, {})
                if config.get("log_scale", False):
                    use_log_scale = True
                    break

            # Plot each metric in the group
            annotation_offset_y = 10
            for metric_key in metric_group:
                if metric_key not in self.metric_arrays:
                    continue

                values = self.metric_arrays[metric_key]
                config = METRIC_CONFIG.get(metric_key, {})
                display_name = config.get("display_name", metric_key.replace("_", " ").title())
                scale = config.get("scale", 1.0)
                unit = config.get("unit", "")

                # Scale values
                plot_values = values * scale
                plot_std = None
                if metric_key in self.metric_stds:
                    plot_std = self.metric_stds[metric_key] * scale

                # Update y limits
                group_y_min = min(group_y_min, np.nanmin(plot_values))
                group_y_max = max(group_y_max, np.nanmax(plot_values))

                # Get line color
                line_color = self.colors.get(metric_key, self.colors["viewing_angle"])

                # Draw auxiliary lines
                aux_lines = config.get("aux_lines", [])
                aux_color_key = config.get("aux_line_color", "text_secondary")
                aux_color = self.colors.get(aux_color_key, self.colors["text_secondary"])

                if self.show_thresholds and aux_lines:
                    for aux_val in aux_lines:
                        ax.axhline(aux_val * scale, color=aux_color,
                                  linestyle='--', alpha=0.6, linewidth=1.5, zorder=0)

                # Plot full data as faded background
                ax.plot(self.time_array, plot_values,
                       color=line_color, alpha=0.3, linewidth=1.5, zorder=1)

                # Plot std band if available
                if self.show_std and plot_std is not None:
                    ax.fill_between(
                        self.time_array,
                        plot_values - plot_std,
                        plot_values + plot_std,
                        alpha=0.1, color=line_color, zorder=0
                    )

                # Plot data up to current timestep (highlighted)
                if current_timestep > 0:
                    line, = ax.plot(
                        self.time_array[:current_timestep + 1],
                        plot_values[:current_timestep + 1],
                        color=line_color, linewidth=2.5, zorder=2,
                        label=display_name
                    )
                    legend_handles.append(line)
                    legend_labels.append(display_name)

                    # Draw current value marker
                    ax.scatter(
                        [current_time],
                        [plot_values[current_timestep]],
                        color=line_color, s=80, zorder=3,
                        edgecolors='white', linewidths=2
                    )

                    # Annotate current value (stagger for multiple metrics)
                    current_val = plot_values[current_timestep]

                    # Format value
                    if current_val < 1.0:
                        val_str = f'{current_val:.2f}{unit}'
                    else:
                        val_str = f'{current_val:.1f}{unit}'

                    ax.annotate(
                        val_str,
                        xy=(current_time, current_val),
                        xytext=(10, annotation_offset_y), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=self.colors["text"],
                        bbox=dict(boxstyle='round,pad=0.2',
                                 facecolor=self.colors["panel"],
                                 edgecolor=line_color, alpha=0.9)
                    )
                    annotation_offset_y += 25  # Stagger annotations

            # Draw playhead line
            ax.axvline(current_time, color=self.colors["playhead"],
                      linestyle='-', linewidth=1.5, alpha=0.7, zorder=4)

            # Configure axes
            ax.set_xlim(0, self.total_time)

            # Set y limits with padding
            if use_log_scale:
                group_y_min = max(group_y_min, 0.01)
                group_y_max = max(group_y_max, group_y_min * 10)
                ax.set_yscale('log')
                ax.set_ylim(group_y_min * 0.5, group_y_max * 2.0)
            else:
                y_range = group_y_max - group_y_min
                if y_range < 0.001:
                    y_range = 1.0
                ax.set_ylim(group_y_min - y_range * 0.15, group_y_max + y_range * 0.15)

            # Labels and styling
            first_config = METRIC_CONFIG.get(metric_group[0], {})
            y_label = first_config.get("y_label", metric_group[0].replace("_", " ").title())
            ax.set_ylabel(y_label, fontsize=12, color=self.colors["text"], fontweight='bold')
            ax.tick_params(colors=self.colors["text_secondary"], labelsize=10)

            # Grid
            ax.grid(True, alpha=0.3, color=self.colors["grid"], linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            # Spine colors
            for spine in ax.spines.values():
                spine.set_color(self.colors["grid"])

            # Only show x-label on bottom plot
            if ax_idx == len(self.axes) - 1:
                ax.set_xlabel("Time (s)", fontsize=12, color=self.colors["text"], fontweight='bold')
            else:
                ax.set_xticklabels([])

            # Title and legend for group
            if len(metric_group) == 1:
                # Single metric - use metric name as title
                display_name = METRIC_CONFIG.get(metric_group[0], {}).get(
                    "display_name", metric_group[0].replace("_", " ").title())
                ax.set_title(display_name, fontsize=14, color=self.colors["text"],
                            fontweight='bold', loc='left', pad=10)
            else:
                # Multiple metrics - use combined title and legend
                group_title = " vs ".join([
                    METRIC_CONFIG.get(m, {}).get("display_name", m.replace("_", " ").title())
                    for m in metric_group
                ])
                ax.set_title(group_title, fontsize=14, color=self.colors["text"],
                            fontweight='bold', loc='left', pad=10)

                # Add legend for overlayed metrics
                if legend_handles:
                    ax.legend(legend_handles, legend_labels,
                             loc='upper right', fontsize=10,
                             facecolor=self.colors["panel"],
                             edgecolor=self.colors["grid"],
                             labelcolor=self.colors["text"])

        # Add time indicator in corner
        time_text = f"t = {current_time:.2f}s"
        self.fig.text(0.95, 0.02, time_text, fontsize=14,
                     color=self.colors["text"], ha='right',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=self.colors["panel"],
                              edgecolor=self.colors["grid"]))

        # Render to numpy array
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        frame = np.asarray(buf)

        # Convert RGBA to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        return frame_bgr

    def generate(self):
        """Generate the full plot video."""
        print(f"Generating plot video: {self.output_path}")
        print(f"  Metrics: {', '.join(self.metric_arrays.keys())}")
        print(f"  Duration: {self.total_time:.2f}s ({self.num_timesteps} timesteps)")
        print(f"  Output FPS: {self.fps}")
        print(f"  Resolution: {self.resolution[0]}x{self.resolution[1]}")

        # Create output directory if needed
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps,
            self.resolution
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_path}")

        # Calculate frame-to-timestep mapping
        video_duration = self.total_time
        total_frames = int(video_duration * self.fps)

        print(f"  Total frames: {total_frames}")
        print("  Rendering...")

        for frame_idx in range(total_frames):
            # Map frame to timestep
            frame_time = frame_idx / self.fps
            timestep = min(int(frame_time / self.step_dt), self.num_timesteps - 1)

            # Draw frame
            frame = self._draw_frame(timestep)

            # Write frame
            writer.write(frame)

            # Progress update
            if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
                progress = 100 * (frame_idx + 1) / total_frames
                print(f"    Frame {frame_idx + 1}/{total_frames} ({progress:.1f}%)")

        writer.release()
        plt.close(self.fig)

        print(f"\nDone! Plot video saved to: {self.output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate animated metrics plot video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input-json", required=True,
        help="Input JSON metrics file path"
    )
    parser.add_argument(
        "--output-video", required=True,
        help="Output video file path"
    )
    parser.add_argument(
        "--data-type",
        choices=["timeseries", "trajectory"],
        default="timeseries",
        help="Type of JSON data (default: timeseries)"
    )
    parser.add_argument(
        "--env-id", type=int, default=0,
        help="Environment ID for trajectory data (default: 0)"
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["viewing_angle", "triangulation_rmse", "visibility"],
        help="Metrics to plot (default: viewing_angle triangulation_rmse visibility)"
    )
    parser.add_argument(
        "--style",
        choices=["dark", "light", "paper"],
        default="dark",
        help="Visual style/color scheme (default: dark)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video FPS (default: 30)"
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[1920, 1080],
        metavar=("WIDTH", "HEIGHT"),
        help="Video resolution (default: 1920 1080)"
    )
    parser.add_argument(
        "--no-std", action="store_true",
        help="Don't show standard deviation bands"
    )
    parser.add_argument(
        "--no-thresholds", action="store_true",
        help="Don't show good/bad threshold regions"
    )
    parser.add_argument(
        "--title", type=str, default="Performance Metrics",
        help="Video title text"
    )
    parser.add_argument(
        "--sync-video", type=str, default=None,
        help="Sync duration with another video file"
    )
    parser.add_argument(
        "--overlay", type=str, nargs="+", action="append", default=[],
        metavar="METRIC",
        help="Overlay metrics on same axes (can be used multiple times). "
             "Example: --overlay ego_aoi other_aoi"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input file
    if not os.path.exists(args.input_json):
        print(f"Error: Input JSON not found: {args.input_json}")
        sys.exit(1)

    # Load JSON data
    print(f"Loading metrics from: {args.input_json}")
    with open(args.input_json, "r") as f:
        json_data = json.load(f)

    # Extract metrics data based on type
    if args.data_type == "timeseries":
        if "timeseries" in json_data:
            metrics_data = json_data["timeseries"]
        else:
            metrics_data = json_data
        step_dt = metrics_data.get("step_dt_seconds", 0.01667)
    else:  # trajectory
        if "trajectories" in json_data:
            traj_root = json_data["trajectories"]
        else:
            traj_root = json_data
        step_dt = traj_root.get("step_dt_seconds", 0.01667)

        # Get specific environment trajectory
        env_id_str = str(args.env_id)
        if "envs" in traj_root and env_id_str in traj_root["envs"]:
            metrics_data = traj_root["envs"][env_id_str]
        else:
            print(f"Error: Environment {args.env_id} not found in trajectory data")
            sys.exit(1)

    print(f"  Data type: {args.data_type}")
    print(f"  Step dt: {step_dt:.5f}s ({1.0/step_dt:.1f} Hz)")

    # Sync duration with another video if specified
    if args.sync_video and os.path.exists(args.sync_video):
        cap = cv2.VideoCapture(args.sync_video)
        if cap.isOpened():
            sync_fps = cap.get(cv2.CAP_PROP_FPS)
            sync_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sync_duration = sync_frames / sync_fps
            print(f"  Syncing with video: {args.sync_video}")
            print(f"    Video duration: {sync_duration:.2f}s")
            # Use same FPS as sync video
            args.fps = int(sync_fps)
        cap.release()

    # Parse overlay groups
    overlay_groups = args.overlay if args.overlay else []
    if overlay_groups:
        print(f"  Overlay groups: {overlay_groups}")

    # Create generator
    generator = PlotVideoGenerator(
        metrics_data=metrics_data,
        step_dt=step_dt,
        output_path=args.output_video,
        metrics_to_plot=args.metrics,
        style=args.style,
        fps=args.fps,
        resolution=tuple(args.resolution),
        show_std=not args.no_std,
        show_thresholds=not args.no_thresholds,
        title=args.title,
        overlay_groups=overlay_groups,
    )

    # Generate video
    generator.generate()


if __name__ == "__main__":
    main()
