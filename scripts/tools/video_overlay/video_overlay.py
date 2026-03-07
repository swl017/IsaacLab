#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Video Overlay Post-Processing Tool for IROS Paper Demonstration Videos.

Overlays real-time tracking metrics onto recorded simulation videos.

Usage examples:

    # Basic overlay with default metrics
    python video_overlay.py \\
        --input-video demo_raw.mp4 \\
        --input-json results.json \\
        --output-video demo_overlay.mp4

    # Specify which metrics to show
    python video_overlay.py \\
        --input-video demo_raw.mp4 \\
        --input-json results.json \\
        --output-video demo_overlay.mp4 \\
        --metrics viewing_angle rmse visibility

    # With animated graph
    python video_overlay.py \\
        --input-video demo_raw.mp4 \\
        --input-json results.json \\
        --output-video demo_overlay.mp4 \\
        --enable-graph \\
        --graph-metrics viewing_angle rmse

    # Using trajectory data instead of timeseries
    python video_overlay.py \\
        --input-video demo_raw.mp4 \\
        --input-json trajectory.json \\
        --output-video demo_overlay.mp4 \\
        --data-type trajectory \\
        --env-id 0

    # Custom thresholds for color coding
    python video_overlay.py \\
        --input-video demo_raw.mp4 \\
        --input-json results.json \\
        --output-video demo_overlay.mp4 \\
        --rmse-thresholds 2.0 5.0 \\
        --angle-thresholds 30.0 20.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add script directory to path for local imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import cv2
import numpy as np

from overlay_config import (
    OverlayConfig,
    PanelLayoutConfig,
    GraphConfig,
    MetricDisplayConfig,
    ColorTheme,
    DEFAULT_METRICS,
)
from metric_synchronizer import MetricSynchronizer
from overlay_renderer import OverlayRenderer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video overlay post-processing tool for IROS paper demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--input-video", required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--input-json", required=True,
        help="Input JSON metrics file path"
    )
    parser.add_argument(
        "--output-video", required=True,
        help="Output video file path"
    )

    # Data type
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

    # Panel configuration
    parser.add_argument(
        "--panel-position",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        default="top_right",
        help="Position of info panel (default: top_right)"
    )
    parser.add_argument(
        "--panel-width", type=int, default=380,
        help="Panel width in pixels (default: 380)"
    )
    parser.add_argument(
        "--panel-alpha", type=float, default=0.7,
        help="Panel background transparency 0-1 (default: 0.7)"
    )

    # Metrics to display
    parser.add_argument(
        "--metrics", nargs="+",
        default=["viewing_angle", "triangulation_rmse", "visibility", "tri_valid"],
        help="Metrics to display in panel (default: viewing_angle rmse visibility tri_valid)"
    )

    # Graph options
    parser.add_argument(
        "--enable-graph", action="store_true",
        help="Enable animated metric graph"
    )
    parser.add_argument(
        "--graph-metrics", nargs="+",
        default=["viewing_angle", "triangulation_rmse"],
        help="Metrics to show in graph (default: viewing_angle rmse)"
    )
    parser.add_argument(
        "--graph-position",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        default="bottom_left",
        help="Position of graph panel (default: bottom_left)"
    )
    parser.add_argument(
        "--graph-window", type=int, default=100,
        help="Number of timesteps in graph window (default: 100)"
    )

    # Thresholds for color coding
    parser.add_argument(
        "--rmse-thresholds", nargs=2, type=float, default=[2.0, 5.0],
        metavar=("GOOD", "BAD"),
        help="RMSE thresholds [good, bad] in meters (default: 2.0 5.0)"
    )
    parser.add_argument(
        "--angle-thresholds", nargs=2, type=float, default=[30.0, 20.0],
        metavar=("GOOD", "BAD"),
        help="Viewing angle thresholds [good, bad] in degrees (default: 30.0 20.0)"
    )
    parser.add_argument(
        "--visibility-thresholds", nargs=2, type=float, default=[0.8, 0.5],
        metavar=("GOOD", "BAD"),
        help="Visibility thresholds [good, bad] as ratio (default: 0.8 0.5)"
    )

    # Timestamp
    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="Disable timestamp display"
    )
    parser.add_argument(
        "--timestamp-position",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        default="bottom_left",
        help="Position of timestamp (default: bottom_left)"
    )

    # Output options
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output FPS (default: match input video)"
    )
    parser.add_argument(
        "--codec", default="mp4v",
        help="Output codec fourcc code (default: mp4v)"
    )

    return parser.parse_args()


def build_metric_configs(args) -> list[MetricDisplayConfig]:
    """Build metric configurations from arguments."""
    configs = []

    metric_defs = {
        "viewing_angle": MetricDisplayConfig(
            name="Viewing Angle",
            key="viewing_angle",
            unit="deg",
            decimals=1,
            good_threshold=args.angle_thresholds[0],
            bad_threshold=args.angle_thresholds[1],
            invert_thresholds=True,
        ),
        "triangulation_rmse": MetricDisplayConfig(
            name="RMSE",
            key="triangulation_rmse",
            unit="m",
            decimals=2,
            good_threshold=args.rmse_thresholds[0],
            bad_threshold=args.rmse_thresholds[1],
            invert_thresholds=False,
        ),
        "rmse": MetricDisplayConfig(
            name="RMSE",
            key="triangulation_rmse",
            unit="m",
            decimals=2,
            good_threshold=args.rmse_thresholds[0],
            bad_threshold=args.rmse_thresholds[1],
            invert_thresholds=False,
        ),
        "visibility": MetricDisplayConfig(
            name="Visibility",
            key="visibility",
            unit="%",
            decimals=0,
            good_threshold=args.visibility_thresholds[0],
            bad_threshold=args.visibility_thresholds[1],
            invert_thresholds=True,
        ),
        "tri_valid": MetricDisplayConfig(
            name="Tri Valid",
            key="tri_valid",
            unit="",
            decimals=0,
            good_threshold=0.5,
            bad_threshold=0.5,
            invert_thresholds=True,
        ),
        "sqrt_trace_sigma": MetricDisplayConfig(
            name="Uncertainty",
            key="sqrt_trace_sigma",
            unit="m",
            decimals=2,
            good_threshold=5.0,
            bad_threshold=20.0,
            invert_thresholds=False,
        ),
    }

    for metric_key in args.metrics:
        if metric_key in metric_defs:
            configs.append(metric_defs[metric_key])
        else:
            # Create generic config for unknown metrics
            configs.append(MetricDisplayConfig(
                name=metric_key.replace("_", " ").title(),
                key=metric_key,
            ))

    return configs


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
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
        # Determine number of timesteps from first metric
        for key in ["viewing_angle", "triangulation_rmse", "visibility"]:
            if key in metrics_data:
                data = metrics_data[key]
                if isinstance(data, dict) and "mean" in data:
                    num_timesteps = len(data["mean"])
                elif isinstance(data, list):
                    num_timesteps = len(data)
                break
        else:
            print("Error: Could not determine number of timesteps from JSON data")
            sys.exit(1)
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

        num_timesteps = metrics_data.get("num_steps", len(metrics_data.get("viewing_angle", [])))

    print(f"  Data type: {args.data_type}")
    print(f"  Step dt: {step_dt:.5f}s ({1.0/step_dt:.1f} Hz)")
    print(f"  Timesteps: {num_timesteps}")

    # Open input video
    print(f"Opening video: {args.input_video}")
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.input_video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")

    # Create synchronizer
    synchronizer = MetricSynchronizer(
        video_fps=video_fps,
        step_dt_seconds=step_dt,
        num_timesteps=num_timesteps,
    )

    # Build configuration
    config = OverlayConfig(
        input_video=args.input_video,
        input_json=args.input_json,
        output_video=args.output_video,
        data_type=args.data_type,
        env_id=args.env_id,
        panel=PanelLayoutConfig(
            position=args.panel_position,
            width=args.panel_width,
            background_alpha=args.panel_alpha,
        ),
        graph=GraphConfig(
            enabled=args.enable_graph,
            position=args.graph_position,
            metrics=args.graph_metrics,
            window_size=args.graph_window,
        ),
        metrics=build_metric_configs(args),
        show_timestamp=not args.no_timestamp,
        timestamp_position=args.timestamp_position,
        output_fps=args.fps,
        codec=args.codec,
    )

    # Create renderer
    renderer = OverlayRenderer(config)

    # Create output video writer
    output_fps = args.fps if args.fps else video_fps
    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    # Ensure output directory exists
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (frame_width, frame_height),
    )

    print(f"\nProcessing video...")
    print(f"  Output: {args.output_video}")
    print(f"  Metrics: {', '.join(args.metrics)}")
    if args.enable_graph:
        print(f"  Graph: {', '.join(args.graph_metrics)}")

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get metrics for this frame
        if args.data_type == "timeseries":
            metrics = synchronizer.get_metrics_for_frame_timeseries(frame_idx, metrics_data)
        else:
            metrics = synchronizer.get_metrics_for_frame_trajectory(frame_idx, metrics_data)

        # Get graph data if enabled
        if args.enable_graph:
            graph_data = synchronizer.get_graph_window(
                frame_idx, metrics_data, args.graph_metrics, args.graph_window
            )
        else:
            graph_data = None

        # Render overlays
        output_frame = renderer.render_frame(frame, metrics, graph_data)

        # Write output
        out.write(output_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            progress = 100 * frame_idx / total_frames
            print(f"  Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    out.release()

    print(f"\nDone! Output saved to: {args.output_video}")
    print(f"  Total frames processed: {frame_idx}")


if __name__ == "__main__":
    main()
