#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Combine simulation video and metrics plot video into a single presentation video.

Layout options:
- side_by_side: Simulation on left, plots on right (16:9 output)
- stacked: Simulation on top, plots on bottom
- picture_in_picture: Plots overlaid on simulation corner

Usage examples:

    # Side-by-side layout (default)
    python combine_videos.py \\
        --simulation-video demo.mp4 \\
        --plot-video metrics_plot.mp4 \\
        --output-video combined.mp4

    # Stacked layout
    python combine_videos.py \\
        --simulation-video demo.mp4 \\
        --plot-video metrics_plot.mp4 \\
        --output-video combined.mp4 \\
        --layout stacked

    # Picture-in-picture
    python combine_videos.py \\
        --simulation-video demo.mp4 \\
        --plot-video metrics_plot.mp4 \\
        --output-video combined.mp4 \\
        --layout pip \\
        --pip-position bottom_right \\
        --pip-scale 0.35
"""

import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine simulation and plot videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--simulation-video", required=True,
        help="Input simulation video path"
    )
    parser.add_argument(
        "--plot-video", required=True,
        help="Input plot video path"
    )
    parser.add_argument(
        "--output-video", required=True,
        help="Output combined video path"
    )
    parser.add_argument(
        "--layout",
        choices=["side_by_side", "stacked", "pip"],
        default="side_by_side",
        help="Layout mode (default: side_by_side)"
    )
    parser.add_argument(
        "--sim-weight", type=float, default=0.6,
        help="Width ratio for simulation video in side_by_side mode (default: 0.6)"
    )
    parser.add_argument(
        "--pip-position",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        default="bottom_right",
        help="Position for PIP mode (default: bottom_right)"
    )
    parser.add_argument(
        "--pip-scale", type=float, default=0.35,
        help="Scale of PIP overlay relative to main video (default: 0.35)"
    )
    parser.add_argument(
        "--pip-margin", type=int, default=20,
        help="Margin for PIP from edge (default: 20)"
    )
    parser.add_argument(
        "--output-width", type=int, default=1920,
        help="Output video width (default: 1920)"
    )
    parser.add_argument(
        "--output-height", type=int, default=None,
        help="Output video height (auto-calculated if not specified)"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output FPS (default: match simulation video)"
    )
    parser.add_argument(
        "--background-color", type=int, nargs=3, default=[30, 30, 30],
        metavar=("B", "G", "R"),
        help="Background color BGR (default: 30 30 30)"
    )

    return parser.parse_args()


def resize_with_aspect(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize frame maintaining aspect ratio with letterboxing."""
    h, w = frame.shape[:2]
    aspect = w / h
    target_aspect = target_width / target_height

    if aspect > target_aspect:
        # Frame is wider - fit to width
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        # Frame is taller - fit to height
        new_h = target_height
        new_w = int(target_height * aspect)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create output with letterboxing
    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return output


def combine_side_by_side(
    sim_frame: np.ndarray,
    plot_frame: np.ndarray,
    output_width: int,
    sim_weight: float,
    bg_color: tuple,
) -> np.ndarray:
    """Combine frames side by side."""
    sim_width = int(output_width * sim_weight)
    plot_width = output_width - sim_width

    # Get output height from simulation frame aspect ratio
    sim_h, sim_w = sim_frame.shape[:2]
    sim_aspect = sim_w / sim_h
    output_height = int(sim_width / sim_aspect)

    # Resize both frames
    sim_resized = resize_with_aspect(sim_frame, sim_width, output_height)
    plot_resized = resize_with_aspect(plot_frame, plot_width, output_height)

    # Combine
    output = np.full((output_height, output_width, 3), bg_color, dtype=np.uint8)
    output[:, :sim_width] = sim_resized
    output[:, sim_width:] = plot_resized

    return output


def combine_stacked(
    sim_frame: np.ndarray,
    plot_frame: np.ndarray,
    output_width: int,
    bg_color: tuple,
) -> np.ndarray:
    """Combine frames vertically stacked."""
    # Calculate heights (60% sim, 40% plot)
    sim_h, sim_w = sim_frame.shape[:2]
    sim_aspect = sim_w / sim_h

    sim_height = int(output_width / sim_aspect)
    plot_height = int(sim_height * 0.6)
    output_height = sim_height + plot_height

    # Resize both frames
    sim_resized = resize_with_aspect(sim_frame, output_width, sim_height)
    plot_resized = resize_with_aspect(plot_frame, output_width, plot_height)

    # Combine
    output = np.full((output_height, output_width, 3), bg_color, dtype=np.uint8)
    output[:sim_height, :] = sim_resized
    output[sim_height:, :] = plot_resized

    return output


def combine_pip(
    sim_frame: np.ndarray,
    plot_frame: np.ndarray,
    output_width: int,
    output_height: int,
    pip_position: str,
    pip_scale: float,
    pip_margin: int,
) -> np.ndarray:
    """Combine with picture-in-picture overlay."""
    # Resize simulation to output size
    output = resize_with_aspect(sim_frame, output_width, output_height)

    # Calculate PIP size
    pip_width = int(output_width * pip_scale)
    pip_height = int(output_height * pip_scale)

    # Resize plot for PIP
    pip_frame = resize_with_aspect(plot_frame, pip_width, pip_height)

    # Calculate position
    if "top" in pip_position:
        y = pip_margin
    else:
        y = output_height - pip_height - pip_margin

    if "left" in pip_position:
        x = pip_margin
    else:
        x = output_width - pip_width - pip_margin

    # Overlay with border
    border = 3
    cv2.rectangle(
        output,
        (x - border, y - border),
        (x + pip_width + border, y + pip_height + border),
        (200, 200, 200), border
    )
    output[y:y + pip_height, x:x + pip_width] = pip_frame

    return output


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.simulation_video):
        print(f"Error: Simulation video not found: {args.simulation_video}")
        sys.exit(1)
    if not os.path.exists(args.plot_video):
        print(f"Error: Plot video not found: {args.plot_video}")
        sys.exit(1)

    # Open input videos
    sim_cap = cv2.VideoCapture(args.simulation_video)
    plot_cap = cv2.VideoCapture(args.plot_video)

    if not sim_cap.isOpened():
        print(f"Error: Cannot open simulation video: {args.simulation_video}")
        sys.exit(1)
    if not plot_cap.isOpened():
        print(f"Error: Cannot open plot video: {args.plot_video}")
        sys.exit(1)

    # Get video properties
    sim_fps = sim_cap.get(cv2.CAP_PROP_FPS)
    sim_frames = int(sim_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sim_width = int(sim_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sim_height = int(sim_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    plot_fps = plot_cap.get(cv2.CAP_PROP_FPS)
    plot_frames = int(plot_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = args.fps if args.fps else sim_fps
    total_frames = min(sim_frames, int(plot_frames * sim_fps / plot_fps))

    print(f"Combining videos:")
    print(f"  Simulation: {args.simulation_video}")
    print(f"    {sim_width}x{sim_height} @ {sim_fps:.1f} fps, {sim_frames} frames")
    print(f"  Plot: {args.plot_video}")
    print(f"    @ {plot_fps:.1f} fps, {plot_frames} frames")
    print(f"  Layout: {args.layout}")
    print(f"  Output frames: {total_frames}")

    # Calculate output dimensions
    output_width = args.output_width
    if args.output_height:
        output_height = args.output_height
    else:
        # Auto-calculate based on layout
        if args.layout == "side_by_side":
            sim_aspect = sim_width / sim_height
            output_height = int((output_width * args.sim_weight) / sim_aspect)
        elif args.layout == "stacked":
            sim_aspect = sim_width / sim_height
            sim_h = int(output_width / sim_aspect)
            output_height = int(sim_h * 1.6)
        else:  # pip
            sim_aspect = sim_width / sim_height
            output_height = int(output_width / sim_aspect)

    print(f"  Output: {output_width}x{output_height} @ {fps:.1f} fps")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        args.output_video, fourcc, fps,
        (output_width, output_height)
    )

    if not out.isOpened():
        print(f"Error: Cannot create output video: {args.output_video}")
        sys.exit(1)

    bg_color = tuple(args.background_color)

    # Process frames
    print("Processing frames...")
    for frame_idx in range(total_frames):
        # Read simulation frame
        ret_sim, sim_frame = sim_cap.read()
        if not ret_sim:
            break

        # Calculate corresponding plot frame
        plot_frame_idx = int(frame_idx * plot_fps / sim_fps)
        plot_cap.set(cv2.CAP_PROP_POS_FRAMES, plot_frame_idx)
        ret_plot, plot_frame = plot_cap.read()

        if not ret_plot:
            # Use last available plot frame
            plot_cap.set(cv2.CAP_PROP_POS_FRAMES, plot_frames - 1)
            _, plot_frame = plot_cap.read()

        # Combine based on layout
        if args.layout == "side_by_side":
            output_frame = combine_side_by_side(
                sim_frame, plot_frame, output_width, args.sim_weight, bg_color
            )
        elif args.layout == "stacked":
            output_frame = combine_stacked(
                sim_frame, plot_frame, output_width, bg_color
            )
        else:  # pip
            output_frame = combine_pip(
                sim_frame, plot_frame, output_width, output_height,
                args.pip_position, args.pip_scale, args.pip_margin
            )

        out.write(output_frame)

        if (frame_idx + 1) % 100 == 0:
            progress = 100 * (frame_idx + 1) / total_frames
            print(f"  {frame_idx + 1}/{total_frames} ({progress:.1f}%)")

    # Cleanup
    sim_cap.release()
    plot_cap.release()
    out.release()

    print(f"\nDone! Combined video saved to: {args.output_video}")


if __name__ == "__main__":
    main()
