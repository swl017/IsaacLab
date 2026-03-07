#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Full pipeline for generating IROS paper demonstration videos.
#
# This script:
# 1. Generates an animated metrics plot video
# 2. Optionally overlays metrics on simulation video
# 3. Optionally combines simulation + plot videos side-by-side
#
# Usage:
#   ./generate_demo_video.sh --sim-video raw_demo.mp4 --json metrics.json --output final.mp4
#
# Options:
#   --sim-video PATH      Input simulation video
#   --json PATH           Input metrics JSON file
#   --output PATH         Output video file
#   --style STYLE         Color style: paper, light, paper (default: paper)
#   --layout LAYOUT       Layout: overlay, plot_only, side_by_side, stacked (default: side_by_side)
#   --metrics M1 M2 ...   Metrics to show (default: viewing_angle triangulation_rmse visibility)
#   --overlay M1 M2       Overlay metrics on same axes (can be used multiple times)
#   --preset PRESET       Use preset configuration: default, delay (default: none)

set -e

# Default values
STYLE="paper"
LAYOUT="side_by_side"
METRICS="viewing_angle triangulation_rmse visibility"
FPS=30
OVERLAY_ARGS=""
PRESET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sim-video)
            SIM_VIDEO="$2"
            shift 2
            ;;
        --json)
            JSON_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --style)
            STYLE="$2"
            shift 2
            ;;
        --layout)
            LAYOUT="$2"
            shift 2
            ;;
        --metrics)
            shift
            METRICS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                METRICS="$METRICS $1"
                shift
            done
            METRICS=$(echo $METRICS | xargs)  # Trim whitespace
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --overlay)
            shift
            OVERLAY_METRICS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                OVERLAY_METRICS="$OVERLAY_METRICS $1"
                shift
            done
            OVERLAY_METRICS=$(echo $OVERLAY_METRICS | xargs)  # Trim whitespace
            OVERLAY_ARGS="$OVERLAY_ARGS --overlay $OVERLAY_METRICS"
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --sim-video VIDEO --json JSON --output OUTPUT [options]"
            echo ""
            echo "Required:"
            echo "  --sim-video PATH      Input simulation video"
            echo "  --json PATH           Input metrics JSON file"
            echo "  --output PATH         Output video file"
            echo ""
            echo "Options:"
            echo "  --style STYLE         Color style: paper, light, paper (default: paper)"
            echo "  --layout LAYOUT       Layout: overlay, plot_only, side_by_side, stacked, pip (default: side_by_side)"
            echo "  --metrics M1 M2 ...   Metrics to show (default: viewing_angle triangulation_rmse visibility)"
            echo "  --overlay M1 M2       Overlay metrics on same axes (can be used multiple times)"
            echo "  --preset PRESET       Use preset: default, delay (overrides metrics and overlay)"
            echo "  --fps FPS             Output FPS (default: 30)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Presets:"
            echo "  default   Standard metrics: viewing_angle, triangulation_rmse, visibility"
            echo "  delay     Delay analysis: viewing_angle, rmse, and overlaid ego_aoi+other_aoi"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$JSON_FILE" ]]; then
    echo "Error: --json is required"
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    echo "Error: --output is required"
    exit 1
fi

# Apply preset configurations (overrides metrics and overlay)
case $PRESET in
    delay)
        METRICS="viewing_angle triangulation_rmse ego_aoi other_aoi"
        OVERLAY_ARGS="--overlay ego_aoi other_aoi"
        echo "Using preset: delay"
        ;;
    default)
        METRICS="viewing_angle triangulation_rmse visibility"
        OVERLAY_ARGS=""
        echo "Using preset: default"
        ;;
    "")
        # No preset, use provided values
        ;;
    *)
        echo "Error: Unknown preset: $PRESET"
        echo "Valid presets: default, delay"
        exit 1
        ;;
esac

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create temp directory for intermediate files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "=============================================="
echo "IROS Demo Video Generation Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Simulation video: ${SIM_VIDEO:-"(none)"}"
echo "  Metrics JSON:     $JSON_FILE"
echo "  Output:           $OUTPUT"
echo "  Style:            $STYLE"
echo "  Layout:           $LAYOUT"
echo "  Metrics:          $METRICS"
if [[ -n "$OVERLAY_ARGS" ]]; then
echo "  Overlay:          $OVERLAY_ARGS"
fi
echo "  FPS:              $FPS"
echo ""

# Handle different layouts
case $LAYOUT in
    plot_only)
        # Just generate the plot video
        echo "Generating plot-only video..."
        python "$SCRIPT_DIR/plot_video.py" \
            --input-json "$JSON_FILE" \
            --output-video "$OUTPUT" \
            --metrics $METRICS \
            --style "$STYLE" \
            --fps $FPS \
            $OVERLAY_ARGS
        ;;

    overlay)
        # Overlay metrics on simulation video
        if [[ -z "$SIM_VIDEO" ]]; then
            echo "Error: --sim-video required for overlay layout"
            exit 1
        fi
        echo "Generating overlay video..."
        python "$SCRIPT_DIR/video_overlay.py" \
            --input-video "$SIM_VIDEO" \
            --input-json "$JSON_FILE" \
            --output-video "$OUTPUT" \
            --metrics $METRICS \
            --panel-position top_right
        ;;

    side_by_side)
        # Generate plot video and combine side-by-side
        if [[ -z "$SIM_VIDEO" ]]; then
            echo "Error: --sim-video required for side_by_side layout"
            exit 1
        fi

        PLOT_VIDEO="$TEMP_DIR/plot_video.mp4"

        echo "Step 1/2: Generating plot video..."
        python "$SCRIPT_DIR/plot_video.py" \
            --input-json "$JSON_FILE" \
            --output-video "$PLOT_VIDEO" \
            --metrics $METRICS \
            --style "$STYLE" \
            --fps $FPS \
            --sync-video "$SIM_VIDEO" \
            $OVERLAY_ARGS

        echo ""
        echo "Step 2/2: Combining videos side-by-side..."
        python "$SCRIPT_DIR/combine_videos.py" \
            --simulation-video "$SIM_VIDEO" \
            --plot-video "$PLOT_VIDEO" \
            --output-video "$OUTPUT" \
            --layout side_by_side \
            --sim-weight 0.55
        ;;

    stacked)
        # Generate plot video and stack vertically
        if [[ -z "$SIM_VIDEO" ]]; then
            echo "Error: --sim-video required for stacked layout"
            exit 1
        fi

        PLOT_VIDEO="$TEMP_DIR/plot_video.mp4"

        echo "Step 1/2: Generating plot video..."
        python "$SCRIPT_DIR/plot_video.py" \
            --input-json "$JSON_FILE" \
            --output-video "$PLOT_VIDEO" \
            --metrics $METRICS \
            --style "$STYLE" \
            --fps $FPS \
            --sync-video "$SIM_VIDEO" \
            $OVERLAY_ARGS

        echo ""
        echo "Step 2/2: Combining videos stacked..."
        python "$SCRIPT_DIR/combine_videos.py" \
            --simulation-video "$SIM_VIDEO" \
            --plot-video "$PLOT_VIDEO" \
            --output-video "$OUTPUT" \
            --layout stacked
        ;;

    pip)
        # Picture-in-picture: plot video overlaid on simulation
        if [[ -z "$SIM_VIDEO" ]]; then
            echo "Error: --sim-video required for pip layout"
            exit 1
        fi

        PLOT_VIDEO="$TEMP_DIR/plot_video.mp4"

        echo "Step 1/2: Generating plot video..."
        python "$SCRIPT_DIR/plot_video.py" \
            --input-json "$JSON_FILE" \
            --output-video "$PLOT_VIDEO" \
            --metrics $METRICS \
            --style "$STYLE" \
            --fps $FPS \
            --sync-video "$SIM_VIDEO" \
            --resolution 960 540 \
            $OVERLAY_ARGS

        echo ""
        echo "Step 2/2: Combining as picture-in-picture..."
        python "$SCRIPT_DIR/combine_videos.py" \
            --simulation-video "$SIM_VIDEO" \
            --plot-video "$PLOT_VIDEO" \
            --output-video "$OUTPUT" \
            --layout pip \
            --pip-position bottom_right \
            --pip-scale 0.35
        ;;

    *)
        echo "Error: Unknown layout: $LAYOUT"
        echo "Valid layouts: overlay, plot_only, side_by_side, stacked, pip"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Done! Output saved to: $OUTPUT"
echo "=============================================="
