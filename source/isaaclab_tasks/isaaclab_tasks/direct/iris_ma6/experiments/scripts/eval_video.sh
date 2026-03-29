#!/usr/bin/env bash
# Video recording: record evaluation video with smooth camera.
#
# Usage:
#   ./eval_video.sh <experiment> <checkpoint> [num_envs] [camera_mode]
#
# Examples:
#   ./eval_video.sh a1_with_aoi /path/to/best_agent.pt
#   ./eval_video.sh a1_with_aoi /path/to/best_agent.pt 64 chase
#   ./eval_video.sh baseline_greedy ""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
EVALUATE="$SCRIPT_DIR/../evaluate.py"

EXPERIMENT="${1:?Usage: $0 <experiment> <checkpoint> [num_envs] [camera_mode]}"
CHECKPOINT="${2:-}"
NUM_ENVS="${3:-64}"
CAMERA_MODE="${4:-formation}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_VIDEO="video_${EXPERIMENT}_${CAMERA_MODE}_${TIMESTAMP}.mp4"
OUTPUT_JSON="eval_${EXPERIMENT}_${TIMESTAMP}.json"

CMD=(
    "$ISAACLAB_ROOT/isaaclab.sh" -p "$EVALUATE"
    --experiment "$EXPERIMENT"
    --num_envs "$NUM_ENVS"
    --num_episodes 1
    --headless
    --no-timeseries
    --record-video "$OUTPUT_VIDEO"
    --camera-mode "$CAMERA_MODE"
    --output "$OUTPUT_JSON"
    --enable_cameras
)

if [[ -n "$CHECKPOINT" ]]; then
    CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "============================================================"
echo "Video Recording — iris_ma6"
echo "============================================================"
echo "  Experiment:   $EXPERIMENT"
echo "  Checkpoint:   ${CHECKPOINT:-greedy (no checkpoint)}"
echo "  Num envs:     $NUM_ENVS"
echo "  Camera mode:  $CAMERA_MODE"
echo "  Output video: $OUTPUT_VIDEO"
echo "  Output JSON:  $OUTPUT_JSON"
echo "============================================================"

"${CMD[@]}"
