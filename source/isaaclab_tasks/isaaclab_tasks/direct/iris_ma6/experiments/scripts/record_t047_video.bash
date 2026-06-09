#!/usr/bin/env bash
# Record one episode of the t047 deployment policy with all agents + target framed.
#
# Renders the deployment baseline (agent_400000.pt) at full curriculum difficulty
# (--step 400000) under the t047 deploy env (validation_task_geom_treatment), using
# the `isometric` camera preset (follow_centroid + dynamic zoom — keeps both observer
# drones and the target in frame). Intended for eyeballing the real-time / sim-to-sim
# behavior (e.g. jerky velocity commands), not for metrics.
#
# Usage:
#   ./record_t047_video.bash [camera_mode] [out.mp4]
#     camera_mode: isometric (default) | formation | wide | overhead

# No `set -u` — IsaacLab conda hook references unbound $ZSH_VERSION.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
EVALUATE="$SCRIPT_DIR/../evaluate.py"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/t047_dualstep"
CKPT_DIR="$ISAACLAB_ROOT/logs/skrl/iris_ma6/2026-06-04_00-36-19_mappo_rnn_torch_ticket047_D_curriculum_cold/checkpoints"

VIDEO_DIR="$OUTPUT_DIR/video"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env_isaaclab
fi
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Recording t047 deployment episode (scene video)"
echo "  checkpoint: agent_400000.pt @ step 400000 (full difficulty)"
echo "  view:       isometric wide (env_cfg.viewer override, all agents+target)"
echo "  -> $VIDEO_DIR/"
echo "============================================================"

mkdir -p "$VIDEO_DIR"
"$ISAACLAB_ROOT/isaaclab.sh" -p "$EVALUATE" \
    --experiment validation_task_geom_treatment \
    --checkpoint "$CKPT_DIR/agent_400000.pt" \
    --step 400000 \
    --num_envs 4 \
    --num_episodes 1 \
    --headless --enable_cameras \
    --record-video-scene "$VIDEO_DIR" \
    --video-resolution 1280 720 \
    --no-record-trajectory --no-record-action-trace \
    --output "$OUTPUT_DIR/_video_run_discard.json"

echo "Video(s) in: $VIDEO_DIR"
ls -la "$VIDEO_DIR"
