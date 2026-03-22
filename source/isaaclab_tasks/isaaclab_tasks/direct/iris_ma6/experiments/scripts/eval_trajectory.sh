#!/usr/bin/env bash
# Trajectory recording: record agent/target XYZ paths for visualization.
#
# Usage:
#   ./eval_trajectory.sh <experiment> <checkpoint> [num_envs] [trajectory_envs]
#
# Examples:
#   ./eval_trajectory.sh a3_delay_only /path/to/best_agent.pt
#   ./eval_trajectory.sh a5_cbf_default /path/to/best_agent.pt 4096 16
#   ./eval_trajectory.sh baseline_greedy ""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
EVALUATE="$SCRIPT_DIR/../evaluate.py"

EXPERIMENT="${1:?Usage: $0 <experiment> <checkpoint> [num_envs] [trajectory_envs]}"
CHECKPOINT="${2:-}"
NUM_ENVS="${3:-4096}"
TRAJ_ENVS="${4:-8}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="traj_${EXPERIMENT}_${TIMESTAMP}.json"

CMD=(
    "$ISAACLAB_ROOT/isaaclab.sh" -p "$EVALUATE"
    --experiment "$EXPERIMENT"
    --num_envs "$NUM_ENVS"
    --num_episodes 1
    --headless
    --no-timeseries
    --record-trajectory
    --trajectory-envs "$TRAJ_ENVS"
    --output "$OUTPUT"
)

if [[ -n "$CHECKPOINT" ]]; then
    CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "============================================================"
echo "Trajectory Recording — iris_ma6"
echo "============================================================"
echo "  Experiment:     $EXPERIMENT"
echo "  Checkpoint:     ${CHECKPOINT:-greedy (no checkpoint)}"
echo "  Num envs:       $NUM_ENVS"
echo "  Trajectory envs: $TRAJ_ENVS"
echo "  Output:         $OUTPUT"
echo "============================================================"

"${CMD[@]}"
