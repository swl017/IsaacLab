#!/usr/bin/env bash
# Standard evaluation: compute paper metrics for a trained checkpoint.
#
# Usage:
#   ./eval_standard.sh <experiment> <checkpoint> [num_envs] [num_episodes]
#
# Examples:
#   ./eval_standard.sh a3_delay_only /path/to/best_agent.pt
#   ./eval_standard.sh a5_cbf_default /path/to/best_agent.pt 2048 3
#   ./eval_standard.sh baseline_greedy ""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
EVALUATE="$SCRIPT_DIR/../evaluate.py"

EXPERIMENT="${1:?Usage: $0 <experiment> <checkpoint> [num_envs] [num_episodes]}"
CHECKPOINT="${2:-}"
NUM_ENVS="${3:-1024}"
NUM_EPISODES="${4:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="eval_${EXPERIMENT}_${TIMESTAMP}.json"

CMD=(
    "$ISAACLAB_ROOT/isaaclab.sh" -p "$EVALUATE"
    --experiment "$EXPERIMENT"
    --num_envs "$NUM_ENVS"
    --num_episodes "$NUM_EPISODES"
    --headless
    --no-timeseries
    --output "$OUTPUT"
)

if [[ -n "$CHECKPOINT" ]]; then
    CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "============================================================"
echo "Standard Evaluation — iris_ma6"
echo "============================================================"
echo "  Experiment:  $EXPERIMENT"
echo "  Checkpoint:  ${CHECKPOINT:-greedy (no checkpoint)}"
echo "  Num envs:    $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES"
echo "  Output:      $OUTPUT"
echo "============================================================"

"${CMD[@]}"
