#!/bin/bash
# Orchestrate all evaluation runs for iris_ma6 policy.
#
# Usage:
#   ./eval_all.sh <experiment> <checkpoint_path> [num_envs_timeseries] [num_envs_trajectory] [seed]
#
# Example:
#   ./eval_all.sh a1_with_aoi logs/skrl/iris_ma6/.../checkpoints/best_agent.pt 1024 64 42

set -euo pipefail

EXPERIMENT="${1:?Usage: eval_all.sh <experiment> <checkpoint_path> [num_envs_ts] [num_envs_traj] [seed]}"
CHECKPOINT="${2:?Usage: eval_all.sh <experiment> <checkpoint_path>}"
NUM_ENVS_TS="${3:-1024}"
NUM_ENVS_TRAJ="${4:-64}"
SEED="${5:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$(dirname "$SCRIPT_DIR")/evaluate.py"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "iris_ma6 Evaluation Pipeline"
echo "============================================================"
echo "Experiment:    $EXPERIMENT"
echo "Checkpoint:    $CHECKPOINT"
echo "Timeseries:    $NUM_ENVS_TS envs"
echo "Trajectory:    $NUM_ENVS_TRAJ envs (8 sampled)"
echo "Seed:          $SEED"
echo "Output dir:    $OUTPUT_DIR"
echo "============================================================"

# 1. Timeseries evaluation
echo ""
echo "[1/3] Running timeseries evaluation..."
"$ISAACLAB_ROOT/isaaclab.sh" -p "$EVAL_SCRIPT" \
    --experiment "$EXPERIMENT" \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS_TS" \
    --num_episodes 1 \
    --seed "$SEED" \
    --headless \
    --no-record-trajectory \
    --output "$OUTPUT_DIR/eval_${EXPERIMENT}.json"

# 2. Trajectory — linear target
echo ""
echo "[2/3] Running trajectory evaluation (linear target)..."
"$ISAACLAB_ROOT/isaaclab.sh" -p "$EVAL_SCRIPT" \
    --experiment "$EXPERIMENT" \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS_TRAJ" \
    --num_episodes 1 \
    --seed "$SEED" \
    --headless \
    --target-trajectory-mode linear \
    --no-timeseries \
    --trajectory-envs 8 \
    --output "$OUTPUT_DIR/eval_traj_linear.json"

# 3. Trajectory — circular target
echo ""
echo "[3/3] Running trajectory evaluation (circular target)..."
"$ISAACLAB_ROOT/isaaclab.sh" -p "$EVAL_SCRIPT" \
    --experiment "$EXPERIMENT" \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS_TRAJ" \
    --num_episodes 1 \
    --seed "$SEED" \
    --headless \
    --target-trajectory-mode circular \
    --no-timeseries \
    --trajectory-envs 8 \
    --output "$OUTPUT_DIR/eval_traj_circular.json"

echo ""
echo "============================================================"
echo "All evaluations complete. Outputs in: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "To generate plots:"
echo "  python $SCRIPT_DIR/plot_timeseries.py --results-dir $OUTPUT_DIR"
echo "  python $SCRIPT_DIR/plot_trajectory.py --results-dir $OUTPUT_DIR"
