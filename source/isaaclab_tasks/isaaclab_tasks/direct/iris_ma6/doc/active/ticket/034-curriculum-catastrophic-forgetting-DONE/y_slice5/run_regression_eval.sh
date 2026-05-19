#!/usr/bin/env bash
# Slice 5 step 5.5 — replay ticket-034's 7-cell regression eval against
# the NEW 400k checkpoint produced by Slice 5's full training run.
# Mirrors r_research/run_regression_eval.sh but the CKPT_ROOT must point
# at the new training run's checkpoints/ directory.
#
# Usage:
#   ./run_regression_eval.sh <PATH/TO/NEW/checkpoints>
#
# Outputs 7 JSON files into this directory, mirroring r_research/.

set -eo pipefail
shopt -s inherit_errexit 2>/dev/null || true

source /home/usrg/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <PATH/TO/NEW/checkpoints>"
    exit 2
fi
CKPT_ROOT="$1"
if [[ ! -f "$CKPT_ROOT/agent_400000.pt" ]]; then
    echo "ERROR: $CKPT_ROOT/agent_400000.pt not found"
    exit 2
fi

ROOT=/home/usrg/IsaacPX4/IsaacLab
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL=$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/evaluate.py

cd "$ROOT"

run_cell () {
    local step="$1"
    local ckpt_step="$2"
    local tag="step${step}_ckpt${ckpt_step}_new"
    local out="$OUT_DIR/eval_${tag}.json"
    local log="$OUT_DIR/eval_${tag}.log"
    local ckpt="$CKPT_ROOT/agent_$(printf '%d' "${ckpt_step}000").pt"

    if [[ ! -f "$ckpt" ]]; then
        echo "SKIP $tag — $ckpt missing"
        return 0
    fi

    echo "==== $tag ==== ckpt=$ckpt step=$((step*1000))"
    set +e
    ./isaaclab.sh -p "$EVAL" \
        --experiment a1_with_aoi \
        --task Isaac-Iris-MA6-Direct-Test-v0 \
        --checkpoint "$ckpt" \
        --step $((step*1000)) \
        --num_envs 1024 --num_episodes 1 \
        --no-record-trajectory --no-record-action-trace --no-timeseries \
        --headless \
        --output "$out" \
        > "$log" 2>&1
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "FAILED ($rc): tail of $log:"
        tail -n 40 "$log"
        exit "$rc"
    fi
    if [[ ! -s "$out" ]]; then
        echo "FAILED: no JSON written at $out:"
        tail -n 40 "$log"
        exit 1
    fi
    echo "OK   $out  ($(wc -c < "$out") bytes)"
}

# Target ckpt (400k) at each eval step
run_cell 39  400
run_cell 99  400
run_cell 199 400
run_cell 399 400

# Reference ckpts (closest pre-boundary intermediate)
run_cell 39  40
run_cell 99  80
run_cell 199 200

echo "==== All cells complete ===="
ls -la "$OUT_DIR"/eval_*_new.json 2>/dev/null
