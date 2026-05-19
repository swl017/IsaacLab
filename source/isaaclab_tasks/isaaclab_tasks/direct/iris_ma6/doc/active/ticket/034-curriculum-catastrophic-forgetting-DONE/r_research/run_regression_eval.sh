#!/usr/bin/env bash
# R0 — regression measurement for ticket 034.
# Runs 7 evaluate.py cells, writing one JSON per cell into this directory.
#
# Usage: bash run_regression_eval.sh
#
# Output: eval_step<S>k_ckpt<C>k.json (7 files).

# Note: not using `set -u` because _isaac_sim/setup_conda_env.sh references
# ZSH_VERSION without a default. Keeping -e and pipefail for error trapping.
set -eo pipefail
shopt -s inherit_errexit 2>/dev/null || true

# Activate the conda env used for fix2 training (has numpy<2.0).
# The default _isaac_sim/python.sh has numpy 2.2.6, which has an ABI
# mismatch with scipy 1.10.1 and crashes at `import gymnasium`.
source /home/usrg/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

ROOT=/home/usrg/IsaacPX4/IsaacLab
CKPT_ROOT=$ROOT/logs/skrl/iris_ma6/2026-05-15_12-00-44_mappo_rnn_torch_7fd070ee09_mappo_rnn_shared_model_scheduler_param_fix2/checkpoints
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL=$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/evaluate.py

cd "$ROOT"

run_cell () {
    local step="$1"
    local ckpt_step="$2"
    local tag="step${step}_ckpt${ckpt_step}"
    local out="$OUT_DIR/eval_${tag}.json"
    local ckpt="$CKPT_ROOT/agent_$(printf '%d' "${ckpt_step}000").pt"

    local log="$OUT_DIR/eval_${tag}.log"
    echo "==== $tag ==== ckpt=$ckpt step=$((step*1000))  log=$log"
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
        echo "FAILED: no JSON written at $out (tail of log):"
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

echo "==== All 7 cells complete ===="
ls -la "$OUT_DIR"/eval_*.json
