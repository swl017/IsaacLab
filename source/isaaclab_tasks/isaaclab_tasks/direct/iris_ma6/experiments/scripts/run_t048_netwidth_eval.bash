#!/usr/bin/env bash
# Ticket 048 — network-width sweep evaluation (overlay at final step 200k).
#
# Evaluates the three net-width runs' final (200k) checkpoints on the IROS
# timeseries metrics and overlays them. Each run differs ONLY in agent width
# (hidden_size = gru_hidden_size = 64 / 128 / 256); the env is identical
# (post-046/047 defaults). The matching `validation_net_width_*` experiment sets
# the network width so the checkpoint loads AND inherits the right env.
#
# Curriculum is pinned to step 200000 (each run's final training step) via
# --step, so every width is scored at the difficulty it actually ended on.
#
# Usage:
#   ./run_t048_netwidth_eval.bash smoke   # baseline only (load/dim check)
#   ./run_t048_netwidth_eval.bash         # all three

# No `set -u` — IsaacLab conda hook references unbound $ZSH_VERSION.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
EVALUATE="$SCRIPT_DIR/../evaluate.py"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/t048_netwidth"
LOGS="$ISAACLAB_ROOT/logs/skrl/iris_ma6"
NUM_ENVS=1024
STEP=200000

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env_isaaclab
fi
mkdir -p "$OUTPUT_DIR"

# "name  experiment  run_dir"
CONFIGS=(
    "baseline validation_net_width_baseline 2026-06-09_23-48-47_mappo_rnn_torch_ticket048_net_width_baseline"
    "mid      validation_net_width_mid      2026-06-10_14-41-45_mappo_rnn_torch_ticket048_net_width_mid"
    "wide     validation_net_width_wide     2026-06-11_05-17-30_mappo_rnn_torch_ticket048_net_width_wide"
)

run_one() {
    local name="$1" exp="$2" run="$3"
    local ckpt="$LOGS/$run/checkpoints/agent_${STEP}.pt"
    local out="$OUTPUT_DIR/${name}.json"
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERROR] Missing checkpoint: $ckpt" >&2; return 1
    fi
    echo "============================================================"
    echo "  ${name}: ${exp}  @ step ${STEP}"
    echo "  $ckpt"
    echo "  -> $out"
    echo "============================================================"
    "$ISAACLAB_ROOT/isaaclab.sh" -p "$EVALUATE" \
        --experiment "$exp" \
        --checkpoint "$ckpt" \
        --step "$STEP" \
        --num_envs "$NUM_ENVS" \
        --num_episodes 1 \
        --headless \
        --no-record-trajectory --no-record-action-trace \
        --output "$out"
}

MODE="${1:-all}"
if [[ "$MODE" == "smoke" ]]; then
    read -r nm ex rn <<< "${CONFIGS[0]}"
    run_one "$nm" "$ex" "$rn"
    echo "[SMOKE] OK — inspect $OUTPUT_DIR/${nm}.json, then run without 'smoke'."
    exit 0
fi

for cfg in "${CONFIGS[@]}"; do
    read -r nm ex rn <<< "$cfg"
    run_one "$nm" "$ex" "$rn"
done
echo "[ALL] Done. JSONs in $OUTPUT_DIR"
echo "      Plot with: python $SCRIPT_DIR/plot_t048_netwidth.py"
