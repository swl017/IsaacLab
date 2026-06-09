#!/usr/bin/env bash
# Ticket 047 — dual-step checkpoint evaluation (IROS timeseries style).
#
# Evaluates the t047 curriculum-cold run's checkpoints in a "dual-step" scheme:
# each snapshot is evaluated at the curriculum step it had just reached (1k before
# the next phase boundary), and the final 400k policy is swept across those same
# difficulties. The two sets feed two IROS-style 3x2 timeseries figures:
#   Figure A (snapshots): 40k@39k, 80k@79k, 120k@119k, 160k@159k, 200k@199k, 400k@400k
#   Figure B (final sweep): 400k@{39k,79k,119k,159k,199k,399k}
#
# Step rationale — each eval step is 1k before a curriculum phase boundary, so each
# snapshot is scored at the hardest regime it fully experienced:
#   39k  -> just before moving_target onset (40k)
#   79k  -> just before moving_target end / task_level_3 (80k)
#   119k -> just before noise end / fixed_delay onset (120k)
#   159k -> just before random_delay end / dropout onset (160k)
#   199k -> just before burst_dropout / zoom_tau end (200k)
#   399k -> full difficulty (terminal)
#
# Env config: validation_task_geom_treatment encodes the t047 deploy env
# (closer spawn + 2D target enable_z_motion=False + t045 envelope + slew clip +
# prev-action obs). t047's reward-scale/entropy overrides do not affect the
# deterministic (policy-mean) eval rollout, so this experiment is the correct env.
#
# Usage:
#   ./run_t047_dualstep_eval.bash smoke   # run ONLY config #1 (dim/obs smoke test)
#   ./run_t047_dualstep_eval.bash         # run all 12 (launch in tmux; ~1h)

# NOTE: no `set -u` — the IsaacLab conda activation hook sources
# _isaac_sim/setup_conda_env.sh, which references unbound $ZSH_VERSION and would
# abort under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
EVALUATE="$SCRIPT_DIR/../evaluate.py"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/t047_dualstep"

CKPT_DIR="$ISAACLAB_ROOT/logs/skrl/iris_ma6/2026-06-04_00-36-19_mappo_rnn_torch_ticket047_D_curriculum_cold/checkpoints"
EXPERIMENT="validation_task_geom_treatment"
NUM_ENVS=1024
NUM_EPISODES=1
# Eval seeds. Each seed redraws the 1024 env initial conditions, so the spread
# across seeds is the reproducibility band (does a snapshot-vs-final gap survive
# a different IC draw?). NOTE: only ONE training seed exists for t047, so this is
# IC-sampling variance, not training-seed variance — see the experiment doc.
SEEDS=(42 123 2024)

# Activate the IsaacLab conda env (numpy ABI mismatch otherwise).
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env_isaaclab
fi

mkdir -p "$OUTPUT_DIR"

# Config table: "checkpoint_step  eval_step  output_basename"
# Figure A = snapshot rows + final_at_400k; Figure B = final_at_* rows.
CONFIGS=(
    # --- Figure A: snapshots at their matched difficulty ---
    "40000   39000  snap_040k_at_039k"
    "80000   79000  snap_080k_at_079k"
    "120000  119000 snap_120k_at_119k"
    "160000  159000 snap_160k_at_159k"
    "200000  199000 snap_200k_at_199k"
    "400000  400000 final_at_400k"
    # --- Figure B: final policy swept across difficulties ---
    "400000  39000  final_at_039k"
    "400000  79000  final_at_079k"
    "400000  119000 final_at_119k"
    "400000  159000 final_at_159k"
    "400000  199000 final_at_199k"
    "400000  399000 final_at_399k"
)

run_one() {
    local ckpt_step="$1" eval_step="$2" name="$3" seed="$4"
    local ckpt="$CKPT_DIR/agent_${ckpt_step}.pt"
    local out="$OUTPUT_DIR/${name}_seed${seed}.json"

    if [[ ! -f "$ckpt" ]]; then
        echo "[ERROR] Missing checkpoint: $ckpt" >&2
        return 1
    fi

    echo "============================================================"
    echo "  ${name} seed=${seed}: agent_${ckpt_step}.pt @ curriculum step ${eval_step}"
    echo "  -> $out"
    echo "============================================================"

    "$ISAACLAB_ROOT/isaaclab.sh" -p "$EVALUATE" \
        --experiment "$EXPERIMENT" \
        --checkpoint "$ckpt" \
        --step "$eval_step" \
        --seed "$seed" \
        --num_envs "$NUM_ENVS" \
        --num_episodes "$NUM_EPISODES" \
        --headless \
        --no-record-trajectory \
        --no-record-action-trace \
        --output "$out"
}

MODE="${1:-all}"

if [[ "$MODE" == "smoke" ]]; then
    echo "[SMOKE] Running only config #1 (seed ${SEEDS[0]}) to verify checkpoint/env dim + tracker quantiles."
    read -r cs es nm <<< "${CONFIGS[0]}"
    run_one "$cs" "$es" "$nm" "${SEEDS[0]}"
    echo "[SMOKE] OK — inspect $OUTPUT_DIR/${nm}_seed${SEEDS[0]}.json then run without 'smoke'."
    exit 0
fi

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
echo "[ALL] Running ${#CONFIGS[@]} configs x ${#SEEDS[@]} seeds = ${TOTAL} evaluations into $OUTPUT_DIR"
n=0
for cfg in "${CONFIGS[@]}"; do
    read -r cs es nm <<< "$cfg"
    for seed in "${SEEDS[@]}"; do
        n=$((n + 1))
        echo "[ALL] ($n/$TOTAL)"
        run_one "$cs" "$es" "$nm" "$seed"
    done
done
echo "[ALL] Done. ${TOTAL} JSONs in $OUTPUT_DIR"
echo "      Plot with: python $SCRIPT_DIR/plot_t047_dualstep.py"
