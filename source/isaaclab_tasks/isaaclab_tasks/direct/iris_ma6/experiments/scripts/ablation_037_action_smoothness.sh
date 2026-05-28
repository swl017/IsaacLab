#!/usr/bin/env bash
# ablation_037_action_smoothness.sh
#
# Three sequential warm-start fine-tune ablations to investigate the
# action-smoothness / attitude-destabilization issue surfaced in ticket 037
# Phase 1 diagnostics. All three share:
#   - warm-start from t034 agent_400000.pt (matched architecture: no priv obs)
#   - curriculum auto-pinned to step 400000 via train_mappo_rnn_hydra.py
#   - 60 k step fine-tune (≈ 2.5 h each, ≈ 7.5 h total)
#   - distinct experiment_name per ablation (no run-dir collisions)
#
# Hydra overrides (struct-mode-safe for scalars; list override tested for
# action_weight). If a list override fails on a given Hydra/OmegaConf version,
# fall back to in-place cfg edit before running ablation B.
#
# Usage:
#   ./ablation_037_action_smoothness.sh             # run all three
#   ./ablation_037_action_smoothness.sh A           # run only A
#   ./ablation_037_action_smoothness.sh B C         # run B then C
#
# Pre-flight checks the script enforces:
#   - conda env `env_isaaclab` activated (per memory feedback)
#   - source checkpoint exists
#   - working tree is clean enough that experiment_name → git hash is meaningful
#     (warns, doesn't block)
#
# Stdout/stderr captured per-ablation in /tmp/ablation_<name>.log.
# TensorBoard logs land in logs/skrl/iris_ma6/<timestamp>_<experiment_name>/.

# NB: `-u` (nounset) is intentionally omitted — Isaac Sim's setup_conda_env.sh
# references unbound vars (e.g. ZSH_VERSION) and would abort the script before
# the first ablation. `-e` + pipefail are kept.
set -eo pipefail

# ---------------------------------------------------------------------------
# Config (edit values here if needed)
# ---------------------------------------------------------------------------

REPO_ROOT="/home/usrg/IsaacPX4/IsaacLab"
CKPT="$REPO_ROOT/logs/skrl/iris_ma6/2026-05-18_11-45-51_mappo_rnn_torch_034_2026-05-18_11-45-45_ticket034_per_env_jitter/checkpoints/agent_400000.pt"
TASK="Isaac-Iris-MA6-Direct-Test-v0"
TIMESTEPS=80000
TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"

# Common CLI overrides applied to ALL ablations.
# Hydra struct mode rejects env.* path overrides in this repo, so train script
# exposes explicit CLI flags that mutate env_cfg attributes after cfg load.
# - Disable Phase 1 priv obs / axis independence so the env shape matches t034
#   (architecture-match for the value-net checkpoint load).
# - Enable the asymmetric z envelope (the new feature under test, ticket 039).
COMMON_FLAGS=(
    --disable_phase1_priv_obs
    --disable_prev_action_obs
    --enable_asymmetric_z
    --timesteps "$TIMESTEPS"
)

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

# Activate conda env (per memory: numpy ABI mismatch without it).
# Use `command -v conda` to detect if not already initialized.
if ! command -v conda >/dev/null 2>&1 || [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "env_isaaclab" ]; then
    # shellcheck disable=SC1091
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
fi

cd "$REPO_ROOT"

if [ ! -f "$CKPT" ]; then
    echo "[ABLATION][FATAL] Checkpoint not found: $CKPT" >&2
    exit 1
fi

# Use a short hash of the implementation commit for experiment-name tagging.
HASH=$(git rev-parse --short=10 HEAD)
DIRTY=$(git status --porcelain | wc -l)
if [ "$DIRTY" -gt 0 ]; then
    echo "[ABLATION][WARN] Working tree has $DIRTY uncommitted changes — experiment_name hash ($HASH) may not reflect the actual code state. Continue? (y/N)"
    read -r -n 1 ans
    echo
    [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
fi

echo "[ABLATION] HEAD=$HASH  timesteps=$TIMESTEPS  task=$TASK"
echo "[ABLATION] Source checkpoint: $CKPT"
echo

# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

run_ablation() {
    local name="$1"
    shift
    local exp_name="${HASH}_ablation_${name}"
    local log="/tmp/ablation_${name}.log"

    echo "=================================================================="
    echo "  Ablation '$name'"
    echo "  Experiment name: $exp_name"
    echo "  Log: $log"
    echo "=================================================================="

    local start=$SECONDS
    ./isaaclab.sh -p "$TRAIN_SCRIPT" \
        --checkpoint "$CKPT" \
        --task "$TASK" \
        --headless \
        --experiment_name "$exp_name" \
        "${COMMON_FLAGS[@]}" \
        "$@" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local elapsed=$((SECONDS - start))
    local hours=$((elapsed / 3600))
    local mins=$(((elapsed % 3600) / 60))

    if [ "$rc" -eq 0 ]; then
        echo "[ABLATION] '$name' completed in ${hours}h ${mins}m"
    else
        echo "[ABLATION][FAIL] '$name' exited $rc after ${hours}h ${mins}m"
        echo "[ABLATION][FAIL] Log: $log"
        exit "$rc"
    fi
    echo
}

# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

ablation_A() {
    # A — asymmetric_z + current cfg (working-tree action_weight, default bbox)
    # Tests asymmetric-z envelope in isolation against the warm-started t034
    # actor. Baseline for the other ablations.
    run_ablation "A_asymz_only"
}

ablation_B() {
    # B — asymmetric_z + action_weight tuning.
    # Lifts vx/vy/vz to 5× so the smoothness penalty applies uniformly across
    # translation. Tests whether action_delta penalty alone collapses the jerk.
    run_ablation "B_asymz_action_w" \
        --action_weight "5,5,5,1,1,1,1"
}

ablation_C() {
    # C — asymmetric_z + bbox reward × 0.5.
    # Halves the bbox_center and bbox_size scales (60 → 30 each). Tests
    # whether weakening the tracking reward lets the policy choose smoother
    # actions at the cost of looser tracking.
    # Reward delta is larger here than in A/B; drop Adam momentum so the
    # optimizer doesn't pull toward the old gradient direction.
    run_ablation "C_asymz_bbox_half" \
        --bbox_center_scale 30.0 \
        --bbox_size_scale 30.0 \
        --skip_load_optimizer
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if [ $# -eq 0 ]; then
    SELECTED=("A" "B" "C")
else
    SELECTED=("$@")
fi

for x in "${SELECTED[@]}"; do
    case "$x" in
        A) ablation_A ;;
        B) ablation_B ;;
        C) ablation_C ;;
        *)
            echo "[ABLATION][FATAL] Unknown ablation: '$x' (valid: A B C)" >&2
            exit 2
            ;;
    esac
done

echo "[ABLATION] All requested ablations complete."
echo "[ABLATION] Compare TensorBoard scalars across:"
for x in "${SELECTED[@]}"; do
    echo "  logs/skrl/iris_ma6/<timestamp>_${HASH}_ablation_*"
done
