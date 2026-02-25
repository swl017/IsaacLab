#!/bin/bash
# IROS 2026 Ablation Experiment Runner
# Runs all MUST experiments (A1-A4 + baselines) x 3 seeds sequentially.
#
# Usage:
#   bash multirun_ablations.bash              # Run all
#   bash multirun_ablations.bash --dry_run    # Print commands only
#   bash multirun_ablations.bash --group A4   # Run only A4 experiments

set -e

RUN_CMD="python /home/usrg/IsaacPX4/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/experiments/run_experiment.py"
NUM_ENVS=4096
# SEEDS=(42 123 456)
SEEDS=(42)

# IROS 2026 MUST experiments
EXPERIMENTS=(
    # A1: Delay modeling
    # "a1_no_delay"
    # "a1_stochastic_delay" # Full delay modeling in a1_with_aoi is sufficient
    # "a1_with_aoi"
    "a1_without_aoi"
    # "a1_curriculum_no_delay"
    # A2: RNN vs MLP
    # "a2_rnn" # Full RNN results already in a1_with_aoi 
    # "a2_mlp"
    # A3: Dual-path
    # "a3_clean_reward" # Duplicate of a1_with_aoi
    # "a3_noisy_reward" 
    # A4: Covariance reward
    # "a4_analytical" # Duplicate of a1_with_aoi (analytical is default reward_mode)
    # "a4_angular_only"
    # "a4_heuristic" # Skip for now
    # "a4_image_only" # Skip for now
    # Baselines
    # "baseline_gavin2024"
)

# Parse args
DRY_RUN=false
GROUP_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run) DRY_RUN=true; shift ;;
        --group)   GROUP_FILTER="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Filter by group prefix if requested
if [[ -n "$GROUP_FILTER" ]]; then
    FILTERED=()
    for exp in "${EXPERIMENTS[@]}"; do
        prefix=$(echo "$exp" | cut -d'_' -f1)
        if [[ "$prefix" == "${GROUP_FILTER,,}" ]] || [[ "$exp" == *"${GROUP_FILTER,,}"* ]]; then
            FILTERED+=("$exp")
        fi
    done
    EXPERIMENTS=("${FILTERED[@]}")
fi

TOTAL=$((${#EXPERIMENTS[@]} * ${#SEEDS[@]}))
echo "============================================================"
echo "IROS 2026 Ablation Experiments"
echo "Experiments: ${#EXPERIMENTS[@]}, Seeds: ${#SEEDS[@]}, Total runs: ${TOTAL}"
echo "Num envs: ${NUM_ENVS}"
if $DRY_RUN; then echo "[DRY RUN - no training will be executed]"; fi
echo "============================================================"

RUN_IDX=0
FAILED=()
for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        echo ""
        echo "[${RUN_IDX}/${TOTAL}] Experiment: ${exp}  Seed: ${seed}"
        echo "------------------------------------------------------------"

        CMD="$RUN_CMD --experiment $exp --seed $seed --num_envs $NUM_ENVS"

        if $DRY_RUN; then
            echo "  CMD: $CMD"
        else
            if $CMD; then
                echo "[${RUN_IDX}/${TOTAL}] DONE: ${exp} seed=${seed}"
            else
                echo "[${RUN_IDX}/${TOTAL}] FAILED: ${exp} seed=${seed}"
                FAILED+=("${exp}_seed${seed}")
            fi
            sleep 5
        fi
    done
done

echo ""
echo "============================================================"
echo "All runs completed. ${#FAILED[@]} failures."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed runs:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
echo "============================================================"
