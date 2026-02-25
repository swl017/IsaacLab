#!/bin/bash
# IROS 2026 Ablation Experiment Runner
# Runs all MUST experiments (A1-A4 + baselines) x 3 seeds sequentially.
#
# Usage:
#   bash multirun_ablations.bash                        # Run all sequentially
#   bash multirun_ablations.bash --dry_run              # Print commands only
#   bash multirun_ablations.bash --group A4             # Run only A4 experiments (MA4)
#   bash multirun_ablations.bash --group ma5            # Run only MA5 N-agent experiments
#   bash multirun_ablations.bash --gpu_ids 0,1,2,3      # Run MA5 in parallel, one per GPU
#   bash multirun_ablations.bash --gpu_ids 0,1 --dry_run  # Preview GPU assignment

set -e

ISAACLAB="${ISAACLAB:-/workspace/isaaclab/isaaclab.sh}"
RUN_CMD="$ISAACLAB -p /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/experiments/run_experiment.py"
RUN_CMD_MA5="$ISAACLAB -p /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma5/experiments/run_experiment.py --task Isaac-Iris-MA5-Direct-v0"
NUM_ENVS=4096
NUM_ENVS_MA5=4096  # Separate var for N-agent runs; reduce to 2048/1024 if GPU OOM persists
# SEEDS=(42 123 456)
SEEDS=(42)

# IROS 2026 MUST experiments (iris_ma4, 2-agent)
EXPERIMENTS=(
    # A1: Delay modeling
    # "a1_no_delay"
    # "a1_stochastic_delay" # Full delay modeling in a1_with_aoi is sufficient
    # "a1_with_aoi"
    # "a1_without_aoi"
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

# iris_ma5 N-agent experiments (registered in iris_ma5/experiments/experiment_registry.py)
# sweep_agents_n2 = 2-agent baseline (same as MA4 default)
# sweep_agents_n3 = 3-agent: full system (AoI + delay + analytical covariance)
# sweep_agents_n4 = 4-agent: full system (AoI + delay + analytical covariance)
EXPERIMENTS_MA5=(
    "a1_with_aoi"
    "a1_without_aoi"
    # "a2_mlp"
    "a3_noisy_reward"
    "a4_angular_only"
    # "sweep_agents_n3"
    # "sweep_agents_n4"
)

# Parse args
DRY_RUN=false
GROUP_FILTER=""
GPU_IDS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)  DRY_RUN=true; shift ;;
        --group)    GROUP_FILTER="$2"; shift 2 ;;
        --gpu_ids)  IFS=',' read -ra GPU_IDS <<< "$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Filter MA4 experiments by group prefix if requested
if [[ -n "$GROUP_FILTER" ]] && [[ "${GROUP_FILTER,,}" != "ma5" ]]; then
    FILTERED=()
    for exp in "${EXPERIMENTS[@]}"; do
        prefix=$(echo "$exp" | cut -d'_' -f1)
        if [[ "$prefix" == "${GROUP_FILTER,,}" ]] || [[ "$exp" == *"${GROUP_FILTER,,}"* ]]; then
            FILTERED+=("$exp")
        fi
    done
    EXPERIMENTS=("${FILTERED[@]}")
fi

# If filtering for ma5 only, clear MA4 list
if [[ "${GROUP_FILTER,,}" == "ma5" ]]; then
    EXPERIMENTS=()
fi

TOTAL=$(( (${#EXPERIMENTS[@]} + ${#EXPERIMENTS_MA5[@]}) * ${#SEEDS[@]} ))
echo "============================================================"
echo "IROS 2026 Ablation Experiments"
echo "MA4 experiments: ${#EXPERIMENTS[@]}, MA5 experiments: ${#EXPERIMENTS_MA5[@]}"
echo "Seeds: ${#SEEDS[@]}, Total runs: ${TOTAL}"
echo "Num envs: MA4=${NUM_ENVS}, MA5=${NUM_ENVS_MA5}"
if [[ ${#GPU_IDS[@]} -gt 0 ]]; then echo "GPU IDs (parallel MA5): ${GPU_IDS[*]}"; fi
if $DRY_RUN; then echo "[DRY RUN - no training will be executed]"; fi
echo "============================================================"

RUN_IDX=0
FAILED=()

# ── MA4 runs (2-agent, iris_ma4) ─────────────────────────────────────────────
for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        echo ""
        echo "[${RUN_IDX}/${TOTAL}] [MA4] Experiment: ${exp}  Seed: ${seed}"
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

# ── MA5 runs (N-agent, iris_ma5) ─────────────────────────────────────────────
MA5_PIDS=()
MA5_TAGS=()
GPU_IDX=0

for exp in "${EXPERIMENTS_MA5[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))

        CMD="$RUN_CMD_MA5 --experiment $exp --seed $seed --num_envs $NUM_ENVS_MA5"

        if [[ ${#GPU_IDS[@]} -gt 0 ]]; then
            GPU_ID="${GPU_IDS[$((GPU_IDX % ${#GPU_IDS[@]}))]}"
            GPU_IDX=$((GPU_IDX + 1))
            TAG="ma5_${exp}_seed${seed}_gpu${GPU_ID}"
            echo ""
            echo "[${RUN_IDX}/${TOTAL}] [MA5] Experiment: ${exp}  Seed: ${seed}  GPU: ${GPU_ID}"
            echo "------------------------------------------------------------"
            if $DRY_RUN; then
                echo "  CMD: CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}"
            else
                LOG_DIR="parallel_logs"
                mkdir -p "$LOG_DIR"
                LOG_FILE="${LOG_DIR}/${TAG}.log"
                CUDA_VISIBLE_DEVICES="${GPU_ID}" $CMD > "$LOG_FILE" 2>&1 &
                MA5_PIDS+=($!)
                MA5_TAGS+=("$TAG")
                echo "  Launched PID $! → ${LOG_FILE}"
            fi
        else
            echo ""
            echo "[${RUN_IDX}/${TOTAL}] [MA5] Experiment: ${exp}  Seed: ${seed}"
            echo "------------------------------------------------------------"
            CMD_FULL="$CMD"
            if $DRY_RUN; then
                echo "  CMD: ${CMD_FULL}"
            else
                if $CMD_FULL; then
                    echo "[${RUN_IDX}/${TOTAL}] DONE: ${exp} seed=${seed}"
                else
                    echo "[${RUN_IDX}/${TOTAL}] FAILED: ${exp} seed=${seed}"
                    FAILED+=("ma5_${exp}_seed${seed}")
                fi
                sleep 5
            fi
        fi
    done
done

# ── Wait for parallel MA5 jobs ────────────────────────────────────────────────
if [[ ${#MA5_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "Waiting for ${#MA5_PIDS[@]} parallel MA5 jobs..."
    for i in "${!MA5_PIDS[@]}"; do
        pid="${MA5_PIDS[$i]}"
        tag="${MA5_TAGS[$i]}"
        if wait "$pid"; then
            echo "  DONE: ${tag}"
        else
            echo "  FAILED: ${tag}  (see parallel_logs/${tag}.log)"
            FAILED+=("$tag")
        fi
    done
fi

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
