#!/bin/bash
# MAPPO-RNN Hyperparameter Tuning Script
# NOTE: Hydra --multirun does NOT work with Isaac Sim because Isaac Sim can only
# be initialized once per process. This script runs each configuration separately.

EXP_DIR="iris_ma4_mappo_rnn_tuning6_$(date +%Y%m%d_%H%M%S)"

set -e  # Exit on error

# Task configuration
TASK="Isaac-Iris-MA4-Direct-v0"
NUM_ENVS=1024
SCRIPT="/home/usrg/IsaacPX4/IsaacLab/scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"

# ============================================================================
# HYPERPARAMETER CONFIGURATION
# To sweep a parameter: use multiple values, e.g., (1e-3 3e-4 1e-4)
# To use a fixed value: use single value, e.g., (0.01)
# To use the YAML default: set to empty, e.g., ()
# ============================================================================

LEARNING_RATES=()               # 0.0003
SEQUENCE_LENGTHS=(32 64)
ENTROPY_SCALES=()               # 0.01
MINI_BATCHES=(8)
ROLLOUTS=(32 64)
DELAY_STEPS=()
GRU_HIDDEN_SIZES=(64)
GRU_NUM_LAYERS=(1)
GRAD_NORM_CLIPS=(0.3)
KL_THRESHOLDS=(0.02)
BURN_IN_STEPS=(0 8)

# ============================================================================
# Handle empty arrays (use placeholder for single iteration)
# ============================================================================
if [[ ${#LEARNING_RATES[@]} -eq 0 ]]; then LEARNING_RATES=("_default_"); fi
if [[ ${#SEQUENCE_LENGTHS[@]} -eq 0 ]]; then SEQUENCE_LENGTHS=("_default_"); fi
if [[ ${#ENTROPY_SCALES[@]} -eq 0 ]]; then ENTROPY_SCALES=("_default_"); fi
if [[ ${#MINI_BATCHES[@]} -eq 0 ]]; then MINI_BATCHES=("_default_"); fi
if [[ ${#ROLLOUTS[@]} -eq 0 ]]; then ROLLOUTS=("_default_"); fi
if [[ ${#DELAY_STEPS[@]} -eq 0 ]]; then DELAY_STEPS=("_default_"); fi
if [[ ${#GRU_HIDDEN_SIZES[@]} -eq 0 ]]; then GRU_HIDDEN_SIZES=("_default_"); fi
if [[ ${#GRU_NUM_LAYERS[@]} -eq 0 ]]; then GRU_NUM_LAYERS=("_default_"); fi
if [[ ${#GRAD_NORM_CLIPS[@]} -eq 0 ]]; then GRAD_NORM_CLIPS=("_default_"); fi
if [[ ${#KL_THRESHOLDS[@]} -eq 0 ]]; then KL_THRESHOLDS=("_default_"); fi
if [[ ${#BURN_IN_STEPS[@]} -eq 0 ]]; then BURN_IN_STEPS=("_default_"); fi

# Counter for tracking progress
total=$((${#LEARNING_RATES[@]} * ${#SEQUENCE_LENGTHS[@]} * ${#ENTROPY_SCALES[@]} * ${#MINI_BATCHES[@]} * ${#ROLLOUTS[@]} * ${#DELAY_STEPS[@]} * ${#GRU_HIDDEN_SIZES[@]} * ${#GRU_NUM_LAYERS[@]} * ${#GRAD_NORM_CLIPS[@]} * ${#KL_THRESHOLDS[@]} * ${#BURN_IN_STEPS[@]}))
current=0
skipped=0

echo "========================================"
echo "MAPPO-RNN Hyperparameter Sweep"
echo "========================================"
echo "Task: $TASK"
echo "Num envs: $NUM_ENVS"
echo "Total configurations: $total"
echo ""
echo "Sweep parameters:"
echo "  learning_rate:      ${LEARNING_RATES[*]}"
echo "  sequence_length:    ${SEQUENCE_LENGTHS[*]}"
echo "  entropy_loss_scale: ${ENTROPY_SCALES[*]}"
echo "  mini_batches:       ${MINI_BATCHES[*]}"
echo "  rollouts:           ${ROLLOUTS[*]}"
echo "  delay_steps:        ${DELAY_STEPS[*]}"
echo "  gru_hidden_size:    ${GRU_HIDDEN_SIZES[*]}"
echo "  gru_num_layers:     ${GRU_NUM_LAYERS[*]}"
echo "  grad_norm_clip:     ${GRAD_NORM_CLIPS[*]}"
echo "  kl_threshold:       ${KL_THRESHOLDS[*]}"
echo "  burn_in_steps:      ${BURN_IN_STEPS[*]}"
echo "========================================"
echo ""

# Copy this script to the experiment directory for reproducibility
FULL_EXP_DIR="/home/usrg/IsaacPX4/IsaacLab/logs/skrl/$EXP_DIR"
mkdir -p "$FULL_EXP_DIR"
cp "$0" "$FULL_EXP_DIR/"
echo "Script copied to: $FULL_EXP_DIR/$(basename "$0")"
echo ""

# Loop through all combinations
for lr in "${LEARNING_RATES[@]}"; do
    for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
        for entropy in "${ENTROPY_SCALES[@]}"; do
            for mini_batch in "${MINI_BATCHES[@]}"; do
                for rollout in "${ROLLOUTS[@]}"; do
                    for delay in "${DELAY_STEPS[@]}"; do
                        for gru_hidden in "${GRU_HIDDEN_SIZES[@]}"; do
                            for gru_layers in "${GRU_NUM_LAYERS[@]}"; do
                                for grad_clip in "${GRAD_NORM_CLIPS[@]}"; do
                                    for kl_thresh in "${KL_THRESHOLDS[@]}"; do
                                        for burn_in in "${BURN_IN_STEPS[@]}"; do
                    current=$((current + 1))

                    # Skip invalid configurations where rollout < seq_len
                    # (rollouts must be >= sequence_length for proper RNN training)
                    if [[ "$seq_len" != "_default_" && "$rollout" != "_default_" ]]; then
                        if (( rollout < seq_len )); then
                            echo "[$current/$total] Skipping: rollout=$rollout < seq_len=$seq_len (invalid)"
                            skipped=$((skipped + 1))
                            continue
                        fi
                    fi

                    # Build experiment name (only include non-default params)
                    exp_name=""
                    if [[ "$lr" != "_default_" ]]; then exp_name+="lr${lr}_"; fi
                    if [[ "$seq_len" != "_default_" ]]; then exp_name+="seq${seq_len}_"; fi
                    if [[ "$entropy" != "_default_" ]]; then exp_name+="ent${entropy}_"; fi
                    if [[ "$mini_batch" != "_default_" ]]; then exp_name+="mb${mini_batch}_"; fi
                    if [[ "$rollout" != "_default_" ]]; then exp_name+="roll${rollout}_"; fi
                    if [[ "$delay" != "_default_" ]]; then exp_name+="delay${delay}_"; fi
                    if [[ "$gru_hidden" != "_default_" ]]; then exp_name+="gru${gru_hidden}_"; fi
                    if [[ "$gru_layers" != "_default_" ]]; then exp_name+="lay${gru_layers}_"; fi
                    if [[ "$grad_clip" != "_default_" ]]; then exp_name+="clip${grad_clip}_"; fi
                    if [[ "$kl_thresh" != "_default_" ]]; then exp_name+="kl${kl_thresh}_"; fi
                    if [[ "$burn_in" != "_default_" ]]; then exp_name+="burnin${burn_in}_"; fi
                    exp_name="${exp_name%_}"  # Remove trailing underscore
                    if [[ -z "$exp_name" ]]; then exp_name="default"; fi

                    echo "========================================"
                    echo "[$current/$total] Starting: $exp_name"
                    echo "========================================"

                    # Build command with only non-default parameters
                    cmd="python \"$SCRIPT\" --task \"$TASK\" --num_envs $NUM_ENVS --headless agent.agent.experiment.directory=\"$EXP_DIR\" agent.agent.experiment.experiment_name=\"$exp_name\""
                    if [[ "$lr" != "_default_" ]]; then cmd+=" agent.agent.learning_rate=$lr"; fi
                    if [[ "$seq_len" != "_default_" ]]; then cmd+=" agent.agent.sequence_length=$seq_len"; fi
                    if [[ "$entropy" != "_default_" ]]; then cmd+=" agent.agent.entropy_loss_scale=$entropy"; fi
                    if [[ "$mini_batch" != "_default_" ]]; then cmd+=" agent.agent.mini_batches=$mini_batch"; fi
                    if [[ "$rollout" != "_default_" ]]; then cmd+=" agent.agent.rollouts=$rollout"; fi
                    if [[ "$delay" != "_default_" ]]; then cmd+=" env.delay_steps=$delay"; fi
                    if [[ "$gru_hidden" != "_default_" ]]; then cmd+=" agent.models.policy.hidden_size=$gru_hidden agent.models.policy.gru_hidden_size=$gru_hidden agent.models.value.hidden_size=$gru_hidden agent.models.value.gru_hidden_size=$gru_hidden"; fi
                    if [[ "$gru_layers" != "_default_" ]]; then cmd+=" agent.models.policy.gru_num_layers=$gru_layers agent.models.value.gru_num_layers=$gru_layers"; fi
                    if [[ "$grad_clip" != "_default_" ]]; then cmd+=" agent.agent.grad_norm_clip=$grad_clip"; fi
                    if [[ "$kl_thresh" != "_default_" ]]; then cmd+=" agent.agent.kl_threshold=$kl_thresh"; fi
                    if [[ "$burn_in" != "_default_" ]]; then cmd+=" agent.agent.episode_start_mask_steps=$burn_in"; fi
                    echo "Command: $cmd"
                    echo ""

                    # Run training
                    eval "$cmd"

                    echo ""
                    echo "[$current/$total] Completed: $exp_name"
                    echo ""

                    # Brief pause between runs to let GPU memory clear
                    sleep 3
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "========================================"
echo "Hyperparameter sweep completed!"
echo "Total configurations: $total"
echo "Skipped (rollout < seq_len): $skipped"
echo "Actual runs: $((total - skipped))"
echo "========================================"
