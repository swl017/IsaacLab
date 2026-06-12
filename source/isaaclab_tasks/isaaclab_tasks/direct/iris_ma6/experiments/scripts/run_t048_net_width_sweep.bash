#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 048 Slice 2 — MAPPO-RNN network-width sweep (sequential A/B).
#
# Varies hidden_size + gru_hidden_size JOINTLY at a 1:1 ratio across three
# scales; gru_num_layers held at 1. Mirrors policy overrides onto value so the
# encoder and GRU scale together (neither becomes a bottleneck).
#
#   baseline : 64/64   (current shipping default;   ~72k combined p+v params)
#   mid      : 128/128  (~3.6x baseline;            ~258k params)
#   wide     : 256/256  (~13.6x baseline;           ~974k params)
#
# These mirror the registered experiments validation_net_width_{baseline,mid,wide}
# in experiments/experiment_registry.py. We launch via train_mappo_rnn_hydra.py
# (the proven path every prior t04x validation run used) rather than
# run_experiment.py, with width passed as hydra overrides.
#
# Env state: BARE cfg defaults (post-045/046/047) — width is the ONLY variable.
#   max_lin_vel=5, slew clip ON (δ_xy=0.040), prev_action obs ON, LP OFF,
#   reward candidate D (bbox_c=90 / bbox_s=30 / tri=8 / action_delta=-24),
#   spawn 30/15/2, agent entropy curriculum (0.01→0.001 over 120k-200k),
#   max_log_std=0.4. NOTE: enable_z_motion defaults to True (3D target motion);
#   t046's 2D-motion treatment was never adopted as the cfg default, so this
#   sweep runs on 3D motion. The baseline (64/64) run is the matched internal
#   control, so the width comparison is clean regardless.
#
# Single GPU, sequential. ~24 wall-h each, ~72h total. seed=42 all three.
# Runs are independent: a failure in one config does NOT abort the others.
#
# Monitoring (per ticket §Slice 2 / §Risk):
#   - KL band: kill a run if kl > 0.04 sustained > 20k steps (LR retune precursor).
#   - Reward-curve slope at 40k / 80k / 120k (wider nets may trail early).
#   - slew_sat_* per channel (does width redistribute saturation?).
#   - If wide is still climbing strongly at 200k, consider extending to 400k.
#
# Usage:
#   tmux new-session -d -s t048_net_width \
#     'bash source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/scripts/run_t048_net_width_sweep.bash'
#   tmux attach -t t048_net_width

set -o pipefail
# Not using -e/-u: setup_conda_env.sh references ZSH_VERSION, and we want the
# sweep to continue to the next width even if one run dies.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab

TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"
TASK="Isaac-Iris-MA6-Direct-Test-v0"
NUM_ENVS=1024
TIMESTEPS=200000
SEED=42

LOG_DIR="${HERE}/t048_sweep_logs"
mkdir -p "${LOG_DIR}"

run_width () {
  local tag="$1" w="$2"
  local exp_name="ticket048_net_width_${tag}"
  local console_log="${LOG_DIR}/${exp_name}.log"
  echo ""
  echo "================================================================================"
  echo "  T048 Slice 2 — ${tag} width  (hidden_size=${w}, gru_hidden_size=${w}, layers=1)"
  echo "  task=${TASK}  num_envs=${NUM_ENVS}  timesteps=${TIMESTEPS}  seed=${SEED}"
  echo "  console log: ${console_log}"
  echo "  $(date)"
  echo "================================================================================"
  ./isaaclab.sh -p "${TRAIN_SCRIPT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --seed "${SEED}" \
    --timesteps "${TIMESTEPS}" \
    --experiment_name "${exp_name}" \
    agent.models.policy.hidden_size="${w}" \
    agent.models.policy.gru_hidden_size="${w}" \
    agent.models.value.hidden_size="${w}" \
    agent.models.value.gru_hidden_size="${w}" \
    2>&1 | tee "${console_log}"
  local rc=${PIPESTATUS[0]}
  echo "  [${tag}] finished rc=${rc} ($(date))"
  return "${rc}"
}

echo "################################################################################"
echo "  TICKET 048 — NETWORK WIDTH SWEEP (sequential: baseline -> mid -> wide)"
echo "  Started: $(date)"
echo "################################################################################"

run_width baseline 64
run_width mid      128
run_width wide     256

echo ""
echo "################################################################################"
echo "  TICKET 048 SWEEP DONE — $(date)"
echo "  Logs: ${LOG_DIR}"
echo "  Next: Pareto analysis + writeup (doc/experiments/), conditional cfg-default flip."
echo "################################################################################"
