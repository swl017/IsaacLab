#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 048 — seed-confirm the wide (256/256) cfg-default flip.
#
# The flip to wide was decided on single-seed (42) + single deploy realization.
# This run repeats wide at a DIFFERENT seed (123) to check the calm-tracker
# behaviour and estimation quality aren't seed luck before we build on wide.
#
# Width is NOT overridden here on purpose: it inherits the (now-default)
# skrl_mappo_rnn_cfg.yaml 256/256 — so this run also confirms the cfg flip is
# wired correctly. Expect combined policy+value scalar params = 974,351
# (the wide count from the original sweep). Everything else = bare cfg defaults
# (post-045/046/047), num_envs=1024, 200k, matching the original sweep.
#
# Usage:
#   tmux new-session -d -s t048_wide_confirm \
#     'bash source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/experiments/scripts/run_t048_wide_seed_confirm.bash'
#   tmux attach -t t048_wide_confirm

set -o pipefail

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
SEED=123
EXP="ticket048_net_width_wide_seed123_confirm"

LOG_DIR="${HERE}/t048_sweep_logs"
mkdir -p "${LOG_DIR}"
CONSOLE="${LOG_DIR}/${EXP}.log"

echo "================================================================================"
echo "  T048 wide seed-confirm — width from cfg DEFAULT (expect 256/256, 974,351 params)"
echo "  task=${TASK}  num_envs=${NUM_ENVS}  timesteps=${TIMESTEPS}  seed=${SEED}"
echo "  console: ${CONSOLE}"
echo "  $(date)"
echo "================================================================================"

./isaaclab.sh -p "${TRAIN_SCRIPT}" \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --seed "${SEED}" \
  --timesteps "${TIMESTEPS}" \
  --experiment_name "${EXP}" \
  2>&1 | tee "${CONSOLE}"

echo "================================================================================"
echo "  T048 wide seed-confirm DONE — $(date)"
echo "  Next: deploy-bag the resulting checkpoint, re-run deploy_tracking_compare.py,"
echo "  and check wide@123 reproduces wide@42's calm/estimation profile."
echo "================================================================================"
