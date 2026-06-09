#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 047 Slice-2 — cold-start full validation of candidate D (visibility-first)
# with the entropy regularizer lowered from 0.01 → 0.001 to allow Policy/std
# to actually collapse during training.
#
# Why this combination:
#   • The warm-start fast-screen (Slice 1) showed Policy/std locked at
#     σ_max = exp(0.7) = 2.014 for all 4 candidates, because the entropy
#     bonus (entropy_loss_scale × entropy ≈ 15% of policy_loss) keeps
#     pushing log_std against its cap regardless of reward shape.
#   • Candidate D was directionally best on task metrics (pair_valid 0.84,
#     tri raw +12%, bbox raw +10%), so its reward shape is worth a full run.
#   • Lowering entropy_loss_scale to 0.001 removes the upward pull on
#     log_std, allowing the policy to commit to a tighter distribution
#     wherever the policy gradient prefers (probably σ ~ 0.3–0.6).
#
# Curriculum: full 0 → 400k cold-start, seed=42, under tmux.
#
# Usage:
#   tmux new-session -d -s t047_D_cold \
#     'bash source/isaaclab_tasks/.../experiments/scripts/run_t047_D_lowent_coldstart.bash'

set -eo pipefail
# Intentionally NOT using -u: setup_conda_env.sh references ZSH_VERSION.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab

TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"

echo ""
echo "================================================================================"
echo "  T047 Slice 2 — Candidate D cold-start (visibility-first + low entropy)"
echo "  Reward weights: bbox_center=90.0  bbox_size=30.0  triangulation=8.0  action_delta=-24.0"
echo "  Agent override: entropy_loss_scale=0.001 (×0.1 from default 0.01)"
echo "  Curriculum:     full 0 → 400k cold-start, seed=42"
echo "  Other env state: post-t046 spawn (30/15/2) + enable_z_motion=False"
echo "                   + slew clip on + prev_action obs on + LP off"
echo "  $(date)"
echo "================================================================================"

./isaaclab.sh -p "${TRAIN_SCRIPT}" \
  --task Isaac-Iris-MA6-Direct-Test-v0 \
  --num_envs 1024 \
  --headless \
  --experiment_name ticket047_D_lowent_cold \
  env.target_controller.enable_z_motion=False \
  env.bbox_center_reward_scale=90.0 \
  env.bbox_size_reward_scale=30.0 \
  env.triangulation_reward_scale=8.0 \
  env.action_delta_penalty_scale=-24.0 \
  agent.agent.entropy_loss_scale=0.001

echo ""
echo "================================================================================"
echo "  T047 Slice 2 — DONE ($(date))"
echo "================================================================================"
