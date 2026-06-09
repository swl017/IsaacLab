#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 047 Slice-2.5 — cold-start full validation of candidate D with the
# *entropy_loss_scale curriculum* added (linear ramp 0.01 → 0.001 over
# 120k → 200k) and max_log_std tightened from 0.7 → 0.4.
#
# Why this combination (vs the prior Slice 2 lowent flat-0.001 run):
#   • Flat entropy_loss_scale=0.001 collapsed σ to 0.24 by step 16k, BEFORE
#     the behavioral curriculum (moving_target 40-80k, dynamics 60-100k,
#     noise 100-120k) had time to bite. The policy then lost grip on harder
#     phases (pair_valid 0.97 → 0.78, triangulation 7.48 → 4.81).
#   • Curriculum schedule keeps entropy_loss_scale=0.01 (full exploration)
#     through 0-120k so the policy explores moving_target / DR / noise with
#     wide σ, then ramps to 0.001 over 120k-200k once those curriculum
#     phases are fully ramped, committing to a tight distribution for the
#     final 200k-400k regime.
#   • max_log_std=0.4 (σ_max=exp(0.4)=1.49) caps the explore-phase σ so it
#     doesn't blow to 2.0 like t046 did. Bounded exploration, not runaway.
#
# Reward weights are candidate D (visibility-first): bbox_c=90, bbox_s=30,
# tri=8, action_delta=-24. All other env state matches t046+t047 baseline.
#
# Usage:
#   tmux new-session -d -s t047_D_curriculum \
#     'bash source/.../experiments/scripts/run_t047_D_curriculum_coldstart.bash'

set -eo pipefail
# Not using -u: setup_conda_env.sh references ZSH_VERSION.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab

TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"

echo ""
echo "================================================================================"
echo "  T047 Slice 2.5 — Candidate D cold-start (curriculum entropy + tighter cap)"
echo "  Reward weights: bbox_center=90.0  bbox_size=30.0  triangulation=8.0  action_delta=-24.0"
echo "  Entropy schedule: 0.01 → 0.001 linear over timestep [120000, 200000]"
echo "  Policy cap:       max_log_std=0.4 (σ_max=exp(0.4)≈1.49)"
echo "  Curriculum:       full 0 → 400k cold-start, seed=42"
echo "  Other env state:  post-t046 spawn (30/15/2) + enable_z_motion=False"
echo "                    + slew clip on + prev_action obs on + LP off"
echo "  $(date)"
echo "================================================================================"

./isaaclab.sh -p "${TRAIN_SCRIPT}" \
  --task Isaac-Iris-MA6-Direct-Test-v0 \
  --num_envs 1024 \
  --headless \
  --experiment_name ticket047_D_curriculum_cold \
  env.target_controller.enable_z_motion=False \
  env.bbox_center_reward_scale=90.0 \
  env.bbox_size_reward_scale=30.0 \
  env.triangulation_reward_scale=8.0 \
  env.action_delta_penalty_scale=-24.0 \
  agent.agent.entropy_loss_scale=0.01 \
  agent.agent.entropy_loss_schedule.end_value=0.001 \
  agent.agent.entropy_loss_schedule.start_step=120000 \
  agent.agent.entropy_loss_schedule.end_step=200000 \
  agent.models.policy.max_log_std=0.4

echo ""
echo "================================================================================"
echo "  T047 Slice 2.5 — DONE ($(date))"
echo "================================================================================"
