#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 047 Slice-1 — warm-start fast-screen across 4 reward-weight candidates.
# Each candidate loads agent_120000.pt from the t046 partial run and trains
# +30k steps under the candidate's reward weights, with the env curriculum
# auto-pinned to step 120000 by --checkpoint plumbing. ~3 wall-h per candidate,
# ~12 wall-h total. Runs sequentially under a single GPU.
#
# Usage:
#   tmux new-session -d -s t047_screen \
#     'bash source/isaaclab_tasks/.../experiments/scripts/run_t047_warmstart_screen.bash'
#
# Or run a single candidate by name:
#   bash run_t047_warmstart_screen.bash A
#
# Candidates (per ticket 047, Slice 1 §"Candidate set"):
#   A — proposed default: bbox_c=60, bbox_s=30, tri=8, action_delta=-24
#   B — tri-led:          bbox_c=60, bbox_s=30, tri=15, action_delta=-24
#   C — smoothness-only:  bbox_c=60, bbox_s=60, tri=5,  action_delta=-36
#   D — visibility-first: bbox_c=90, bbox_s=30, tri=8,  action_delta=-24
#
# Each run writes to a fresh log dir:
#   logs/skrl/iris_ma6/<timestamp>_mappo_rnn_torch_t047_warmstart_<X>/

set -eo pipefail
# Intentionally NOT using `-u`: the isaaclab conda env's activate.d scripts
# (e.g. setup_conda_env.sh) reference ZSH_VERSION when running under bash,
# which strict-no-unset would trip.

# ---------- locate repo root ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------- environment ----------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab

# ---------- shared launch params ----------
CKPT="${REPO_ROOT}/logs/skrl/iris_ma6/2026-06-02_00-29-06_mappo_rnn_torch_ticket046_closer_spawn_2d_target/checkpoints/agent_120000.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found at ${CKPT}" >&2
  exit 1
fi

TIMESTEPS=30000
TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"

# Common args (positional first; hydra-style env.* overrides go LAST per
# train_mappo_rnn_hydra.py's argparse + sys.argv handover).
common_argparse_args=(
  --task Isaac-Iris-MA6-Direct-Test-v0
  --num_envs 1024
  --headless
  --checkpoint "${CKPT}"
  --timesteps "${TIMESTEPS}"
)

# t046 cfg state preserved in every warm-start: 2D target motion.
# (Spawn geometry is already the post-046 cfg default.)
common_hydra_args=(
  "env.target_controller.enable_z_motion=False"
)

# ---------- candidate launcher ----------
run_candidate() {
  local name="$1"
  local bbox_c="$2"
  local bbox_s="$3"
  local tri="$4"
  local ad="$5"
  shift 5
  local description="$*"

  echo ""
  echo "================================================================================"
  echo "  T047 WARM-START SCREEN — Candidate ${name}"
  echo "  ${description}"
  echo "  bbox_center=${bbox_c}  bbox_size=${bbox_s}  triangulation=${tri}  action_delta=${ad}"
  echo "  +${TIMESTEPS} steps from agent_120000.pt"
  echo "  $(date)"
  echo "================================================================================"

  ./isaaclab.sh -p "${TRAIN_SCRIPT}" \
    "${common_argparse_args[@]}" \
    --experiment_name "t047_warmstart_${name}" \
    "${common_hydra_args[@]}" \
    "env.bbox_center_reward_scale=${bbox_c}" \
    "env.bbox_size_reward_scale=${bbox_s}" \
    "env.triangulation_reward_scale=${tri}" \
    "env.action_delta_penalty_scale=${ad}"
}

# ---------- dispatch ----------
which="${1:-all}"

case "${which}" in
  A|a)
    run_candidate A 60.0 30.0 8.0 -24.0 "proposed default — halve bbox_size, modest tri lift, double smoothness"
    ;;
  B|b)
    run_candidate B 60.0 30.0 15.0 -24.0 "tri-led — same as A but stronger tri push"
    ;;
  C|c)
    run_candidate C 60.0 60.0 5.0 -36.0 "smoothness-only — no reward retune; triple action_delta"
    ;;
  D|d)
    run_candidate D 90.0 30.0 8.0 -24.0 "visibility-first — boost bbox_center to push pair_valid"
    ;;
  all)
    run_candidate A 60.0 30.0 8.0 -24.0 "proposed default — halve bbox_size, modest tri lift, double smoothness"
    run_candidate B 60.0 30.0 15.0 -24.0 "tri-led — same as A but stronger tri push"
    run_candidate C 60.0 60.0 5.0 -36.0 "smoothness-only — no reward retune; triple action_delta"
    run_candidate D 90.0 30.0 8.0 -24.0 "visibility-first — boost bbox_center to push pair_valid"
    ;;
  *)
    echo "Unknown candidate '${which}'. Valid: A|B|C|D|all" >&2
    exit 2
    ;;
esac

echo ""
echo "================================================================================"
echo "  T047 WARM-START SCREEN — DONE ($(date))"
echo "================================================================================"
