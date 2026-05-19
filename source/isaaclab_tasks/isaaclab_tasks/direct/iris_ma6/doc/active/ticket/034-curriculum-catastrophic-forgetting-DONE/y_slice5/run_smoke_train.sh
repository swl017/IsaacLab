#!/usr/bin/env bash
# Slice 5 step 5.3 — 10 k-step smoke training to verify Slices 1-4 don't
# crash or NaN the trainer. Uses the canonical iris_ma6 MAPPO-RNN cfg with
# trainer.timesteps overridden to 10k.
#
# Pass criteria (from p_plan.md):
#   - no NaN in loss
#   - KL parks in [0.012, 0.028] (matches fix2 early-training)
#   - no entropy / std drift > 0.30

# Don't use `set -u`: _isaac_sim/setup_conda_env.sh references unbound ZSH_VERSION.
set -eo pipefail

source /home/usrg/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

ROOT=/home/usrg/IsaacPX4/IsaacLab
cd "$ROOT"

OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$OUT_DIR/smoke_train_$(date +%Y%m%d_%H%M%S).log"

echo "Launching 10k smoke; log=$LOG"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py \
    --task Isaac-Iris-MA6-Direct-Test-v0 \
    --num_envs 256 \
    --seed 0 \
    --headless \
    agent.trainer.timesteps=10000 \
    > "$LOG" 2>&1

echo "Smoke training complete. Inspect $LOG."
