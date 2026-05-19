#!/usr/bin/env bash
# Slice 5 step 5.4 — Full 400k MAPPO-RNN training under the ticket-034
# anti-forgetting curriculum. Run inside a persistent tmux session so the
# job survives the launching shell and the engineer can attach to monitor.
#
# tmux session name: ticket034_train
# Attach with: `tmux attach -t ticket034_train`
# Detach with: `Ctrl-B then D`
# Log file: $ROOT/logs/skrl/iris_ma6/ticket034_train_<timestamp>.log
#   (skrl trainer also auto-creates its own run dir under the same parent;
#    this top-level log captures stdout/stderr for tail -f convenience.)

set -eo pipefail

SESSION="ticket034_train"
ROOT=/home/usrg/IsaacPX4/IsaacLab
LOG_DIR="$ROOT/logs/skrl/iris_ma6"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$LOG_DIR/ticket034_train_${STAMP}.log"

# Number of envs — match fix2's nominal training scale. Override via
# argv[1] if available. Seed is argv[2], experiment-name suffix is argv[3].
NUM_ENVS="${1:-1024}"
SEED="${2:-0}"
EXP_SUFFIX="${3:-ticket034_per_env_jitter}"
# The skrl experiment_name yaml default is hardcoded as
# `7fd070ee09_mappo_rnn_shared_model_scheduler_param_fix2`, which causes
# the auto-named run dir to collide across runs. Override via hydra so
# every ticket-034 run gets a unique name. The hydra path is
# `agent.experiment.experiment_name`.
EXP_NAME="034_${STAMP}_${EXP_SUFFIX}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Attach with"
    echo "       tmux attach -t $SESSION"
    echo "  or kill it with: tmux kill-session -t $SESSION"
    exit 1
fi

CMD="cd $ROOT && \
source /home/usrg/miniconda3/etc/profile.d/conda.sh && \
conda activate env_isaaclab && \
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py \
    --task Isaac-Iris-MA6-Direct-Test-v0 \
    --num_envs $NUM_ENVS \
    --seed $SEED \
    --experiment_name '$EXP_NAME' \
    --headless \
    2>&1 | tee $LOG"

echo "Launching tmux session: $SESSION"
echo "  num_envs:        $NUM_ENVS"
echo "  seed:            $SEED"
echo "  experiment_name: $EXP_NAME"
echo "  log file:        $LOG"
echo "  attach:          tmux attach -t $SESSION"
echo "  detach:          Ctrl-B then D"

tmux new-session -d -s "$SESSION" "$CMD"
echo "Session launched."
echo
echo "First-pass health checks (run in ~2 min after launch):"
echo "  tail -100 $LOG | grep -E 'Setting seed|Number of environments|NaN|loss|kl'"
echo "  tmux ls"
