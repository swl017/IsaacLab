#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 045 post-training |Δa| probe sweep — 9 calls across 5 curriculum
# points × dual checkpoints (40k+400k @ 39k, 80k+400k @ 79k,
# 120k+400k @ 119k, 200k+400k @ 199k, 400k @ 399k).
#
# The probe env is built with the same envelope downshift the t045 run
# trained under (max_lin_vel=5.0, action_slew_vel_xy=0.040,
# target_controller.max_lin_vel=2.5) so the measured Δa distribution
# reflects the training regime — not what the policy *would* do under
# the default cfg.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PROBE_SCRIPT="${HERE}/probe_action_delta_distribution.py"
if [[ ! -f "${PROBE_SCRIPT}" ]]; then
    echo "[ERROR] cannot find ${PROBE_SCRIPT}" >&2
    exit 1
fi

PROBE_RUN_DIR="${PROBE_RUN_DIR:-${ISAACLAB_ROOT}/logs/skrl/iris_ma6/2026-05-29_11-25-06_mappo_rnn_torch_09762f259f_ticket045_task_difficulty_tuning}"
EPISODES="${EPISODES:-10}"
NUM_ENVS="${NUM_ENVS:-1024}"
STEPS_PER_EP="${STEPS_PER_EP:-500}"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/probe_action_delta_t045_${TS}}"

CKPT_DIR="${PROBE_RUN_DIR}/checkpoints"
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

# Overrides matching the t045 training env (see params/env.yaml of the run).
CFG_OVERRIDES=(
    "--cfg_override" "max_lin_vel=5.0"
    "--cfg_override" "max_lin_vel_min=5.0"
    "--cfg_override" "action_slew_vel_xy=0.040"
    "--cfg_override" "enable_action_slew_clip=true"
    "--cfg_override" "enable_prev_action_obs=true"
    "--cfg_override" "enable_action_lowpass=false"
    "--cfg_override" "target_controller.max_lin_vel=2.5"
)

echo "==================================================================="
echo "Ticket 045 post-training |Δa| probe sweep"
echo "==================================================================="
echo "Run dir          : ${PROBE_RUN_DIR}"
echo "Output dir       : ${OUTPUT_DIR}"
echo "episodes         : ${EPISODES}"
echo "num_envs         : ${NUM_ENVS}"
echo "steps_per_ep     : ${STEPS_PER_EP}"
echo "cfg overrides    : ${CFG_OVERRIDES[*]}"
echo "==================================================================="

# 9-call probe matrix: (debug_step, ckpt_step) tuples.
# Dual-checkpoint at 4 curriculum points + final-checkpoint at the 399k point.
PROBES=(
    "39000   40000"
    "39000   400000"
    "79000   80000"
    "79000   400000"
    "119000  120000"
    "119000  400000"
    "199000  200000"
    "199000  400000"
    "399000  400000"
)

T_START=$(date +%s)
for entry in "${PROBES[@]}"; do
    # shellcheck disable=SC2206
    arr=($entry)
    DEBUG_STEP="${arr[0]}"
    CKPT_STEP="${arr[1]}"
    CKPT="${CKPT_DIR}/agent_${CKPT_STEP}.pt"
    if [[ ! -f "${CKPT}" ]]; then
        echo "[WARN] missing checkpoint ${CKPT}, skipping" >&2
        continue
    fi
    DEBUG_TAG=$(printf "%dk" $((DEBUG_STEP / 1000)))
    CKPT_TAG=$(printf "%dk" $((CKPT_STEP / 1000)))
    OUT_CSV="${OUTPUT_DIR}/probe_step${DEBUG_TAG}_ckpt${CKPT_TAG}.csv"
    LOG="${OUTPUT_DIR}/probe_step${DEBUG_TAG}_ckpt${CKPT_TAG}.log"

    echo
    echo "-------------------------------------------------------------------"
    echo "[probe] debug_step=${DEBUG_STEP} | ckpt=agent_${CKPT_STEP}.pt"
    echo "-------------------------------------------------------------------"

    cd "${ISAACLAB_ROOT}"
    ./isaaclab.sh -p "${PROBE_SCRIPT}" \
        --checkpoint "${CKPT}" \
        --debug_step "${DEBUG_STEP}" \
        --episodes "${EPISODES}" \
        --num_envs "${NUM_ENVS}" \
        --steps_per_episode "${STEPS_PER_EP}" \
        --output_csv "${OUT_CSV}" \
        "${CFG_OVERRIDES[@]}" \
        --headless \
        2>&1 | tee "${LOG}"
done

# Aggregate
SUMMARY="${OUTPUT_DIR}/summary.csv"
PIVOT="${OUTPUT_DIR}/summary_pivot.txt"

echo
echo "==================================================================="
echo "[probe] aggregating into ${SUMMARY}"
echo "==================================================================="

FIRST=1
for csv in "${OUTPUT_DIR}"/probe_*.csv; do
    [[ -f "$csv" ]] || continue
    if [[ $FIRST -eq 1 ]]; then
        cat "$csv" > "${SUMMARY}"
        FIRST=0
    else
        tail -n +2 "$csv" >> "${SUMMARY}"
    fi
done
ROWS=$(($(wc -l < "${SUMMARY}") - 1))
echo "[probe] aggregated ${ROWS} rows"

PIVOT_SCRIPT=$(mktemp /tmp/probe_pivot_XXXXXX.py)
cat > "${PIVOT_SCRIPT}" <<'PYEOF'
import csv
import sys
summary_path = sys.argv[1]
rows = list(csv.DictReader(open(summary_path)))
keys = sorted({(r["channel_label"], r["checkpoint"]) for r in rows})
steps = sorted({int(r["debug_step"]) for r in rows})
# t045-effective slew limits (action units). Note the doubled δ_xy for the
# envelope-downshifted treatment so physical accel stays at 5 m/s².
T045_DELTA = {
    "vx": 0.040, "vy": 0.040, "vz": 0.053,
    "yaw_rate": 0.300,
    "gimbal_yaw_rate": 0.400, "gimbal_pitch_rate": 0.400,
    "zoom_rate": 0.200,
}
print("Ticket 045 post-training - per-channel p90 of |delta_action|, max over agents")
print("Probed under t045 cfg: max_lin_vel=5.0, action_slew_vel_xy=0.040, target.max_lin_vel=2.5")
print("=" * 110)
header = f"{'channel':<20} {'checkpoint':<25}"
for s in steps:
    header += f"{s//1000}k".rjust(10)
header += f"{'t045 delta':>14}"
print(header)
print("-" * 110)
for (ch, ckpt) in keys:
    line = f"{ch:<20} {ckpt:<25}"
    for s in steps:
        match = [r for r in rows if r["channel_label"] == ch
                 and r["checkpoint"] == ckpt and int(r["debug_step"]) == s]
        if match:
            p90 = max(float(r["p90"]) for r in match)
            line += f"{p90:>10.4f}"
        else:
            line += f"{'-':>10}"
    ref = T045_DELTA.get(ch, float('nan'))
    line += f"{ref:>14.4f}"
    print(line)
print("=" * 110)
print()
print("Interpretation (per ticket 045 acceptance bar):")
print("  - p90 >= delta on a channel  -> policy SATURATING that channel (clip is binding)")
print("  - p90 << delta on a channel  -> policy has slack on that channel")
print("  - Compare across checkpoints to see if late-stage policy uses more or less of the bandwidth than early-stage at the same curriculum.")
PYEOF
python3 "${PIVOT_SCRIPT}" "${SUMMARY}" > "${PIVOT}" 2>&1
rm -f "${PIVOT_SCRIPT}"

T_ELAPSED=$(( $(date +%s) - T_START ))
echo
echo "==================================================================="
echo "[probe] DONE - total wall-time ${T_ELAPSED}s"
echo "  per-call CSVs : ${OUTPUT_DIR}/probe_step*.csv"
echo "  summary CSV   : ${SUMMARY}"
echo "  pivot table   : ${PIVOT}"
echo "==================================================================="
cat "${PIVOT}"
