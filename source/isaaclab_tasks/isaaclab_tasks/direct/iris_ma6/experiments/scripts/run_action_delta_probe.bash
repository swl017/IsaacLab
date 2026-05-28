#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ticket 044 Slice-0 — run 7 probe calls (4 curriculum points × dual checkpoints,
# except the 199k point where final == nearest-greater) and aggregate the
# per-channel percentiles into one summary CSV used to size the per-channel
# action_slew_* defaults.
#
# Usage:
#   bash source/isaaclab_tasks/.../experiments/scripts/run_action_delta_probe.bash
#
# Override the checkpoint run via env var:
#   PROBE_RUN_DIR=/path/to/run bash run_action_delta_probe.bash
#
# Output dir layout:
#   <OUTPUT_DIR>/
#     probe_step39k_ckpt40k.csv
#     probe_step39k_ckpt200k.csv
#     probe_step79k_ckpt80k.csv
#     ...
#     summary.csv          (concatenated per-row CSV, all 7 calls)
#     summary_pivot.txt    (human-readable comparison table)

set -euo pipefail

# ---------- locate repo root + script ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PROBE_SCRIPT="${HERE}/probe_action_delta_distribution.py"
if [[ ! -f "${PROBE_SCRIPT}" ]]; then
    echo "[ERROR] cannot find ${PROBE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "[ERROR] cannot find ${ISAACLAB_ROOT}/isaaclab.sh" >&2
    exit 1
fi

# ---------- defaults ----------
PROBE_RUN_DIR="${PROBE_RUN_DIR:-${ISAACLAB_ROOT}/logs/skrl/iris_ma6/2026-05-26_22-27-56_mappo_rnn_torch_78038af2c2_ticket043_prev_action_obs_true_action_lowpass_false}"
EPISODES="${EPISODES:-10}"
NUM_ENVS="${NUM_ENVS:-1024}"
STEPS_PER_EP="${STEPS_PER_EP:-500}"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/probe_action_delta_${TS}}"

CKPT_DIR="${PROBE_RUN_DIR}/checkpoints"
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

echo "==================================================================="
echo "Ticket 044 Slice-0 — |Δaction| distribution probe sweep"
echo "==================================================================="
echo "Run dir       : ${PROBE_RUN_DIR}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "episodes      : ${EPISODES}"
echo "num_envs      : ${NUM_ENVS}"
echo "steps_per_ep  : ${STEPS_PER_EP}"
echo "==================================================================="

# ---------- probe matrix: (debug_step, ckpt_step) tuples ----------
# At debug_step 199k there is no checkpoint beyond 200k, so we use 200k only.
PROBES=(
    "39000   40000"
    "39000   200000"
    "79000   80000"
    "79000   200000"
    "119000  120000"
    "119000  200000"
    "199000  200000"
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
    # Compact tags for filenames (e.g., 39000 -> 39k, 200000 -> 200k)
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
        --headless \
        2>&1 | tee "${LOG}"
done

# ---------- aggregate ----------
SUMMARY="${OUTPUT_DIR}/summary.csv"
PIVOT="${OUTPUT_DIR}/summary_pivot.txt"

echo
echo "==================================================================="
echo "[probe] aggregating into ${SUMMARY}"
echo "==================================================================="

# Concatenate all per-call CSVs into one. Keep one header from the first file.
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

# Human-readable pivot table via Python — per-channel p90 across all probes.
PIVOT_SCRIPT=$(mktemp /tmp/probe_pivot_XXXXXX.py)
cat > "${PIVOT_SCRIPT}" <<'PYEOF'
import csv
import sys
summary_path = sys.argv[1]
rows = list(csv.DictReader(open(summary_path)))
keys = sorted({(r["channel_label"], r["checkpoint"]) for r in rows})
steps = sorted({int(r["debug_step"]) for r in rows})
PX4_REF = {
    "vx": 0.020, "vy": 0.020, "vz": 0.053,
    "yaw_rate": 0.300,
    "gimbal_yaw_rate": 0.400, "gimbal_pitch_rate": 0.400,
    "zoom_rate": 0.200,
}
print("Ticket 044 Slice-0 - per-channel p90 of |delta_action|, max over agents")
print("=" * 100)
header = f"{'channel':<20} {'checkpoint':<25}"
for s in steps:
    header += f"{s//1000}k".rjust(10)
header += f"{'PX4-strict':>14}"
print(header)
print("-" * 100)
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
    ref = PX4_REF.get(ch, float('nan'))
    line += f"{ref:>14.4f}"
    print(line)
print("=" * 100)
print()
print("Interpretation:")
print("  - p90 <= PX4-strict      -> Case A (ship strict)")
print("  - p90 >  PX4-strict      -> Case B (loosen or accept regression)")
print("  - p90 << PX4-strict      -> Case C (slew clip is a guardrail only)")
PYEOF
python3 "${PIVOT_SCRIPT}" "${SUMMARY}" > "${PIVOT}" 2>&1 || \
    "${ISAACLAB_ROOT}/_isaac_sim/python.sh" "${PIVOT_SCRIPT}" "${SUMMARY}" > "${PIVOT}" 2>&1
rm -f "${PIVOT_SCRIPT}"

T_ELAPSED=$(( $(date +%s) - T_START ))
echo
echo "==================================================================="
echo "[probe] DONE — total wall-time ${T_ELAPSED}s"
echo "  per-call CSVs : ${OUTPUT_DIR}/probe_step*.csv"
echo "  summary CSV   : ${SUMMARY}"
echo "  pivot table   : ${PIVOT}"
echo "==================================================================="
cat "${PIVOT}"
