#!/usr/bin/env bash
# Run sysid_replicator.py with configurable trial count, aero level, and plant mode.
#
# Usage:
#   bash run_sysid_replicator.bash                         # 1024 trials, aero-3, default plant
#   bash run_sysid_replicator.bash 2048                    # 2048 trials, aero-3, default plant
#   bash run_sysid_replicator.bash 4096 3                  # 4096 trials, aero-3, default plant
#   bash run_sysid_replicator.bash 2048 0                  # 2048 trials, no aero, default plant
#   bash run_sysid_replicator.bash 4096 0 pegasus          # 4096 trials, pegasus plant (ticket 040)
#   bash run_sysid_replicator.bash 4096 0 pegasus lag      # + EKF state lag (ticket 041)
#   bash run_sysid_replicator.bash 4096 0 pegasus lag asym # + asymmetric scoring (penalize iris ahead of PX4)
#
# Plant modes (ticket 040):
#   default — pre-040 racing-class numerics (k_f=1.2e-5, omega_max=5000, T/W ~82, tau=10ms
#             symmetric, quadratic body drag + optional Dryden/rotor effects via aero-level).
#   pegasus — PegasusSimulator IrisConfig parity (k_f=8.55e-6, omega_max=1100, T/W ~2.81,
#             tau ~ 0, linear-diag body drag (0.5, 0.3, 0), no wind/rotor effects regardless
#             of aero-level). Use this to match the iris_ma6 closed-loop step response to the
#             PX4 SITL CSVs (which were recorded against a Pegasus + PX4 plant).
#
# Output files are plant-tagged so a pegasus run does not overwrite the default baseline:
#   replicator_metrics.json          (default plant)
#   replicator_metrics_pegasus.json  (pegasus plant)
#   replicator_<test>.pdf            (default plant)
#   replicator_<test>_pegasus.pdf    (pegasus plant)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tuning(0) → controller(1) → iris_ma6(2) → direct(3) → isaaclab_tasks(4) →
# isaaclab_tasks(5) → source(6) → IsaacLab(7)  — isaaclab.sh lives at depth 7.
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"

NUM_TRIALS="${1:-1024}"
AERO_LEVEL="${2:-3}"
PLANT="${3:-default}"
# 4th arg: 'lag' to engage ticket-041 EKF state lag (18 ms attitude, 15 ms angular vel).
# 5th arg: 'asym' to engage asymmetric scoring (penalize iris-ma6 ahead of PX4).
EKF_LAG_FLAG=""
ASYMMETRIC_FLAG=""
for extra in "${4:-}" "${5:-}"; do
    case "${extra}" in
        lag)  EKF_LAG_FLAG="--ekf-lag" ;;
        asym) ASYMMETRIC_FLAG="--asymmetric-score" ;;
        "") ;;
        *) echo "ERROR: unknown extra flag '${extra}' (expected: 'lag' or 'asym')" >&2; exit 1 ;;
    esac
done

if [[ "${PLANT}" != "default" && "${PLANT}" != "pegasus" ]]; then
    echo "ERROR: PLANT must be 'default' or 'pegasus' (got: ${PLANT})" >&2
    exit 1
fi

echo "=== Sysid Replicator ==="
echo "  Trials:     ${NUM_TRIALS}"
echo "  Aero level: ${AERO_LEVEL}  $([[ "${PLANT}" == "pegasus" ]] && echo '(ignored — pegasus plant forces linear-diag drag)')"
echo "  Plant:      ${PLANT}"
echo "  EKF lag:    $([[ -n "${EKF_LAG_FLAG}" ]] && echo 'ON (ticket 041)' || echo 'off')"
echo "  Asym score: $([[ -n "${ASYMMETRIC_FLAG}" ]] && echo 'ON (3:1 ahead:behind)' || echo 'off')"
echo "  IsaacLab:   ${ISAACLAB_ROOT}"
echo ""

cd "${ISAACLAB_ROOT}"

# Use isaaclab.sh -p so the Isaac Sim conda env (env_isaaclab) is active.
# Bare `python3` works only if the caller already sourced the env.
./isaaclab.sh -p \
    "${SCRIPT_DIR}/sysid_replicator.py" \
    --headless \
    --num-trials "${NUM_TRIALS}" \
    --aero-level "${AERO_LEVEL}" \
    --plant "${PLANT}" \
    ${EKF_LAG_FLAG} \
    ${ASYMMETRIC_FLAG}
