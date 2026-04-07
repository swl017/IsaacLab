#!/usr/bin/env bash
# Run sysid_replicator.py with configurable trial count and aero level.
#
# Usage:
#   bash run_sysid_replicator.bash              # 1024 trials, aero-level 3
#   bash run_sysid_replicator.bash 2048          # 2048 trials, aero-level 3
#   bash run_sysid_replicator.bash 4096 3        # 4096 trials, aero-level 3
#   bash run_sysid_replicator.bash 2048 0        # 2048 trials, no aero

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

NUM_TRIALS="${1:-1024}"
AERO_LEVEL="${2:-3}"

echo "=== Sysid Replicator ==="
echo "  Trials:     ${NUM_TRIALS}"
echo "  Aero level: ${AERO_LEVEL}"
echo "  IsaacLab:   ${ISAACLAB_ROOT}"
echo ""

cd "${ISAACLAB_ROOT}"

python3 \
    "${SCRIPT_DIR}/sysid_replicator.py" \
    --headless \
    --num-trials "${NUM_TRIALS}" \
    --aero-level "${AERO_LEVEL}"
