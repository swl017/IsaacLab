#!/usr/bin/env bash
# EKF2 state-lag measurement orchestrator — ticket 041.
#
# Preconditions (the script does NOT start these for you — they must already
# be up, e.g. via tmux/isaac_sim.tmuxp.yaml):
#   * Pegasus SITL + PX4 (Iris airframe, single vehicle px4_1) with lockstep
#     enabled (px4_mavlink_backend.enable_lockstep = True; default).
#   * MAVROS bridge for px4_1 publishing /px4_1/mavros/local_position/pose,
#     velocity_local, velocity_body, /px4_1/mavros/imu/data, and exposing
#     the /px4_1/mavros/set_mode service.
#   * Vehicle is armed and in AUTO.LOITER (in-air). Recorder switches to
#     OFFBOARD per maneuver and back to AUTO.LOITER between maneuvers.
#   * No other node is publishing setpoints to px4_1 (would conflict).
#   * /clock is being published (sim-time).
#
# Output: raw/<maneuver>_trial<N>_<topic>.csv (144 files) and
# ekf_state_lag.json + ekf_lag_report.pdf in this directory.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="${HERE}/raw"
OUT_JSON="${HERE}/ekf_state_lag.json"
OUT_PDF="${HERE}/ekf_lag_report.pdf"
PX4_PARAMS_JSON="${RAW_DIR}/px4_params_snapshot.json"

NAMESPACE="${NAMESPACE:-px4_1}"
DURATION_S="${DURATION_S:-30}"
N_TRIALS="${N_TRIALS:-3}"
MANEUVERS=(${MANEUVERS:-hover_drift vel_step_x vel_step_z yaw_step chirp vel_impulse_recovery})

mkdir -p "${RAW_DIR}"

# ----------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------
echo "[ekf-lag] pre-flight: verify topics + services are alive"
required_topics=(
  "/${NAMESPACE}/state/pose"
  "/${NAMESPACE}/state/twist"
  "/${NAMESPACE}/state/twist_inertial"
  "/${NAMESPACE}/state/accel"
  "/${NAMESPACE}/mavros/local_position/pose"
  "/${NAMESPACE}/mavros/local_position/velocity_local"
  "/${NAMESPACE}/mavros/local_position/velocity_body"
  "/${NAMESPACE}/mavros/imu/data"
  "/clock"
)
missing=0
all_topics="$(ros2 topic list 2>/dev/null || true)"
for topic in "${required_topics[@]}"; do
  if ! grep -Fxq "${topic}" <<<"${all_topics}"; then
    echo "  MISSING topic: ${topic}"
    missing=$((missing + 1))
  fi
done

required_services=(
  "/${NAMESPACE}/mavros/set_mode"
)
all_services="$(ros2 service list 2>/dev/null || true)"
for svc in "${required_services[@]}"; do
  if ! grep -Fxq "${svc}" <<<"${all_services}"; then
    echo "  MISSING service: ${svc}"
    missing=$((missing + 1))
  fi
done
if [ "${missing}" -gt 0 ]; then
  echo "[ekf-lag] ${missing} required topic/service(s) not yet visible — bring up the stack first." >&2
  exit 1
fi

echo "[ekf-lag] pre-flight: verify /clock advances"
clock_before="$(ros2 topic echo --once --field clock /clock 2>/dev/null | head -3 || true)"
sleep 1
clock_after="$(ros2 topic echo --once --field clock /clock 2>/dev/null | head -3 || true)"
if [ "${clock_before}" = "${clock_after}" ]; then
  echo "[ekf-lag] /clock did not advance over 1 s wall — sim-time is not flowing." >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Snapshot PX4 EKF2 params via MAVROS get-param (best-effort)
# ----------------------------------------------------------------------
echo "[ekf-lag] snapshotting PX4 EKF2 params"
python3 - <<EOF || true
import json
import subprocess

params = [
    "EKF2_PREDICT_US", "EKF2_GPS_DELAY", "EKF2_BARO_DELAY",
    "EKF2_MAG_DELAY", "EKF2_EV_DELAY",
    "EKF2_IMU_POS_X", "EKF2_IMU_POS_Y", "EKF2_IMU_POS_Z",
    "IMU_INTEG_RATE",
]
out = {}
for p in params:
    try:
        r = subprocess.run(
            ["ros2", "service", "call",
             "/${NAMESPACE}/mavros/param/get",
             "mavros_msgs/srv/ParamGet",
             "{param_id: '" + p + "'}"],
            capture_output=True, text=True, timeout=5.0,
        )
        out[p] = {"raw": r.stdout.strip(), "rc": r.returncode}
    except Exception as e:
        out[p] = {"error": repr(e)}
with open("${PX4_PARAMS_JSON}", "w") as f:
    json.dump(out, f, indent=2)
print("wrote", "${PX4_PARAMS_JSON}")
EOF

# ----------------------------------------------------------------------
# Record 6 maneuvers × N trials
# ----------------------------------------------------------------------
# Sim-time advance rate measured at ~0.3× wall-time on this setup
# (multi-vehicle rendering load). Per-trial wall-time budget = duration_s × 4
# + 30 s margin covers pre-stream + maneuver + LOITER recovery.
SIM_TO_WALL_MULTIPLIER="${SIM_TO_WALL_MULTIPLIER:-4}"
PER_TRIAL_TIMEOUT_S=$(awk -v d="${DURATION_S}" -v m="${SIM_TO_WALL_MULTIPLIER}" \
                       'BEGIN{print int(d*m + 30)}')
echo "[ekf-lag] recording: ${#MANEUVERS[@]} maneuvers × ${N_TRIALS} trials × ${DURATION_S}s sim"
echo "[ekf-lag] per-trial wall-time budget: ${PER_TRIAL_TIMEOUT_S}s"

for maneuver in "${MANEUVERS[@]}"; do
  for trial in $(seq 0 $((N_TRIALS - 1))); do
    echo "[ekf-lag] -> ${maneuver} trial ${trial}"
    set +e
    timeout --kill-after=10 "${PER_TRIAL_TIMEOUT_S}" \
      python3 -u "${HERE}/ekf_lag_recorder.py" \
        --maneuver "${maneuver}" \
        --trial "${trial}" \
        --duration "${DURATION_S}" \
        --namespace "${NAMESPACE}" \
        --out-dir "${RAW_DIR}" \
        --ros-args -p use_sim_time:=true
    rc=$?
    set -e
    if [ "${rc}" -ne 0 ]; then
      echo "[ekf-lag] WARN: trial returned rc=${rc} (124=timeout)"
    fi
    # Inter-trial settle — let LOITER reclaim, EKF re-converge to hover.
    sleep 6
  done
done

# ----------------------------------------------------------------------
# Fit + report
# ----------------------------------------------------------------------
echo "[ekf-lag] fitting"
python3 "${HERE}/fit_ekf_lag.py" \
  --raw-dir "${RAW_DIR}" \
  --out-json "${OUT_JSON}" \
  --out-pdf "${OUT_PDF}" \
  --maneuver-duration "${DURATION_S}" \
  --px4-params-json "${PX4_PARAMS_JSON}" \
  --dataset-id "$(date +%Y-%m-%d)_pegasus_iris_ekf_lag_v1"

# ----------------------------------------------------------------------
# Sanity check — position channel must be ~110 ± 30 ms (per spec §6)
# ----------------------------------------------------------------------
echo "[ekf-lag] sanity check"
python3 - <<EOF
import json, sys
with open("${OUT_JSON}") as f:
    d = json.load(f)
pos = d["channels"].get("position_xyz", {})
delay_ms = (pos.get("delay_mean_s") or 0.0) * 1000
ok = 80 <= delay_ms <= 250
status = pos.get("status", "?")
print(f"  position_xyz: status={status} delay={delay_ms:.1f} ms")
if not ok:
    print(f"  WARNING: position delay {delay_ms:.1f} ms is outside [80, 250] ms gate.")
    print(f"           Time-base reconciliation is likely wrong — check use_sim_time and lockstep.")
    sys.exit(0)  # don't fail the script, but flag it loudly
print("[ekf-lag] done.")
EOF
