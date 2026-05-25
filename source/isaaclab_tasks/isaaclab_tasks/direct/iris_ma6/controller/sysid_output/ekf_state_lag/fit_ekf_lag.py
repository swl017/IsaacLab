#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""EKF2 state-lag fit — ticket 041.

Reads raw/<maneuver>_trial<N>_<topic>.csv files written by ekf_lag_recorder.py,
fits per-channel delay via xcorr / step_50 / chirp_phase_delay, and writes
ekf_state_lag.json + ekf_lag_report.pdf.

Pure-Python; runs under any Python with numpy, scipy, pandas, matplotlib.
No Isaac Lab / ROS dependency at fit time.

Usage::

    python3 fit_ekf_lag.py \\
        --raw-dir raw/ \\
        --out-json ekf_state_lag.json \\
        --out-pdf ekf_lag_report.pdf

See ekf_state_lag_spec.md §4–7 for fit-method definitions, validation gates,
and the JSON schema this script produces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

# matplotlib is imported lazily — PDF generation is optional (skip with --no-pdf).


# ----------------------------------------------------------------------
# Schema constants
# ----------------------------------------------------------------------

SCHEMA_VERSION = 1

# Per-channel "expected delay" range (seconds) — a *sanity* check, not a
# correctness gate. PX4's EKF2 is lag-compensated: the output state is
# extrapolated forward from delayed measurements to t=now, so the observable
# output-vs-truth delay in lockstep SITL is typically << EKF2_GPS_DELAY.
# We allow [-50, +250] ms on all channels (the negative end catches gross
# time-base inversion; the positive end catches "no lag compensation").
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "position_xyz":           (-0.050, 0.250),
    "velocity_world_xyz":     (-0.050, 0.250),
    "velocity_body_xyz":      (-0.050, 0.250),
    "attitude_roll":          (-0.050, 0.250),
    "attitude_pitch":         (-0.050, 0.250),
    "attitude_yaw":           (-0.050, 0.250),
    "body_rate_xyz":          (-0.050, 0.100),
    "linear_acceleration_xyz": (-0.050, 0.250),
}

XCORR_PEAK_GATE = 0.8
# Per-axis gate: when a maneuver doesn't excite a given axis, the truth and
# EKF signals on that axis are both near-zero noise. Their xcorr fits noise
# correlations and emits garbage delay numbers that pollute the channel mean.
# Require per-axis peak ≥ 0.95 AND signal std ≥ a channel-specific floor
# before including that axis in the aggregate. The std floor distinguishes
# real excitation from slow drift / noise.
PER_AXIS_PEAK_GATE = 0.95
PER_CHANNEL_EXCITATION_FLOOR: Dict[str, float] = {
    "position_xyz":           0.5,    # m
    "velocity_world_xyz":     0.3,    # m/s
    "velocity_body_xyz":      0.3,    # m/s
    "attitude_roll":          0.003,  # rad (vel_step_x produces ~0.004 rad roll)
    "attitude_pitch":         0.03,   # rad (vel_step_x produces ~0.04 rad pitch)
    "attitude_yaw":           0.05,   # rad (yaw_step produces ~0.24 rad yaw)
    "body_rate_xyz":          0.1,    # rad/s
    "linear_acceleration_xyz": 0.05,  # m/s² — IMU spec-force has low signal during gentle maneuvers
}
# Per-channel xcorr peak gate override (defaults to PER_AXIS_PEAK_GATE if unset).
# linear_acceleration has lower SNR — small specific-force magnitudes during
# our maneuvers + noisy dv/dt-derived truth — so a lower peak gate is honest.
PER_CHANNEL_PEAK_GATE: Dict[str, float] = {
    "linear_acceleration_xyz": 0.65,
}
SEARCH_LAG_S_MIN = -0.050
SEARCH_LAG_S_MAX = +0.500
RESAMPLE_HZ = 200.0


# ----------------------------------------------------------------------
# Maneuver → fit-method routing (per spec §4.4)
# ----------------------------------------------------------------------

@dataclass
class ChannelPlan:
    """Pairing of (truth_topic, truth_cols, ekf_topic, ekf_cols, per-axis labels)."""
    name: str
    truth_topic: str
    truth_cols: List[str]
    ekf_topic: str
    ekf_cols: List[str]
    axes: List[str]
    primary_maneuvers: List[str]
    secondary_maneuvers: List[str] = field(default_factory=list)
    chirp_maneuver: Optional[str] = None  # if set, also run chirp_phase_delay
    # Optional auxiliary topic used to transform the truth signal before
    # comparison. Currently only used by linear_acceleration_xyz to fetch
    # the orientation quaternion needed to rotate inertial-frame accel
    # into body frame + add gravity (matching the IMU's spec-force output).
    truth_aux_topic: Optional[str] = None
    truth_transform: Optional[str] = None  # name of the transform in TRUTH_TRANSFORMS


def build_channel_plans() -> List[ChannelPlan]:
    return [
        ChannelPlan(
            name="position_xyz",
            truth_topic="state_pose",
            truth_cols=["px", "py", "pz"],
            ekf_topic="mavros_local_position_pose",
            ekf_cols=["px", "py", "pz"],
            axes=["x", "y", "z"],
            primary_maneuvers=["vel_step_x", "vel_step_z"],
        ),
        ChannelPlan(
            name="velocity_world_xyz",
            truth_topic="state_twist_inertial",
            truth_cols=["vx_world", "vy_world", "vz_world"],
            ekf_topic="mavros_local_position_velocity_local",
            ekf_cols=["vx_world", "vy_world", "vz_world"],
            axes=["x", "y", "z"],
            primary_maneuvers=["vel_step_x", "vel_step_z"],
        ),
        ChannelPlan(
            name="velocity_body_xyz",
            truth_topic="state_twist",
            truth_cols=["vx_body", "vy_body", "vz_body"],
            ekf_topic="mavros_local_position_velocity_body",
            ekf_cols=["vx_body", "vy_body", "vz_body"],
            axes=["x", "y", "z"],
            primary_maneuvers=["vel_step_x", "vel_step_z"],
        ),
        ChannelPlan(
            name="attitude_roll",
            truth_topic="state_pose",
            truth_cols=["__quat_to_roll__"],
            ekf_topic="mavros_imu_data",
            ekf_cols=["__quat_to_roll__"],
            axes=["roll"],
            primary_maneuvers=["vel_step_x"],
            chirp_maneuver="chirp",
        ),
        ChannelPlan(
            name="attitude_pitch",
            truth_topic="state_pose",
            truth_cols=["__quat_to_pitch__"],
            ekf_topic="mavros_imu_data",
            ekf_cols=["__quat_to_pitch__"],
            axes=["pitch"],
            primary_maneuvers=["vel_step_x", "vel_step_z"],
            chirp_maneuver="chirp",
        ),
        ChannelPlan(
            name="attitude_yaw",
            truth_topic="state_pose",
            truth_cols=["__quat_to_yaw__"],
            ekf_topic="mavros_imu_data",
            ekf_cols=["__quat_to_yaw__"],
            axes=["yaw"],
            primary_maneuvers=["yaw_step"],
        ),
        ChannelPlan(
            name="body_rate_xyz",
            truth_topic="state_twist",
            truth_cols=["wx_body", "wy_body", "wz_body"],
            ekf_topic="mavros_imu_data",
            ekf_cols=["wx_body", "wy_body", "wz_body"],
            axes=["x", "y", "z"],
            primary_maneuvers=["vel_step_x", "yaw_step"],
            chirp_maneuver="chirp",
        ),
        ChannelPlan(
            name="linear_acceleration_xyz",
            # Truth is inertial-frame, gravity-removed (pure dv/dt) from Pegasus.
            # The transform `accel_inertial_to_body_plus_g` rotates it into body
            # frame using the same-stamp orientation from state_pose and adds
            # +g_body so the result matches the IMU's spec-force convention.
            truth_topic="state_accel",
            truth_cols=["ax_world", "ay_world", "az_world"],
            truth_aux_topic="state_pose",
            truth_transform="accel_inertial_to_body_plus_g",
            ekf_topic="mavros_imu_data",
            ekf_cols=["ax_body", "ay_body", "az_body"],
            axes=["x", "y", "z"],
            primary_maneuvers=["vel_impulse_recovery"],
            chirp_maneuver="chirp",
        ),
    ]


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def load_raw(raw_dir: str, maneuver: str, trial: int, topic_slug: str) -> Optional[pd.DataFrame]:
    path = os.path.join(raw_dir, f"{maneuver}_trial{trial}_{topic_slug}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # Drop duplicate stamps (would break interp1d monotonicity).
    df = df.drop_duplicates(subset="stamp_s", keep="first").reset_index(drop=True)
    df = df.sort_values("stamp_s").reset_index(drop=True)
    return df


def discover_trials(raw_dir: str, maneuver: str, topic_slug: str) -> List[int]:
    """Return trial indices that have a CSV present for (maneuver, topic)."""
    prefix = f"{maneuver}_trial"
    suffix = f"_{topic_slug}.csv"
    trials = []
    if not os.path.isdir(raw_dir):
        return trials
    for fn in os.listdir(raw_dir):
        if fn.startswith(prefix) and fn.endswith(suffix):
            try:
                trial = int(fn[len(prefix):-len(suffix)])
                trials.append(trial)
            except ValueError:
                continue
    return sorted(set(trials))


# ----------------------------------------------------------------------
# Quaternion helpers — ROS convention (qx, qy, qz, qw); ZYX intrinsic euler.
# ----------------------------------------------------------------------

def quat_to_euler(df: pd.DataFrame) -> pd.DataFrame:
    qx, qy, qz, qw = df["qx"].to_numpy(), df["qy"].to_numpy(), df["qz"].to_numpy(), df["qw"].to_numpy()
    # Standard ZYX intrinsic (roll = X, pitch = Y, yaw = Z); same convention as
    # iris_ma6's frame_conventions.md and what MAVROS uses internally.
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    # Unwrap yaw on the raw samples so subsequent interpolation does not smooth
    # ±π discontinuities into spurious ramps.
    yaw = np.unwrap(yaw)
    return pd.DataFrame({"roll": roll, "pitch": pitch, "yaw": yaw})


G_ENU = 9.80665  # m/s², ENU convention: gravity acceleration acts -z_world.


def accel_inertial_to_body_plus_g(df_accel: pd.DataFrame, df_pose: pd.DataFrame) -> pd.DataFrame:
    """Rotate inertial-frame accel into body frame and add gravity-in-body.

    Pegasus state/accel is `dv_world/dt` (gravity removed). The IMU's
    linear_acceleration is the specific force `a_body - g_body`, where
    `g_body = R_world_to_body @ g_world` and `g_world = (0, 0, -G_ENU)`.
    For a stationary hovering vehicle (body z = world z up), this evaluates
    to `(0, 0, +G_ENU)` — matching the observed ~9.81 m/s² az_body at hover.

    Orientations are sampled from state_pose at each accel timestamp via
    nearest-neighbour matching on Header.stamp; quaternions are (qx, qy, qz, qw).
    Returns a DataFrame with `stamp_s` and `ax_body, ay_body, az_body` columns
    (so the rest of the pipeline can index by axis name).
    """
    # Build a quat lookup keyed by the pose timestamps.
    t_pose = df_pose["stamp_s"].to_numpy()
    qx_p = df_pose["qx"].to_numpy()
    qy_p = df_pose["qy"].to_numpy()
    qz_p = df_pose["qz"].to_numpy()
    qw_p = df_pose["qw"].to_numpy()
    # For each accel sample, find the nearest pose sample by stamp.
    t_acc = df_accel["stamp_s"].to_numpy()
    idx = np.searchsorted(t_pose, t_acc)
    idx = np.clip(idx, 0, len(t_pose) - 1)
    # Prefer the closer of idx-1 and idx.
    idx_lo = np.clip(idx - 1, 0, len(t_pose) - 1)
    pick_lo = (t_acc - t_pose[idx_lo]) < (t_pose[idx] - t_acc)
    idx_use = np.where(pick_lo, idx_lo, idx)
    qx = qx_p[idx_use]; qy = qy_p[idx_use]; qz = qz_p[idx_use]; qw = qw_p[idx_use]

    # Build R_body_to_world from quaternion (Hamilton, scalar-last). Standard formula.
    # Then transpose to get R_world_to_body since we want body-frame components.
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    # R_body_to_world rows (3x3) — column-major rotation matrix entries.
    r00 = 1 - 2*(yy + zz);  r01 = 2*(xy - wz);      r02 = 2*(xz + wy)
    r10 = 2*(xy + wz);      r11 = 1 - 2*(xx + zz);  r12 = 2*(yz - wx)
    r20 = 2*(xz - wy);      r21 = 2*(yz + wx);      r22 = 1 - 2*(xx + yy)

    # Inertial-frame accel + gravity-in-world (so subtraction by g_world matches IMU spec).
    # Specific force a_spec = a_inertial - g_world = a_inertial - (0, 0, -G) = a_inertial + (0, 0, +G).
    ax_w = df_accel["ax_world"].to_numpy()
    ay_w = df_accel["ay_world"].to_numpy()
    az_w = df_accel["az_world"].to_numpy() + G_ENU
    # Rotate world -> body by applying R_world_to_body = R_body_to_world.T (element-wise).
    ax_b = r00*ax_w + r10*ay_w + r20*az_w
    ay_b = r01*ax_w + r11*ay_w + r21*az_w
    az_b = r02*ax_w + r12*ay_w + r22*az_w
    out = pd.DataFrame({
        "stamp_s": t_acc,
        "recv_stamp_s": df_accel["recv_stamp_s"].to_numpy() if "recv_stamp_s" in df_accel.columns else t_acc,
        "ax_body": ax_b, "ay_body": ay_b, "az_body": az_b,
    })
    return out


TRUTH_TRANSFORMS: Dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    "accel_inertial_to_body_plus_g": accel_inertial_to_body_plus_g,
}


def resolve_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Resolve `__quat_to_{roll|pitch|yaw}__` magic strings to actual columns."""
    out = pd.DataFrame()
    quat_cache: Optional[pd.DataFrame] = None
    for c in cols:
        if c.startswith("__quat_to_") and c.endswith("__"):
            axis = c[len("__quat_to_"):-len("__")]
            if quat_cache is None:
                quat_cache = quat_to_euler(df)
            out[axis] = quat_cache[axis].to_numpy()
        else:
            out[c] = df[c].to_numpy()
    return out


# ----------------------------------------------------------------------
# Resampling onto a common grid
# ----------------------------------------------------------------------

def resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interp y(t) onto t_grid; extrapolate by clipping to endpoint values."""
    return np.interp(t_grid, t, y, left=y[0], right=y[-1])


def common_grid(t_truth: np.ndarray, t_ekf: np.ndarray, hz: float = RESAMPLE_HZ) -> np.ndarray:
    t_lo = max(t_truth.min(), t_ekf.min())
    t_hi = min(t_truth.max(), t_ekf.max())
    if t_hi - t_lo < 1.0:
        return np.empty(0)
    n = int(np.floor((t_hi - t_lo) * hz))
    return t_lo + np.arange(n) / hz


# ----------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------

def xcorr_delay(truth: np.ndarray, ekf: np.ndarray, dt: float) -> Tuple[float, float]:
    """Returns (delay_s, normalized_peak)."""
    t = truth - truth.mean()
    e = ekf - ekf.mean()
    t_norm = np.linalg.norm(t)
    e_norm = np.linalg.norm(e)
    if t_norm < 1e-9 or e_norm < 1e-9:
        return float("nan"), 0.0
    corr = correlate(e, t, mode="full") / (t_norm * e_norm)
    lags = correlation_lags(len(e), len(t), mode="full")
    # Restrict to the physical search window.
    lo_idx = int(np.searchsorted(lags, int(np.floor(SEARCH_LAG_S_MIN / dt))))
    hi_idx = int(np.searchsorted(lags, int(np.ceil(SEARCH_LAG_S_MAX / dt))))
    if hi_idx <= lo_idx + 1:
        return float("nan"), 0.0
    sub = corr[lo_idx:hi_idx]
    sub_lags = lags[lo_idx:hi_idx]
    k = int(np.argmax(sub))
    return float(sub_lags[k] * dt), float(sub[k])


def step_50_delay(t: np.ndarray, truth: np.ndarray, ekf: np.ndarray, command_edges_s: List[float]) -> float:
    """Delay between truth's 50% crossing and EKF's 50% crossing for each command edge.

    Returns mean across all edges. Edges must lie inside [t.min(), t.max()-window].
    """
    window = 4.0  # seconds after the command edge to look for the 50% crossing
    deltas: List[float] = []
    for edge in command_edges_s:
        m = (t >= edge) & (t <= edge + window)
        if m.sum() < 4:
            continue
        ts = t[m]
        tr = truth[m]
        ek = ekf[m]
        # Use baseline = first sample, peak = last sample within window.
        t_tr_50 = t_ek_50 = None
        for series_name, series in (("tr", tr), ("ek", ek)):
            base = series[0]
            peak = series[-1]
            if abs(peak - base) < 1e-6:
                # Channel didn't move on this edge — skip.
                deltas.append(float("nan"))
                break
            mid = 0.5 * (base + peak)
            # Find first crossing of `mid` going from base toward peak.
            sign = np.sign(peak - base)
            crossed = (sign * (series - mid)) >= 0
            if not crossed.any():
                deltas.append(float("nan"))
                break
            i = int(np.argmax(crossed))
            if i == 0:
                t_cross = ts[0]
            else:
                # Linear interp between samples i-1 and i.
                y0, y1 = series[i - 1], series[i]
                if abs(y1 - y0) < 1e-12:
                    t_cross = ts[i]
                else:
                    frac = (mid - y0) / (y1 - y0)
                    t_cross = ts[i - 1] + frac * (ts[i] - ts[i - 1])
            if series_name == "tr":
                t_tr_50 = t_cross
            else:
                t_ek_50 = t_cross
        else:
            # the inner for-loop completed without `break`
            deltas.append(t_ek_50 - t_tr_50)
            continue
    deltas = [d for d in deltas if np.isfinite(d)]
    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


def chirp_group_delay(t: np.ndarray, truth: np.ndarray, ekf: np.ndarray,
                      f_lo: float = 0.5, f_hi: float = 2.0) -> Tuple[float, float]:
    """Average group delay over [f_lo, f_hi]. Returns (delay_s, flatness_rel)."""
    n = len(t)
    if n < 64:
        return float("nan"), float("nan")
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    T = (truth - truth.mean())
    E = (ekf - ekf.mean())
    # Apply a window to reduce spectral leakage.
    win = np.hanning(n)
    Tf = np.fft.rfft(T * win)
    Ef = np.fft.rfft(E * win)
    freqs = np.fft.rfftfreq(n, dt)
    cross = Ef * np.conj(Tf)
    phase = np.unwrap(np.angle(cross))
    in_band = (freqs >= f_lo) & (freqs <= f_hi)
    if in_band.sum() < 4:
        return float("nan"), float("nan")
    delays = -phase[in_band] / (2 * np.pi * freqs[in_band])
    mean = float(np.mean(delays))
    if abs(mean) < 1e-9:
        flat = float("inf")
    else:
        flat = float(np.std(delays) / abs(mean))
    return mean, flat


# ----------------------------------------------------------------------
# Maneuver command-edge timestamps (for step_50)
# ----------------------------------------------------------------------

def command_edges_for(maneuver: str, duration_s: float) -> List[float]:
    """Times (offset from t0) at which the maneuver command changes."""
    if maneuver in ("vel_step_x", "vel_step_z"):
        # Blocks at 0, 5, 10, 15, 20, 25 s (within 30 s window).
        return [5.0, 10.0, 15.0, 20.0]
    if maneuver == "yaw_step":
        # +pulse at 0.0/0.5 and -pulse at 5.5/6.0 every 11 s.
        edges = []
        for k in range(int(duration_s // 11.0) + 1):
            base = 11.0 * k
            edges.extend([base + 0.0, base + 0.5, base + 5.5, base + 6.0])
        return [e for e in edges if e < duration_s - 1.0]
    return []


# ----------------------------------------------------------------------
# Channel fit (loops over trials, maneuvers, axes)
# ----------------------------------------------------------------------

@dataclass
class AxisResult:
    delay_xcorr_s: List[float] = field(default_factory=list)
    xcorr_peak: List[float] = field(default_factory=list)
    delay_step50_s: List[float] = field(default_factory=list)
    delay_chirp_s: List[float] = field(default_factory=list)
    chirp_flatness: List[float] = field(default_factory=list)
    residual_rmse: List[float] = field(default_factory=list)
    source_maneuvers: List[str] = field(default_factory=list)


def fit_channel(raw_dir: str, plan: ChannelPlan, duration_s: float) -> dict:
    per_axis_results: Dict[str, AxisResult] = {ax: AxisResult() for ax in plan.axes}

    all_maneuvers = list(dict.fromkeys(plan.primary_maneuvers + plan.secondary_maneuvers))
    if plan.chirp_maneuver and plan.chirp_maneuver not in all_maneuvers:
        all_maneuvers.append(plan.chirp_maneuver)

    truth_trials = discover_trials(raw_dir, all_maneuvers[0], plan.truth_topic)
    if not truth_trials:
        # No data — fall through to a 'skipped' result.
        pass

    for maneuver in all_maneuvers:
        trials = discover_trials(raw_dir, maneuver, plan.truth_topic)
        for trial in trials:
            df_truth = load_raw(raw_dir, maneuver, trial, plan.truth_topic)
            df_ekf = load_raw(raw_dir, maneuver, trial, plan.ekf_topic)
            if df_truth is None or df_ekf is None or len(df_truth) < 8 or len(df_ekf) < 8:
                continue
            # Apply truth-side transform if the plan defines one (e.g.
            # linear_acceleration_xyz needs to be rotated from inertial to
            # body and have gravity added before it can be compared against
            # the IMU's spec-force output). Replaces df_truth's payload
            # columns; the rest of the pipeline runs unchanged.
            if plan.truth_transform is not None:
                df_aux = load_raw(raw_dir, maneuver, trial, plan.truth_aux_topic)
                if df_aux is None or len(df_aux) < 8:
                    continue
                transform = TRUTH_TRANSFORMS[plan.truth_transform]
                df_truth = transform(df_truth, df_aux)
                # Repoint truth_cols and ekf_cols to the transformed payload.
                local_truth_cols = list(df_truth.columns[2:])  # skip stamp_s, recv_stamp_s
            else:
                local_truth_cols = plan.truth_cols
            truth_vals = resolve_columns(df_truth, local_truth_cols)
            ekf_vals = resolve_columns(df_ekf, plan.ekf_cols)
            t_truth = df_truth["stamp_s"].to_numpy()
            t_ekf = df_ekf["stamp_s"].to_numpy()
            t_grid = common_grid(t_truth, t_ekf)
            if len(t_grid) < 64:
                continue
            dt = 1.0 / RESAMPLE_HZ

            for axis, truth_col, ekf_col in zip(plan.axes, truth_vals.columns, ekf_vals.columns):
                tr = resample_to_grid(t_truth, truth_vals[truth_col].to_numpy(), t_grid)
                ek = resample_to_grid(t_ekf, ekf_vals[ekf_col].to_numpy(), t_grid)
                # Yaw already unwrapped on raw samples in quat_to_euler() —
                # re-unwrapping post-interpolation is a no-op.

                # xcorr is universally applicable, but only keep samples whose
                # per-axis peak passes PER_AXIS_PEAK_GATE AND whose truth signal
                # std exceeds the channel's minimum-excitation floor (skip axes
                # that the maneuver does not drive — their xcorr fits slow drift).
                d_xc, peak = xcorr_delay(tr, ek, dt)
                truth_std = float(np.std(tr))
                excitation_floor = PER_CHANNEL_EXCITATION_FLOOR.get(plan.name, 0.0)
                peak_gate = PER_CHANNEL_PEAK_GATE.get(plan.name, PER_AXIS_PEAK_GATE)
                excited = truth_std >= excitation_floor
                if np.isfinite(d_xc) and peak >= peak_gate and excited:
                    per_axis_results[axis].delay_xcorr_s.append(d_xc)
                    per_axis_results[axis].xcorr_peak.append(peak)
                    per_axis_results[axis].source_maneuvers.append(f"{maneuver}/trial{trial}")
                    # Residual after compensation: shift EKF back by d_xc, compare to truth.
                    shift = int(round(d_xc / dt))
                    if shift > 0 and shift < len(ek) - 8:
                        rmse = float(np.sqrt(np.mean((tr[:-shift] - ek[shift:]) ** 2)))
                        per_axis_results[axis].residual_rmse.append(rmse)

                # step_50 for step maneuvers.
                edges = command_edges_for(maneuver, duration_s)
                if edges:
                    # Re-index edges relative to the actual recording's t_grid start.
                    edge_abs = [t_grid[0] + e for e in edges]
                    d_s50 = step_50_delay(t_grid, tr, ek, edge_abs)
                    if np.isfinite(d_s50):
                        per_axis_results[axis].delay_step50_s.append(d_s50)

                # chirp_phase_delay for the chirp maneuver on chirp-capable channels.
                if plan.chirp_maneuver and maneuver == plan.chirp_maneuver:
                    d_ch, flat = chirp_group_delay(t_grid, tr, ek)
                    if np.isfinite(d_ch):
                        per_axis_results[axis].delay_chirp_s.append(d_ch)
                        per_axis_results[axis].chirp_flatness.append(flat)

    return _aggregate_axis_results(plan, per_axis_results)


def _aggregate_axis_results(plan: ChannelPlan, per_axis: Dict[str, AxisResult]) -> dict:
    # Aggregate per-axis to channel-level summary.
    axis_summary: Dict[str, dict] = {}
    aggregate_xcorr: List[float] = []
    aggregate_peak: List[float] = []
    aggregate_step50: List[float] = []
    aggregate_chirp: List[float] = []
    aggregate_chirp_flat: List[float] = []
    aggregate_residual: List[float] = []
    source_maneuvers: List[str] = []

    for axis, res in per_axis.items():
        axis_summary[axis] = {
            "delay_xcorr_mean_s": _safe_mean(res.delay_xcorr_s),
            "delay_xcorr_std_s": _safe_std(res.delay_xcorr_s),
            "xcorr_peak_mean": _safe_mean(res.xcorr_peak),
            "delay_step50_mean_s": _safe_mean(res.delay_step50_s),
            "delay_chirp_mean_s": _safe_mean(res.delay_chirp_s),
            "chirp_flatness_rel_mean": _safe_mean(res.chirp_flatness),
            "residual_rmse_mean": _safe_mean(res.residual_rmse),
            "n_xcorr_samples": len(res.delay_xcorr_s),
            "n_step50_samples": len(res.delay_step50_s),
            "n_chirp_samples": len(res.delay_chirp_s),
        }
        aggregate_xcorr.extend(res.delay_xcorr_s)
        aggregate_peak.extend(res.xcorr_peak)
        aggregate_step50.extend(res.delay_step50_s)
        aggregate_chirp.extend(res.delay_chirp_s)
        aggregate_chirp_flat.extend(res.chirp_flatness)
        aggregate_residual.extend(res.residual_rmse)
        source_maneuvers.extend(res.source_maneuvers)

    if not aggregate_xcorr:
        return {
            "status": "skipped",
            "reason": "no usable xcorr samples (no CSVs found or no excitation in window)",
            "fit_method": "xcorr",
            "source_maneuvers": list(dict.fromkeys(plan.primary_maneuvers + plan.secondary_maneuvers)),
            "per_axis": axis_summary,
        }

    delay_mean = float(np.mean(aggregate_xcorr))
    delay_std = float(np.std(aggregate_xcorr))
    delay_p50 = float(np.percentile(aggregate_xcorr, 50))
    delay_p95 = float(np.percentile(aggregate_xcorr, 95))
    delay_p99 = float(np.percentile(aggregate_xcorr, 99))
    xpeak_mean = float(np.mean(aggregate_peak)) if aggregate_peak else 0.0
    residual = float(np.mean(aggregate_residual)) if aggregate_residual else float("nan")
    step50_mean = float(np.mean(aggregate_step50)) if aggregate_step50 else float("nan")
    chirp_mean = float(np.mean(aggregate_chirp)) if aggregate_chirp else float("nan")
    chirp_flat = float(np.mean(aggregate_chirp_flat)) if aggregate_chirp_flat else float("nan")

    # Validation gates.
    notes: List[str] = []
    bounds = PHYSICAL_BOUNDS.get(plan.name)
    gate_status = "fit"
    channel_peak_gate = PER_CHANNEL_PEAK_GATE.get(plan.name, XCORR_PEAK_GATE)
    if xpeak_mean < channel_peak_gate:
        gate_status = "skipped"
        notes.append(f"xcorr_peak {xpeak_mean:.2f} < gate {channel_peak_gate}")
    if bounds is not None and not (bounds[0] <= delay_mean <= bounds[1]):
        # Don't drop, but flag it loudly.
        notes.append(
            f"delay_mean {delay_mean*1000:.1f} ms outside expected bounds "
            f"[{bounds[0]*1000:.0f}, {bounds[1]*1000:.0f}] ms — investigate"
        )
    # Note the lag-compensation finding for near-zero results (a real PX4
    # behavior, not a bug — but worth surfacing so downstream tickets don't
    # silently assume EKF lag where none exists).
    if abs(delay_mean) < 0.010:
        notes.append(
            f"delay_mean {delay_mean*1000:.1f} ms (≤10 ms) — within measurement "
            "resolution; consistent with PX4's lag-compensated EKF output in "
            "Pegasus lockstep SITL"
        )
    if np.isfinite(chirp_flat) and chirp_flat > 0.2:
        notes.append(
            f"chirp group-delay flatness {chirp_flat:.2f} > 0.2 — constant-delay model may be insufficient; "
            "consider escalating to a first-order channel model in a follow-up ticket"
        )

    result = {
        "status": gate_status,
        "delay_mean_s": delay_mean,
        "delay_std_s": delay_std,
        "delay_p50_s": delay_p50,
        "delay_p95_s": delay_p95,
        "delay_p99_s": delay_p99,
        "residual_rmse_after_compensation": residual,
        "fit_method": "xcorr",
        "fit_method_secondary": "step_50" if np.isfinite(step50_mean) else None,
        "fit_method_secondary_value_s": step50_mean if np.isfinite(step50_mean) else None,
        "fit_method_tertiary": "chirp_phase_delay" if np.isfinite(chirp_mean) else None,
        "fit_method_tertiary_value_s": chirp_mean if np.isfinite(chirp_mean) else None,
        "chirp_flatness_rel": chirp_flat if np.isfinite(chirp_flat) else None,
        "xcorr_peak": xpeak_mean,
        "source_maneuvers": sorted(set(source_maneuvers)),
        "per_axis": axis_summary,
    }
    if gate_status == "skipped":
        result["reason"] = "; ".join(notes) or "validation gate failed"
    if notes:
        result.setdefault("notes", []).extend(notes)
    return result


def _safe_mean(xs: List[float]) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def _safe_std(xs: List[float]) -> Optional[float]:
    return float(np.std(xs)) if xs else None


# ----------------------------------------------------------------------
# PDF report (lazy import; skip with --no-pdf)
# ----------------------------------------------------------------------

def write_pdf(raw_dir: str, plans: List[ChannelPlan], channel_results: Dict[str, dict],
              out_pdf: str, duration_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        for plan in plans:
            res = channel_results.get(plan.name, {})
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 11.0))
            fig.suptitle(f"{plan.name}    status={res.get('status', '?')}    "
                         f"delay_mean={res.get('delay_mean_s', float('nan')):.4f}s    "
                         f"xcorr_peak={res.get('xcorr_peak', float('nan')):.2f}",
                         fontsize=12)

            # Panel 1: overlay truth vs EKF on a trial that actually contributed
            # to the fit. Picks (maneuver, axis) by:
            #   - axis with the most xcorr samples (per `per_axis.n_xcorr_samples`)
            #   - first maneuver in `source_maneuvers` from that result
            # Falls back to plan.primary_maneuvers if no fit samples exist.
            ax = axes[0]
            shown = False

            # Pick the most-contributing axis (highest n_xcorr_samples).
            per_axis_d = res.get("per_axis", {})
            axes_ranked = sorted(per_axis_d.items(),
                                 key=lambda kv: kv[1].get("n_xcorr_samples", 0),
                                 reverse=True)
            preferred_axis = axes_ranked[0][0] if axes_ranked and axes_ranked[0][1].get("n_xcorr_samples", 0) > 0 else plan.axes[0]
            preferred_axis_idx = plan.axes.index(preferred_axis) if preferred_axis in plan.axes else 0

            # Pick maneuvers in priority order: source_maneuvers > chirp_maneuver > primary_maneuvers.
            source_list = res.get("source_maneuvers", [])
            source_maneuver_names = list(dict.fromkeys(s.split("/")[0] for s in source_list))
            maneuver_order = source_maneuver_names if source_maneuver_names else list(plan.primary_maneuvers)
            if plan.chirp_maneuver and plan.chirp_maneuver not in maneuver_order:
                maneuver_order.append(plan.chirp_maneuver)

            for maneuver in maneuver_order:
                trials = discover_trials(raw_dir, maneuver, plan.truth_topic)
                if not trials:
                    continue
                trial = trials[0]
                df_t = load_raw(raw_dir, maneuver, trial, plan.truth_topic)
                df_e = load_raw(raw_dir, maneuver, trial, plan.ekf_topic)
                if df_t is None or df_e is None:
                    continue
                # Apply truth_transform if the plan defines one (e.g.
                # linear_acceleration_xyz needs body-frame + gravity rotation).
                if plan.truth_transform is not None:
                    df_aux = load_raw(raw_dir, maneuver, trial, plan.truth_aux_topic)
                    if df_aux is None or len(df_aux) < 8:
                        continue
                    df_t_used = TRUTH_TRANSFORMS[plan.truth_transform](df_t, df_aux)
                    local_truth_cols = list(df_t_used.columns[2:])
                else:
                    df_t_used = df_t
                    local_truth_cols = plan.truth_cols

                tr_vals = resolve_columns(df_t_used, local_truth_cols)
                ek_vals = resolve_columns(df_e, plan.ekf_cols)
                # Clamp the preferred axis index to the available columns.
                col_idx = min(preferred_axis_idx, tr_vals.shape[1] - 1)
                t0 = df_t_used["stamp_s"].iloc[0]
                tt = df_t_used["stamp_s"].to_numpy() - t0
                te = df_e["stamp_s"].to_numpy() - t0
                ax.plot(tt, tr_vals.iloc[:, col_idx].to_numpy(),
                        label=f"truth/{preferred_axis}", lw=1.2)
                ax.plot(te, ek_vals.iloc[:, col_idx].to_numpy(),
                        label=f"ekf/{preferred_axis}", lw=1.2, alpha=0.85)
                title_extra = " (truth transformed)" if plan.truth_transform else ""
                ax.set_title(f"{maneuver} trial {trial} — axis={preferred_axis}{title_extra}")
                ax.set_xlabel("t (s)")
                ax.legend(fontsize=8)
                shown = True
                break
            if not shown:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")

            # Panel 2: per-axis summary table.
            ax = axes[1]
            ax.axis("off")
            per_axis = res.get("per_axis", {})
            lines = [f"channel: {plan.name}"]
            lines.append(f"source_maneuvers: {res.get('source_maneuvers', [])}")
            lines.append("")
            for axis, summary in per_axis.items():
                lines.append(f"  {axis}:")
                for k, v in summary.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.5f}")
                    else:
                        lines.append(f"    {k}: {v}")
                lines.append("")
            for note in res.get("notes", []):
                lines.append(f"NOTE: {note}")
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                    va="top", ha="left", family="monospace", fontsize=8)

            pdf.savefig(fig)
            plt.close(fig)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EKF2 state-lag fit (ticket 041)")
    parser.add_argument("--raw-dir", required=True, help="Directory with raw/*.csv recordings.")
    parser.add_argument("--out-json", required=True, help="Output JSON path (the canonical fit).")
    parser.add_argument("--out-pdf", default=None, help="Output PDF path (one page per channel). Pass empty to skip.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation entirely.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id (default: directory mtime).")
    parser.add_argument("--maneuver-duration", type=float, default=30.0, help="Per-maneuver duration in seconds (must match recorder).")
    parser.add_argument("--px4-params-json", default=None, help="Optional path to a JSON snapshot of PX4 EKF2 params.")
    args = parser.parse_args(argv)

    plans = build_channel_plans()
    raw_dir = os.path.abspath(args.raw_dir)

    channel_results: Dict[str, dict] = {}
    for plan in plans:
        channel_results[plan.name] = fit_channel(raw_dir, plan, duration_s=args.maneuver_duration)

    px4_params = {}
    if args.px4_params_json and os.path.isfile(args.px4_params_json):
        with open(args.px4_params_json) as f:
            px4_params = json.load(f)
    if not px4_params:
        # Document defaults from ekf2_params.c so consumers can compare.
        px4_params = {
            "EKF2_PREDICT_US": 10000,
            "EKF2_GPS_DELAY": 110.0,
            "EKF2_BARO_DELAY": 0.0,
            "EKF2_MAG_DELAY": 0.0,
            "EKF2_EV_DELAY": 0.0,
            "_note": "defaults from PX4-Autopilot/src/modules/ekf2/ekf2_params.c (not measured from running FCU)",
        }

    dataset_id = args.dataset_id or os.path.basename(raw_dir.rstrip("/").rstrip("\\"))
    output = {
        "dataset_id": dataset_id,
        "schema_version": SCHEMA_VERSION,
        "setup": {
            "raw_dir": raw_dir,
            "num_trials_per_maneuver": _guess_num_trials(raw_dir),
            "maneuver_duration_s": args.maneuver_duration,
        },
        "px4_params_snapshot": px4_params,
        "channels": channel_results,
        "notes": _collect_global_notes(channel_results),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2, sort_keys=False)
    print(f"wrote {args.out_json}")

    if not args.no_pdf:
        pdf_path = args.out_pdf or os.path.join(os.path.dirname(args.out_json), "ekf_lag_report.pdf")
        try:
            write_pdf(raw_dir, plans, channel_results, pdf_path, duration_s=args.maneuver_duration)
            print(f"wrote {pdf_path}")
        except Exception as e:
            print(f"PDF generation failed ({e!r}); JSON is still authoritative", file=sys.stderr)

    return 0


def _guess_num_trials(raw_dir: str) -> int:
    # Look for vel_step_x trials as a proxy.
    return len(discover_trials(raw_dir, "vel_step_x", "state_pose"))


def _collect_global_notes(channel_results: Dict[str, dict]) -> List[str]:
    notes: List[str] = []
    for ch, res in channel_results.items():
        for n in res.get("notes", []):
            notes.append(f"[{ch}] {n}")
        if res.get("status") == "skipped":
            notes.append(f"[{ch}] SKIPPED: {res.get('reason', 'unknown')}")
    return notes


if __name__ == "__main__":
    sys.exit(main())
