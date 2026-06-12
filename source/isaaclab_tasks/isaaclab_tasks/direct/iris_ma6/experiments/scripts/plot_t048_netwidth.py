#!/usr/bin/env python3
"""Overlay the three t048 net-width final (200k) policies on one IROS 3x2 figure.

Single eval seed per config (the runs are single training seed). Heavy-tailed
metrics (RMSE, sqrt_trace_sigma) use the per-step env MEDIAN with a p25-p75 band;
the bounded metrics use env MEAN with a +/-1 sigma band.

Inputs: baseline.json / mid.json / wide.json in outputs/t048_netwidth/
        (from run_t048_netwidth_eval.bash).
Output: timeseries_t048_netwidth.pdf
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "t048_netwidth")
OUTPUT = os.path.join(RESULTS_DIR, "timeseries_t048_netwidth.pdf")

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

METRICS = [
    ("triangulation_rmse", "(a) Triangulation RMSE", "RMSE (m)", 1.0),
    ("sqrt_trace_sigma", "(b) Position Uncertainty", "√tr(Σ) (m)", 1.0),
    ("tri_valid", "(c) Triangulation Visibility", "Visibility (%)", 100.0),
    ("distance_to_target", "(d) Agent–Target Distance", "Distance (m)", 1.0),
    ("viewing_angle", "(e) Viewing Angle", "Angle (deg)", 1.0),
]
MEDIAN_METRICS = {"triangulation_rmse", "sqrt_trace_sigma"}
SMOOTH_METRICS = {"triangulation_rmse", "sqrt_trace_sigma"}
SMOOTH_WINDOW = 5

EXPERIMENTS = [
    ("baseline.json", "baseline (64/64)", "#7f7f7f"),
    ("mid.json", "mid (128/128)", "#1f77b4"),
    ("wide.json", "wide (256/256)", "#d62728"),
]


def smooth(v, w=SMOOTH_WINDOW):
    if len(v) <= w:
        return v
    return np.convolve(v, np.ones(w) / w, mode="same")


def series(ts, key, scale):
    """Return (line, lo, hi) for one metric: env-median + p25/p75 for heavy-tailed,
    env-mean +/- std otherwise. Drops the last (post-reset) timestep."""
    m = ts.get(key)
    if m is None:
        return None, None, None
    if key in MEDIAN_METRICS and "median" in m:
        line = np.array(m["median"])[:-1] * scale
        lo = np.array(m.get("p25", m["median"]))[:-1] * scale
        hi = np.array(m.get("p75", m["median"]))[:-1] * scale
    else:
        line = np.array(m["mean"])[:-1] * scale
        sd = np.array(m["std"])[:-1] * scale
        lo, hi = line - sd, line + sd
    return line, lo, hi


def main():
    if not os.path.isdir(RESULTS_DIR):
        raise SystemExit(f"Results dir not found: {RESULTS_DIR}\nRun run_t048_netwidth_eval.bash first.")
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.75))
    axes = axes.flatten()
    axes[5].set_visible(False)

    for fname, label, color in EXPERIMENTS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        ts = json.load(open(path)).get("timeseries")
        if ts is None:
            continue
        dt = ts.get("step_dt_seconds", 0.04)
        t = np.array(ts["timesteps"])[:-1] * dt
        for i, (key, _title, _yl, scale) in enumerate(METRICS):
            ax = axes[i]
            line, lo, hi = series(ts, key, scale)
            if line is None:
                continue
            n = min(len(t), len(line))
            tt, line, lo, hi = t[:n], line[:n], lo[:n], hi[:n]
            if key in SMOOTH_METRICS:
                line, lo, hi = smooth(line), smooth(lo), smooth(hi)
            ax.plot(tt, line, color=color, lw=1.3, label=label)
            hi_c = np.minimum(hi, 100.0) if key == "tri_valid" else hi
            ax.fill_between(tt, np.maximum(lo, 0), hi_c, color=color, alpha=0.12)

    for i, (key, title, ylabel, _s) in enumerate(METRICS):
        ax = axes[i]
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 20); ax.grid(True, alpha=0.3, which="both")
        if key in ("triangulation_rmse", "sqrt_trace_sigma"):
            ax.set_yscale("log")
        elif key == "tri_valid":
            ax.set_ylim(-5, 105)
        elif key == "distance_to_target":
            ax.set_ylim(bottom=0)
        elif key == "viewing_angle":
            ax.set_ylim(0, 150); ax.set_yticks([0, 45, 90, 135])
    axes[0].set_ylim(0.01, 30)
    axes[1].set_ylim(0.1, 30)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[5].set_visible(True); axes[5].axis("off")
        axes[5].legend(handles, labels, loc="center", title="MAPPO-RNN width @ 200k\n(line=median RMSE/mean; band=IQR/σ)",
                       frameon=True, framealpha=0.9, fontsize=8, title_fontsize=7)

    fig.suptitle("t048 network-width sweep — final (200k) policies, step-200k difficulty", fontsize=9, y=1.0)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fig.savefig(OUTPUT)
    plt.close()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
