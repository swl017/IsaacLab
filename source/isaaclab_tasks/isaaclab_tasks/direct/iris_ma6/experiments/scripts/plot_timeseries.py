#!/usr/bin/env python3
"""Plot per-timestep metrics within an episode (3x2 subplots).

Shows how RMSE, position uncertainty, visibility, agent-target distance,
viewing angle, and CBF penalty evolve over time within an episode, averaged
across envs/episodes. Supports overlaying multiple experiments for comparison.

Requires evaluation JSONs with a "timeseries" key (produced by evaluate.py
with TimeseriesTracker enabled).

Usage:
    python plot_timeseries.py
    python plot_timeseries.py --results-dir path/to/outputs --output timeseries.pdf
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "outputs", "timeseries.pdf")

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# Experiments to plot: (json_filename, display_label, color, linestyle)
# Update for iris_ma6 experiments.
EXPERIMENTS = [
    ("eval_a1_with_aoi.json", "a1_with_aoi (default)", "#1f77b4", "-"),
]

# Metrics to plot: (timeseries_key, subplot_title, ylabel, scale_factor)
METRICS = [
    ("triangulation_rmse", "(a) Triangulation RMSE", "RMSE (m)", 1.0),
    ("sqrt_trace_sigma", "(b) Position Uncertainty", "\u221atr(\u03a3) (m)", 1.0),
    ("tri_valid", "(c) Triangulation Visibility", "Visibility (%)", 100.0),
    ("distance_to_target", "(d) Agent\u2013Target Distance", "Distance (m)", 1.0),
    ("viewing_angle", "(e) Viewing Angle", "Angle (deg)", 1.0),
    ("cbf_penalty", "(f) CBF Penalty", "Penalty", 1.0),
]

SMOOTH_METRICS = {"triangulation_rmse", "sqrt_trace_sigma"}
SMOOTH_WINDOW = 5


def smooth(values, window=SMOOTH_WINDOW):
    """Simple moving average."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def load_timeseries(json_path):
    """Load timeseries data from an evaluation JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    ts = data.get("timeseries")
    if ts is None:
        print(f"  WARNING: no 'timeseries' key in {json_path}")
        return None, None

    step_dt = ts.get("step_dt_seconds", 0.04)
    timesteps = np.array(ts["timesteps"])[:-1]  # drop last (post-reset) point
    time_s = timesteps * step_dt

    return time_s, ts


def plot_metric(ax, time_s, ts_data, metric_key, scale=1.0):
    """Extract mean and std for a single metric, apply scaling."""
    m = ts_data.get(metric_key)
    if m is None:
        return None, None, None

    mean = np.array(m["mean"])[:-1] * scale
    std = np.array(m["std"])[:-1] * scale
    return mean, mean - std, mean + std


def main():
    parser = argparse.ArgumentParser(description="Plot per-timestep evaluation metrics.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing evaluation JSON files")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output PDF path")
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="JSON filenames to plot (overrides EXPERIMENTS list)")
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(7.16, 7.5))
    axes = axes.flatten()

    experiments = EXPERIMENTS
    if args.experiments:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        styles = ["-", "--", "-.", ":", "-", "--"]
        experiments = [
            (fn, fn.replace("eval_", "").replace(".json", ""),
             colors[i % len(colors)], styles[i % len(styles)])
            for i, fn in enumerate(args.experiments)
        ]

    for json_fn, label, color, ls in experiments:
        json_path = os.path.join(args.results_dir, json_fn)
        if not os.path.exists(json_path):
            print(f"  SKIP: {json_path} not found")
            continue

        time_s, ts_data = load_timeseries(json_path)
        if ts_data is None:
            continue

        for i, (metric_key, title, ylabel, scale) in enumerate(METRICS):
            ax = axes[i]
            mean, lo, hi = plot_metric(ax, time_s, ts_data, metric_key, scale)
            if mean is None:
                continue

            if metric_key in SMOOTH_METRICS:
                mean, lo, hi = smooth(mean), smooth(lo), smooth(hi)

            ax.plot(time_s, mean, color=color, linestyle=ls, linewidth=1.2, label=label)
            hi_clamped = np.minimum(hi, 100.0) if metric_key == "tri_valid" else hi
            ax.fill_between(time_s, np.maximum(lo, 0), hi_clamped, color=color, alpha=0.15)

    # Configure axes
    for i, (metric_key, title, ylabel, scale) in enumerate(METRICS):
        ax = axes[i]
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 20)
        ax.grid(True, alpha=0.3, which="both")
        if metric_key in ("triangulation_rmse", "sqrt_trace_sigma"):
            ax.set_yscale("log")
            ax.set_ylim(0.1, 30)
        elif metric_key == "tri_valid":
            ax.set_ylim(-5, 105)
        elif metric_key == "distance_to_target":
            ax.set_ylim(bottom=0)
        elif metric_key == "viewing_angle":
            ax.set_ylim(0, 150)
            ax.set_yticks([0, 45, 90, 135])
        elif metric_key == "cbf_penalty":
            ax.set_ylim(bottom=0)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 3),
                   frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 1.0), fontsize=7)

    fig.tight_layout(h_pad=1.0, w_pad=0.8, rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
