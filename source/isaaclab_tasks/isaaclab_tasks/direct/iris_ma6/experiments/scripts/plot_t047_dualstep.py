#!/usr/bin/env python3
"""Plot the t047 dual-step checkpoint evaluation as two IROS-style 3x2 figures.

Multi-seed aggregation:
    For each config we run several eval seeds (each redraws the 1024 env initial
    conditions). Per seed we take a per-timestep central statistic across envs:
      - triangulation_rmse, sqrt_trace_sigma -> MEDIAN across envs (heavy-tailed;
        the mean is outlier-dominated, so median is the honest "typical env").
      - everything else -> MEAN across envs (visibility/tri_valid are fractions).
    The plotted line is the MEAN of that statistic across seeds; the shaded band
    is the across-seed MIN..MAX envelope = reproducibility (does a gap survive a
    different IC draw?). NOTE: only one TRAINING seed exists for t047, so this is
    IC-sampling variance, not training-seed variance.

Figures:
  Figure A  timeseries_t047_snapshots.pdf
    Snapshots at the difficulty each had just reached:
    40k@39k, 80k@79k, 120k@119k, 160k@159k, 200k@199k, 400k@400k.
  Figure B  timeseries_t047_final_sweep.pdf
    The final 400k policy swept across the same difficulties:
    400k@{39k, 79k, 119k, 159k, 199k, 399k}.

Both share the per-eval-step color order, so panel (x) of A vs panel (x) of B at
a matching color is the dual-step comparison.

Usage:
    python plot_t047_dualstep.py
"""

import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "t047_dualstep")

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

# (timeseries_key, subplot_title, ylabel, scale_factor)
METRICS = [
    ("triangulation_rmse", "(a) Triangulation RMSE", "RMSE (m)", 1.0),
    ("sqrt_trace_sigma", "(b) Position Uncertainty", "√tr(Σ) (m)", 1.0),
    ("tri_valid", "(c) Triangulation Visibility", "Visibility (%)", 100.0),
    ("distance_to_target", "(d) Agent–Target Distance", "Distance (m)", 1.0),
    ("viewing_angle", "(e) Viewing Angle", "Angle (deg)", 1.0),
]

# Heavy-tailed across envs -> use the per-step env MEDIAN as the per-seed stat.
MEDIAN_METRICS = {"triangulation_rmse", "sqrt_trace_sigma"}
SMOOTH_METRICS = {"triangulation_rmse", "sqrt_trace_sigma"}
SMOOTH_WINDOW = 5

# Per-eval-step colors (shared across both figures so panels pair by color).
_STEP_COLORS = [plt.cm.viridis(v) for v in np.linspace(0.0, 0.85, 6)]

FIG_A = {
    "output": os.path.join(RESULTS_DIR, "timeseries_t047_snapshots.pdf"),
    "suptitle": "t047 dual-step — snapshots at their matched curriculum difficulty",
    "experiments": [
        ("snap_040k_at_039k", "40k @ 39k", _STEP_COLORS[0]),
        ("snap_080k_at_079k", "80k @ 79k", _STEP_COLORS[1]),
        ("snap_120k_at_119k", "120k @ 119k", _STEP_COLORS[2]),
        ("snap_160k_at_159k", "160k @ 159k", _STEP_COLORS[3]),
        ("snap_200k_at_199k", "200k @ 199k", _STEP_COLORS[4]),
        ("final_at_400k", "400k @ 400k", _STEP_COLORS[5]),
    ],
}

FIG_B = {
    "output": os.path.join(RESULTS_DIR, "timeseries_t047_final_sweep.pdf"),
    "suptitle": "t047 dual-step — final 400k policy across curriculum difficulties",
    "experiments": [
        ("final_at_039k", "400k @ 39k", _STEP_COLORS[0]),
        ("final_at_079k", "400k @ 79k", _STEP_COLORS[1]),
        ("final_at_119k", "400k @ 119k", _STEP_COLORS[2]),
        ("final_at_159k", "400k @ 159k", _STEP_COLORS[3]),
        ("final_at_199k", "400k @ 199k", _STEP_COLORS[4]),
        ("final_at_399k", "400k @ 399k", _STEP_COLORS[5]),
    ],
}


def smooth(values, window=SMOOTH_WINDOW):
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _per_seed_series(ts, metric_key, scale):
    """Return the per-timestep central stat for one seed (median or mean), scaled.

    Drops the last (post-reset) timestep. Returns None if the metric is absent.
    """
    m = ts.get(metric_key)
    if m is None:
        return None
    stat = "median" if (metric_key in MEDIAN_METRICS and "median" in m) else "mean"
    return np.array(m[stat], dtype=float)[:-1] * scale


def load_seed_files(prefix):
    """Load all seed JSONs for a config prefix. Returns (time_s, list_of_ts_dicts)."""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_seed*.json")))
    # Back-compat: also accept a single un-suffixed file.
    if not paths and os.path.exists(os.path.join(RESULTS_DIR, f"{prefix}.json")):
        paths = [os.path.join(RESULTS_DIR, f"{prefix}.json")]
    ts_list, time_s = [], None
    for p in paths:
        with open(p) as f:
            ts = json.load(f).get("timeseries")
        if ts is None:
            continue
        ts_list.append(ts)
        if time_s is None:
            dt = ts.get("step_dt_seconds", 0.04)
            time_s = np.array(ts["timesteps"])[:-1] * dt
    return time_s, ts_list


def aggregate(ts_list, metric_key, scale):
    """Across-seed aggregate of the per-seed central series.

    Returns (line, lo, hi): mean across seeds, and min/max envelope across seeds.
    Series are trimmed to the shortest seed length before stacking.
    """
    series = [s for ts in ts_list if (s := _per_seed_series(ts, metric_key, scale)) is not None]
    if not series:
        return None, None, None
    n = min(len(s) for s in series)
    arr = np.stack([s[:n] for s in series], axis=0)  # (seeds, T)
    return arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def render(spec):
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.75))
    axes = axes.flatten()
    axes[5].set_visible(False)

    n_seeds_seen = set()
    for prefix, label, color in spec["experiments"]:
        time_s, ts_list = load_seed_files(prefix)
        if not ts_list:
            print(f"  SKIP: no seed files for {prefix}")
            continue
        n_seeds_seen.add(len(ts_list))

        for i, (metric_key, _t, _y, scale) in enumerate(METRICS):
            ax = axes[i]
            line, lo, hi = aggregate(ts_list, metric_key, scale)
            if line is None:
                continue
            t = time_s[:len(line)]
            if metric_key in SMOOTH_METRICS:
                line, lo, hi = smooth(line), smooth(lo), smooth(hi)
            ax.plot(t, line, color=color, linewidth=1.2, label=label)
            hi_c = np.minimum(hi, 100.0) if metric_key == "tri_valid" else hi
            ax.fill_between(t, np.maximum(lo, 0), hi_c, color=color, alpha=0.18)

    for i, (metric_key, title, ylabel, _s) in enumerate(METRICS):
        ax = axes[i]
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 20)
        ax.grid(True, alpha=0.3, which="both")
        if metric_key in ("triangulation_rmse", "sqrt_trace_sigma"):
            ax.set_yscale("log")
        elif metric_key == "tri_valid":
            ax.set_ylim(-5, 105)
        elif metric_key == "distance_to_target":
            ax.set_ylim(bottom=0)
        elif metric_key == "viewing_angle":
            ax.set_ylim(0, 150)
            ax.set_yticks([0, 45, 90, 135])
    axes[0].set_ylim(0.01, 30)  # RMSE dips to ~1-2 cm at easy regimes; show it
    axes[1].set_ylim(0.1, 30)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[5].set_visible(True)
        axes[5].axis("off")
        nseed = max(n_seeds_seen) if n_seeds_seen else 0
        axes[5].legend(handles, labels, loc="center", ncol=1,
                       title=f"ckpt @ step  (n={nseed} seeds)\nline=median(RMSE)/mean; band=seed min–max",
                       frameon=True, framealpha=0.9, fontsize=8, title_fontsize=7)

    fig.suptitle(spec["suptitle"], fontsize=9, y=1.0)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fig.savefig(spec["output"])
    plt.close()
    print(f"Saved: {spec['output']}")


def main():
    if not os.path.isdir(RESULTS_DIR):
        raise SystemExit(f"Results dir not found: {RESULTS_DIR}\n"
                         "Run run_t047_dualstep_eval.bash first.")
    render(FIG_A)
    render(FIG_B)


if __name__ == "__main__":
    main()
