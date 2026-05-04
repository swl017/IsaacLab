#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-hoc plot of evaluate.py's `action_diagnostics` block.

Produces two figures:
  <stem>_traces.png  — per-axis × per-agent time series (one line per recorded env).
  <stem>_stats.png   — per-axis bar charts of action RMS, ΔRMS, and dominant FFT freq.

Usage:
    python plot_action_diagnostics.py path/to/diag.json
    python plot_action_diagnostics.py diag1.json diag2.json --label-from-stem  # overlay multiple
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_diag(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    if "action_diagnostics" not in d:
        raise ValueError(f"{path}: missing 'action_diagnostics' block")
    return d["action_diagnostics"]


def plot_traces(ad: dict, out_path: str, title_suffix: str = "") -> None:
    """7×A grid: per-axis time series, one line per recorded env."""
    axes_names: List[str] = ad["axis_names"]
    units: List[str] = ad["axis_units"]
    dt = float(ad["step_dt_seconds"])
    rate = float(ad["policy_rate_hz"])
    agents = sorted(ad["agents"].keys())

    # Skip if no traces
    if not any("trace" in ad["agents"][aid] for aid in agents):
        print(f"  skipping traces plot (no trace arrays in JSON)")
        return

    n_axes = len(axes_names)
    n_agents = len(agents)
    fig, axs = plt.subplots(n_axes, n_agents, figsize=(5.5 * n_agents, 1.8 * n_axes),
                            sharex=True, squeeze=False)

    for col, aid in enumerate(agents):
        a = ad["agents"][aid]
        if "trace" not in a:
            continue
        traces = a["trace"]  # list of (T, 7) lists
        for row, (ax_name, unit) in enumerate(zip(axes_names, units)):
            ax = axs[row, col]
            for env_i, tr in enumerate(traces):
                arr = np.asarray(tr, dtype=np.float32)  # (T, 7)
                t = np.arange(arr.shape[0]) * dt
                ax.plot(t, arr[:, row], lw=0.6, alpha=0.55,
                        label=f"env{env_i}" if (row == 0 and col == 0) else None)
            # Per-axis ΔRMS and peak Δfreq annotations
            drms = a["action_delta_rms"][ax_name]
            peak = a["dominant_freq_delta_hz"][ax_name]
            ax.set_ylabel(f"{ax_name}\n[{unit}]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            ax.text(0.99, 0.95,
                    f"ΔRMS={drms:.2f} {unit}/s\npeak Δf={peak:.1f} Hz",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
            if row == 0:
                ax.set_title(aid, fontsize=10)
            if row == n_axes - 1:
                ax.set_xlabel("time [s]", fontsize=8)

    nyq = rate / 2.0
    fig.suptitle(
        f"Per-axis action command traces{(' — ' + title_suffix) if title_suffix else ''}\n"
        f"policy rate {rate:.1f} Hz (Nyquist {nyq:.1f} Hz), {len(traces)} envs sampled",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_stats(diags: List[Tuple[str, dict]], out_path: str) -> None:
    """4-panel bar chart of action_rms, action_delta_rms, dominant_freq_hz, dominant_freq_delta_hz.

    Bars are grouped per axis; one bar per (run, agent).
    """
    # Use first run's axis ordering as canonical
    label0, ad0 = diags[0]
    axes_names: List[str] = ad0["axis_names"]
    units: List[str] = ad0["axis_units"]
    rate = float(ad0["policy_rate_hz"])
    nyq = rate / 2.0

    stat_keys = [
        ("action_rms",              "action RMS",          "[axis units]"),
        ("action_delta_rms",        "Δaction RMS",         "[axis units / s]"),
        ("dominant_freq_hz",        "dominant freq of a",  "[Hz]"),
        ("dominant_freq_delta_hz",  "dominant freq of Δa", "[Hz]"),
    ]

    # Collect series per stat: list of (label, agent_id, values_per_axis)
    series: Dict[str, List[Tuple[str, str, List[float]]]] = {k: [] for k, _, _ in stat_keys}
    for run_label, ad in diags:
        for aid in sorted(ad["agents"].keys()):
            for stat_key, _, _ in stat_keys:
                vals = [ad["agents"][aid][stat_key].get(ax, np.nan) for ax in axes_names]
                lbl = f"{run_label}/{aid}" if len(diags) > 1 else aid
                series[stat_key].append((run_label, aid, vals))
                # second tuple element is agent_id; use lbl when plotting
                series[stat_key][-1] = (lbl, aid, vals)

    n_axes = len(axes_names)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    for panel_idx, (stat_key, title, ylab) in enumerate(stat_keys):
        ax = axs[panel_idx // 2, panel_idx % 2]
        bars_for_panel = series[stat_key]
        n_bars = len(bars_for_panel)
        x = np.arange(n_axes)
        width = 0.8 / max(n_bars, 1)
        for i, (lbl, aid, vals) in enumerate(bars_for_panel):
            offset = (i - (n_bars - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=lbl, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(axes_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if "freq" in stat_key:
            ax.axhline(nyq, color="red", lw=0.8, ls="--", alpha=0.6,
                       label=f"Nyquist {nyq:.1f} Hz" if panel_idx == 2 else None)
        if panel_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    title_suffix = ", ".join(label for label, _ in diags) if len(diags) > 1 else diags[0][0]
    fig.suptitle(f"Action diagnostics — {title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="evaluate.py JSON file(s) with action_diagnostics")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: alongside first input)")
    parser.add_argument("--label-from-stem", action="store_true",
                        help="Use input filename stem as run label (auto-on for >1 input)")
    args = parser.parse_args()

    diags: List[Tuple[str, dict]] = []
    for p in args.inputs:
        ad = load_diag(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        diags.append((stem, ad))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.inputs[0]))
    os.makedirs(out_dir, exist_ok=True)

    # Stats plot — supports overlay across multiple runs
    if len(diags) > 1:
        stats_path = os.path.join(out_dir, "action_diag_stats_compare.png")
    else:
        stem = diags[0][0]
        stats_path = os.path.join(out_dir, f"{stem}_stats.png")
    plot_stats(diags, stats_path)

    # Traces plot — only first input (overlay across runs would be unreadable)
    stem = diags[0][0]
    traces_path = os.path.join(out_dir, f"{stem}_traces.png")
    plot_traces(diags[0][1], traces_path, title_suffix=stem)


if __name__ == "__main__":
    main()
