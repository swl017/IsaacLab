#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize burst dropout characteristics for parameter tuning.

Generates plots showing:
1. Markov state timeline (Good/Bad) for multiple envs
2. Dropout event raster (showing burst clusters)
3. Burst length distribution histogram
4. Good-run length distribution histogram
5. Comparison across 4 parameter configurations

Self-contained — does not require Isaac Sim.

Usage:
    conda run -n env_isaaclab python plot_burst_dropout.py
    conda run -n env_isaaclab python plot_burst_dropout.py --steps 3000 --envs 128
    conda run -n env_isaaclab python plot_burst_dropout.py --output /tmp/burst_analysis.png
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def simulate_burst(p_onset, p_recovery, good_drop, bad_drop,
                   num_envs, num_steps, device="cpu"):
    """Simulate Gilbert-Elliott burst dropout and return histories.

    Returns:
        state_history: (steps, envs) int, 0=Good 1=Bad
        dropout_history: (steps, envs) bool, True=dropped
    """
    state = torch.zeros(num_envs, dtype=torch.long, device=device)
    state_hist = []
    drop_hist = []

    for _ in range(num_steps):
        # Markov transition
        rand_t = torch.rand(num_envs, device=device)
        good_to_bad = (state == 0) & (rand_t < p_onset)
        bad_to_good = (state == 1) & (rand_t < p_recovery)
        state = torch.where(good_to_bad, torch.ones_like(state), state)
        state = torch.where(bad_to_good, torch.zeros_like(state), state)

        # Dropout sampling
        rand_d = torch.rand(num_envs, device=device)
        prob = torch.where(state == 0,
                           torch.tensor(good_drop, device=device),
                           torch.tensor(bad_drop, device=device))
        dropped = rand_d < prob

        state_hist.append(state.cpu().clone())
        drop_hist.append(dropped.cpu().clone())

    return torch.stack(state_hist).numpy(), torch.stack(drop_hist).numpy()


def compute_run_lengths(binary_seq):
    """Compute run lengths of 1s and 0s in a binary sequence."""
    burst_lens = []
    good_lens = []
    current_val = binary_seq[0]
    run_len = 1

    for i in range(1, len(binary_seq)):
        if binary_seq[i] == current_val:
            run_len += 1
        else:
            if current_val == 1:
                burst_lens.append(run_len)
            else:
                good_lens.append(run_len)
            current_val = binary_seq[i]
            run_len = 1

    # Final run
    if current_val == 1:
        burst_lens.append(run_len)
    else:
        good_lens.append(run_len)

    return burst_lens, good_lens


def plot_analysis(args):
    device = "cpu"

    configs = [
        {
            "label": "Default\np_on=0.01, p_rec=0.1\n(burst~10, good~100)",
            "p_onset": 0.01, "p_recovery": 0.1,
            "good_drop": 0.01, "bad_drop": 0.9,
        },
        {
            "label": "Frequent short\np_on=0.05, p_rec=0.2\n(burst~5, good~20)",
            "p_onset": 0.05, "p_recovery": 0.2,
            "good_drop": 0.01, "bad_drop": 0.9,
        },
        {
            "label": "Rare long\np_on=0.005, p_rec=0.05\n(burst~20, good~200)",
            "p_onset": 0.005, "p_recovery": 0.05,
            "good_drop": 0.01, "bad_drop": 0.9,
        },
        {
            "label": "i.i.d. 5%\n(baseline comparison)",
            "p_onset": 0.0, "p_recovery": 1.0,
            "good_drop": 0.05, "bad_drop": 0.05,
        },
    ]

    num_envs = args.envs
    num_steps = args.steps

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("Burst Dropout Characteristics — Gilbert-Elliott Model\n"
                 f"({num_steps} steps, {num_envs} envs, channel 0→1)",
                 fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(4, len(configs), hspace=0.5, wspace=0.3,
                           top=0.90, bottom=0.05, left=0.07, right=0.97)

    for col, c in enumerate(configs):
        state_hist, drop_hist = simulate_burst(
            c["p_onset"], c["p_recovery"], c["good_drop"], c["bad_drop"],
            num_envs, num_steps, device,
        )

        # Collect stats across all envs
        all_burst_lens = []
        all_good_lens = []
        for e in range(num_envs):
            bl, gl = compute_run_lengths(state_hist[:, e])
            all_burst_lens.extend(bl)
            all_good_lens.extend(gl)

        total_drops = drop_hist.sum()
        total_possible = num_steps * num_envs
        drop_rate_pct = total_drops / total_possible * 100
        bad_frac_pct = state_hist.sum() / total_possible * 100

        # --- Row 1: Markov state timeline ---
        ax1 = fig.add_subplot(gs[0, col])
        show_envs = min(4, num_envs)
        for e in range(show_envs):
            ax1.fill_between(
                range(num_steps),
                e + state_hist[:, e] * 0.8,
                e,
                alpha=0.7, color=f"C{e}", step="post",
            )
        ax1.set_yticks(range(show_envs))
        ax1.set_yticklabels([f"Env {e}" for e in range(show_envs)], fontsize=8)
        ax1.set_title(c["label"], fontsize=9, fontweight="bold")
        if col == 0:
            ax1.set_ylabel("Markov State\n(filled = Bad)", fontsize=9)
        ax1.set_xlim(0, num_steps)
        ax1.tick_params(labelsize=7)

        # --- Row 2: Dropout event raster ---
        ax2 = fig.add_subplot(gs[1, col])
        for e in range(show_envs):
            drops = np.where(drop_hist[:, e])[0]
            ax2.scatter(drops, [e] * len(drops), s=0.3, c=f"C{e}",
                        marker="|", linewidths=0.5)
        ax2.set_yticks(range(show_envs))
        ax2.set_yticklabels([f"Env {e}" for e in range(show_envs)], fontsize=8)
        if col == 0:
            ax2.set_ylabel("Dropout Events\n(tick = drop)", fontsize=9)
        ax2.set_xlim(0, num_steps)
        ax2.tick_params(labelsize=7)

        # --- Row 3: Burst length histogram ---
        ax3 = fig.add_subplot(gs[2, col])
        if len(all_burst_lens) > 0:
            max_bin = min(max(all_burst_lens), 80)
            bins = np.arange(1, max_bin + 2) - 0.5
            ax3.hist(all_burst_lens, bins=bins, color="coral", edgecolor="darkred",
                     alpha=0.8, density=True)
            mean_bl = np.mean(all_burst_lens)
            ax3.axvline(mean_bl, color="red", ls="--", lw=1.5,
                        label=f"Mean={mean_bl:.1f}")
            if c["p_recovery"] > 0:
                expected = 1.0 / c["p_recovery"]
                ax3.axvline(expected, color="blue", ls=":", lw=1.5,
                            label=f"1/p_rec={expected:.1f}")
            ax3.legend(fontsize=7, loc="upper right")
        else:
            ax3.text(0.5, 0.5, "No bursts\n(p_onset=0)", transform=ax3.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
        if col == 0:
            ax3.set_ylabel("Burst Length\nDistribution (density)", fontsize=9)
        ax3.set_xlabel("Burst length (steps)", fontsize=8)
        ax3.tick_params(labelsize=7)

        # --- Row 4: Good-run length histogram ---
        ax4 = fig.add_subplot(gs[3, col])
        if len(all_good_lens) > 0 and c["p_onset"] > 0:
            max_bin = min(max(all_good_lens), 400)
            n_bins = min(60, max_bin)
            bins = np.linspace(0.5, max_bin + 0.5, n_bins)
            ax4.hist(all_good_lens, bins=bins, color="lightgreen", edgecolor="darkgreen",
                     alpha=0.8, density=True)
            mean_gl = np.mean(all_good_lens)
            ax4.axvline(mean_gl, color="green", ls="--", lw=1.5,
                        label=f"Mean={mean_gl:.1f}")
            expected = 1.0 / c["p_onset"]
            if expected < 500:
                ax4.axvline(expected, color="blue", ls=":", lw=1.5,
                            label=f"1/p_on={expected:.1f}")
            ax4.legend(fontsize=7, loc="upper right")
        else:
            ax4.text(0.5, 0.5, "N/A (i.i.d. mode)", transform=ax4.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
        if col == 0:
            ax4.set_ylabel("Good-Run Length\nDistribution (density)", fontsize=9)
        ax4.set_xlabel("Good-run length (steps)", fontsize=8)
        ax4.tick_params(labelsize=7)

        # Stats annotation
        ax4.annotate(
            f"Overall drop rate: {drop_rate_pct:.1f}%\n"
            f"Time in Bad state: {bad_frac_pct:.1f}%\n"
            f"Burst count: {len(all_burst_lens)}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    output_path = args.output
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Also try to show interactively
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize burst dropout characteristics")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of sim steps to simulate")
    parser.add_argument("--envs", type=int, default=1024,
                        help="Number of environments")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "burst_dropout_analysis.png"),
                        help="Output image path")
    args = parser.parse_args()

    plot_analysis(args)


if __name__ == "__main__":
    main()
