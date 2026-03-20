#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot curriculum schedule and parameter distributions over training progress.

Generates two figures:
  1. curriculum_overview.png  — Phase Gantt chart, heatmap, delay mode timeline
  2. curriculum_details.png   — 6 detail panels (latency, noise, dropout,
                                task levels, gain randomization, per-agent)

Values are read directly from the environment config files
(CurriculumCfg, DelaySystemKeyParams, GainRandomizationCfg, PerAgentDelayCfg).
No Isaac Sim dependency required.

Usage:
    python .../curriculum/test/plot_curriculum.py
    python ... --out-dir /tmp                  # custom output directory
    python ... --dark                          # dark theme
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Import configs directly from the codebase (bypassing __init__.py chains
# that pull in gymnasium/Isaac Sim)
# ---------------------------------------------------------------------------
import importlib.util

_ma6_root = Path(__file__).resolve().parents[2]  # iris_ma6/


def _load_module(name: str, filepath: Path):
    """Load a single .py file as a module without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub out isaaclab.utils.configclass so the decorator is a no-op
import types
_stub = types.ModuleType("isaaclab")
_stub_utils = types.ModuleType("isaaclab.utils")
_stub_utils.configclass = lambda cls: cls  # no-op decorator
_stub.utils = _stub_utils
sys.modules.setdefault("isaaclab", _stub)
sys.modules.setdefault("isaaclab.utils", _stub_utils)

_curr_mod = _load_module(
    "curriculum_cfg", _ma6_root / "curriculum" / "curriculum_cfg.py"
)
_delay_mod = _load_module(
    "delay_cfg_v3", _ma6_root / "delay_system_v3" / "delay_cfg_v3.py"
)
_gain_mod = _load_module(
    "gain_randomization_cfg", _ma6_root / "controller" / "gain_randomization_cfg.py"
)

CurriculumCfg = _curr_mod.CurriculumCfg
DelaySystemKeyParams = _delay_mod.DelaySystemKeyParams
PerAgentDelayCfg = _delay_mod.PerAgentDelayCfg
GainRandomizationCfg = _gain_mod.GainRandomizationCfg

# Instantiate configs matching iris_ma_env6_test_cfg.py overrides
_curr = CurriculumCfg()
_delay = DelaySystemKeyParams(
    ego_motion_latency_enabled=True,
    ego_motion_fol_tau=0.005,
    ego_detection_latency_mean=0.1,
    ego_detection_latency_std=0.015,
    other_latency_mean=0.5,
    other_latency_std=0.08,
    staleness_fps_mean=25.0,
    staleness_fps_range=5.0,
    dropout_prob=0.05,
    noise_enabled=True,
    noise_position_std=0.1,
    noise_velocity_std=0.05,
    noise_orientation_std=0.01,
    noise_angular_velocity_std=0.02,
    noise_acceleration_std=0.1,
    noise_bbox_std=7.0,
    reward_use_delay=True,
    reward_use_noise=False,
)
_gain = GainRandomizationCfg()
_pa = PerAgentDelayCfg()

# ---------------------------------------------------------------------------
# Build plot-friendly dicts from config objects
# ---------------------------------------------------------------------------

CURRICULUM = dict(
    all_end_step=_curr.all_end_step,
    safety_start=_curr.safety_start_step,
    safety_end=_curr.safety_end_step,
    tracking_start=_curr.tracking_start_step,
    tracking_end=_curr.tracking_end_step,
    target_start=_curr.moving_target_start_step,
    target_end=_curr.moving_target_end_step,
    coord_start=_curr.coordination_start_step,
    coord_end=_curr.coordination_end_step,
    noise_start=_curr.noise_start_step,
    noise_end=_curr.noise_end_step,
    fixed_delay_start=_curr.fixed_delay_start_step,
    fixed_delay_end=_curr.fixed_delay_end_step,
    random_delay_start=_curr.random_delay_start_step,
    random_delay_end=_curr.random_delay_end_step,
    dropout_start=_curr.dropout_start_step,
    dropout_end=_curr.dropout_end_step,
    dynamics_start=_curr.dynamics_start_step,
    dynamics_end=_curr.dynamics_end_step,
    task_l2_start=_curr.task_level_2_start_step,
    task_l2_end=_curr.task_level_2_end_step,
    task_l3_start=_curr.task_level_3_start_step,
    task_l3_end=_curr.task_level_3_end_step,
)

DELAY_PARAMS = dict(
    ego_latency_mean=_delay.ego_detection_latency_mean,
    ego_latency_std=_delay.ego_detection_latency_std,
    other_latency_mean=_delay.other_latency_mean,
    other_latency_std=_delay.other_latency_std,
    staleness_fps_mean=_delay.staleness_fps_mean,
    staleness_fps_half_range=_delay.staleness_fps_range,
    dropout_prob=_delay.dropout_prob,
    noise_position_std=_delay.noise_position_std,
    noise_velocity_std=_delay.noise_velocity_std,
    noise_orientation_std=_delay.noise_orientation_std,
    noise_angular_velocity_std=_delay.noise_angular_velocity_std,
    noise_acceleration_std=_delay.noise_acceleration_std,
    noise_bbox_std=_delay.noise_bbox_std,
)

GAIN_RAND = dict(scale_lo=_gain.scale_range[0], scale_hi=_gain.scale_range[1])

PER_AGENT = dict(
    latency_scale_range=_pa.latency_scale_range,
    noise_scale_range=_pa.noise_scale_range,
    dropout_offset_range=_pa.dropout_offset_range,
)

COLORS = {
    "safety": "#e74c3c",
    "tracking": "#3498db",
    "target": "#2ecc71",
    "coord": "#9b59b6",
    "noise": "#f39c12",
    "fixed_delay": "#e67e22",
    "random_delay": "#d35400",
    "dropout": "#c0392b",
    "dynamics": "#1abc9c",
    "gain_lo": "#16a085",
    "gain_hi": "#e74c3c",
    "latency_ego": "#3498db",
    "latency_other": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lp(step: np.ndarray, start: int, end: int) -> np.ndarray:
    """Linear progress [0, 1]."""
    return np.clip((step - start) / max(end - start, 1), 0.0, 1.0)


def _delay_mode(step: np.ndarray) -> np.ndarray:
    """Return 0=none, 1=fixed, 2=random."""
    mode = np.zeros_like(step, dtype=float)
    mode[step >= CURRICULUM["fixed_delay_start"]] = 1.0
    mode[step >= CURRICULUM["random_delay_start"]] = 2.0
    return mode


def _step_axis():
    return np.arange(0, CURRICULUM["all_end_step"] + 1, 200)


def _format_step_axis(ax):
    """Format x-axis with k-suffix tick labels."""
    ax.set_xlim(0, CURRICULUM["all_end_step"])
    ticks = np.arange(0, CURRICULUM["all_end_step"] + 1, 20_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // 1000)}k" for t in ticks])
    ax.set_xlabel("Training step", fontsize=12)


# ===========================================================================
# Overview figure
# ===========================================================================

def plot_overview(out_dir: Path, dark: bool):
    C = CURRICULUM
    steps = _step_axis()

    fig, axes = plt.subplots(3, 1, figsize=(20, 16),
                             gridspec_kw={"height_ratios": [4, 3, 2]})
    fig.subplots_adjust(hspace=0.35)

    # ----- Panel 1: Gantt chart -----
    ax = axes[0]
    phases = [
        ("Safety",        "safety",       C["safety_start"],       C["safety_end"]),
        ("Tracking",      "tracking",     C["tracking_start"],     C["tracking_end"]),
        ("Target motion", "target",       C["target_start"],       C["target_end"]),
        ("Coordination",  "coord",        C["coord_start"],        C["coord_end"]),
        ("Noise",         "noise",        C["noise_start"],        C["noise_end"]),
        ("Fixed delay",   "fixed_delay",  C["fixed_delay_start"],  C["fixed_delay_end"]),
        ("Random delay",  "random_delay", C["random_delay_start"], C["random_delay_end"]),
        ("Dropout",       "dropout",      C["dropout_start"],      C["dropout_end"]),
        ("Dynamics",      "dynamics",     C["dynamics_start"],     C["dynamics_end"]),
    ]
    for i, (label, key, start, end) in enumerate(phases):
        prog = _lp(steps, start, end)
        ax.fill_between(steps, i, i + prog, alpha=0.7, color=COLORS[key])
        ax.axvline(start, color=COLORS[key], alpha=0.25, lw=0.7, ls="--")
        ax.axvline(end,   color=COLORS[key], alpha=0.25, lw=0.7, ls="--")
        # Label on the right at full height
        ax.text(C["all_end_step"] + 1500, i + 0.5, label, va="center", fontsize=11)
    ax.set_yticks(np.arange(len(phases)) + 0.5)
    ax.set_yticklabels([p[0] for p in phases], fontsize=11)
    _format_step_axis(ax)
    ax.set_title("Curriculum Phase Schedule", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # ----- Panel 2: Heatmap -----
    ax = axes[1]
    key_steps = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 130_000, 160_000, 200_000]
    param_names = [
        "Safety", "Tracking", "Target motion", "Coordination",
        "Noise scale", "Fixed delay", "Random delay",
        "Dropout rate", "Dynamics", "Gain range",
    ]
    table_data = []
    for s in key_steps:
        row = [
            _lp(np.array([s]), C["safety_start"], C["safety_end"])[0],
            _lp(np.array([s]), C["tracking_start"], C["tracking_end"])[0],
            _lp(np.array([s]), C["target_start"], C["target_end"])[0],
            _lp(np.array([s]), C["coord_start"], C["coord_end"])[0],
            _lp(np.array([s]), C["noise_start"], C["noise_end"])[0],
            _lp(np.array([s]), C["fixed_delay_start"], C["fixed_delay_end"])[0],
            _lp(np.array([s]), C["random_delay_start"], C["random_delay_end"])[0],
            _lp(np.array([s]), C["dropout_start"], C["dropout_end"])[0],
            _lp(np.array([s]), C["dynamics_start"], C["dynamics_end"])[0],
            _lp(np.array([s]), C["dynamics_start"], C["dynamics_end"])[0],
        ]
        table_data.append(row)

    td = np.array(table_data)
    im = ax.imshow(td.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(key_steps)))
    ax.set_xticklabels([f"{s // 1000}k" for s in key_steps], fontsize=11)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=11)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_title("Parameter Progress Heatmap  (0 = off, 1 = full)", fontsize=14, fontweight="bold")
    for i in range(len(param_names)):
        for j in range(len(key_steps)):
            v = td[j, i]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, label="Progress")

    # ----- Panel 3: Delay mode timeline -----
    ax = axes[2]
    mode_arr = _delay_mode(steps)
    mode_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    mode_labels = {0: "None", 1: "Fixed", 2: "Random"}
    for m in [0, 1, 2]:
        mask = mode_arr == m
        if mask.any():
            ax.fill_between(steps, 0, 1, where=mask, alpha=0.4,
                            color=mode_colors[m], label=f"Mode: {mode_labels[m]}")

    ax2 = ax.twinx()
    noise_prog = _lp(steps, C["noise_start"], C["noise_end"])
    dropout_prog = _lp(steps, C["dropout_start"], C["dropout_end"])
    dyn_prog = _lp(steps, C["dynamics_start"], C["dynamics_end"])
    ax2.plot(steps, noise_prog, lw=2, ls="--", color=COLORS["noise"], label="Noise scale")
    ax2.plot(steps, dropout_prog * DELAY_PARAMS["dropout_prob"],
             lw=2, ls=":", color=COLORS["dropout"], label="Dropout rate")
    ax2.plot(steps, dyn_prog, lw=2, ls="-.", color=COLORS["dynamics"], label="Dynamics prog")
    ax2.set_ylabel("Progress / Rate", fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    _format_step_axis(ax)
    ax.set_title("Delay Mode + Observability Timeline", fontsize=14, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=11, loc="upper left", ncol=3)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("iris_ma6 — Curriculum Overview",
                 fontsize=17, fontweight="bold", y=0.995)

    path = out_dir / "curriculum_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# Detail figure (6 panels, each ~3x larger than before)
# ===========================================================================

def plot_details(out_dir: Path, dark: bool):
    C = CURRICULUM
    D = DELAY_PARAMS
    steps = _step_axis()

    # Each panel is ~9x7 inches (was ~4.5x3 in the combined figure)
    fig, axes = plt.subplots(3, 2, figsize=(24, 27))
    fig.subplots_adjust(hspace=0.30, wspace=0.28)

    # ---- 1) Delay latency ----
    ax = axes[0, 0]
    mode = _delay_mode(steps)
    fixed_prog = _lp(steps, C["fixed_delay_start"], C["fixed_delay_end"])
    random_prog = _lp(steps, C["random_delay_start"], C["random_delay_end"])
    ego_lat = np.where(mode == 0, 0.0,
                       np.where(mode == 1, fixed_prog * D["ego_latency_mean"],
                                D["ego_latency_mean"]))
    other_lat = np.where(mode == 0, 0.0,
                         np.where(mode == 1, fixed_prog * D["other_latency_mean"],
                                  D["other_latency_mean"]))
    ego_std = np.where(mode == 2, random_prog * D["ego_latency_std"], 0.0)
    other_std = np.where(mode == 2, random_prog * D["other_latency_std"], 0.0)

    # Background mode bands
    ax.axvspan(C["fixed_delay_start"], C["random_delay_start"],
               alpha=0.08, color=COLORS["fixed_delay"], label="Fixed delay phase")
    ax.axvspan(C["random_delay_start"], C["all_end_step"],
               alpha=0.08, color=COLORS["random_delay"], label="Random delay phase")

    ax.plot(steps, ego_lat * 1000, lw=2.5, color=COLORS["latency_ego"],
            label=f"Ego mean ({D['ego_latency_mean']*1000:.0f} ms)")
    ax.fill_between(steps, (ego_lat - ego_std) * 1000, (ego_lat + ego_std) * 1000,
                    alpha=0.15, color=COLORS["latency_ego"])
    ax.plot(steps, other_lat * 1000, lw=2.5, color=COLORS["latency_other"],
            label=f"Other mean ({D['other_latency_mean']*1000:.0f} ms)")
    ax.fill_between(steps, (other_lat - other_std) * 1000, (other_lat + other_std) * 1000,
                    alpha=0.15, color=COLORS["latency_other"])
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("Delay System: Latency", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    _format_step_axis(ax)
    ax.grid(alpha=0.3)

    # ---- 2) Observation noise ----
    ax = axes[0, 1]
    noise_prog = _lp(steps, C["noise_start"], C["noise_end"])
    ax.plot(steps, noise_prog * D["noise_position_std"] * 100, lw=2.5,
            label=f"Position ({D['noise_position_std']*100:.0f} cm)", color="#3498db")
    ax.plot(steps, noise_prog * D["noise_velocity_std"] * 100, lw=2.5,
            label=f"Velocity ({D['noise_velocity_std']*100:.0f} cm/s)", color="#2ecc71")
    ax.plot(steps, noise_prog * D["noise_orientation_std"] * (180 / np.pi), lw=2.5,
            label=f"Orientation ({D['noise_orientation_std']*180/np.pi:.2f}\u00b0)", color="#9b59b6")
    ax.plot(steps, noise_prog * D["noise_angular_velocity_std"] * (180 / np.pi), lw=2.5, ls="--",
            label=f"Ang vel ({D['noise_angular_velocity_std']*180/np.pi:.1f}\u00b0/s)", color="#1abc9c")
    ax.plot(steps, noise_prog * D["noise_acceleration_std"] * 100, lw=2.5, ls="-.",
            label=f"Accel ({D['noise_acceleration_std']*100:.0f} cm/s\u00b2)", color="#e67e22")
    ax2 = ax.twinx()
    ax2.plot(steps, noise_prog * D["noise_bbox_std"], lw=2.5, ls="--",
             color="#e74c3c", label=f"BBox ({D['noise_bbox_std']:.0f} px)")
    ax2.set_ylabel("BBox noise (px)", fontsize=13, color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c", labelsize=11)

    ax.set_ylabel("Noise std (cm / cm\u00b7s\u207b\u00b9 / deg)", fontsize=13)
    ax.set_title("Observation Noise Ramp", fontsize=15, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper left")
    _format_step_axis(ax)
    ax.grid(alpha=0.3)

    # ---- 3) Dropout ----
    ax = axes[1, 0]
    dropout_prog = _lp(steps, C["dropout_start"], C["dropout_end"])
    base_dropout = dropout_prog * D["dropout_prob"]
    lo_off, hi_off = PER_AGENT["dropout_offset_range"]
    ax.plot(steps, base_dropout * 100, lw=2.5, color=COLORS["dropout"],
            label="Base dropout rate")
    # Per-agent offset only visible when base dropout > 0
    per_agent_lo = np.maximum((base_dropout + lo_off * dropout_prog) * 100, 0)
    per_agent_hi = np.minimum((base_dropout + hi_off * dropout_prog) * 100, 100)
    ax.fill_between(steps, per_agent_lo, per_agent_hi,
                    alpha=0.25, color=COLORS["dropout"],
                    label=f"Per-agent offset [{lo_off:.0%}, +{hi_off:.0%}]")
    ax.set_ylabel("Dropout rate (%)", fontsize=13)
    ax.set_title("Dropout Rate", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    _format_step_axis(ax)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # ---- 4) Task reward levels ----
    ax = axes[1, 1]
    l2_prog = _lp(steps, C["task_l2_start"], C["task_l2_end"])
    l3_prog = _lp(steps, C["task_l3_start"], C["task_l3_end"])
    w_l1 = 1.0 - l2_prog
    w_l2 = l2_prog * (1.0 - l3_prog)
    w_l3 = l3_prog
    ax.stackplot(steps, w_l1, w_l2, w_l3,
                 labels=["L1: FIM-only", "L2: GT-anchored",
                         "L3: Composite (GT+E2E)"],
                 colors=["#3498db", "#2ecc71", "#9b59b6"], alpha=0.7)
    ax.set_ylabel("Reward weight", fontsize=13)
    ax.set_title("Task Reward Level Curriculum", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="center right")
    _format_step_axis(ax)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # ---- 5) Gain randomization ----
    ax = axes[2, 0]
    dyn_prog = _lp(steps, C["dynamics_start"], C["dynamics_end"])
    gain_lo = 1.0 - dyn_prog * (1.0 - GAIN_RAND["scale_lo"])
    gain_hi = 1.0 + dyn_prog * (GAIN_RAND["scale_hi"] - 1.0)
    ax.fill_between(steps, gain_lo, gain_hi, alpha=0.3, color=COLORS["gain_lo"],
                    label="Gain scale range")
    ax.plot(steps, gain_lo, lw=2, color=COLORS["gain_lo"])
    ax.plot(steps, gain_hi, lw=2, color=COLORS["gain_hi"])
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("Gain multiplier", fontsize=13)
    ax.set_title("Controller Gain Randomization Range", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    _format_step_axis(ax)
    ax.set_ylim(0.7, 1.35)
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.05, "Affects: Vel PID, Att P, Rate PID, Motor \u03c4",
            transform=ax.transAxes, fontsize=11, ha="right", va="bottom",
            style="italic", alpha=0.6)

    # ---- 6) Per-agent heterogeneity ----
    ax = axes[2, 1]
    lat_lo, lat_hi = PER_AGENT["latency_scale_range"]
    noise_lo, noise_hi = PER_AGENT["noise_scale_range"]
    drp_lo, drp_hi = PER_AGENT["dropout_offset_range"]
    categories = ["Latency\nscale", "Noise\nscale", "Dropout\noffset"]
    x = np.arange(len(categories))
    bar_colors = [COLORS["latency_ego"], COLORS["noise"], COLORS["dropout"]]
    ranges_data = [(lat_lo, lat_hi), (noise_lo, noise_hi), (drp_lo, drp_hi)]
    for i, ((lo, hi), c) in enumerate(zip(ranges_data, bar_colors)):
        ax.bar(i, hi - lo, bottom=lo, width=0.45, color=c, alpha=0.6, edgecolor=c, lw=1.5)
        ax.text(i, lo - 0.04, f"{lo:.2f}", ha="center", va="top", fontsize=12,
                fontweight="bold")
        ax.text(i, hi + 0.04, f"{hi:.2f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Value range", fontsize=13)
    ax.set_title("Per-Agent Heterogeneity (sampled at reset)", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # Add note
    ax.text(0.98, 0.95,
            "Latency/noise: multiplicative on base\nDropout: additive offset on base rate",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            style="italic", alpha=0.5)

    # Tick sizes
    for row in axes:
        for a in row:
            a.tick_params(labelsize=11)

    fig.suptitle("iris_ma6 — Curriculum Detail Panels",
                 fontsize=18, fontweight="bold", y=0.995)

    path = out_dir / "curriculum_details.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot iris_ma6 curriculum schedule")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same as this script)")
    parser.add_argument("--dark", action="store_true", help="Use dark theme")
    args = parser.parse_args()

    if args.dark:
        plt.style.use("dark_background")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overview(out_dir, args.dark)
    plot_details(out_dir, args.dark)


if __name__ == "__main__":
    main()
