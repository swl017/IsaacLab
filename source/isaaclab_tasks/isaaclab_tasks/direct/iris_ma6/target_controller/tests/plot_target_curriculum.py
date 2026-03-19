#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize how target motion difficulty increases along curriculum.

Generates one figure:
  target_curriculum.png — Speed, direction-change interval, evasion,
                          geofence, and behavior profile panels.

Values are read from TargetControllerCfg and CurriculumCfg.
No Isaac Sim dependency required.

Usage:
    python .../target_controller/tests/plot_target_curriculum.py
    python ... --out-dir /tmp
    python ... --dark
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Import configs without Isaac Sim
# ---------------------------------------------------------------------------
_ma6_root = Path(__file__).resolve().parents[2]  # iris_ma6/

# Stub isaaclab.utils.configclass
_stub = types.ModuleType("isaaclab")
_stub_utils = types.ModuleType("isaaclab.utils")
_stub_utils.configclass = lambda cls: cls
_stub.utils = _stub_utils
sys.modules.setdefault("isaaclab", _stub)
sys.modules.setdefault("isaaclab.utils", _stub_utils)

# Stub torch
try:
    import torch  # noqa: F401
except ImportError:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = _torch_stub


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tc_cfg_mod = _load_module(
    "target_controller_cfg",
    _ma6_root / "target_controller" / "target_controller_cfg.py",
)
TargetControllerCfg = _tc_cfg_mod.TargetControllerCfg
BEHAVIOR_PROFILES = _tc_cfg_mod.BEHAVIOR_PROFILES

_curr_mod = _load_module(
    "curriculum_cfg", _ma6_root / "curriculum" / "curriculum_cfg.py"
)
CurriculumCfg = _curr_mod.CurriculumCfg

cfg = TargetControllerCfg()
curr = CurriculumCfg()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lp(step: np.ndarray, start: int, end: int) -> np.ndarray:
    """Linear progress, clamped to [0, 1]."""
    return np.clip((step - start) / max(end - start, 1), 0.0, 1.0)


def _format_step_axis(ax, end_step: int):
    ax.set_xlim(0, end_step)
    ticks = np.arange(0, end_step + 1, 20_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // 1000)}k" for t in ticks])
    ax.set_xlabel("Training step", fontsize=12)


COLORS = {
    "speed": "#e74c3c",
    "interval": "#3498db",
    "evasion": "#9b59b6",
    "geofence": "#2ecc71",
    "profile": "#f39c12",
    "phase": "#95a5a6",
    "kamikaze": "#e74c3c",
    "standard": "#3498db",
    "evasive": "#9b59b6",
    "stealth": "#1abc9c",
}


def _add_phase_spans(ax, alpha: float = 0.06):
    """Add light shading for the moving-target curriculum phase."""
    ax.axvspan(
        curr.moving_target_start_step,
        curr.moving_target_end_step,
        alpha=alpha,
        color=COLORS["phase"],
        zorder=0,
    )
    ax.axvline(
        curr.moving_target_start_step,
        color=COLORS["phase"],
        ls=":",
        lw=1,
        alpha=0.4,
    )
    ax.axvline(
        curr.moving_target_end_step,
        color=COLORS["phase"],
        ls=":",
        lw=1,
        alpha=0.4,
    )


# ===========================================================================
# Main figure
# ===========================================================================

def plot_target_curriculum(out_dir: Path, dark: bool):
    steps = np.arange(0, curr.all_end_step + 1, 200)
    progress = _lp(steps, curr.moving_target_start_step, curr.moving_target_end_step)

    fig, axes = plt.subplots(3, 2, figsize=(24, 24))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    # ------------------------------------------------------------------
    # Panel 1: Max speed vs curriculum
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    _add_phase_spans(ax)

    max_speed = cfg.max_speed_start + progress * (cfg.max_speed_end - cfg.max_speed_start)
    ax.fill_between(steps, 0, max_speed, alpha=0.25, color=COLORS["speed"])
    ax.plot(steps, max_speed, lw=2.5, color=COLORS["speed"], label="Max target speed")

    # Show per-profile effective speeds
    for name, prof in BEHAVIOR_PROFILES.items():
        effective = max_speed * prof.speed_multiplier
        ax.plot(
            steps, effective, lw=1.5, ls="--", color=COLORS.get(name, "gray"),
            alpha=0.7, label=f"{name} ({prof.speed_multiplier:.1f}x)",
        )

    ax.set_ylabel("Speed (m/s)", fontsize=13)
    ax.set_title("Target Max Speed vs Curriculum", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    _format_step_axis(ax, curr.all_end_step)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Annotate key values
    ax.annotate(
        f"{cfg.max_speed_start:.0f} m/s",
        xy=(curr.moving_target_start_step, cfg.max_speed_start),
        xytext=(curr.moving_target_start_step + 5000, cfg.max_speed_start + 2),
        fontsize=10, color=COLORS["speed"],
        arrowprops=dict(arrowstyle="->", color=COLORS["speed"], lw=1),
    )
    ax.annotate(
        f"{cfg.max_speed_end:.0f} m/s",
        xy=(curr.moving_target_end_step, cfg.max_speed_end),
        xytext=(curr.moving_target_end_step + 5000, cfg.max_speed_end - 2),
        fontsize=10, color=COLORS["speed"],
        arrowprops=dict(arrowstyle="->", color=COLORS["speed"], lw=1),
    )

    # ------------------------------------------------------------------
    # Panel 2: Direction change interval vs curriculum
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    _add_phase_spans(ax)

    int_min = cfg.update_interval_min_start + progress * (
        cfg.update_interval_min_end - cfg.update_interval_min_start
    )
    int_max = cfg.update_interval_max_start + progress * (
        cfg.update_interval_max_end - cfg.update_interval_max_start
    )

    ax.fill_between(steps, int_min, int_max, alpha=0.25, color=COLORS["interval"],
                    label="Sampling range")
    ax.plot(steps, int_min, lw=2.5, color=COLORS["interval"], label="Min interval")
    ax.plot(steps, int_max, lw=2.5, color=COLORS["interval"], ls="--",
            label="Max interval")

    ax.set_ylabel("Direction change interval (s)", fontsize=13)
    ax.set_title("Direction Change Frequency vs Curriculum", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    _format_step_axis(ax, curr.all_end_step)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Annotate start / end
    ax.annotate(
        f"{cfg.update_interval_min_start:.0f}–{cfg.update_interval_max_start:.0f}s\n(easy)",
        xy=(curr.moving_target_start_step, cfg.update_interval_max_start),
        xytext=(curr.moving_target_start_step - 2000, cfg.update_interval_max_start + 1),
        fontsize=10, ha="right", color=COLORS["interval"],
    )
    ax.annotate(
        f"{cfg.update_interval_min_end:.0f}–{cfg.update_interval_max_end:.0f}s\n(hard)",
        xy=(curr.moving_target_end_step, cfg.update_interval_max_end),
        xytext=(curr.moving_target_end_step + 5000, cfg.update_interval_max_end + 1.5),
        fontsize=10, color=COLORS["interval"],
        arrowprops=dict(arrowstyle="->", color=COLORS["interval"], lw=1),
    )

    # ------------------------------------------------------------------
    # Panel 3: Evasion availability
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    _add_phase_spans(ax)

    evasion_enabled = (progress >= cfg.evasion_start_progress).astype(float)
    ax.fill_between(steps, 0, evasion_enabled, alpha=0.25, color=COLORS["evasion"])
    ax.plot(steps, evasion_enabled, lw=2.5, color=COLORS["evasion"],
            label="Evasion enabled", drawstyle="steps-post")

    # Trigger distance line
    ax2 = ax.twinx()
    ax2.axhline(cfg.evade_trigger_distance, color=COLORS["evasion"], ls="--",
                lw=1.5, alpha=0.5)
    ax2.set_ylabel("Trigger distance (m)", fontsize=12, color=COLORS["evasion"],
                    alpha=0.6)
    ax2.set_ylim(0, cfg.evade_trigger_distance * 2.5)
    ax2.tick_params(axis="y", labelcolor=COLORS["evasion"], labelsize=10)

    # Evasion start marker
    evasion_step = curr.moving_target_start_step + cfg.evasion_start_progress * (
        curr.moving_target_end_step - curr.moving_target_start_step
    )
    ax.axvline(evasion_step, color=COLORS["evasion"], ls="--", lw=1.5, alpha=0.6)
    ax.annotate(
        f"Evasion ON\n(progress={cfg.evasion_start_progress:.0%})",
        xy=(evasion_step, 0.5),
        xytext=(evasion_step + 10000, 0.6),
        fontsize=10, color=COLORS["evasion"],
        arrowprops=dict(arrowstyle="->", color=COLORS["evasion"], lw=1),
    )

    # Show evasion duration range
    ax.text(
        0.02, 0.05,
        f"Duration: [{cfg.evade_duration_min:.0f}, {cfg.evade_duration_max:.0f}]s  |  "
        f"Trigger: {cfg.evade_trigger_distance:.0f}m  |  "
        f"Agility: [{cfg.evasion_agility_min:.1f}, {cfg.evasion_agility_max:.1f}]",
        transform=ax.transAxes, fontsize=10, style="italic", alpha=0.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3),
    )

    ax.set_ylabel("Evasion enabled (0/1)", fontsize=13)
    ax.set_title("Evasion Mode Availability vs Curriculum", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(-0.1, 1.3)
    _format_step_axis(ax, curr.all_end_step)
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 4: Geofence size vs curriculum
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    _add_phase_spans(ax)

    geofence = cfg.geofence_min_size + progress * (
        cfg.geofence_max_size - cfg.geofence_min_size
    )
    ax.fill_between(steps, 0, geofence, alpha=0.15, color=COLORS["geofence"])
    ax.plot(steps, geofence, lw=2.5, color=COLORS["geofence"],
            label="Geofence half-width")

    # Margin line
    ax.plot(steps, geofence - cfg.geofence_margin, lw=1.5, ls="--",
            color=COLORS["geofence"], alpha=0.5,
            label=f"Slowdown boundary (−{cfg.geofence_margin:.0f}m)")

    ax.set_ylabel("Geofence half-width (m)", fontsize=13)
    ax.set_title("Geofence Size vs Curriculum", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    _format_step_axis(ax, curr.all_end_step)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    ax.annotate(
        f"{cfg.geofence_min_size:.0f}m",
        xy=(curr.moving_target_start_step, cfg.geofence_min_size),
        xytext=(curr.moving_target_start_step + 5000, cfg.geofence_min_size + 15),
        fontsize=10, color=COLORS["geofence"],
        arrowprops=dict(arrowstyle="->", color=COLORS["geofence"], lw=1),
    )
    ax.annotate(
        f"{cfg.geofence_max_size:.0f}m",
        xy=(curr.moving_target_end_step, cfg.geofence_max_size),
        xytext=(curr.moving_target_end_step + 5000, cfg.geofence_max_size - 15),
        fontsize=10, color=COLORS["geofence"],
        arrowprops=dict(arrowstyle="->", color=COLORS["geofence"], lw=1),
    )

    # ------------------------------------------------------------------
    # Panel 5: Behavior profiles (pie + speed comparison)
    # ------------------------------------------------------------------
    ax = axes[2, 0]

    profile_names = list(BEHAVIOR_PROFILES.keys())
    weights = [
        cfg.behavior_kamikaze_weight,
        cfg.behavior_standard_weight,
        cfg.behavior_evasive_weight,
        cfg.behavior_stealth_weight,
    ]
    profile_colors = [COLORS[n] for n in profile_names]

    # Left half: pie chart
    wedges, texts = ax.pie(
        weights,
        labels=[f"{n}\n({w:.0%})" for n, w in zip(profile_names, weights)],
        colors=profile_colors,
        autopct=None,
        startangle=90,
        pctdistance=0.7,
        textprops={"fontsize": 11},
    )

    # Add speed/evasion info as a secondary annotation
    profile_info = []
    for name in profile_names:
        p = BEHAVIOR_PROFILES[name]
        spd_at_full = cfg.max_speed_end * p.speed_multiplier
        profile_info.append(
            f"{name}: {p.speed_multiplier:.1f}x speed "
            f"({spd_at_full:.0f} m/s @full), "
            f"evasion={p.evasion_agility:.1f}"
        )

    info_text = "\n".join(profile_info)
    ax.text(
        0.0, -0.12, info_text,
        transform=ax.transAxes, fontsize=10, ha="center", va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.4),
    )

    ax.set_title("Behavior Profile Distribution", fontsize=15, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel 6: Combined difficulty summary
    # ------------------------------------------------------------------
    ax = axes[2, 1]

    # Normalize each parameter to [0, 1] for a combined view
    speed_norm = (max_speed - cfg.max_speed_start) / (
        cfg.max_speed_end - cfg.max_speed_start + 1e-8
    )
    # Interval difficulty: shorter = harder, so invert
    mean_interval = (int_min + int_max) / 2.0
    mean_interval_start = (cfg.update_interval_min_start + cfg.update_interval_max_start) / 2
    mean_interval_end = (cfg.update_interval_min_end + cfg.update_interval_max_end) / 2
    interval_norm = 1.0 - (mean_interval - mean_interval_end) / (
        mean_interval_start - mean_interval_end + 1e-8
    )
    geofence_norm = (geofence - cfg.geofence_min_size) / (
        cfg.geofence_max_size - cfg.geofence_min_size + 1e-8
    )

    ax.plot(steps, speed_norm, lw=2.5, color=COLORS["speed"],
            label="Speed difficulty")
    ax.plot(steps, interval_norm, lw=2.5, color=COLORS["interval"], ls="--",
            label="Direction-change difficulty")
    ax.plot(steps, evasion_enabled, lw=2.5, color=COLORS["evasion"], ls="-.",
            label="Evasion enabled", drawstyle="steps-post")
    ax.plot(steps, geofence_norm, lw=2.5, color=COLORS["geofence"], ls=":",
            label="Geofence expansion")

    # Composite difficulty (weighted average)
    composite = 0.4 * speed_norm + 0.3 * interval_norm + 0.15 * evasion_enabled + 0.15 * geofence_norm
    ax.fill_between(steps, 0, composite, alpha=0.12, color="gray")
    ax.plot(steps, composite, lw=3, color="black", alpha=0.5,
            label="Composite difficulty")

    # Phase annotations
    _add_phase_spans(ax, alpha=0.08)
    ax.text(
        (curr.moving_target_start_step + curr.moving_target_end_step) / 2,
        1.08, "Target motion ramp",
        ha="center", fontsize=10, color=COLORS["phase"], style="italic",
    )

    ax.set_ylabel("Normalized difficulty (0–1)", fontsize=13)
    ax.set_title("Combined Target Difficulty Summary", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="center right")
    ax.set_ylim(-0.05, 1.15)
    _format_step_axis(ax, curr.all_end_step)
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    for row in axes:
        for a in row:
            a.tick_params(labelsize=11)

    fig.suptitle(
        "iris_ma6 — Target Motion Difficulty vs Curriculum",
        fontsize=18, fontweight="bold", y=0.995,
    )

    path = out_dir / "target_curriculum.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot iris_ma6 target motion difficulty vs curriculum"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: same as this script)",
    )
    parser.add_argument("--dark", action="store_true", help="Use dark theme")
    args = parser.parse_args()

    if args.dark:
        plt.style.use("dark_background")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_target_curriculum(out_dir, args.dark)


if __name__ == "__main__":
    main()
