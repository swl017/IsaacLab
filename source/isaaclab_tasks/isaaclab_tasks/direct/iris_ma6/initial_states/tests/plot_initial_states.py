#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize initial_states module: placement geometry, curriculum sampling,
gimbal modes, and parameter distributions.

Generates two figures:
  1. initial_states_overview.png  — Placement geometry, curriculum ranges, gimbal modes
  2. initial_states_details.png   — Per-parameter distributions and sampling behavior

Values are read from InitialStatesCfg. No Isaac Sim dependency required.

Usage:
    python .../initial_states/tests/plot_initial_states.py
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
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

# ---------------------------------------------------------------------------
# Import InitialStatesCfg without Isaac Sim
# ---------------------------------------------------------------------------
_ma6_root = Path(__file__).resolve().parents[2]  # iris_ma6/

# Stub isaaclab.utils.configclass
_stub = types.ModuleType("isaaclab")
_stub_utils = types.ModuleType("isaaclab.utils")
_stub_utils.configclass = lambda cls: cls
_stub.utils = _stub_utils
sys.modules.setdefault("isaaclab", _stub)
sys.modules.setdefault("isaaclab.utils", _stub_utils)

# Stub torch (only needed for type annotations in the cfg dataclass)
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


_cfg_mod = _load_module(
    "initial_states_cfg", _ma6_root / "initial_states" / "initial_states_cfg.py"
)
InitialStatesCfg = _cfg_mod.InitialStatesCfg

# Also load CurriculumCfg for phase timing
_curr_mod = _load_module(
    "curriculum_cfg", _ma6_root / "curriculum" / "curriculum_cfg.py"
)
CurriculumCfg = _curr_mod.CurriculumCfg

cfg = InitialStatesCfg()
curr = CurriculumCfg()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lp(step: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.clip((step - start) / max(end - start, 1), 0.0, 1.0)


def _curriculum_sample(progress: np.ndarray, lo: float, hi: float) -> tuple:
    """Return (sampled_lo, sampled_hi) for uniform(lo, lo + progress*(hi-lo))."""
    return lo, lo + progress * (hi - lo)


def _format_step_axis(ax, end_step: int):
    ax.set_xlim(0, end_step)
    ticks = np.arange(0, end_step + 1, 20_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // 1000)}k" for t in ticks])
    ax.set_xlabel("Training step", fontsize=12)


COLORS = {
    "cylinder": "#3498db",
    "target": "#e74c3c",
    "agent": "#2ecc71",
    "observer": "#f39c12",
    "gimbal": "#9b59b6",
    "velocity": "#1abc9c",
    "zoom": "#e67e22",
    "height": "#34495e",
}


# ===========================================================================
# Overview figure
# ===========================================================================

def plot_overview(out_dir: Path, dark: bool):
    steps = np.arange(0, curr.all_end_step + 1, 200)
    progress = _lp(steps, curr.tracking_start_step, curr.tracking_end_step)

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.subplots_adjust(hspace=0.30, wspace=0.28)

    # ---- Panel 1: Top-down placement geometry (example at 3 progress levels) ----
    ax = axes[0, 0]
    progress_examples = [0.0, 0.5, 1.0]
    alphas = [0.3, 0.5, 0.8]
    rng = np.random.RandomState(42)

    for prog, alpha in zip(progress_examples, alphas):
        # Cylinder diameter
        _, diam_hi = _curriculum_sample(prog, cfg.cylinder_diameter_min,
                                        cfg.cylinder_diameter_max)
        radius = diam_hi / 2.0

        # Draw cylinder (top-down = circle)
        circle = Circle((0, 0), radius, fill=False, edgecolor=COLORS["cylinder"],
                         lw=2.0, alpha=alpha, ls="--" if prog < 1.0 else "-",
                         label=f"Cylinder p={prog:.0%} (d={diam_hi:.0f}m)")
        ax.add_patch(circle)

        # Target distance
        _, dist_hi = _curriculum_sample(prog, cfg.target_distance_min,
                                        cfg.target_distance_max)
        # Draw target ring
        target_circle = Circle((0, 0), dist_hi, fill=False, edgecolor=COLORS["target"],
                               lw=1.5, alpha=alpha * 0.7, ls=":")
        ax.add_patch(target_circle)

        # Sample some agent positions inside cylinder
        n_agents = 3
        angles_a = rng.uniform(0, 2 * np.pi, n_agents)
        radii_a = rng.uniform(0, radius, n_agents)
        ax_pos = radii_a * np.cos(angles_a)
        ay_pos = radii_a * np.sin(angles_a)

        # Sample a target
        t_angle = rng.uniform(0, 2 * np.pi)
        t_dist = rng.uniform(cfg.target_distance_min, dist_hi)
        tx, ty = t_dist * np.cos(t_angle), t_dist * np.sin(t_angle)

        marker_size = 8 + prog * 4
        ax.scatter(ax_pos, ay_pos, s=marker_size ** 2, c=COLORS["agent"],
                   marker="^", alpha=alpha, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.scatter(tx, ty, s=(marker_size + 2) ** 2, c=COLORS["target"],
                   marker="*", alpha=alpha, zorder=5,
                   edgecolors="black", linewidths=0.5)

    # Clearance annotation
    ax.annotate(f"Agent clearance: {cfg.agent_clearance:.0f}m",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=11, style="italic", alpha=0.6)

    ax.set_xlim(-220, 220)
    ax.set_ylim(-220, 220)
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title("Top-Down Placement Geometry (3 progress levels)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)

    # ---- Panel 2: Cylinder & target distance curriculum ----
    ax = axes[0, 1]
    # Cylinder diameter range
    diam_lo = np.full_like(progress, cfg.cylinder_diameter_min)
    diam_hi = cfg.cylinder_diameter_min + progress * (
        cfg.cylinder_diameter_max - cfg.cylinder_diameter_min
    )
    ax.fill_between(steps, diam_lo, diam_hi, alpha=0.3, color=COLORS["cylinder"],
                    label="Cylinder diameter range")
    ax.plot(steps, diam_lo, lw=2, color=COLORS["cylinder"])
    ax.plot(steps, diam_hi, lw=2, color=COLORS["cylinder"], ls="--")

    # Target distance range
    tdist_lo = np.full_like(progress, cfg.target_distance_min)
    tdist_hi = cfg.target_distance_min + progress * (
        cfg.target_distance_max - cfg.target_distance_min
    )
    ax.fill_between(steps, tdist_lo, tdist_hi, alpha=0.3, color=COLORS["target"],
                    label="Target distance range")
    ax.plot(steps, tdist_lo, lw=2, color=COLORS["target"])
    ax.plot(steps, tdist_hi, lw=2, color=COLORS["target"], ls="--")

    ax.set_ylabel("Distance (meters)", fontsize=13)
    ax.set_title("Cylinder Diameter & Target Distance vs Curriculum",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    _format_step_axis(ax, curr.all_end_step)
    ax.grid(alpha=0.3)

    # ---- Panel 3: Gimbal curriculum modes ----
    ax = axes[1, 0]
    # "gradual" mode: P(pointing) = 1 - progress
    p_pointing_gradual = 1.0 - progress
    # "threshold" mode: P(pointing) = 1 if progress < thresh else 0
    threshold = cfg.gimbal_randomization_threshold
    p_pointing_threshold = np.where(progress < threshold, 1.0, 0.0)
    # "always_pointing" mode
    p_pointing_always = np.ones_like(progress)

    ax.plot(steps, p_pointing_gradual * 100, lw=2.5, color=COLORS["gimbal"],
            label='Mode: "gradual"')
    ax.plot(steps, p_pointing_threshold * 100, lw=2.5, color=COLORS["target"],
            ls="--", label=f'Mode: "threshold" (t={threshold:.0%})')
    ax.plot(steps, p_pointing_always * 100, lw=2.5, color=COLORS["agent"],
            ls=":", label='Mode: "always_pointing"')

    # Shade designated observer (always 100%)
    ax.axhline(100, color=COLORS["observer"], lw=1.5, ls="-.", alpha=0.5,
               label="Designated observer (always)")

    ax.set_ylabel("P(gimbal points at target) %", fontsize=13)
    ax.set_title("Gimbal Curriculum Modes (non-observer agents)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="center right")
    ax.set_ylim(-5, 110)
    _format_step_axis(ax, curr.all_end_step)
    ax.grid(alpha=0.3)

    # Annotate current mode
    ax.text(0.02, 0.05, f'Current mode: "{cfg.gimbal_curriculum_mode}"',
            transform=ax.transAxes, fontsize=11, style="italic",
            alpha=0.6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    # ---- Panel 4: Height and clearance diagram (side view) ----
    ax = axes[1, 1]
    # Draw height range
    h_min = cfg.cylinder_height_min
    h_max = cfg.cylinder_height_max
    h_range = cfg.cylinder_height_range_max

    # Ground
    ax.axhline(0, color="brown", lw=3, alpha=0.5, label="Ground")
    ax.fill_between([-1, 1], 0, 0, color="brown", alpha=0.1)

    # Height range band for cylinder center
    ax.fill_between([-0.6, 0.6], h_min, h_max, alpha=0.15, color=COLORS["height"],
                    label=f"Center height [{h_min:.0f}, {h_max:.0f}]m")
    ax.axhline(h_min, color=COLORS["height"], ls="--", lw=1, alpha=0.5)
    ax.axhline(h_max, color=COLORS["height"], ls="--", lw=1, alpha=0.5)

    # Example cylinder center and vertical spread
    cz = (h_min + h_max) / 2
    ax.fill_between([-0.4, 0.4], cz, cz + h_range, alpha=0.3, color=COLORS["cylinder"],
                    label=f"Agent vertical spread ({h_range:.0f}m)")

    # Example agents at different heights
    agent_heights = [cz + 2, cz + 10, cz + 17]
    for i, h in enumerate(agent_heights):
        marker = "s" if i == 0 else "^"
        color = COLORS["observer"] if i == 0 else COLORS["agent"]
        label = "Designated observer" if i == 0 else ("Other agents" if i == 1 else None)
        ax.scatter(0.0, h, s=200, c=color, marker=marker, zorder=5,
                   edgecolors="black", linewidths=1, label=label)

    # Clearance annotation
    ax.annotate("", xy=(0.25, agent_heights[0]), xytext=(0.25, agent_heights[1]),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(0.30, (agent_heights[0] + agent_heights[1]) / 2,
            f"clearance\n>={cfg.agent_clearance:.0f}m",
            fontsize=10, va="center", color="gray")

    # Target height offset
    t_h_lo = cz + cfg.target_height_offset_min
    t_h_hi = cz + cfg.target_height_offset_max
    ax.fill_between([0.55, 0.85], t_h_lo, t_h_hi, alpha=0.2, color=COLORS["target"],
                    label=f"Target height offset [{cfg.target_height_offset_min:.0f}, "
                          f"+{cfg.target_height_offset_max:.0f}]m")
    ax.scatter(0.7, cz, s=300, c=COLORS["target"], marker="*", zorder=5,
               edgecolors="black", linewidths=1)

    ax.set_xlim(-1, 1.2)
    ax.set_ylim(-5, h_max + h_range + 10)
    ax.set_ylabel("Height (meters)", fontsize=13)
    ax.set_title("Side View: Height Configuration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xticks([0.0, 0.7])
    ax.set_xticklabels(["Agents", "Target"], fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("iris_ma6 — Initial States Overview",
                 fontsize=18, fontweight="bold", y=0.995)

    path = out_dir / "initial_states_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# Detail figure
# ===========================================================================

def plot_details(out_dir: Path, dark: bool):
    steps = np.arange(0, curr.all_end_step + 1, 200)
    progress = _lp(steps, curr.tracking_start_step, curr.tracking_end_step)

    fig, axes = plt.subplots(3, 2, figsize=(24, 27))
    fig.subplots_adjust(hspace=0.30, wspace=0.28)

    # ---- 1) Velocity curriculum ----
    ax = axes[0, 0]
    # Agent velocity: magnitude ~ uniform(0, progress * scale_max * max_vel)
    agent_vel_max = progress * cfg.agent_velocity_scale_max * cfg.agent_max_velocity
    ax.fill_between(steps, 0, agent_vel_max, alpha=0.3, color=COLORS["velocity"],
                    label=f"Agent vel range (max={cfg.agent_max_velocity:.0f} m/s)")
    ax.plot(steps, agent_vel_max, lw=2.5, color=COLORS["velocity"])

    # Target velocity
    target_vel_max = progress * cfg.target_velocity_scale_max * cfg.target_max_velocity
    ax.fill_between(steps, 0, target_vel_max, alpha=0.2, color=COLORS["target"],
                    label=f"Target vel range (max={cfg.target_max_velocity:.0f} m/s)")
    ax.plot(steps, target_vel_max, lw=2.5, color=COLORS["target"], ls="--")

    # Yaw rate
    ax2 = ax.twinx()
    yaw_max_deg = progress * math.degrees(cfg.max_yaw_rate)
    ax2.plot(steps, yaw_max_deg, lw=2, ls=":", color=COLORS["observer"],
             label=f"Max yaw rate ({math.degrees(cfg.max_yaw_rate):.0f}\u00b0/s)")
    ax2.set_ylabel("Yaw rate (\u00b0/s)", fontsize=13, color=COLORS["observer"])
    ax2.tick_params(axis="y", labelcolor=COLORS["observer"], labelsize=11)

    ax.set_ylabel("Velocity (m/s)", fontsize=13)
    ax.set_title("Initial Velocity Curriculum", fontsize=15, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=11, loc="upper left")
    _format_step_axis(ax, curr.all_end_step)
    ax.grid(alpha=0.3)

    # ---- 2) Zoom level range ----
    ax = axes[0, 1]
    zoom_lo = np.full_like(steps, cfg.zoom_initial_min, dtype=float)
    zoom_hi = cfg.zoom_initial_max_start + progress * (
        cfg.zoom_initial_max_end - cfg.zoom_initial_max_start
    )
    ax.fill_between(steps, zoom_lo, zoom_hi, alpha=0.3, color=COLORS["zoom"],
                    label=f"Initial zoom [{cfg.zoom_initial_min:.0f}, "
                          f"{cfg.zoom_initial_max_start:.0f}\u2192{cfg.zoom_initial_max_end:.0f}]")
    ax.plot(steps, zoom_lo, lw=2, color=COLORS["zoom"])
    ax.plot(steps, zoom_hi, lw=2, color=COLORS["zoom"], ls="--")

    # Show full zoom range
    ax.axhline(cfg.zoom_min, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(cfg.zoom_max, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(steps[-1] * 0.98, cfg.zoom_max + 0.5, f"Hardware max = {cfg.zoom_max:.0f}x",
            ha="right", fontsize=10, color="gray", alpha=0.6)

    ax.set_ylabel("Zoom level", fontsize=13)
    ax.set_title("Zoom Level Sampling Range", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    _format_step_axis(ax, curr.all_end_step)
    ax.set_ylim(0, cfg.zoom_max + 3)
    ax.grid(alpha=0.3)

    # ---- 3) Gimbal joint angle ranges ----
    ax = axes[1, 0]
    yaw_range = [math.degrees(cfg.gimbal_yaw_min), math.degrees(cfg.gimbal_yaw_max)]
    pitch_range = [math.degrees(cfg.gimbal_pitch_min), math.degrees(cfg.gimbal_pitch_max)]

    categories = ["Yaw", "Roll", "Pitch"]
    ranges_data = [
        (yaw_range[0], yaw_range[1]),
        (0.0, 0.0),  # Roll always 0
        (pitch_range[0], pitch_range[1]),
    ]
    bar_colors = [COLORS["gimbal"], "#95a5a6", COLORS["observer"]]

    x = np.arange(len(categories))
    for i, ((lo, hi), c) in enumerate(zip(ranges_data, bar_colors)):
        if lo == hi == 0:
            ax.bar(i, 0.5, bottom=-0.25, width=0.5, color=c, alpha=0.3,
                   edgecolor=c, lw=1.5)
            ax.text(i, 0, "0\n(stabilized)", ha="center", va="center",
                    fontsize=11, color=c, fontweight="bold")
        else:
            ax.bar(i, hi - lo, bottom=lo, width=0.5, color=c, alpha=0.5,
                   edgecolor=c, lw=1.5)
            ax.text(i, lo - 5, f"{lo:.0f}\u00b0", ha="center", va="top",
                    fontsize=12, fontweight="bold")
            ax.text(i, hi + 5, f"{hi:.0f}\u00b0", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylabel("Angle (degrees)", fontsize=13)
    ax.set_title("Gimbal Joint Angle Ranges (random mode)",
                 fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.3)

    # ---- 4) Orientation noise visualization ----
    ax = axes[1, 1]
    # Show heading distribution: uniform(0, 2pi) for random, or N(toward_target, noise_std)
    angles = np.linspace(-np.pi, np.pi, 500)

    # Random yaw: uniform
    uniform_pdf = np.ones_like(angles) / (2 * np.pi)

    # Face-target yaw: gaussian centered at 0 with noise_std
    sigma = cfg.orientation_noise_std
    gaussian_pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (angles / sigma) ** 2
    )
    # Normalize for visual comparison
    gaussian_pdf = gaussian_pdf / gaussian_pdf.max() * 3.0
    uniform_pdf = uniform_pdf / uniform_pdf.max() * 1.0

    ax.fill_between(np.degrees(angles), 0, uniform_pdf, alpha=0.2,
                    color=COLORS["agent"], label="Random yaw (other agents)")
    ax.plot(np.degrees(angles), uniform_pdf, lw=2, color=COLORS["agent"])

    ax.fill_between(np.degrees(angles), 0, gaussian_pdf, alpha=0.3,
                    color=COLORS["observer"],
                    label=f"Face-target + noise (\u03c3={math.degrees(sigma):.1f}\u00b0)")
    ax.plot(np.degrees(angles), gaussian_pdf, lw=2.5, color=COLORS["observer"])

    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.5, label="Toward target")

    ax.set_xlabel("Heading offset from target direction (\u00b0)", fontsize=12)
    ax.set_ylabel("Probability density (normalized)", fontsize=13)
    ax.set_title("Body Orientation Distribution", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(-180, 180)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Annotate current config
    obs_mode = ("Faces target" if cfg.designated_observer_faces_target
                else "Random")
    other_mode = cfg.other_agents_orientation_mode
    ax.text(0.02, 0.95,
            f"Observer: {obs_mode}\nOthers: {other_mode}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))

    # ---- 5) Designated observer mode comparison ----
    ax = axes[2, 0]
    n_resets = 30
    n_agents = 3
    rng = np.random.RandomState(123)

    modes = ["random", "fixed", "rotating"]
    mode_colors = [COLORS["agent"], COLORS["target"], COLORS["observer"]]

    for row_idx, (mode, color) in enumerate(zip(modes, mode_colors)):
        observers = []
        rotate_counter = 0
        for r in range(n_resets):
            if mode == "random":
                observers.append(rng.randint(0, n_agents))
            elif mode == "fixed":
                observers.append(0)
            elif mode == "rotating":
                observers.append(rotate_counter % n_agents)
                rotate_counter += 1

        y_base = row_idx * 1.5
        for r, obs in enumerate(observers):
            for a in range(n_agents):
                c = color if a == obs else "#d5dbdb"
                alpha = 1.0 if a == obs else 0.3
                ax.scatter(r, y_base + a * 0.35, s=80, c=c, alpha=alpha,
                           edgecolors="black" if a == obs else "gray",
                           linewidths=0.5, zorder=5)

        ax.text(-2.5, y_base + 0.35, f'"{mode}"', ha="right", va="center",
                fontsize=12, fontweight="bold", color=color)

    ax.set_xlabel("Reset index", fontsize=12)
    ax.set_yticks([0.35, 1.85, 3.35])
    ax.set_yticklabels(["Agent 0\nAgent 1\nAgent 2"] * 3, fontsize=8)
    ax.set_title("Designated Observer Selection Modes",
                 fontsize=15, fontweight="bold")
    ax.set_xlim(-4, n_resets)
    ax.grid(axis="x", alpha=0.2)

    # Annotate current mode
    ax.text(0.98, 0.05, f'Current: "{cfg.designated_observer_mode}"',
            transform=ax.transAxes, fontsize=11, ha="right", va="bottom",
            style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    # ---- 6) Summary parameter table ----
    ax = axes[2, 1]
    ax.axis("off")

    params = [
        ("Cylinder", ""),
        ("  Diameter", f"[{cfg.cylinder_diameter_min:.0f}, "
                       f"{cfg.cylinder_diameter_max:.0f}] m"),
        ("  Height (center)", f"[{cfg.cylinder_height_min:.0f}, "
                              f"{cfg.cylinder_height_max:.0f}] m"),
        ("  Vertical spread", f"[{cfg.cylinder_height_range_min:.0f}, "
                              f"{cfg.cylinder_height_range_max:.0f}] m"),
        ("  Agent clearance", f"{cfg.agent_clearance:.0f} m"),
        ("", ""),
        ("Target", ""),
        ("  Distance", f"[{cfg.target_distance_min:.0f}, "
                       f"{cfg.target_distance_max:.0f}] m"),
        ("  Height offset", f"[{cfg.target_height_offset_min:.0f}, "
                            f"+{cfg.target_height_offset_max:.0f}] m"),
        ("", ""),
        ("Velocity", ""),
        ("  Agent max", f"{cfg.agent_max_velocity:.0f} m/s"),
        ("  Target max", f"{cfg.target_max_velocity:.0f} m/s"),
        ("  Max yaw rate", f"{math.degrees(cfg.max_yaw_rate):.0f}\u00b0/s"),
        ("", ""),
        ("Gimbal", ""),
        ("  Yaw range", f"[{math.degrees(cfg.gimbal_yaw_min):.0f}, "
                        f"+{math.degrees(cfg.gimbal_yaw_max):.0f}]\u00b0"),
        ("  Pitch range", f"[{math.degrees(cfg.gimbal_pitch_min):.0f}, "
                          f"+{math.degrees(cfg.gimbal_pitch_max):.0f}]\u00b0"),
        ("  Curriculum mode", f'"{cfg.gimbal_curriculum_mode}"'),
        ("  Orientation noise", f"\u03c3 = {math.degrees(cfg.orientation_noise_std):.1f}\u00b0"),
        ("", ""),
        ("Zoom", ""),
        ("  Initial range", f"[{cfg.zoom_initial_min:.0f}, {cfg.zoom_initial_max_start:.0f}\u2192{cfg.zoom_initial_max_end:.0f}]x"),
        ("  Hardware range", f"[{cfg.zoom_min:.0f}, {cfg.zoom_max:.0f}]x"),
        ("", ""),
        ("Observer mode", f'"{cfg.designated_observer_mode}"'),
    ]

    y = 0.98
    for name, value in params:
        if name == "" and value == "":
            y -= 0.015
            continue
        if value == "":
            # Section header
            ax.text(0.05, y, name, transform=ax.transAxes, fontsize=13,
                    fontweight="bold", va="top")
        else:
            ax.text(0.05, y, name, transform=ax.transAxes, fontsize=11,
                    va="top", color="#555555")
            ax.text(0.55, y, value, transform=ax.transAxes, fontsize=11,
                    va="top", fontfamily="monospace")
        y -= 0.035

    ax.set_title("Parameter Summary (from InitialStatesCfg)",
                 fontsize=15, fontweight="bold")

    for row in axes:
        for a in row:
            a.tick_params(labelsize=11)

    fig.suptitle("iris_ma6 — Initial States Detail Panels",
                 fontsize=18, fontweight="bold", y=0.995)

    path = out_dir / "initial_states_details.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot iris_ma6 initial states configuration"
    )
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
