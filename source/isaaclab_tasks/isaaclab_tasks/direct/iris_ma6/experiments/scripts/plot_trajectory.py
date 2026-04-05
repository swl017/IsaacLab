#!/usr/bin/env python3
"""Plot trajectories of agents and target in 3D with viewing angle and gimbal/zoom panels.

Creates a 3-row x N-column figure:
    Row 0: 3D trajectory (with LOS lines every 2s)
    Row 1: Viewing angle + triangulation RMSE (dual axis)
    Row 2: Gimbal angles (yaw/pitch) + zoom level (dual axis)

Requires evaluation JSONs with a "trajectories" key (produced by evaluate.py
with --record-trajectory enabled).

Usage:
    python plot_trajectory.py
    python plot_trajectory.py --linear path/to/linear.json --circular path/to/circular.json
    python plot_trajectory.py --env-idx 3

Output: iris_ma6/experiments/scripts/outputs/trajectory.pdf
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "outputs", "trajectory.pdf")

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

# Colors
COLOR_TARGET = "#d62728"     # red
COLOR_DRONE0 = "#1f77b4"    # blue
COLOR_DRONE1 = "#ff7f0e"    # orange
COLOR_TRI_EST = "#2ca02c"   # green
AGENT_COLORS = [COLOR_DRONE0, COLOR_DRONE1, "#9467bd", "#8c564b"]

TIME_MARKER_INTERVAL = 10.0
LOS_INTERVAL = 2.0


def load_trajectories(json_path):
    """Load trajectory data from an evaluation JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    traj = data.get("trajectories")
    if traj is None:
        print(f"  WARNING: no 'trajectories' key in {json_path}")
        return None
    return traj


def select_best_env(traj_data):
    """Select the environment with the best trajectory for visualization."""
    best_score = -1
    best_env = None

    for env_id, env_traj in traj_data["envs"].items():
        n_steps = env_traj["num_steps"]
        if n_steps == 0:
            continue
        valid_count = sum(1 for v in env_traj["tri_valid"] if v)
        tri_ratio = valid_count / n_steps
        travel = 0.0
        for aid in traj_data["agent_ids"]:
            x = np.array(env_traj["agents"][aid]["x"])
            y = np.array(env_traj["agents"][aid]["y"])
            if len(x) > 1:
                travel += np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        score = n_steps * 1000 + tri_ratio * 100 + travel * 0.1
        if score > best_score:
            best_score = score
            best_env = (env_id, env_traj)
    return best_env


def _get_positions(env_traj, agent_ids):
    """Extract numpy arrays for target and agent positions, dropping the last point."""
    tgt = {k: np.array(env_traj["target"][k])[:-1] for k in ("x", "y", "z")}
    agents = {}
    for aid in agent_ids:
        agents[aid] = {k: np.array(env_traj["agents"][aid][k])[:-1] for k in ("x", "y", "z")}
    return tgt, agents


def _marker_and_los_indices(n_steps, step_dt):
    marker_steps = max(1, int(TIME_MARKER_INTERVAL / step_dt))
    los_steps = max(1, int(LOS_INTERVAL / step_dt))
    return (list(range(0, n_steps, marker_steps)),
            list(range(0, n_steps, los_steps)))


# ── 3D ─────────────────────────────────────────────────────────────────────

def plot_trajectory_3d(ax, env_traj, agent_ids, step_dt, title):
    """3D (XYZ) trajectory with LOS lines."""
    n_steps = env_traj["num_steps"] - 1
    tgt, agents = _get_positions(env_traj, agent_ids)
    marker_idx, los_idx = _marker_and_los_indices(n_steps, step_dt)

    # Target
    ax.plot(tgt["x"], tgt["y"], tgt["z"], color=COLOR_TARGET, linewidth=2.0, zorder=3)
    ax.scatter(tgt["x"][0], tgt["y"][0], tgt["z"][0],
               color=COLOR_TARGET, s=20, marker="o", zorder=5)
    ax.scatter(tgt["x"][-1], tgt["y"][-1], tgt["z"][-1],
               color=COLOR_TARGET, s=20, marker="^", zorder=5)

    # Agents
    for i, aid in enumerate(agent_ids):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        a = agents[aid]
        ax.plot(a["x"], a["y"], a["z"], color=c, linewidth=1.2, zorder=2)
        ax.scatter(a["x"][0], a["y"][0], a["z"][0], color=c, s=15, marker="o", zorder=5)
        ax.scatter(a["x"][-1], a["y"][-1], a["z"][-1], color=c, s=15, marker="^", zorder=5)

    # LOS lines
    for idx in los_idx:
        for i, aid in enumerate(agent_ids):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            a = agents[aid]
            ax.plot([a["x"][idx], tgt["x"][idx]],
                    [a["y"][idx], tgt["y"][idx]],
                    [a["z"][idx], tgt["z"][idx]],
                    color=c, linewidth=0.8, alpha=0.5, zorder=1)

    ax.view_init(elev=ax.elev, azim=ax.azim - 45)
    ax.set_xlabel("X (m)", labelpad=2)
    ax.set_ylabel("Y (m)", labelpad=2)
    ax.set_zlabel("Z (m)", labelpad=2)
    ax.tick_params(axis="both", pad=1, labelsize=6)
    ax.tick_params(axis="z", pad=1, labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.text2D(0.5, -0.2, title, transform=ax.transAxes, ha="center", fontsize=9)


# ── Viewing angle ──────────────────────────────────────────────────────────

def _smooth(arr, window):
    """Moving average that preserves NaN gaps."""
    kernel = np.ones(window) / window
    valid = ~np.isnan(arr)
    out = np.full_like(arr, np.nan)
    if valid.any():
        smoothed = np.convolve(np.where(valid, arr, 0.0), kernel, mode="same")
        counts = np.convolve(valid.astype(float), kernel, mode="same")
        mask = counts > 0
        out[mask] = smoothed[mask] / counts[mask]
    return out


def plot_viewing_angle(ax, env_traj, agent_ids, step_dt, title):
    """Viewing angle and per-step triangulation error over time."""
    n_steps = env_traj["num_steps"] - 1
    time_s = np.arange(n_steps) * step_dt
    angles = np.array(env_traj["viewing_angle"])[:-1]

    # Viewing angle (left axis)
    ln1 = ax.plot(time_s, angles, color="#008080", linewidth=1.0, label="Viewing angle")
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Viewing Angle (\u00b0)", color="#008080")
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.tick_params(axis="y", labelcolor="#008080")

    # Per-step RMSE (right axis), smoothed
    if "rmse" in env_traj:
        raw_rmse = env_traj["rmse"][:-1]
        rmse = np.array([v if v is not None else np.nan for v in raw_rmse], dtype=float)
    else:
        tri_x = np.array(env_traj["tri_estimate"]["x"][:-1], dtype=float)
        tri_y = np.array(env_traj["tri_estimate"]["y"][:-1], dtype=float)
        tri_z = np.array(env_traj["tri_estimate"]["z"][:-1], dtype=float)
        tgt_x = np.array(env_traj["target"]["x"][:-1], dtype=float)
        tgt_y = np.array(env_traj["target"]["y"][:-1], dtype=float)
        tgt_z = np.array(env_traj["target"]["z"][:-1], dtype=float)
        tri_valid = np.array(env_traj["tri_valid"][:-1], dtype=bool)
        rmse = np.sqrt((tri_x - tgt_x)**2 + (tri_y - tgt_y)**2 + (tri_z - tgt_z)**2)
        rmse[~tri_valid] = np.nan
    smooth_win = max(1, int(4.0 / step_dt))
    error_smooth = _smooth(rmse, smooth_win)

    ax2 = ax.twinx()
    ln2 = ax2.plot(time_s, error_smooth, linestyle="--", color="#8B008B", linewidth=0.8, alpha=0.7, label="RMSE")
    ax2.set_ylabel("Estimated Error (m)", color="#8B008B")
    ax2.tick_params(axis="y", labelcolor="#8B008B")

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=6)

    ax.set_xlabel(f"Time (s)\n{title}")
    ax.grid(True, alpha=0.3)


# ── Gimbal / Zoom ─────────────────────────────────────────────────────────

def plot_gimbal_zoom(ax, env_traj, agent_ids, step_dt, title):
    """Plot gimbal yaw/pitch angles and zoom level over time per agent."""
    n_steps = env_traj["num_steps"] - 1
    time_s = np.arange(n_steps) * step_dt

    lines_all = []

    # Gimbal angles (left axis, degrees)
    for i, aid in enumerate(agent_ids):
        agent_data = env_traj["agents"][aid]
        c = AGENT_COLORS[i % len(AGENT_COLORS)]

        has_gimbal = ("gimbal_yaw" in agent_data and len(agent_data["gimbal_yaw"]) > 1)
        if has_gimbal:
            yaw_deg = np.degrees(np.array(agent_data["gimbal_yaw"])[:-1])
            pitch_deg = np.degrees(np.array(agent_data["gimbal_pitch"])[:-1])
            ln_y = ax.plot(time_s, yaw_deg, color=c, linewidth=0.9,
                           linestyle="-", label=f"{aid} yaw")
            ln_p = ax.plot(time_s, pitch_deg, color=c, linewidth=0.9,
                           linestyle="--", alpha=0.7, label=f"{aid} pitch")
            lines_all.extend(ln_y + ln_p)

    ax.set_ylabel("Gimbal Angle (\u00b0)")
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.grid(True, alpha=0.3)

    # Zoom level (right axis)
    ax2 = ax.twinx()
    for i, aid in enumerate(agent_ids):
        agent_data = env_traj["agents"][aid]
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        has_zoom = ("zoom" in agent_data and len(agent_data["zoom"]) > 1)
        if has_zoom:
            zoom = np.array(agent_data["zoom"])[:-1]
            ln_z = ax2.plot(time_s, zoom, color=c, linewidth=0.8,
                            linestyle=":", alpha=0.6, label=f"{aid} zoom")
            lines_all.extend(ln_z)

    ax2.set_ylabel("Zoom Level")
    ax2.set_ylim(bottom=0)

    if lines_all:
        labels = [l.get_label() for l in lines_all]
        ax.legend(lines_all, labels, loc="upper right", fontsize=5, ncol=2)

    ax.set_xlabel(f"Time (s)\n{title}")


# ── Velocity ──────────────────────────────────────────────────────────────

def plot_velocity(ax, env_traj, agent_ids, step_dt, title):
    """Plot agent and target speed (velocity norm) over time."""
    n_steps = env_traj["num_steps"] - 1
    time_s = np.arange(n_steps) * step_dt

    # Target speed
    tgt = env_traj["target"]
    has_tgt_vel = ("vx" in tgt and len(tgt["vx"]) > 1)
    if has_tgt_vel:
        tvx = np.array(tgt["vx"])[:-1]
        tvy = np.array(tgt["vy"])[:-1]
        tvz = np.array(tgt["vz"])[:-1]
        tgt_speed = np.sqrt(tvx**2 + tvy**2 + tvz**2)
        ax.plot(time_s, tgt_speed, color=COLOR_TARGET, linewidth=1.5, label="Target")

    # Agent speeds
    for i, aid in enumerate(agent_ids):
        agent_data = env_traj["agents"][aid]
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        has_vel = ("vx" in agent_data and len(agent_data["vx"]) > 1)
        if has_vel:
            avx = np.array(agent_data["vx"])[:-1]
            avy = np.array(agent_data["vy"])[:-1]
            avz = np.array(agent_data["vz"])[:-1]
            agent_speed = np.sqrt(avx**2 + avy**2 + avz**2)
            ax.plot(time_s, agent_speed, color=c, linewidth=1.0, label=f"{aid}")

    ax.set_ylabel("Speed (m/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(f"Time (s)\n{title}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot trajectory visualizations.")
    parser.add_argument("--linear", type=str, default=None,
                        help="Path to linear-target eval JSON")
    parser.add_argument("--circular", type=str, default=None,
                        help="Path to circular-target eval JSON")
    parser.add_argument("--env-idx", type=int, default=None,
                        help="Force specific env index to plot")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output PDF path")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory to search for trajectory JSONs")
    args = parser.parse_args()

    # Auto-discover
    if args.linear is None:
        c = os.path.join(args.results_dir, "eval_traj_linear.json")
        if os.path.exists(c):
            args.linear = c
    if args.circular is None:
        c = os.path.join(args.results_dir, "eval_traj_circular.json")
        if os.path.exists(c):
            args.circular = c

    # Load
    panels = []
    if args.linear:
        traj = load_trajectories(args.linear)
        if traj is not None:
            panels.append((traj,
                           "(a) Linear target 3D trajectory",
                           "(c) Linear: Angle & RMSE",
                           "(e) Linear: Gimbal & Zoom",
                           "(g) Linear: Speed"))
    if args.circular:
        traj = load_trajectories(args.circular)
        if traj is not None:
            panels.append((traj,
                           "(b) Circular target 3D trajectory",
                           "(d) Circular: Angle & RMSE",
                           "(f) Circular: Gimbal & Zoom",
                           "(h) Circular: Speed"))

    if not panels:
        print("ERROR: No trajectory data found. Provide --linear and/or --circular paths.")
        sys.exit(1)

    n_cols = len(panels)

    # 4 rows: 3D, viewing angle, gimbal/zoom, velocity
    fig = plt.figure(figsize=(3.5 * n_cols + 0.16, 10.5))
    gs = fig.add_gridspec(4, n_cols, height_ratios=[1.3, 0.7, 0.7, 0.7],
                          hspace=0.35, wspace=0.55)

    for col, (traj_data, title_3d, angle_title, gimbal_title, vel_title) in enumerate(panels):
        step_dt = traj_data.get("step_dt_seconds", 0.04)
        agent_ids = traj_data["agent_ids"]

        # Select env
        if args.env_idx is not None:
            env_key = str(args.env_idx)
            if env_key not in traj_data["envs"]:
                sample_ids = traj_data["sample_env_ids"]
                if args.env_idx < len(sample_ids):
                    env_key = str(sample_ids[args.env_idx])
                else:
                    env_key, env_traj = select_best_env(traj_data)
            if env_key in traj_data["envs"]:
                env_traj = traj_data["envs"][env_key]
            else:
                env_key, env_traj = select_best_env(traj_data)
        else:
            env_key, env_traj = select_best_env(traj_data)

        print(f"  {title_3d}: env={env_key}, steps={env_traj['num_steps']}, "
              f"tri_valid={sum(1 for v in env_traj['tri_valid'] if v)}/{env_traj['num_steps']}")

        # Row 0: 3D
        ax_3d = fig.add_subplot(gs[0, col], projection="3d")
        plot_trajectory_3d(ax_3d, env_traj, agent_ids, step_dt, title_3d)

        # Row 1: Viewing angle
        ax_angle = fig.add_subplot(gs[1, col])
        plot_viewing_angle(ax_angle, env_traj, agent_ids, step_dt, angle_title)

        # Row 2: Gimbal / Zoom
        ax_gimbal = fig.add_subplot(gs[2, col])
        plot_gimbal_zoom(ax_gimbal, env_traj, agent_ids, step_dt, gimbal_title)

        # Row 3: Velocity
        ax_vel = fig.add_subplot(gs[3, col])
        plot_velocity(ax_vel, env_traj, agent_ids, step_dt, vel_title)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=COLOR_TARGET, linewidth=2.0, label="Target"),
    ]
    for i, aid in enumerate(panels[0][0]["agent_ids"]):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        legend_elements.append(Line2D([0], [0], color=c, linewidth=1.2, label=f"Agent {i}"))
    legend_elements.append(
        Line2D([0], [0], color="gray", linewidth=0.8, alpha=0.5, label="LOS (2 s)"))
    legend_elements.append(
        Line2D([0], [0], marker="o", color="gray", markersize=4, linewidth=0, label="Start"))
    legend_elements.append(
        Line2D([0], [0], marker="^", color="gray", markersize=4, linewidth=0, label="End"))

    fig.legend(handles=legend_elements, loc="upper center",
               ncol=len(legend_elements), frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.97), fontsize=7)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
