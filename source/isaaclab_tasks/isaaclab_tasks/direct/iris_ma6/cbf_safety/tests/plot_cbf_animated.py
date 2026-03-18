#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Animated CBF/CPA Visualization Script

Generates animated plots showing how CPA barrier and penalty evolve over time
as drones move. This helps build intuition about:
1. Why penalty converges to ~0.3 for head-on approach at any distance
2. How CPA "looks ahead" to predict collisions
3. The difference between current distance and predicted CPA distance

Animations:
1. Head-on approach: Two drones approaching, showing CPA point moving
2. Evasive maneuver: One drone turns to avoid collision
3. Parallel flight: Close but safe parallel trajectory

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_animated.py
    ./isaaclab.sh -p ... --output-dir ./cbf_animations --fps 30
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Animated CBF/CPA visualization")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./cbf_animations",
    help="Directory to save animations",
)
parser.add_argument(
    "--fps",
    type=int,
    default=20,
    help="Frames per second for animations",
)
parser.add_argument(
    "--duration",
    type=float,
    default=4.0,
    help="Animation duration in seconds",
)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab_tasks.direct.iris_ma6.cbf_safety import CPARewardShaper, CPARewardShaperCfg


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_shaper(device: torch.device) -> CPARewardShaper:
    """Create CPA reward shaper with default config."""
    cfg = CPARewardShaperCfg(
        D_s=2.0,      # 2m safety distance
        gamma=2.0,    # CBF decay rate
        T=1.0,        # 1s look-ahead
        lambda_cbf=1.0,
    )
    return CPARewardShaper(cfg, num_envs=1, num_agents=2, device=device)


def compute_cpa_manually(p0, p1, v0, v1, T=1.0, eps=1e-8):
    """Compute CPA information manually for visualization.

    Returns:
        dict with: tau, cpa_pos_0, cpa_pos_1, d_cpa, h_cpa
    """
    dp = np.array(p1) - np.array(p0)  # p1 - p0
    dv = np.array(v1) - np.array(v0)  # v1 - v0

    dv_sq = np.dot(dv, dv) + eps
    tau_star = -np.dot(dp, dv) / dv_sq
    tau = np.clip(tau_star, 0, T)

    # Positions at CPA
    cpa_pos_0 = np.array(p0) + tau * np.array(v0)
    cpa_pos_1 = np.array(p1) + tau * np.array(v1)

    # CPA distance
    d_cpa = np.linalg.norm(cpa_pos_1 - cpa_pos_0)

    # Barrier value
    D_s = 2.0
    h_cpa = d_cpa**2 - D_s**2

    return {
        "tau": tau,
        "cpa_pos_0": cpa_pos_0,
        "cpa_pos_1": cpa_pos_1,
        "d_cpa": d_cpa,
        "h_cpa": h_cpa,
    }


def animate_head_on_approach(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Animation 1: Head-on approach showing CPA prediction.

    Shows:
    - Two drones approaching each other
    - The CPA point (where they would be closest)
    - Real-time penalty value
    - Why penalty is constant even when far apart
    """
    print("Generating: Head-on approach animation...")

    # Parameters
    d_init = 12.0  # Initial separation
    v_rel = 6.0    # Relative speed (each moves at v_rel/2)
    dt_sim = 0.04  # Simulation timestep
    duration = args_cli.duration
    fps = args_cli.fps

    n_frames = int(duration * fps)
    dt_anim = duration / n_frames

    # Setup figure with multiple panels
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Main visualization panel
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlim(-2, 14)
    ax_main.set_ylim(-4, 4)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel("X (m)", fontsize=12)
    ax_main.set_ylabel("Y (m)", fontsize=12)
    ax_main.set_title("Head-on Approach: CPA Prediction", fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    # Initialize plot elements
    drone0, = ax_main.plot([], [], 'bo', markersize=15, label="Drone 0")
    drone1, = ax_main.plot([], [], 'ro', markersize=15, label="Drone 1")
    velocity0 = ax_main.quiver([], [], [], [], color='blue', scale=0.8, width=0.015, alpha=0.7)
    velocity1 = ax_main.quiver([], [], [], [], color='red', scale=0.8, width=0.015, alpha=0.7)
    cpa_point0, = ax_main.plot([], [], 'b^', markersize=10, alpha=0.5, label="CPA position")
    cpa_point1, = ax_main.plot([], [], 'r^', markersize=10, alpha=0.5)
    cpa_line, = ax_main.plot([], [], 'g--', linewidth=2, alpha=0.7, label="CPA distance")
    trajectory0, = ax_main.plot([], [], 'b--', alpha=0.3, linewidth=1)
    trajectory1, = ax_main.plot([], [], 'r--', alpha=0.3, linewidth=1)

    # Safety circles
    theta = np.linspace(0, 2*np.pi, 50)
    safety_circle0, = ax_main.plot([], [], 'b-', alpha=0.3, linewidth=1)
    safety_circle1, = ax_main.plot([], [], 'r-', alpha=0.3, linewidth=1)

    # Time and info text
    time_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes, fontsize=11,
                             verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_main.legend(loc='upper right')

    # Metrics panel - Distance
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_xlim(0, duration)
    ax_dist.set_ylim(0, d_init + 1)
    ax_dist.set_xlabel("Time (s)", fontsize=10)
    ax_dist.set_ylabel("Distance (m)", fontsize=10)
    ax_dist.set_title("Current Distance vs CPA Distance", fontsize=11)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.axhline(y=shaper.cfg.D_s, color='gray', linestyle=':', alpha=0.7, label=f"D_s={shaper.cfg.D_s}m")

    dist_line, = ax_dist.plot([], [], 'b-', linewidth=2, label="Current distance")
    cpa_dist_line, = ax_dist.plot([], [], 'g--', linewidth=2, label="CPA distance")
    dist_marker, = ax_dist.plot([], [], 'bo', markersize=8)
    cpa_dist_marker, = ax_dist.plot([], [], 'g^', markersize=8)
    ax_dist.legend(loc='upper right', fontsize=9)

    # Metrics panel - Penalty and Barrier
    ax_penalty = fig.add_subplot(gs[1, 1])
    ax_penalty.set_xlim(0, duration)
    ax_penalty.set_xlabel("Time (s)", fontsize=10)
    ax_penalty.set_ylabel("Penalty / Barrier", fontsize=10)
    ax_penalty.set_title("CPA Penalty & Barrier Value", fontsize=11)
    ax_penalty.grid(True, alpha=0.3)
    ax_penalty.axhline(y=0, color='gray', linestyle=':', alpha=0.7)

    penalty_line, = ax_penalty.plot([], [], 'r-', linewidth=2, label="Penalty")
    barrier_line, = ax_penalty.plot([], [], 'purple', linestyle='--', linewidth=2, label="h_CPA")
    penalty_marker, = ax_penalty.plot([], [], 'ro', markersize=8)
    barrier_marker, = ax_penalty.plot([], [], 'purple', marker='s', markersize=8, linestyle='')
    ax_penalty.legend(loc='upper right', fontsize=9)

    # Explanation panel
    ax_explain = fig.add_subplot(gs[2, :])
    ax_explain.axis('off')
    explanation_text = ax_explain.text(0.5, 0.5, '', transform=ax_explain.transAxes,
                                        fontsize=11, ha='center', va='center',
                                        fontfamily='monospace',
                                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Data storage for time series
    times = []
    distances = []
    cpa_distances = []
    penalties = []
    barriers = []

    def init():
        return (drone0, drone1, cpa_point0, cpa_point1, cpa_line, trajectory0, trajectory1,
                safety_circle0, safety_circle1, time_text, dist_line, cpa_dist_line,
                dist_marker, cpa_dist_marker, penalty_line, barrier_line, penalty_marker,
                barrier_marker, explanation_text)

    def animate(frame):
        t = frame * dt_anim

        # Positions (approaching from both sides)
        x0 = v_rel/2 * t
        x1 = d_init - v_rel/2 * t
        p0 = [x0, 0, 3]
        p1 = [x1, 0, 3]
        v0 = [v_rel/2, 0, 0]
        v1 = [-v_rel/2, 0, 0]

        # Current distance
        d_current = abs(x1 - x0)

        # Compute CPA
        cpa_info = compute_cpa_manually(p0, p1, v0, v1, T=shaper.cfg.T)

        # Compute penalty using the actual shaper
        positions = torch.tensor([[p0, p1]], dtype=torch.float32, device=device)
        velocities = torch.tensor([[v0, v1]], dtype=torch.float32, device=device)
        penalty = shaper.compute_penalty(positions, velocities, dt_sim).item()

        # Store data
        times.append(t)
        distances.append(d_current)
        cpa_distances.append(cpa_info["d_cpa"])
        penalties.append(penalty)
        barriers.append(cpa_info["h_cpa"])

        # Update main plot
        drone0.set_data([x0], [0])
        drone1.set_data([x1], [0])

        # Velocity arrows
        velocity0.set_offsets([[x0, 0]])
        velocity0.set_UVC([v_rel/2], [0])
        velocity1.set_offsets([[x1, 0]])
        velocity1.set_UVC([-v_rel/2], [0])

        # CPA points and line
        cpa_x0 = cpa_info["cpa_pos_0"][0]
        cpa_x1 = cpa_info["cpa_pos_1"][0]
        cpa_point0.set_data([cpa_x0], [0])
        cpa_point1.set_data([cpa_x1], [0])
        cpa_line.set_data([cpa_x0, cpa_x1], [0, 0])

        # Trajectories (showing where they'll go in next T seconds)
        t_future = np.linspace(0, shaper.cfg.T, 20)
        traj0_x = x0 + v_rel/2 * t_future
        traj1_x = x1 - v_rel/2 * t_future
        trajectory0.set_data(traj0_x, np.zeros_like(traj0_x))
        trajectory1.set_data(traj1_x, np.zeros_like(traj1_x))

        # Safety circles
        safety_circle0.set_data(x0 + shaper.cfg.D_s * np.cos(theta),
                                shaper.cfg.D_s * np.sin(theta))
        safety_circle1.set_data(x1 + shaper.cfg.D_s * np.cos(theta),
                                shaper.cfg.D_s * np.sin(theta))

        # Info text
        info = (f"Time: {t:.2f}s\n"
                f"Current distance: {d_current:.2f}m\n"
                f"CPA distance: {cpa_info['d_cpa']:.2f}m\n"
                f"Time to CPA: τ={cpa_info['tau']:.2f}s\n"
                f"Barrier h_CPA: {cpa_info['h_cpa']:.2f}\n"
                f"Penalty: {penalty:.3f}")
        time_text.set_text(info)

        # Update time series plots
        dist_line.set_data(times, distances)
        cpa_dist_line.set_data(times, cpa_distances)
        dist_marker.set_data([t], [d_current])
        cpa_dist_marker.set_data([t], [cpa_info["d_cpa"]])

        penalty_line.set_data(times, penalties)
        barrier_line.set_data(times, barriers)
        penalty_marker.set_data([t], [penalty])
        barrier_marker.set_data([t], [cpa_info["h_cpa"]])

        # Auto-scale penalty axis
        if len(penalties) > 1:
            p_min = min(min(penalties), min(barriers)) - 1
            p_max = max(max(penalties), max(barriers)) + 1
            ax_penalty.set_ylim(p_min, p_max)

        # Explanation based on phase
        if d_current > 8:
            explain = ("FAR APART but penalty ≈ 0.3\n"
                       "Why? CPA predicts collision!\n"
                       f"d_CPA = {cpa_info['d_cpa']:.2f}m < D_s = {shaper.cfg.D_s}m\n"
                       "Barrier h_CPA < 0 (unsafe trajectory)")
        elif d_current > 4:
            explain = ("APPROACHING\n"
                       f"Current: {d_current:.1f}m vs CPA: {cpa_info['d_cpa']:.2f}m\n"
                       "CPA always shows predicted collision\n"
                       "Penalty stays constant ≈ 0.3")
        else:
            explain = ("CLOSE! But CPA already knew...\n"
                       f"Current: {d_current:.1f}m, CPA: {cpa_info['d_cpa']:.2f}m\n"
                       "Penalty reflects rate of barrier decrease:\n"
                       "penalty = relu((1-γΔt)·h - h_next)")

        explanation_text.set_text(explain)

        return (drone0, drone1, cpa_point0, cpa_point1, cpa_line, trajectory0, trajectory1,
                safety_circle0, safety_circle1, time_text, dist_line, cpa_dist_line,
                dist_marker, cpa_dist_marker, penalty_line, barrier_line, penalty_marker,
                barrier_marker, explanation_text, velocity0, velocity1)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                   interval=1000/fps, blit=False)

    # Save animation
    output_path = output_dir / "anim_01_head_on_approach.mp4"
    print(f"  Saving to {output_path}...")
    anim.save(str(output_path), writer='ffmpeg', fps=fps, dpi=100)
    plt.close()

    # Also save a static summary plot
    fig_static, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, distances, 'b-', linewidth=2, label="Current distance")
    ax.plot(times, cpa_distances, 'g--', linewidth=2, label="CPA distance (predicted)")
    ax.fill_between(times, 0, shaper.cfg.D_s, alpha=0.2, color='red', label="Unsafe zone (d < D_s)")
    ax.axhline(y=shaper.cfg.D_s, color='r', linestyle=':', alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(times, penalties, 'r-', linewidth=2, alpha=0.7, label="Penalty")
    ax2.set_ylabel("Penalty", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Head-on Approach: CPA Predicts Collision Early")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate("CPA shows d=0\n(collision predicted)\neven at large distance",
                xy=(0.5, 0), xytext=(1.0, 4),
                arrowprops=dict(arrowstyle="->", color='green'),
                fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(output_dir / "static_01_head_on_summary.png", dpi=150)
    plt.close()


def animate_evasive_maneuver(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Animation 2: Evasive maneuver showing penalty reduction.

    One drone turns to avoid collision, CPA point moves, penalty drops.
    """
    print("Generating: Evasive maneuver animation...")

    # Parameters
    d_init = 8.0
    speed = 3.0
    dt_sim = 0.04
    duration = args_cli.duration
    fps = args_cli.fps
    n_frames = int(duration * fps)
    dt_anim = duration / n_frames

    # Setup figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlim(-2, 12)
    ax_main.set_ylim(-2, 8)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel("X (m)", fontsize=12)
    ax_main.set_ylabel("Y (m)", fontsize=12)
    ax_main.set_title("Evasive Maneuver: Turn Reduces Penalty", fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    # Plot elements
    drone0, = ax_main.plot([], [], 'bo', markersize=15, label="Drone 0 (evading)")
    drone1, = ax_main.plot([], [], 'ro', markersize=15, label="Drone 1 (straight)")
    cpa_line, = ax_main.plot([], [], 'g--', linewidth=2, alpha=0.7, label="CPA distance")
    trajectory0, = ax_main.plot([], [], 'b-', alpha=0.5, linewidth=1)
    trajectory1, = ax_main.plot([], [], 'r-', alpha=0.5, linewidth=1)

    theta_circle = np.linspace(0, 2*np.pi, 50)
    safety_circle0, = ax_main.plot([], [], 'b-', alpha=0.3, linewidth=1)

    time_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes, fontsize=11,
                             verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_main.legend(loc='upper right')

    # Penalty plot
    ax_penalty = fig.add_subplot(gs[1, 0])
    ax_penalty.set_xlim(0, duration)
    ax_penalty.set_xlabel("Time (s)")
    ax_penalty.set_ylabel("Penalty")
    ax_penalty.set_title("Penalty Over Time")
    ax_penalty.grid(True, alpha=0.3)

    penalty_line, = ax_penalty.plot([], [], 'r-', linewidth=2)
    penalty_marker, = ax_penalty.plot([], [], 'ro', markersize=8)

    # CPA distance plot
    ax_cpa = fig.add_subplot(gs[1, 1])
    ax_cpa.set_xlim(0, duration)
    ax_cpa.set_xlabel("Time (s)")
    ax_cpa.set_ylabel("CPA Distance (m)")
    ax_cpa.set_title("Predicted CPA Distance")
    ax_cpa.grid(True, alpha=0.3)
    ax_cpa.axhline(y=shaper.cfg.D_s, color='r', linestyle=':', alpha=0.7, label=f"D_s={shaper.cfg.D_s}m")

    cpa_dist_line, = ax_cpa.plot([], [], 'g-', linewidth=2)
    cpa_dist_marker, = ax_cpa.plot([], [], 'g^', markersize=8)
    ax_cpa.legend()

    # Data storage
    times = []
    penalties = []
    cpa_distances = []
    traj0_x = []
    traj0_y = []
    traj1_x = []
    traj1_y = []

    def get_turn_angle(t):
        """Drone 0 starts straight, then turns at t=1s."""
        turn_start = 1.0
        turn_duration = 0.8
        max_turn = 70  # degrees

        if t < turn_start:
            return 0
        elif t < turn_start + turn_duration:
            progress = (t - turn_start) / turn_duration
            return max_turn * progress
        else:
            return max_turn

    def init():
        return (drone0, drone1, cpa_line, trajectory0, trajectory1, safety_circle0,
                time_text, penalty_line, penalty_marker, cpa_dist_line, cpa_dist_marker)

    def animate(frame):
        t = frame * dt_anim

        # Drone 1: straight line from right
        x1 = d_init - speed * t
        y1 = 0
        v1 = [-speed, 0, 0]

        # Drone 0: starts going right, then turns up
        turn_angle = get_turn_angle(t)
        turn_rad = np.radians(turn_angle)

        # Integrate position based on accumulated trajectory
        if frame == 0:
            x0, y0 = 0, 0
        else:
            # Use stored trajectory
            x0 = traj0_x[-1] + speed * np.cos(turn_rad) * dt_anim
            y0 = traj0_y[-1] + speed * np.sin(turn_rad) * dt_anim

        v0 = [speed * np.cos(turn_rad), speed * np.sin(turn_rad), 0]

        p0 = [x0, y0, 3]
        p1 = [x1, y1, 3]

        # Store trajectories
        traj0_x.append(x0)
        traj0_y.append(y0)
        traj1_x.append(x1)
        traj1_y.append(y1)

        # Compute CPA and penalty
        cpa_info = compute_cpa_manually(p0, p1, v0, v1, T=shaper.cfg.T)

        positions = torch.tensor([[p0, p1]], dtype=torch.float32, device=device)
        velocities = torch.tensor([[v0, v1]], dtype=torch.float32, device=device)
        penalty = shaper.compute_penalty(positions, velocities, dt_sim).item()

        times.append(t)
        penalties.append(penalty)
        cpa_distances.append(cpa_info["d_cpa"])

        # Update plots
        drone0.set_data([x0], [y0])
        drone1.set_data([x1], [y1])

        cpa_x0 = cpa_info["cpa_pos_0"][0]
        cpa_y0 = cpa_info["cpa_pos_0"][1]
        cpa_x1 = cpa_info["cpa_pos_1"][0]
        cpa_y1 = cpa_info["cpa_pos_1"][1]
        cpa_line.set_data([cpa_x0, cpa_x1], [cpa_y0, cpa_y1])

        trajectory0.set_data(traj0_x, traj0_y)
        trajectory1.set_data(traj1_x, traj1_y)

        safety_circle0.set_data(x0 + shaper.cfg.D_s * np.cos(theta_circle),
                                y0 + shaper.cfg.D_s * np.sin(theta_circle))

        info = (f"Time: {t:.2f}s\n"
                f"Turn angle: {turn_angle:.0f}°\n"
                f"CPA distance: {cpa_info['d_cpa']:.2f}m\n"
                f"Penalty: {penalty:.3f}")
        time_text.set_text(info)

        penalty_line.set_data(times, penalties)
        penalty_marker.set_data([t], [penalty])

        cpa_dist_line.set_data(times, cpa_distances)
        cpa_dist_marker.set_data([t], [cpa_info["d_cpa"]])

        # Auto-scale
        if len(penalties) > 1:
            ax_penalty.set_ylim(0, max(penalties) * 1.2 + 0.1)
            ax_cpa.set_ylim(0, max(cpa_distances) * 1.2 + 1)

        return (drone0, drone1, cpa_line, trajectory0, trajectory1, safety_circle0,
                time_text, penalty_line, penalty_marker, cpa_dist_line, cpa_dist_marker)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                   interval=1000/fps, blit=False)

    output_path = output_dir / "anim_02_evasive_maneuver.mp4"
    print(f"  Saving to {output_path}...")
    anim.save(str(output_path), writer='ffmpeg', fps=fps, dpi=100)
    plt.close()


def animate_penalty_explanation(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Animation 3: Step-by-step penalty computation explanation.

    Shows exactly how the discrete-time CBF penalty formula works.
    """
    print("Generating: Penalty explanation animation...")

    # Parameters
    d_init = 10.0
    v_rel = 6.0
    dt_sim = 0.04
    duration = args_cli.duration
    fps = args_cli.fps
    n_frames = int(duration * fps)
    dt_anim = duration / n_frames

    gamma = shaper.cfg.gamma
    D_s = shaper.cfg.D_s
    T = shaper.cfg.T

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Understanding CPA Penalty Computation", fontsize=14, fontweight='bold')

    # Panel 1: Position diagram
    ax1 = axes[0, 0]
    ax1.set_xlim(-2, 12)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_title("Position & CPA")
    ax1.grid(True, alpha=0.3)

    drone0, = ax1.plot([], [], 'bo', markersize=12)
    drone1, = ax1.plot([], [], 'ro', markersize=12)
    cpa_pt0, = ax1.plot([], [], 'b^', markersize=8, alpha=0.6)
    cpa_pt1, = ax1.plot([], [], 'r^', markersize=8, alpha=0.6)
    cpa_line, = ax1.plot([], [], 'g--', linewidth=2, alpha=0.7)

    # Panel 2: Barrier values
    ax2 = axes[0, 1]
    ax2.set_title("Barrier Values")
    ax2.set_ylabel("Value")

    # Panel 3: Formula breakdown
    ax3 = axes[1, 0]
    ax3.axis('off')
    formula_text = ax3.text(0.5, 0.5, '', ha='center', va='center',
                            transform=ax3.transAxes, fontsize=11,
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Panel 4: Penalty over time
    ax4 = axes[1, 1]
    ax4.set_xlim(0, duration)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Penalty")
    ax4.set_title("Penalty Over Time")
    ax4.grid(True, alpha=0.3)

    penalty_line, = ax4.plot([], [], 'r-', linewidth=2)

    times = []
    penalties = []

    def init():
        return drone0, drone1, cpa_pt0, cpa_pt1, cpa_line, formula_text, penalty_line

    def animate(frame):
        t = frame * dt_anim

        # Positions
        x0 = v_rel/2 * t
        x1 = d_init - v_rel/2 * t

        p0 = [x0, 0, 3]
        p1 = [x1, 0, 3]
        v0 = [v_rel/2, 0, 0]
        v1 = [-v_rel/2, 0, 0]

        # CPA info
        cpa_info = compute_cpa_manually(p0, p1, v0, v1, T=T)

        # Compute h_current
        h_current = cpa_info["h_cpa"]

        # Compute h_next (after one dt)
        p0_next = [x0 + v_rel/2 * dt_sim, 0, 3]
        p1_next = [x1 - v_rel/2 * dt_sim, 0, 3]
        cpa_info_next = compute_cpa_manually(p0_next, p1_next, v0, v1, T=T)
        h_next = cpa_info_next["h_cpa"]

        # CBF condition
        required = (1.0 - gamma * dt_sim) * h_current
        violation = max(0, required - h_next)

        positions = torch.tensor([[p0, p1]], dtype=torch.float32, device=device)
        velocities = torch.tensor([[v0, v1]], dtype=torch.float32, device=device)
        penalty_actual = shaper.compute_penalty(positions, velocities, dt_sim).item()

        times.append(t)
        penalties.append(penalty_actual)

        # Update panel 1
        drone0.set_data([x0], [0])
        drone1.set_data([x1], [0])
        cpa_pt0.set_data([cpa_info["cpa_pos_0"][0]], [0])
        cpa_pt1.set_data([cpa_info["cpa_pos_1"][0]], [0])
        cpa_line.set_data([cpa_info["cpa_pos_0"][0], cpa_info["cpa_pos_1"][0]], [0, 0])

        # Update panel 2: Bar chart
        ax2.clear()
        ax2.set_title("Barrier Values")
        ax2.set_ylabel("Value")

        labels = ['h_current', f'(1-γΔt)·h', 'h_next', 'violation']
        values = [h_current, required, h_next, violation]
        colors = ['blue', 'orange', 'green', 'red']

        bars = ax2.bar(labels, values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

        # Update panel 3: Formula
        d_current = abs(x1 - x0)
        formula = (
            f"═══════════════════════════════════════════\n"
            f"           PENALTY COMPUTATION\n"
            f"═══════════════════════════════════════════\n\n"
            f"Given:\n"
            f"  Current distance:     d = {d_current:.2f} m\n"
            f"  CPA distance:         d_CPA = {cpa_info['d_cpa']:.2f} m\n"
            f"  Time to CPA:          τ = {cpa_info['tau']:.3f} s\n\n"
            f"Barrier values:\n"
            f"  h_current = d_CPA² - D_s² = {cpa_info['d_cpa']:.2f}² - {D_s}² = {h_current:.2f}\n"
            f"  h_next    = {h_next:.2f}\n\n"
            f"CBF condition:\n"
            f"  required = (1 - γΔt) · h_current\n"
            f"           = (1 - {gamma}×{dt_sim}) × {h_current:.2f}\n"
            f"           = {1-gamma*dt_sim:.3f} × {h_current:.2f} = {required:.2f}\n\n"
            f"  violation = max(0, required - h_next)\n"
            f"            = max(0, {required:.2f} - {h_next:.2f})\n"
            f"            = max(0, {required - h_next:.2f}) = {violation:.3f}\n"
            f"═══════════════════════════════════════════"
        )
        formula_text.set_text(formula)

        # Update panel 4
        penalty_line.set_data(times, penalties)
        if len(penalties) > 1:
            ax4.set_ylim(0, max(penalties) * 1.2 + 0.1)

        return drone0, drone1, cpa_pt0, cpa_pt1, cpa_line, formula_text, penalty_line

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                   interval=1000/fps, blit=False)

    output_path = output_dir / "anim_03_penalty_explained.mp4"
    print(f"  Saving to {output_path}...")
    anim.save(str(output_path), writer='ffmpeg', fps=fps, dpi=100)
    plt.close()


def create_static_explanation(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """Create static diagram explaining why penalty converges to ~0.3."""
    print("Generating: Static explanation diagram...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Why CPA Penalty Converges to ~0.3 at Large Distances",
                 fontsize=14, fontweight='bold')

    dt = 0.04
    gamma = shaper.cfg.gamma
    D_s = shaper.cfg.D_s

    # Panel 1: CPA prediction at different distances
    ax1 = axes[0, 0]
    distances = np.linspace(3, 15, 50)
    v_rel = 10.0  # High speed

    cpa_dists = []
    taus = []
    h_values = []

    for d in distances:
        dp = np.array([d, 0, 0])
        dv = np.array([-v_rel, 0, 0])

        tau_star = -np.dot(dp, dv) / (np.dot(dv, dv) + 1e-8)
        tau = np.clip(tau_star, 0, shaper.cfg.T)

        dp_cpa = dp + tau * dv
        d_cpa = np.linalg.norm(dp_cpa)
        h = d_cpa**2 - D_s**2

        cpa_dists.append(d_cpa)
        taus.append(tau)
        h_values.append(h)

    ax1.plot(distances, distances, 'b--', alpha=0.5, label='Current distance')
    ax1.plot(distances, cpa_dists, 'g-', linewidth=2, label='CPA distance')
    ax1.axhline(y=D_s, color='r', linestyle=':', alpha=0.7, label=f'D_s = {D_s}m')
    ax1.fill_between(distances, 0, D_s, alpha=0.1, color='red')
    ax1.set_xlabel("Current Distance (m)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("CPA Predicts Collision at All Distances")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.annotate("CPA = 0 (collision)\nfor head-on approach",
                xy=(10, 0), xytext=(8, 3),
                arrowprops=dict(arrowstyle="->", color='green'),
                fontsize=10, color='green')

    # Panel 2: Barrier value h
    ax2 = axes[0, 1]
    ax2.plot(distances, h_values, 'purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(y=-D_s**2, color='r', linestyle='--', alpha=0.7, label=f'h = -D_s² = -{D_s**2}')
    ax2.fill_between(distances, min(h_values), 0, alpha=0.1, color='red', label='Unsafe (h < 0)')
    ax2.set_xlabel("Current Distance (m)")
    ax2.set_ylabel("Barrier Value h")
    ax2.set_title("Barrier is Negative (Unsafe) for All Distances")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Penalty computation breakdown
    ax3 = axes[1, 0]

    # For fixed scenario: d=10, v_rel=10
    d = 10.0
    v_rel = 10.0
    h_current = 0 - D_s**2  # d_CPA = 0 for head-on
    h_next = 0 - D_s**2  # Same prediction next step

    required = (1.0 - gamma * dt) * h_current
    violation = max(0, required - h_next)

    # Bar chart showing computation
    categories = ['h_current', f'(1-γΔt)·h\n={1-gamma*dt:.3f}×h', 'h_next', 'violation']
    values = [h_current, required, h_next, violation]
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']

    bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linewidth=1)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')

    ax3.set_ylabel("Value")
    ax3.set_title(f"Penalty Computation (d={d}m, v_rel={v_rel}m/s)")
    ax3.set_ylim(-5, 1)

    # Add formula
    ax3.text(0.02, 0.95,
             f"penalty = max(0, (1-γΔt)·h - h_next)\n"
             f"        = max(0, {required:.2f} - ({h_next:.2f}))\n"
             f"        = max(0, {required - h_next:.2f})\n"
             f"        = {violation:.2f}",
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Panel 4: Penalty vs speed at large distance
    ax4 = axes[1, 1]

    d = 15.0  # Large distance
    speeds = np.linspace(1, 15, 50)
    penalties = []

    for v_rel in speeds:
        positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)
        velocities = torch.tensor([[[v_rel/2, 0, 0], [-v_rel/2, 0, 0]]], dtype=torch.float32, device=device)
        penalty = shaper.compute_penalty(positions, velocities, dt).item()
        penalties.append(penalty)

    ax4.plot(speeds, penalties, 'r-', linewidth=2)
    ax4.axhline(y=gamma * dt * D_s**2, color='gray', linestyle='--', alpha=0.7,
                label=f'≈ γ·Δt·D_s² = {gamma*dt*D_s**2:.2f}')
    ax4.set_xlabel("Relative Approach Speed (m/s)")
    ax4.set_ylabel("Penalty")
    ax4.set_title(f"Penalty at Large Distance (d={d}m)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax4.annotate(f"Converges to ~{gamma*dt*D_s**2:.2f}\n(for head-on approach)",
                xy=(10, gamma*dt*D_s**2), xytext=(6, 0.5),
                arrowprops=dict(arrowstyle="->", color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / "static_penalty_explanation.png", dpi=150)
    plt.close()

    # Create text explanation
    explanation = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    WHY CPA PENALTY CONVERGES TO ~0.32                           ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  For HEAD-ON APPROACH at any distance d, with relative velocity v_rel:          ║
║                                                                                 ║
║  1. TIME TO CPA:  τ* = d / v_rel                                               ║
║     - If τ* <= T (look-ahead horizon), τ = τ*                                  ║
║     - If τ* > T, τ = T (clamped)                                               ║
║                                                                                 ║
║  2. CPA DISTANCE: For head-on approach, CPA predicts collision                  ║
║     d_CPA = |d - v_rel × τ|                                                    ║
║     - If τ = τ* = d/v_rel: d_CPA = |d - v_rel × d/v_rel| = 0                   ║
║     - The CPA distance is ZERO regardless of current distance!                  ║
║                                                                                 ║
║  3. BARRIER VALUE:                                                              ║
║     h_CPA = d_CPA² - D_s² = 0 - 4 = -4  (for D_s = 2m)                         ║
║     h is NEGATIVE = unsafe trajectory predicted                                 ║
║                                                                                 ║
║  4. DISCRETE-TIME CBF CONDITION:                                                ║
║     violation = max(0, (1 - γ·Δt)·h_current - h_next)                          ║
║                                                                                 ║
║     With γ=2.0, Δt=0.04:                                                       ║
║     (1 - γ·Δt) = 1 - 2×0.04 = 0.92                                             ║
║                                                                                 ║
║     Since h_current ≈ h_next ≈ -4:                                              ║
║     violation = max(0, 0.92×(-4) - (-4))                                        ║
║              = max(0, -3.68 + 4)                                                ║
║              = max(0, 0.32)                                                     ║
║              = 0.32 ✓                                                           ║
║                                                                                 ║
║  INTERPRETATION:                                                                ║
║  The penalty ~0.32 represents the RATE at which the barrier is becoming         ║
║  more negative - i.e., how fast the system is heading toward collision.         ║
║  This is γ·Δt·|h| = 2×0.04×4 = 0.32 when h is stable at -4.                    ║
║                                                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

    with open(output_dir / "penalty_explanation.txt", 'w') as f:
        f.write(explanation)

    print(f"  Saved explanation to {output_dir}/penalty_explanation.txt")


def main():
    """Main function to generate all animations."""
    print("=" * 80)
    print("CBF ANIMATED VISUALIZATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = setup_output_dir(args_cli.output_dir)
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args_cli.fps}")
    print(f"Duration: {args_cli.duration}s")

    # Create shaper
    shaper = create_shaper(device)
    print(f"\nCPA Reward Shaper Config:")
    print(f"  D_s (safety distance): {shaper.cfg.D_s}m")
    print(f"  gamma (decay rate): {shaper.cfg.gamma}")
    print(f"  T (look-ahead): {shaper.cfg.T}s")

    print("\nGenerating visualizations...")

    # Static explanation first
    create_static_explanation(shaper, output_dir, device)

    # Animations
    animate_head_on_approach(shaper, output_dir, device)
    animate_evasive_maneuver(shaper, output_dir, device)
    animate_penalty_explanation(shaper, output_dir, device)

    print(f"\n{'=' * 80}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'=' * 80}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
