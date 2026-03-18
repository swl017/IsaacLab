#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CBF Reward Shaper Visualization Script

Generates plots to understand how the CPA-based CBF penalty works under different
scenarios. This helps build intuition about the velocity-aware barrier behavior.

Scenarios visualized:
1. Head-on approach at varying speeds
2. Parallel flight at varying separations
3. Crossing trajectories at different angles
4. CPA distance vs current distance comparison
5. Penalty landscape as function of relative position/velocity
6. Time evolution during approach

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_behavior.py
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cbf_safety/tests/plot_cbf_behavior.py --output-dir ./cbf_plots
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Visualize CBF reward shaper behavior")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./cbf_behavior_plots",
    help="Directory to save plots",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Show plots interactively (requires display)",
)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
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
    # 2 agents for pairwise analysis
    return CPARewardShaper(cfg, num_envs=1, num_agents=2, device=device)


def plot_head_on_approach(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 1: Head-on approach at varying speeds.

    Two drones facing each other, approaching at different relative velocities.
    Shows how penalty increases with approach speed and decreases with distance.
    """
    print("Generating: Head-on approach plot...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Parameters
    distances = np.linspace(1.0, 10.0, 50)  # Distance between drones
    approach_speeds = [0.5, 1.0, 2.0, 5.0, 10.0]  # Relative approach speeds (m/s)
    dt = 0.04  # 25 Hz

    # Plot 1a: Penalty vs Distance at different speeds
    ax = axes[0]
    for v_rel in approach_speeds:
        penalties = []
        for d in distances:
            # Agent 0 at origin, Agent 1 at (d, 0, 0)
            positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)
            # Approaching each other
            velocities = torch.tensor([[[v_rel/2, 0, 0], [-v_rel/2, 0, 0]]], dtype=torch.float32, device=device)

            penalty = shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        ax.plot(distances, penalties, label=f"v_rel={v_rel} m/s")

    ax.axvline(x=shaper.cfg.D_s, color='r', linestyle='--', alpha=0.5, label=f"D_s={shaper.cfg.D_s}m")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title("Head-on Approach: Penalty vs Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 10])

    # Plot 1b: Penalty vs Approach Speed at different distances
    ax = axes[1]
    speeds = np.linspace(0, 15, 50)
    fixed_distances = [2.5, 3.0, 4.0, 5.0, 7.0]

    for d in fixed_distances:
        penalties = []
        for v_rel in speeds:
            positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)
            velocities = torch.tensor([[[v_rel/2, 0, 0], [-v_rel/2, 0, 0]]], dtype=torch.float32, device=device)

            penalty = shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        ax.plot(speeds, penalties, label=f"d={d}m")

    ax.set_xlabel("Relative Approach Speed (m/s)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title("Head-on Approach: Penalty vs Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 1c: CPA distance visualization
    ax = axes[2]
    d = 5.0  # Fixed initial distance
    v_rel_values = np.linspace(0, 10, 50)

    cpa_distances = []
    tau_values = []

    for v_rel in v_rel_values:
        # CPA calculation
        dp = np.array([d, 0, 0])
        dv = np.array([-v_rel, 0, 0])  # Approaching

        dv_sq = np.dot(dv, dv) + 1e-8
        tau_star = -np.dot(dp, dv) / dv_sq
        tau = np.clip(tau_star, 0, shaper.cfg.T)

        dp_cpa = dp + tau * dv
        d_cpa = np.linalg.norm(dp_cpa)

        cpa_distances.append(d_cpa)
        tau_values.append(tau)

    ax.plot(v_rel_values, cpa_distances, 'b-', label="CPA Distance")
    ax.axhline(y=d, color='gray', linestyle=':', alpha=0.5, label=f"Initial Distance ({d}m)")
    ax.axhline(y=shaper.cfg.D_s, color='r', linestyle='--', alpha=0.5, label=f"D_s={shaper.cfg.D_s}m")

    ax2 = ax.twinx()
    ax2.plot(v_rel_values, tau_values, 'g--', alpha=0.7, label="τ (time to CPA)")
    ax2.set_ylabel("Time to CPA (s)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax.set_xlabel("Relative Approach Speed (m/s)")
    ax.set_ylabel("CPA Distance (m)")
    ax.set_title(f"CPA Analysis (initial d={d}m)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "01_head_on_approach.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def plot_parallel_flight(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 2: Parallel flight at varying separations.

    Two drones flying in the same direction at varying lateral separations.
    Shows that CPA barrier has minimal penalty for parallel trajectories.
    """
    print("Generating: Parallel flight plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Parameters
    separations = np.linspace(1.0, 10.0, 50)
    speeds = [1.0, 5.0, 10.0, 15.0]
    dt = 0.04

    # Plot 2a: Penalty vs Separation for parallel flight
    ax = axes[0]
    for speed in speeds:
        penalties = []
        for sep in separations:
            # Both agents moving in +x direction at same speed
            positions = torch.tensor([[[0, 0, 3], [0, sep, 3]]], dtype=torch.float32, device=device)
            velocities = torch.tensor([[[speed, 0, 0], [speed, 0, 0]]], dtype=torch.float32, device=device)

            penalty = shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        ax.plot(separations, penalties, label=f"speed={speed} m/s")

    ax.axvline(x=shaper.cfg.D_s, color='r', linestyle='--', alpha=0.5, label=f"D_s={shaper.cfg.D_s}m")
    ax.set_xlabel("Lateral Separation (m)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title("Parallel Flight: Penalty vs Separation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2b: Compare parallel vs approaching at same distance
    ax = axes[1]
    separations = np.linspace(1.5, 8.0, 50)
    speed = 5.0

    parallel_penalties = []
    approaching_penalties = []

    for sep in separations:
        # Parallel
        positions = torch.tensor([[[0, 0, 3], [0, sep, 3]]], dtype=torch.float32, device=device)
        velocities_par = torch.tensor([[[speed, 0, 0], [speed, 0, 0]]], dtype=torch.float32, device=device)
        penalty_par = shaper.compute_penalty(positions, velocities_par, dt)
        parallel_penalties.append(penalty_par.item())

        # Approaching (head-on in y-direction)
        velocities_app = torch.tensor([[[0, speed/2, 0], [0, -speed/2, 0]]], dtype=torch.float32, device=device)
        penalty_app = shaper.compute_penalty(positions, velocities_app, dt)
        approaching_penalties.append(penalty_app.item())

    ax.plot(separations, parallel_penalties, 'b-', label="Parallel flight", linewidth=2)
    ax.plot(separations, approaching_penalties, 'r-', label="Approaching", linewidth=2)
    ax.axvline(x=shaper.cfg.D_s, color='gray', linestyle='--', alpha=0.5, label=f"D_s={shaper.cfg.D_s}m")
    ax.set_xlabel("Separation (m)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title(f"Parallel vs Approaching (speed={speed} m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=0.01)

    plt.tight_layout()
    plt.savefig(output_dir / "02_parallel_flight.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def plot_crossing_trajectories(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 3: Crossing trajectories at different angles.

    Two drones with crossing paths at various intersection angles.
    Shows how the CPA barrier captures collision risk from velocity vectors.
    """
    print("Generating: Crossing trajectories plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dt = 0.04
    speed = 5.0

    # Plot 3a: Penalty vs crossing angle
    ax = axes[0]
    angles = np.linspace(0, 180, 100)  # Degrees
    initial_distances = [3.0, 4.0, 5.0, 6.0]

    for d_init in initial_distances:
        penalties = []
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)

            # Agent 0 at origin moving in +x
            # Agent 1 at (d_init, 0) moving at angle
            positions = torch.tensor([[[0, 0, 3], [d_init, 0, 3]]], dtype=torch.float32, device=device)

            v0 = [speed, 0, 0]
            v1 = [-speed * np.cos(angle_rad), speed * np.sin(angle_rad), 0]
            velocities = torch.tensor([[v0, v1]], dtype=torch.float32, device=device)

            penalty = shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        ax.plot(angles, penalties, label=f"d={d_init}m")

    ax.set_xlabel("Crossing Angle (degrees)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title(f"Crossing Trajectories (speed={speed} m/s each)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 180])

    # Plot 3b: 2D visualization of crossing scenario
    ax = axes[1]

    # Show trajectory lines for a specific scenario
    d_init = 4.0
    crossing_angle = 45  # degrees
    angle_rad = np.radians(crossing_angle)

    t_range = np.linspace(-0.5, 1.5, 100)

    # Agent 0 trajectory
    x0 = speed * t_range
    y0 = np.zeros_like(t_range)

    # Agent 1 trajectory
    x1 = d_init - speed * np.cos(angle_rad) * t_range
    y1 = speed * np.sin(angle_rad) * t_range

    ax.plot(x0, y0, 'b-', label="Agent 0", linewidth=2)
    ax.plot(x1, y1, 'r-', label="Agent 1", linewidth=2)

    # Mark initial positions
    ax.scatter([0, d_init], [0, 0], s=100, c=['blue', 'red'], marker='o', zorder=5)

    # Calculate and mark CPA point
    dp = np.array([d_init, 0, 0])
    dv = np.array([-speed - speed * np.cos(angle_rad), speed * np.sin(angle_rad), 0])
    dv_sq = np.dot(dv, dv) + 1e-8
    tau_star = -np.dot(dp, dv) / dv_sq
    tau = np.clip(tau_star, 0, shaper.cfg.T)

    # Positions at CPA
    x0_cpa = speed * tau
    y0_cpa = 0
    x1_cpa = d_init - speed * np.cos(angle_rad) * tau
    y1_cpa = speed * np.sin(angle_rad) * tau

    ax.scatter([x0_cpa, x1_cpa], [y0_cpa, y1_cpa], s=150, c=['blue', 'red'], marker='x', zorder=5)
    ax.plot([x0_cpa, x1_cpa], [y0_cpa, y1_cpa], 'k--', alpha=0.5, label=f"CPA (d={np.sqrt((x1_cpa-x0_cpa)**2 + (y1_cpa-y0_cpa)**2):.2f}m)")

    # Safety circle around agent 0 at CPA
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(x0_cpa + shaper.cfg.D_s * np.cos(theta), y0_cpa + shaper.cfg.D_s * np.sin(theta),
            'g--', alpha=0.5, label=f"Safety radius D_s={shaper.cfg.D_s}m")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Crossing at {crossing_angle}° (τ={tau:.2f}s)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-2, 8])
    ax.set_ylim([-2, 6])

    plt.tight_layout()
    plt.savefig(output_dir / "03_crossing_trajectories.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def plot_penalty_landscape(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 4: 2D penalty landscape as function of relative position.

    Heatmap showing penalty for different relative positions when
    approaching at a fixed velocity.
    """
    print("Generating: Penalty landscape plot...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dt = 0.04

    # Grid of relative positions
    x_range = np.linspace(-8, 8, 80)
    y_range = np.linspace(-8, 8, 80)
    X, Y = np.meshgrid(x_range, y_range)

    scenarios = [
        ("Agent 1 moving +X", [5.0, 0, 0]),
        ("Agent 1 stationary", [0.0, 0, 0]),
        ("Agent 1 moving toward origin", None),  # Computed dynamically
    ]

    for ax_idx, (title, v1_base) in enumerate(scenarios):
        ax = axes[ax_idx]
        penalties = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                d = np.sqrt(x**2 + y**2)
                if d < 0.5:  # Skip very close positions
                    penalties[i, j] = np.nan
                    continue

                positions = torch.tensor([[[0, 0, 3], [x, y, 3]]], dtype=torch.float32, device=device)

                if v1_base is None:
                    # Moving toward origin
                    speed = 5.0
                    dir_x = -x / d if d > 0 else 0
                    dir_y = -y / d if d > 0 else 0
                    v1 = [speed * dir_x, speed * dir_y, 0]
                else:
                    v1 = v1_base

                velocities = torch.tensor([[[0, 0, 0], v1]], dtype=torch.float32, device=device)
                penalty = shaper.compute_penalty(positions, velocities, dt)
                penalties[i, j] = penalty.item()

        # Plot heatmap
        im = ax.pcolormesh(X, Y, penalties, cmap='hot_r', shading='auto')
        plt.colorbar(im, ax=ax, label="CPA Penalty")

        # Safety circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(shaper.cfg.D_s * np.cos(theta), shaper.cfg.D_s * np.sin(theta),
                'g-', linewidth=2, label=f"D_s={shaper.cfg.D_s}m")

        # Agent 0 position
        ax.scatter([0], [0], c='blue', s=100, marker='o', label="Agent 0", zorder=5)

        ax.set_xlabel("Relative X (m)")
        ax.set_ylabel("Relative Y (m)")
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])

    plt.tight_layout()
    plt.savefig(output_dir / "04_penalty_landscape.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def plot_time_evolution(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 5: Time evolution of penalty during approach.

    Shows how penalty changes over time as two drones approach and pass each other.
    """
    print("Generating: Time evolution plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    dt = 0.04

    # Scenario: Two drones starting 10m apart, approaching
    d_init = 10.0
    speeds = [2.0, 5.0, 10.0]

    # Plot 5a: Distance and CPA distance over time
    ax = axes[0, 0]

    for speed in speeds:
        time_steps = np.arange(0, d_init / speed + 0.5, dt)
        distances = []
        cpa_distances = []

        for t in time_steps:
            # Current positions
            x0 = speed * t
            x1 = d_init - speed * t
            d = abs(x1 - x0)
            distances.append(d)

            # CPA calculation
            dp = np.array([x1 - x0, 0, 0])
            dv = np.array([-2 * speed, 0, 0])
            dv_sq = np.dot(dv, dv) + 1e-8
            tau_star = -np.dot(dp, dv) / dv_sq
            tau = np.clip(tau_star, 0, shaper.cfg.T)
            dp_cpa = dp + tau * dv
            d_cpa = np.linalg.norm(dp_cpa)
            cpa_distances.append(d_cpa)

        ax.plot(time_steps, distances, '-', label=f"Distance (v={speed} m/s)")
        ax.plot(time_steps, cpa_distances, '--', alpha=0.7, label=f"CPA dist (v={speed} m/s)")

    ax.axhline(y=shaper.cfg.D_s, color='r', linestyle=':', alpha=0.5, label=f"D_s={shaper.cfg.D_s}m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance and CPA Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5b: Penalty over time
    ax = axes[0, 1]

    for speed in speeds:
        time_steps = np.arange(0, d_init / speed + 0.5, dt)
        penalties = []

        for t in time_steps:
            x0 = speed * t
            x1 = d_init - speed * t

            positions = torch.tensor([[[x0, 0, 3], [x1, 0, 3]]], dtype=torch.float32, device=device)
            velocities = torch.tensor([[[speed, 0, 0], [-speed, 0, 0]]], dtype=torch.float32, device=device)

            penalty = shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        ax.plot(time_steps, penalties, label=f"v={speed} m/s")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title("Penalty Evolution During Approach")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5c: Cumulative penalty
    ax = axes[1, 0]

    for speed in speeds:
        time_steps = np.arange(0, d_init / speed + 0.5, dt)
        penalties = []

        for t in time_steps:
            x0 = speed * t
            x1 = d_init - speed * t

            positions = torch.tensor([[[x0, 0, 3], [x1, 0, 3]]], dtype=torch.float32, device=device)
            velocities = torch.tensor([[[speed, 0, 0], [-speed, 0, 0]]], dtype=torch.float32, device=device)

            penalty = shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        cumulative = np.cumsum(penalties)
        ax.plot(time_steps, cumulative, label=f"v={speed} m/s")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Penalty")
    ax.set_title("Cumulative Penalty (Total Cost)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5d: Barrier value h over time
    ax = axes[1, 1]
    speed = 5.0
    time_steps = np.arange(0, d_init / speed + 0.5, dt)

    h_distance = []  # Simple distance barrier
    h_cpa = []       # CPA barrier

    for t in time_steps:
        x0 = speed * t
        x1 = d_init - speed * t

        # Distance barrier: h = d^2 - D_s^2
        d = abs(x1 - x0)
        h_d = d**2 - shaper.cfg.D_s**2
        h_distance.append(h_d)

        # CPA barrier
        dp = np.array([x1 - x0, 0, 0])
        dv = np.array([-2 * speed, 0, 0])
        dv_sq = np.dot(dv, dv) + 1e-8
        tau_star = -np.dot(dp, dv) / dv_sq
        tau = np.clip(tau_star, 0, shaper.cfg.T)
        dp_cpa = dp + tau * dv
        d_cpa_sq = np.dot(dp_cpa, dp_cpa)
        h_c = d_cpa_sq - shaper.cfg.D_s**2
        h_cpa.append(h_c)

    ax.plot(time_steps, h_distance, 'b-', label="h_distance = d² - D_s²", linewidth=2)
    ax.plot(time_steps, h_cpa, 'r-', label="h_CPA = d_CPA² - D_s²", linewidth=2)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, label="h=0 (boundary)")
    ax.fill_between(time_steps, h_cpa, 0, where=np.array(h_cpa) < 0, alpha=0.3, color='red', label="Unsafe region")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Barrier Value h")
    ax.set_title(f"Barrier Values (v={speed} m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "05_time_evolution.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def plot_gamma_effect(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 6: Effect of gamma (CBF decay rate) on penalty.

    Shows how different gamma values affect the penalty magnitude.
    """
    print("Generating: Gamma effect plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dt = 0.04

    # Plot 6a: Penalty vs gamma
    ax = axes[0]
    gammas = [0.5, 1.0, 2.0, 3.0, 5.0]
    distances = np.linspace(2.0, 8.0, 50)
    speed = 5.0

    for gamma in gammas:
        cfg = CPARewardShaperCfg(D_s=2.0, gamma=gamma, T=1.0)
        temp_shaper = CPARewardShaper(cfg, num_envs=1, num_agents=2, device=device)

        penalties = []
        for d in distances:
            positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)
            velocities = torch.tensor([[[speed/2, 0, 0], [-speed/2, 0, 0]]], dtype=torch.float32, device=device)

            penalty = temp_shaper.compute_penalty(positions, velocities, dt)
            penalties.append(penalty.item())

        ax.plot(distances, penalties, label=f"γ={gamma}")

    ax.axvline(x=2.0, color='r', linestyle='--', alpha=0.5, label="D_s=2.0m")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title(f"Effect of γ on Penalty (approach speed={speed} m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6b: Penalty contours in (distance, gamma) space
    ax = axes[1]
    gammas = np.linspace(0.5, 5.0, 50)
    distances = np.linspace(2.0, 8.0, 50)
    G, D = np.meshgrid(gammas, distances)
    penalties = np.zeros_like(G)

    for i, d in enumerate(distances):
        for j, gamma in enumerate(gammas):
            cfg = CPARewardShaperCfg(D_s=2.0, gamma=gamma, T=1.0)
            temp_shaper = CPARewardShaper(cfg, num_envs=1, num_agents=2, device=device)

            positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)
            velocities = torch.tensor([[[2.5, 0, 0], [-2.5, 0, 0]]], dtype=torch.float32, device=device)

            penalty = temp_shaper.compute_penalty(positions, velocities, dt)
            penalties[i, j] = penalty.item()

    cs = ax.contourf(G, D, penalties, levels=20, cmap='hot_r')
    plt.colorbar(cs, ax=ax, label="CPA Penalty")

    ax.set_xlabel("Gamma (γ)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Penalty Contours (approach speed=5 m/s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "06_gamma_effect.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def plot_distance_vs_cpa_comparison(shaper: CPARewardShaper, output_dir: Path, device: torch.device):
    """
    Plot 7: Comparison of distance-based vs CPA-based penalties.

    Illustrates the key advantage of CPA: velocity awareness.
    """
    print("Generating: Distance vs CPA comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    dt = 0.04

    # Scenario 1: Side-by-side at close range (parallel flight)
    ax = axes[0, 0]
    separation = 2.5  # Close to D_s
    speeds = np.linspace(0, 15, 50)

    # Same direction
    cpa_same_dir = []
    # Opposite direction (approaching)
    cpa_opposite = []
    # Distance-based penalty (constant regardless of velocity)
    dist_penalty = (shaper.cfg.D_s**2 - separation**2)  # Would be constant

    for speed in speeds:
        positions = torch.tensor([[[0, 0, 3], [0, separation, 3]]], dtype=torch.float32, device=device)

        # Same direction
        vel_same = torch.tensor([[[speed, 0, 0], [speed, 0, 0]]], dtype=torch.float32, device=device)
        p_same = shaper.compute_penalty(positions, vel_same, dt)
        cpa_same_dir.append(p_same.item())

        # Opposite (approaching in y)
        vel_opp = torch.tensor([[[0, speed/2, 0], [0, -speed/2, 0]]], dtype=torch.float32, device=device)
        p_opp = shaper.compute_penalty(positions, vel_opp, dt)
        cpa_opposite.append(p_opp.item())

    ax.plot(speeds, cpa_same_dir, 'b-', label="CPA: Parallel flight", linewidth=2)
    ax.plot(speeds, cpa_opposite, 'r-', label="CPA: Approaching", linewidth=2)
    ax.axhline(y=max(0, -dist_penalty) * 0.1, color='gray', linestyle='--', alpha=0.5,
               label="Distance-only (constant)")

    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Penalty")
    ax.set_title(f"Parallel vs Approaching (separation={separation}m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scenario 2: Different approach angles
    ax = axes[0, 1]
    d = 4.0
    speed = 5.0
    angles = np.linspace(0, 180, 100)

    cpa_penalties = []
    for angle in angles:
        angle_rad = np.radians(angle)
        positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)

        # Agent 0 stationary, Agent 1 approaching at angle
        v1 = [-speed * np.cos(angle_rad), speed * np.sin(angle_rad), 0]
        velocities = torch.tensor([[[0, 0, 0], v1]], dtype=torch.float32, device=device)

        penalty = shaper.compute_penalty(positions, velocities, dt)
        cpa_penalties.append(penalty.item())

    ax.plot(angles, cpa_penalties, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Approach Angle (degrees)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title(f"Penalty vs Approach Angle (d={d}m, speed={speed}m/s)")
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate("Head-on\n(max penalty)", xy=(0, cpa_penalties[0]), xytext=(30, cpa_penalties[0]*0.8),
                arrowprops=dict(arrowstyle="->", color='red'), fontsize=9)
    ax.annotate("Perpendicular\n(moderate)", xy=(90, cpa_penalties[50]), xytext=(70, cpa_penalties[50]*1.5),
                arrowprops=dict(arrowstyle="->", color='orange'), fontsize=9)

    # Scenario 3: Evasive maneuver
    ax = axes[1, 0]

    # Initial: approaching
    # Then: turn to evade
    d = 5.0
    speed = 5.0
    turn_angles = np.linspace(0, 90, 50)  # Agent 1 turns from head-on to perpendicular

    penalties_evasion = []
    for turn in turn_angles:
        turn_rad = np.radians(turn)
        positions = torch.tensor([[[0, 0, 3], [d, 0, 3]]], dtype=torch.float32, device=device)

        # Agent 0 moving +x, Agent 1 starts moving -x then turns
        v0 = [speed, 0, 0]
        v1 = [-speed * np.cos(turn_rad), speed * np.sin(turn_rad), 0]
        velocities = torch.tensor([[v0, v1]], dtype=torch.float32, device=device)

        penalty = shaper.compute_penalty(positions, velocities, dt)
        penalties_evasion.append(penalty.item())

    ax.plot(turn_angles, penalties_evasion, 'g-', linewidth=2)
    ax.set_xlabel("Evasion Turn Angle (degrees)")
    ax.set_ylabel("CPA Penalty")
    ax.set_title("Evasive Maneuver: Turn Reduces Penalty")
    ax.grid(True, alpha=0.3)
    ax.annotate("Head-on", xy=(0, penalties_evasion[0]), xytext=(15, penalties_evasion[0]*0.9),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.annotate("90° turn\n(safe)", xy=(90, penalties_evasion[-1]), xytext=(70, penalties_evasion[0]*0.3),
                arrowprops=dict(arrowstyle="->"), fontsize=9)

    # Scenario 4: Text summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = """
    CPA Barrier Advantages over Distance-Based:

    1. Velocity-Aware:
       • Parallel flight at close range: LOW penalty
       • Approaching at same range: HIGH penalty
       • Distance-only would penalize both equally

    2. Directional Information:
       • Distinguishes approach angle (0° vs 90° vs 180°)
       • Rewards evasive maneuvers immediately
       • Provides gradient for collision avoidance learning

    3. Predictive:
       • Looks ahead T seconds (horizon)
       • Catches approaching threats early
       • Allows smoother avoidance trajectories

    4. Less Conservative:
       • Allows close parallel operation
       • Enables formation flying
       • Better task-safety tradeoff
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title("Summary: Why CPA > Distance-Only")

    plt.tight_layout()
    plt.savefig(output_dir / "07_distance_vs_cpa.png", dpi=150)
    if args_cli.show:
        plt.show()
    plt.close()


def main():
    """Main function to generate all plots."""
    print("=" * 80)
    print("CBF REWARD SHAPER VISUALIZATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = setup_output_dir(args_cli.output_dir)
    print(f"Output directory: {output_dir}")

    # Create shaper
    shaper = create_shaper(device)
    print(f"\nCPA Reward Shaper Config:")
    print(f"  D_s (safety distance): {shaper.cfg.D_s}m")
    print(f"  gamma (decay rate): {shaper.cfg.gamma}")
    print(f"  T (look-ahead): {shaper.cfg.T}s")

    print("\nGenerating plots...")

    # Generate all plots
    plot_head_on_approach(shaper, output_dir, device)
    plot_parallel_flight(shaper, output_dir, device)
    plot_crossing_trajectories(shaper, output_dir, device)
    plot_penalty_landscape(shaper, output_dir, device)
    plot_time_evolution(shaper, output_dir, device)
    plot_gamma_effect(shaper, output_dir, device)
    plot_distance_vs_cpa_comparison(shaper, output_dir, device)

    print(f"\n{'=' * 80}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'=' * 80}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
