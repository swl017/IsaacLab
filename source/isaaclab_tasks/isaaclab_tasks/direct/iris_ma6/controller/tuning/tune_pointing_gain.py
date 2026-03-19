#!/usr/bin/env python3
"""Parallel sweep of gimbal pointing_gain using N envs.

Each environment tests a different pointing_gain value simultaneously.
All envs receive identical maneuver commands; only the gimbal gain differs.

Usage:
    ./isaaclab.sh -p .../tune_pointing_gain.py
    ./isaaclab.sh -p .../tune_pointing_gain.py --num-envs 64
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tune gimbal pointing_gain (parallel)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel envs (= number of gain values)")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import sys
import time
import torch
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_rotate


def gimbal_world_direction(q_body, gimbal_jp, joint_offset):
    """Compute camera world-frame pointing direction from joint positions.

    Rotation chain: Yaw(Z) -> Roll(X) -> Pitch(Y) applied to [1,0,0].
    Must match GimbalController._apply_gimbal_rotation exactly.
    """
    a_yaw = gimbal_jp[:, 1] - joint_offset
    a_roll = gimbal_jp[:, 2]
    a_pitch = gimbal_jp[:, 0]
    cy, sy = torch.cos(a_yaw), torch.sin(a_yaw)
    cr, sr = torch.cos(a_roll), torch.sin(a_roll)
    cp, sp = torch.cos(a_pitch), torch.sin(a_pitch)
    # Yaw(Z)[1,0,0]=[cy,sy,0] -> Roll(X)=[cy, sy*cr, sy*sr] -> Pitch(Y):
    fwd_body = torch.stack([cy*cp + sy*sr*sp, sy*cr, -cy*sp + sy*sr*cp], dim=-1)
    return quat_rotate(q_body, fwd_body)


def main():
    print("=" * 80)
    print("GIMBAL POINTING_GAIN PARALLEL SWEEP")
    print("=" * 80)

    N = args_cli.num_envs
    device = "cuda:0"

    # Generate gain values: linearly spaced from 1 to 95
    # Upper bound < 100 to avoid instability (K*dt=1 at K=100 for dt=0.01)
    gains = torch.linspace(20.0, 200.0, N, device=device)
    print(f"Testing {N} gains: [{gains[0]:.1f}, ..., {gains[-1]:.1f}]")

    # Create environment
    print("Creating environment...", flush=True)
    task_name = "Isaac-Iris-MA6-Direct-Test-v0"
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=N)
    # Disable subsystems not needed for gimbal tuning
    env_cfg.enable_target_controller = False  # type: ignore[attr-defined]
    env_cfg.enable_tiled_cameras = False  # type: ignore[attr-defined]
    env = gym.make(task_name, cfg=env_cfg)
    unwrapped = env.unwrapped

    # Get references
    agent_id = unwrapped.cfg.possible_agents[0]
    robot = unwrapped._robots[agent_id]
    gimbal_ctrl = unwrapped._controllers[agent_id]._gimbal

    # Override pointing_gain per env (only works with jacobian mode)
    from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller_jacobian import GimbalController as JacobianGimbal
    assert isinstance(gimbal_ctrl, JacobianGimbal), (
        f"Expected jacobian gimbal controller, got {type(gimbal_ctrl).__name__}. "
        f"Set mode='jacobian' in GimbalControllerCfg."
    )
    gimbal_ctrl._pointing_gain = gains.unsqueeze(-1)  # (N, 1)

    from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller import YAW_JOINT_OFFSET

    # Get gimbal joint indices
    gimbal_jids = unwrapped.gimbal_joint_idx[agent_id]

    dt = unwrapped.cfg.sim.dt * unwrapped.cfg.decimation  # policy dt
    sim_dt = unwrapped.cfg.sim.dt

    print(f"Policy dt: {dt:.3f}s, Sim dt: {sim_dt:.3f}s")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # Maneuvers: (name, duration_s, action_fn) where action_fn(t) -> (N, 7) actions
    # Action layout: [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
    # Actions 0-3 are scaled by max_lin_vel / max_yaw_rate; actions 4-6 are normalized [-1,1]
    vel_amp = 3.0
    freq = 0.5
    max_lin_vel = unwrapped.cfg.max_lin_vel

    def make_action(t, vx=0.0, vy=0.0, vz=0.0, yr=0.0, g_yaw=0.0, g_pitch=0.0):
        """Create normalized action tensor for all envs."""
        a = torch.zeros(N, 7, device=device)
        a[:, 0] = vx / max_lin_vel
        a[:, 1] = vy / max_lin_vel
        a[:, 2] = vz / max_lin_vel
        a[:, 3] = yr / unwrapped.cfg.max_yaw_rate if hasattr(unwrapped.cfg, 'max_yaw_rate') else yr
        a[:, 4] = g_yaw    # gimbal yaw rate (normalized [-1, 1])
        a[:, 5] = g_pitch  # gimbal pitch rate (normalized [-1, 1])
        return torch.clamp(a, -1.0, 1.0)

    # pointing_gain controls how fast the gimbal converges to its world-frame target.
    # To test it, we need maneuvers that CREATE tracking error — i.e., active gimbal
    # slewing while the body moves and rotates. The feedforward (-omega_body) handles
    # pure stabilization regardless of gain; the gain drives residual error to zero.
    # Moderate gimbal amplitudes to stay well within joint limits.
    # At amp=0.1, max_rate=2π: peak rate ≈ 0.63 rad/s, peak excursion ≈ 18 deg.
    # This keeps the gimbal in the regime where the gain is the limiting factor.
    gimbal_amp = 0.1  # normalized gimbal rate amplitude
    maneuvers = [
        # Slew gimbal while body pitches (forward/back) — tests yaw/pitch tracking
        ("SlewPitch", 5.0, lambda t: make_action(t,
            vx=vel_amp*math.sin(2*math.pi*freq*t),
            g_yaw=gimbal_amp*math.sin(2*math.pi*0.3*t),
            g_pitch=gimbal_amp*math.cos(2*math.pi*0.2*t))),
        # Slew gimbal while body rolls (lateral) — tests cross-coupling
        ("SlewRoll",  5.0, lambda t: make_action(t,
            vy=vel_amp*math.sin(2*math.pi*freq*t),
            g_yaw=gimbal_amp*math.sin(2*math.pi*0.25*t))),
        # Circular body motion with gimbal scanning
        ("SlewCirc",  5.0, lambda t: make_action(t,
            vx=vel_amp*math.sin(2*math.pi*freq*t),
            vy=vel_amp*math.cos(2*math.pi*freq*t),
            g_yaw=gimbal_amp*math.sin(2*math.pi*0.4*t),
            g_pitch=gimbal_amp*0.5*math.sin(2*math.pi*0.3*t))),
        # Pure gimbal slew (no body motion) — baseline tracking quality
        ("SlewOnly",  4.0, lambda t: make_action(t,
            g_yaw=gimbal_amp*math.sin(2*math.pi*0.5*t),
            g_pitch=gimbal_amp*math.sin(2*math.pi*0.3*t))),
        # Step response: body suddenly moves, gimbal holds zero command
        ("StepHold",  5.0, lambda t: make_action(t,
            vx=vel_amp*math.sin(2*math.pi*0.3*t))),
    ]

    all_rms = {}  # maneuver_name -> (N,) rms per gain

    start_time = time.time()

    for man_name, man_dur, man_action_fn in maneuvers:
        print(f"\n--- {man_name} ({man_dur}s) ---", flush=True)

        # Reset env
        obs, _ = env.reset()

        # Re-apply per-env gains (reset may have cleared them)
        gimbal_ctrl._pointing_gain = gains.unsqueeze(-1)

        # Settle for 1 second
        settle_steps = int(1.0 / dt)
        for _ in range(settle_steps):
            action_dict = {aid: torch.zeros(N, 7, device=device) for aid in unwrapped.cfg.possible_agents}
            obs, _, _, _, _ = env.step(action_dict)

        # Run maneuver — measure error between controller's desired world-frame
        # pointing and the actual gimbal pointing direction each step.
        num_steps = int(man_dur / dt)
        error_sq_sum = torch.zeros(N, device=device)
        error_count = 0

        for step in range(num_steps):
            t = step * dt
            action = man_action_fn(t)
            action_dict = {aid: action for aid in unwrapped.cfg.possible_agents}
            obs, _, _, _, _ = env.step(action_dict)

            # Actual gimbal pointing direction in world frame
            q_body = robot.data.root_quat_w
            jp = torch.stack([
                robot.data.joint_pos[:, gimbal_jids["pitch"]],
                robot.data.joint_pos[:, gimbal_jids["yaw"]],
                robot.data.joint_pos[:, gimbal_jids["roll"]],
            ], dim=-1)
            actual_dir = gimbal_world_direction(q_body, jp, YAW_JOINT_OFFSET)

            # Desired pointing direction from controller's world-frame target
            # The controller integrates gimbal rate commands into azimuth/elevation
            az = gimbal_ctrl.azimuth_world   # (N,)
            el = gimbal_ctrl.elevation_world  # (N,)
            desired_dir = torch.stack([
                torch.cos(el) * torch.cos(az),
                torch.cos(el) * torch.sin(az),
                torch.sin(el),
            ], dim=-1)  # (N, 3)

            dot = (desired_dir * actual_dir).sum(dim=-1).clamp(-1.0, 1.0)
            error_rad = torch.acos(dot)
            error_deg = torch.rad2deg(error_rad)

            error_sq_sum += error_deg ** 2
            error_count += 1

        rms = torch.sqrt(error_sq_sum / max(error_count, 1))
        all_rms[man_name] = rms

        # Print best/worst for this maneuver
        best_idx = int(rms.argmin().item())
        worst_idx = int(rms.argmax().item())
        print(f"  Best:  gain={gains[best_idx]:.2f} -> {rms[best_idx]:.2f} deg")
        print(f"  Worst: gain={gains[worst_idx]:.2f} -> {rms[worst_idx]:.2f} deg")

    elapsed = time.time() - start_time

    # Composite score: weighted average across maneuvers
    weights = {"SlewPitch": 2.0, "SlewRoll": 1.0, "SlewCirc": 2.0, "SlewOnly": 1.0, "StepHold": 1.0}
    composite = torch.zeros(N, device=device)
    total_weight = 0.0
    for name, rms in all_rms.items():
        w = weights.get(name, 1.0)
        composite += w * rms
        total_weight += w
    composite /= total_weight

    # Summary table
    maneuver_names = [m[0] for m in maneuvers]
    print("\n" + "=" * 80)
    print(f"RESULTS ({elapsed:.1f}s total)")
    print("=" * 80)

    header = f"{'Gain':>8s} |" + "".join(f" {n:>8s}" for n in maneuver_names) + f" | {'Score':>6s}"
    print(header)
    print("-" * len(header))

    # Print top 15 gains sorted by composite score
    sorted_idx = composite.argsort()
    for rank, idx_tensor in enumerate(sorted_idx[:15]):
        i = int(idx_tensor.item())
        vals = [all_rms[n][i].item() for n in maneuver_names]
        row = f"{gains[i]:8.2f} |" + "".join(f" {v:8.2f}" for v in vals) + f" | {composite[i]:6.2f}"
        if rank == 0:
            row += " <-- BEST"
        print(row)

    if N > 15:
        print(f"  ... ({N - 15} more)")

    best_i = int(sorted_idx[0].item())
    print(f"\nBest: pointing_gain = {gains[best_i]:.2f} (composite score = {composite[best_i]:.3f} deg)")
    print("=" * 80)

    env.close()
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
