#!/usr/bin/env python3
"""Isaac Sim integration test for gimbal stabilization.

Tests two gimbal actuation modes:
1. Direct state: write_joint_state_to_sim (zero lag, kinematic)
2. Implicit actuator: set_joint_position/velocity_target (PD servo with lag)

Uses the DroneController for flight stabilization while commanding sinusoidal
velocity inputs. Measures world-frame gimbal pointing error.

All angle measurements use quaternion math (no Euler decomposition).
Body tilt is measured as the angle between body Z-axis and world Z-axis.

Usage:
    ./isaaclab.sh -p .../test_gimbal_sim.py
    ./isaaclab.sh -p .../test_gimbal_sim.py --test-verbose
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gimbal stabilization in-sim test")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Print per-step debug info")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import math
import torch
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.utils.math import quat_rotate

from isaaclab_assets import IRIS_GIMBAL3_CFG
from isaaclab_tasks.direct.iris_ma6.controller import (
    DroneController,
    GimbalController,
    GimbalControllerCfg,
)
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller import YAW_JOINT_OFFSET
from isaaclab_tasks.direct.iris_ma6.controller.tuning import TUNED_CONTROLLER_CFG


def body_tilt_deg(q_body: torch.Tensor) -> float:
    """Measure body tilt as angle between body Z and world Z (quaternion-based)."""
    body_z = quat_rotate(q_body, torch.tensor([[0.0, 0.0, 1.0]], device=q_body.device))
    cos_tilt = body_z[:, 2].clamp(-1.0, 1.0)
    tilt_rad = torch.acos(cos_tilt)
    return math.degrees(tilt_rad[0].item())


def body_yaw_deg(q_body: torch.Tensor) -> float:
    """Measure body yaw as angle of body X projected onto world XY plane."""
    body_x = quat_rotate(q_body, torch.tensor([[1.0, 0.0, 0.0]], device=q_body.device))
    yaw = torch.atan2(body_x[:, 1], body_x[:, 0])
    return math.degrees(yaw[0].item())


def gimbal_world_direction(q_body, gimbal_jp, joint_offset):
    """Compute camera world-frame pointing direction from actual joint positions."""
    a_yaw = gimbal_jp[:, 1] - joint_offset
    a_roll = gimbal_jp[:, 2]
    a_pitch = gimbal_jp[:, 0]
    cy, sy = torch.cos(a_yaw), torch.sin(a_yaw)
    cr, sr = torch.cos(a_roll), torch.sin(a_roll)
    cp, sp = torch.cos(a_pitch), torch.sin(a_pitch)
    fwd_body = torch.stack([cy*cp + sy*sr*sp, sy*cp - cy*sr*sp, -cr*sp], dim=-1)
    return quat_rotate(q_body, fwd_body)


def run_test(label, v_cmd_func, duration_s, dt, robot, drone_ctrl, gimbal,
             body_id, joint_ids, device, use_direct_state=True):
    """Run a single stabilization test.

    Args:
        use_direct_state: If True, use write_joint_state_to_sim (zero lag).
                          If False, use set_joint_position/velocity_target (implicit actuator).
    """
    num_steps = int(duration_s / dt)
    settle_steps = int(1.0 / dt)
    sim_dt_decimated = dt * 4  # decimation=4

    gimbal.reset()
    drone_ctrl.reset()
    robot.reset()
    sim_ctx = sim_utils.SimulationContext.instance()

    gimbal_joint_id_list = [joint_ids["pitch"], joint_ids["yaw"], joint_ids["roll"]]

    def step_sim(vx=0, vy=0, vz=0, yr=0):
        """One physics step with DroneController."""
        gimbal_jp = torch.stack([
            robot.data.joint_pos[:, joint_ids["pitch"]],
            robot.data.joint_pos[:, joint_ids["yaw"]],
            robot.data.joint_pos[:, joint_ids["roll"]],
        ], dim=-1)

        F, tau, gp, gv, _ = drone_ctrl.step_policy(
            v_cmd=torch.tensor([[vx, vy, vz]], device=device),
            yaw_rate_cmd=torch.tensor([yr], device=device),
            gimbal_yaw_rate_cmd=torch.zeros(1, device=device),
            gimbal_pitch_rate_cmd=torch.zeros(1, device=device),
            zoom_rate_cmd=torch.zeros(1, device=device),
            q_body=robot.data.root_quat_w,
            v_body=robot.data.root_lin_vel_w,
            omega_body=robot.data.root_ang_vel_b,
            sim_dt=sim_dt_decimated,
            gimbal_joint_positions=gimbal_jp,
            physics_dt=dt,
        )
        robot.set_external_force_and_torque(
            forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=body_id,
        )

        g_yaw, g_roll, g_pitch = gp
        g_yaw_v, g_roll_v, g_pitch_v = gv
        pos_cmd = torch.stack([g_pitch, g_yaw + YAW_JOINT_OFFSET, g_roll], dim=-1)
        vel_cmd = torch.stack([g_pitch_v, g_yaw_v, g_roll_v], dim=-1)

        if use_direct_state:
            # Bypass implicit actuator: set joint state directly in PhysX.
            # Also set targets to match so the PD controller applies zero force
            # during the step (otherwise stale targets cause shaking).
            robot.write_joint_state_to_sim(
                position=pos_cmd, velocity=vel_cmd, joint_ids=gimbal_joint_id_list,
            )
            robot.set_joint_position_target(
                target=pos_cmd, joint_ids=gimbal_joint_id_list,
            )
            robot.set_joint_velocity_target(
                target=vel_cmd, joint_ids=gimbal_joint_id_list,
            )
        else:
            # Use implicit actuator PD loop
            robot.set_joint_position_target(
                target=pos_cmd, joint_ids=gimbal_joint_id_list,
            )
            robot.set_joint_velocity_target(
                target=vel_cmd, joint_ids=gimbal_joint_id_list,
            )

        robot.write_data_to_sim()
        sim_ctx.step()
        robot.update(dt)
        return gimbal_jp

    # Settle at hover
    for _ in range(100):
        step_sim()

    # Record initial world direction
    q0 = robot.data.root_quat_w
    jp0 = torch.stack([
        robot.data.joint_pos[:, joint_ids["pitch"]],
        robot.data.joint_pos[:, joint_ids["yaw"]],
        robot.data.joint_pos[:, joint_ids["roll"]],
    ], dim=-1)
    initial_dir = gimbal_world_direction(q0, jp0, YAW_JOINT_OFFSET)

    errors_deg = []
    tilts_deg = []
    yaws_deg = []

    for step in range(num_steps):
        t = step * dt
        vx, vy, vz, yr = v_cmd_func(t)
        jp = step_sim(vx, vy, vz, yr)

        if step >= settle_steps:
            q_body = robot.data.root_quat_w

            tilts_deg.append(body_tilt_deg(q_body))
            yaws_deg.append(body_yaw_deg(q_body))

            fwd_world = gimbal_world_direction(q_body, jp, YAW_JOINT_OFFSET)
            dot = (initial_dir * fwd_world).sum(dim=-1).clamp(-1.0, 1.0)
            errors_deg.append(math.degrees(torch.acos(dot)[0].item()))

    if not errors_deg:
        return {"label": label, "mean": 0, "max": 0, "rms": 0, "tilt_max": 0, "yaw_max": 0}

    return {
        "label": label,
        "mean": sum(errors_deg) / len(errors_deg),
        "max": max(errors_deg),
        "rms": (sum(e**2 for e in errors_deg) / len(errors_deg)) ** 0.5,
        "tilt_max": max(tilts_deg),
        "yaw_max": max(abs(y) for y in yaws_deg),
    }


def main():
    print("=" * 80)
    print("GIMBAL STABILIZATION -- ISAAC SIM INTEGRATION TEST")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sim_dt = 0.01
    with build_simulation_context(
        dt=sim_dt, gravity_enabled=True, device=device_str,
        add_ground_plane=True, add_lighting=True,
    ) as sim:
        robot_cfg = copy.deepcopy(IRIS_GIMBAL3_CFG)
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,
        )
        robot = Articulation(robot_cfg)

        sim.reset()
        robot.reset()

        joint_ids = {
            "yaw": robot.find_joints("yaw_joint")[0][0],
            "roll": robot.find_joints("roll_joint")[0][0],
            "pitch": robot.find_joints("pitch_joint")[0][0],
        }
        body_id, _ = robot.find_bodies("body")

        mass = robot.root_physx_view.get_masses().sum().item()
        print(f"Robot mass: {mass:.3f} kg")
        print(f"Sim dt: {sim_dt}s")

        vel_amp = 3.0
        freq = 0.5
        duration = 6.0

        test_maneuvers = [
            ("Pure Pitch (vx)", lambda t: (vel_amp*math.sin(2*math.pi*freq*t), 0, 0, 0)),
            ("Pure Roll (vy)",  lambda t: (0, vel_amp*math.sin(2*math.pi*freq*t), 0, 0)),
            ("Pure Yaw (yr)",   lambda t: (0, 0, 0, 1.5*math.sin(2*math.pi*freq*t))),
            ("Cross vx+vy",    lambda t: (vel_amp*math.sin(2*math.pi*freq*t),
                                           vel_amp*math.cos(2*math.pi*freq*t), 0, 0)),
        ]

        # ============================================================
        # Mode 1: Direct state setting (zero actuator lag)
        # ============================================================
        print("\n" + "=" * 80)
        print("MODE 1: DIRECT STATE (write_joint_state_to_sim)")
        print("  feedback_blend=0.0 (no actuator drift to correct)")
        print("=" * 80)

        drone_ctrl = DroneController(
            cfg=TUNED_CONTROLLER_CFG, mass=mass, gravity=9.81,
            num_envs=1, device=device,
        )
        gimbal_direct = GimbalController(
            cfg=GimbalControllerCfg(feedback_blend=0.0), num_envs=1, device=device,
        )

        results_direct = []
        for label, cmd_fn in test_maneuvers:
            print(f"\n--- {label} ---", flush=True)
            r = run_test(label, cmd_fn, duration, sim_dt, robot, drone_ctrl,
                         gimbal_direct, body_id, joint_ids, device,
                         use_direct_state=True)
            results_direct.append(r)
            print(f"  Body tilt max: {r['tilt_max']:.1f} deg  yaw max: {r['yaw_max']:.1f} deg")
            print(f"  Gimbal error -- mean:{r['mean']:.2f} deg  RMS:{r['rms']:.2f} deg  max:{r['max']:.2f} deg")

        # ============================================================
        # Mode 2: Implicit actuator (PD servo with lag)
        # ============================================================
        print("\n" + "=" * 80)
        print("MODE 2: IMPLICIT ACTUATOR (set_joint_position/velocity_target)")
        print("  feedback_blend=0.1 (corrects actuator tracking drift)")
        print("=" * 80)

        drone_ctrl2 = DroneController(
            cfg=TUNED_CONTROLLER_CFG, mass=mass, gravity=9.81,
            num_envs=1, device=device,
        )
        gimbal_implicit = GimbalController(
            cfg=GimbalControllerCfg(feedback_blend=0.1), num_envs=1, device=device,
        )

        results_implicit = []
        for label, cmd_fn in test_maneuvers:
            print(f"\n--- {label} ---", flush=True)
            r = run_test(label, cmd_fn, duration, sim_dt, robot, drone_ctrl2,
                         gimbal_implicit, body_id, joint_ids, device,
                         use_direct_state=False)
            results_implicit.append(r)
            print(f"  Body tilt max: {r['tilt_max']:.1f} deg  yaw max: {r['yaw_max']:.1f} deg")
            print(f"  Gimbal error -- mean:{r['mean']:.2f} deg  RMS:{r['rms']:.2f} deg  max:{r['max']:.2f} deg")

        # ============================================================
        # Comparison summary
        # ============================================================
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        header = f"{'Test':<22s} {'|':>2s} {'Direct Mean':>11s} {'Direct RMS':>10s} {'Direct Max':>10s} {'|':>2s} {'Implicit Mean':>13s} {'Implicit RMS':>12s} {'Implicit Max':>12s}"
        print(header)
        print("-" * len(header))
        for rd, ri in zip(results_direct, results_implicit):
            print(
                f"{rd['label']:<22s} {'|':>2s} "
                f"{rd['mean']:10.2f}d {rd['rms']:9.2f}d {rd['max']:9.2f}d {'|':>2s} "
                f"{ri['mean']:12.2f}d {ri['rms']:11.2f}d {ri['max']:11.2f}d"
            )

        # Verify direct state mode is strictly better
        print("\n" + "-" * 80)
        all_pass = True
        for rd, ri in zip(results_direct, results_implicit):
            direct_better = rd['rms'] <= ri['rms'] + 0.5  # allow 0.5 deg tolerance
            status = "PASS" if direct_better else "FAIL"
            if not direct_better:
                all_pass = False
            improvement = ri['rms'] - rd['rms']
            print(f"  [{status}] {rd['label']}: direct RMS {rd['rms']:.2f} deg vs implicit RMS {ri['rms']:.2f} deg "
                  f"(improvement: {improvement:+.2f} deg)")

        print("-" * 80)
        if all_pass:
            print("RESULT: PASS -- Direct state mode matches or outperforms implicit actuator")
        else:
            print("RESULT: FAIL -- Direct state mode worse than implicit on some tests")
        print("=" * 80)

    simulation_app.close()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
