#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mas/036 LOS-stabilization-unaffected invariant test.

Architectural property under test (extends mas/035's bypass invariant):
even with the dead-time buffer fully active at the rate-loop input, body
motion compensation must remain instantaneous. Failing this test means
the dead-time buffer was wrongly placed on the IK path.

Acceptance: with body rotated at 60 deg/s for 0.5 s, policy zero rate,
and dead time at 100 ms (well above the 50–66 ms ranges used in mas/036
defaults), camera world-frame azimuth/elevation remain within ±2° of
the starting value.

Negative-control note: if the buffer were on the IK path, body yawing at
60 deg/s × 0.5 s = 30° would propagate directly into the camera; with
100 ms dead time it would lag, producing 6+ degrees residual at 0.1 s.
The 2° gate catches that mode comfortably.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/036 LOS-stabilization-unaffected test")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import sys
import traceback
from datetime import datetime

import torch

from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller_jacobian import (
    GimbalController,
    YAW_JOINT_OFFSET,
)
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller_cfg import GimbalControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop import GimbalRateLoop
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop_cfg import GimbalRateLoopCfg


VERBOSE = args_cli.test_verbose
DEG = math.pi / 180.0


# ---------------------------------------------------------------- TestResults


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def add_fail(self, name: str, err: str):
        self.failed.append((name, err))
        print(f"  ✗ {name}")
        for line in err.split("\n")[:5]:
            print(f"    {line}")

    def add_error(self, name: str, err: str):
        self.errors.append((name, err))
        print(f"  ERROR {name}")
        print(f"    {err.splitlines()[-1] if err else ''}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY (mas/036 LOS-stabilization-unaffected)")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        if self.failed:
            for n, e in self.failed:
                print(f"  {n}: {e.splitlines()[0] if e else ''}")
        return len(self.failed) == 0 and len(self.errors) == 0


# ---------------------------------------------------------------- helpers


def euler_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return quat_from_euler_xyz(roll, pitch, yaw)


def run_body_rotation_with_dead_time(
    name: str,
    body_yaw_rate: float,
    body_pitch_rate: float,
    body_roll_rate: float,
    duration_s: float,
    dt: float,
    dead_time_mean_s: float,
    dead_time_curriculum_scale: float,
    rate_loop_progress: float,
    device: torch.device,
):
    """Body rotates while policy emits zero rate. Returns (max|d_az|,
    max|d_el|) in degrees. Dead-time buffer is fully active."""
    N = 1
    rl_cfg = GimbalRateLoopCfg(
        dead_time_mean_s=dead_time_mean_s,
        dead_time_std_s=0.0,  # deterministic for this test
        dead_time_max_s=max(dead_time_mean_s + 0.05, 0.15),
    )
    rate_loop = GimbalRateLoop(rl_cfg, num_envs=N, device=device)
    rate_loop.set_progress(rate_loop_progress)
    rate_loop.set_dead_time_curriculum_scale(dead_time_curriculum_scale)
    rate_loop.reset()

    g_cfg = GimbalControllerCfg(mode="jacobian")
    gimbal = GimbalController(g_cfg, num_envs=N, device=device)

    body_roll = torch.zeros(N, device=device)
    body_pitch = torch.zeros(N, device=device)
    body_yaw = torch.zeros(N, device=device)
    q_body = euler_quat(body_roll, body_pitch, body_yaw)

    az0 = torch.zeros(N, device=device)
    el0 = torch.zeros(N, device=device)
    gimbal.reset(az_initial=az0, el_initial=el0)

    actual_pitch = torch.zeros(N, device=device)
    actual_yaw = torch.zeros(N, device=device)
    actual_roll = torch.zeros(N, device=device)

    omega_body = torch.tensor(
        [[body_roll_rate, body_pitch_rate, body_yaw_rate]],
        dtype=torch.float32, device=device,
    )

    n_steps = int(duration_s / dt)
    devs_az = []
    devs_el = []
    for i in range(n_steps):
        body_roll = body_roll + body_roll_rate * dt
        body_pitch = body_pitch + body_pitch_rate * dt
        body_yaw = body_yaw + body_yaw_rate * dt
        q_body = euler_quat(body_roll, body_pitch, body_yaw)

        # Zero policy rate command. With dead-time buffer on, output is
        # still zero (zero in → zero out).
        omega_user_radps = torch.zeros(N, 2, device=device)
        omega_user_radps_actual = rate_loop.step(omega_user_radps, dt)
        gimbal_yaw_rate_cmd = omega_user_radps_actual[:, 0] / g_cfg.max_gimbal_rate
        gimbal_pitch_rate_cmd = omega_user_radps_actual[:, 1] / g_cfg.max_gimbal_rate

        joint_positions_actual = torch.stack(
            [actual_pitch, actual_yaw + YAW_JOINT_OFFSET, actual_roll], dim=-1
        )
        pos_targets, _ = gimbal.compute_control(
            gimbal_yaw_rate_cmd=gimbal_yaw_rate_cmd,
            gimbal_pitch_rate_cmd=gimbal_pitch_rate_cmd,
            q_body=q_body,
            dt=dt,
            omega_body=omega_body,
            joint_positions_actual=joint_positions_actual,
        )
        target_yaw, target_roll, target_pitch = pos_targets
        actual_yaw = target_yaw
        actual_roll = target_roll
        actual_pitch = target_pitch

        camera_dir_w = gimbal.get_camera_direction_world(q_body)
        cam_x, cam_y, cam_z = camera_dir_w[0, 0], camera_dir_w[0, 1], camera_dir_w[0, 2]
        az_world = torch.atan2(cam_y, cam_x).item()
        el_world = torch.atan2(cam_z, torch.sqrt(cam_x ** 2 + cam_y ** 2)).item()

        d_az = az_world - az0.item()
        d_az = math.atan2(math.sin(d_az), math.cos(d_az))
        d_el = el_world - el0.item()
        devs_az.append(abs(d_az) / DEG)
        devs_el.append(abs(d_el) / DEG)

    if VERBOSE:
        print(
            f"    {name}: max|d_az|={max(devs_az):.3f} deg, "
            f"max|d_el|={max(devs_el):.3f} deg over {n_steps} steps"
        )
    return max(devs_az), max(devs_el)


# ---------------------------------------------------------------- tests


def run_yaw_with_dead_time(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Body yaws at 60 deg/s, dead-time = 100 ms; camera holds within 2°")
    print("=" * 80)
    try:
        max_d_az, max_d_el = run_body_rotation_with_dead_time(
            "body yaw +60 deg/s, dead_time=100ms",
            body_yaw_rate=60.0 * DEG,
            body_pitch_rate=0.0, body_roll_rate=0.0,
            duration_s=0.5, dt=0.01,
            dead_time_mean_s=0.10, dead_time_curriculum_scale=1.0,
            rate_loop_progress=1.0, device=device,
        )
        assert max_d_az <= 2.0, f"yaw camera drift {max_d_az:.3f} deg > 2 deg"
        assert max_d_el <= 2.0, f"yaw camera el drift {max_d_el:.3f} deg > 2 deg"
        results.add_pass(f"body yaw, dead-time 100ms: max|d_az|={max_d_az:.2f}, "
                         f"max|d_el|={max_d_el:.2f} ≤ 2")
    except Exception:
        results.add_fail("yaw with dead-time", traceback.format_exc())


def run_pitch_with_dead_time(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Body pitches at 60 deg/s, dead-time = 100 ms; camera holds within 2°")
    print("=" * 80)
    try:
        max_d_az, max_d_el = run_body_rotation_with_dead_time(
            "body pitch +60 deg/s, dead_time=100ms",
            body_yaw_rate=0.0, body_pitch_rate=60.0 * DEG, body_roll_rate=0.0,
            duration_s=0.5, dt=0.01,
            dead_time_mean_s=0.10, dead_time_curriculum_scale=1.0,
            rate_loop_progress=1.0, device=device,
        )
        assert max_d_az <= 2.0, f"pitch camera az drift {max_d_az:.3f} deg > 2 deg"
        assert max_d_el <= 2.0, f"pitch camera el drift {max_d_el:.3f} deg > 2 deg"
        results.add_pass(f"body pitch, dead-time 100ms: max|d_az|={max_d_az:.2f}, "
                         f"max|d_el|={max_d_el:.2f} ≤ 2")
    except Exception:
        results.add_fail("pitch with dead-time", traceback.format_exc())


def run_dead_time_invariance(results: TestResults, device: torch.device):
    """Architectural property: with policy command = 0, the dead-time
    buffer's presence (or absence) MUST NOT change body-motion
    compensation. Run identical scenarios at scale=0 and scale=1; the
    stabilization residual must be bit-exact."""
    print("\n" + "=" * 80)
    print("Invariance: dead-time scale 0 vs 1 produces same body-comp residual")
    print("=" * 80)
    try:
        max_d_az_s0, max_d_el_s0 = run_body_rotation_with_dead_time(
            "scale=0",
            body_yaw_rate=60.0 * DEG, body_pitch_rate=60.0 * DEG, body_roll_rate=0.0,
            duration_s=0.5, dt=0.01,
            dead_time_mean_s=0.10, dead_time_curriculum_scale=0.0,
            rate_loop_progress=1.0, device=device,
        )
        max_d_az_s1, max_d_el_s1 = run_body_rotation_with_dead_time(
            "scale=1",
            body_yaw_rate=60.0 * DEG, body_pitch_rate=60.0 * DEG, body_roll_rate=0.0,
            duration_s=0.5, dt=0.01,
            dead_time_mean_s=0.10, dead_time_curriculum_scale=1.0,
            rate_loop_progress=1.0, device=device,
        )
        assert abs(max_d_az_s0 - max_d_az_s1) < 1e-6, (
            f"dead-time scale changes az drift: s0={max_d_az_s0:.6f}, s1={max_d_az_s1:.6f}"
        )
        assert abs(max_d_el_s0 - max_d_el_s1) < 1e-6, (
            f"dead-time scale changes el drift: s0={max_d_el_s0:.6f}, s1={max_d_el_s1:.6f}"
        )
        results.add_pass(
            f"dead-time invariance: |Δ_az|={abs(max_d_az_s0 - max_d_az_s1):.2e}, "
            f"|Δ_el|={abs(max_d_el_s0 - max_d_el_s1):.2e}"
        )
    except Exception:
        results.add_fail("dead-time invariance", traceback.format_exc())


def main() -> int:
    print("=" * 80)
    print("LOS-STABILIZATION-UNAFFECTED TEST SUITE (mas/036)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    torch.manual_seed(0)

    results = TestResults()
    for fn in (
        run_yaw_with_dead_time,
        run_pitch_with_dead_time,
        run_dead_time_invariance,
    ):
        try:
            fn(results, device)
        except Exception:
            results.add_error(fn.__name__, traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
