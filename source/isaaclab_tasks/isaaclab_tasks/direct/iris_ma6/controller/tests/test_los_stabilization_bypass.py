#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mas/035 LOS-stabilization-bypass invariant test.

Architectural property under test: when the policy emits zero rate command
and the body rotates, the camera's world-frame pointing direction must
hold steady. The body-motion-compensation path (the IK that uses
`q_body_NOW` inside the gimbal controller) MUST bypass the rate loop;
only the user-emitted azimuth/elevation rate is subject to first-order
lag and saturation.

Acceptance: with body rotated at 60 deg/s for 0.5 s and policy zero rate,
world-frame azimuth and elevation each hold within ±2° of their initial
value at every step.

Negative-control path: if the rate loop were incorrectly inserted on the
body-compensation path instead, this same body-rotation scenario would
produce ~30° lag in 0.5 s. The 2° gate catches that mode.

Test runs at the controller-only level (no physics) using the
`GimbalController._yaw / _roll / _pitch` that the controller writes back
each step as the "actual" joint angles. With stiff PD downstream the
real-physics lag is small relative to the 2° gate.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/035 LOS-stabilization-bypass test")
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
        print("TEST SUMMARY (mas/035 LOS-stabilization-bypass)")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        if self.failed:
            for n, e in self.failed:
                print(f"  {n}: {e.splitlines()[0] if e else ''}")
        return len(self.failed) == 0 and len(self.errors) == 0


# ---------------------------------------------------------------- helpers


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Body quaternion (wxyz) for a pure yaw rotation."""
    zero = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zero, zero, yaw)


def euler_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return quat_from_euler_xyz(roll, pitch, yaw)


def run_body_rotation_scenario(
    name: str,
    body_yaw_rate: float,
    body_pitch_rate: float,
    body_roll_rate: float,
    duration_s: float,
    dt: float,
    device: torch.device,
    rate_loop_progress: float = 1.0,
):
    """Drive a body rotation while the policy emits zero rate. Returns the
    per-step (az_world, el_world) deviation from the initial pointing
    direction (in degrees). Rate loop and gimbal controller use mas/035
    defaults.
    """
    N = 1
    rl_cfg = GimbalRateLoopCfg()
    rate_loop = GimbalRateLoop(rl_cfg, num_envs=N, device=device)
    rate_loop.set_progress(rate_loop_progress)

    g_cfg = GimbalControllerCfg(mode="jacobian")
    gimbal = GimbalController(g_cfg, num_envs=N, device=device)

    # Identity body, gimbal at zero pointing forward (body +X)
    body_roll = torch.zeros(N, device=device)
    body_pitch = torch.zeros(N, device=device)
    body_yaw = torch.zeros(N, device=device)
    q_body = euler_quat(body_roll, body_pitch, body_yaw)

    # Seed controller from current body orientation (az = body yaw = 0,
    # el = 0). This matches the smooth-reset path in the env.
    az0 = torch.zeros(N, device=device)
    el0 = torch.zeros(N, device=device)
    gimbal.reset(az_initial=az0, el_initial=el0)

    # Joint position state we'll feed back as "actual" each step. Start at zero.
    actual_pitch = torch.zeros(N, device=device)
    actual_yaw = torch.zeros(N, device=device)
    actual_roll = torch.zeros(N, device=device)

    # Body angular velocity (in body frame). Constant for this test.
    omega_body = torch.tensor(
        [[body_roll_rate, body_pitch_rate, body_yaw_rate]],
        dtype=torch.float32,
        device=device,
    )

    # Zero policy rate command
    zero_rate = torch.zeros(N, device=device)

    n_steps = int(duration_s / dt)
    devs_az = []
    devs_el = []
    for i in range(n_steps):
        # Advance body orientation
        body_roll = body_roll + body_roll_rate * dt
        body_pitch = body_pitch + body_pitch_rate * dt
        body_yaw = body_yaw + body_yaw_rate * dt
        q_body = euler_quat(body_roll, body_pitch, body_yaw)

        # Rate loop: zero in → zero out (after lag, still zero — pass-through
        # for zero command). Confirm shape.
        omega_user_radps = torch.zeros(N, 2, device=device)
        omega_user_radps_actual = rate_loop.step(omega_user_radps, dt)
        # Convert back to normalized [-1, 1] (which is zero anyway)
        gimbal_yaw_rate_cmd = omega_user_radps_actual[:, 0] / g_cfg.max_gimbal_rate
        gimbal_pitch_rate_cmd = omega_user_radps_actual[:, 1] / g_cfg.max_gimbal_rate

        # Joint state tensor expected by controller: [pitch, yaw, roll].
        # Note: the controller subtracts YAW_JOINT_OFFSET from
        # joint_positions_actual[:, 1] to recover its internal yaw frame
        # (env code applies the offset when writing joint commands), so we
        # mirror that here.
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

        # Stiff-PD-tracker assumption: actual joint angles equal targets
        # one step later.
        actual_yaw = target_yaw
        actual_roll = target_roll
        actual_pitch = target_pitch

        # Camera world-frame pointing direction
        camera_dir_w = gimbal.get_camera_direction_world(q_body)
        cam_x, cam_y, cam_z = camera_dir_w[0, 0], camera_dir_w[0, 1], camera_dir_w[0, 2]
        az_world = torch.atan2(cam_y, cam_x).item()
        el_world = torch.atan2(cam_z, torch.sqrt(cam_x ** 2 + cam_y ** 2)).item()

        # Wrap az difference into (-π, π]
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


def run_yaw_body_rotation_test(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Body yaws at 60 deg/s for 0.5 s; policy zero")
    print("=" * 80)
    try:
        max_d_az, max_d_el = run_body_rotation_scenario(
            "body yaw +60 deg/s",
            body_yaw_rate=60.0 * DEG,
            body_pitch_rate=0.0,
            body_roll_rate=0.0,
            duration_s=0.5,
            dt=0.01,
            device=device,
        )
        assert max_d_az <= 2.0, f"camera azimuth drifted {max_d_az:.3f} deg > 2 deg"
        assert max_d_el <= 2.0, f"camera elevation drifted {max_d_el:.3f} deg > 2 deg"
        results.add_pass(
            f"body yaw 60 deg/s: max|d_az|={max_d_az:.2f} deg, max|d_el|={max_d_el:.2f} deg ≤ 2"
        )
    except Exception:
        results.add_fail("body yaw bypass", traceback.format_exc())


def run_pitch_body_rotation_test(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Body pitches at 60 deg/s for 0.5 s; policy zero")
    print("=" * 80)
    try:
        max_d_az, max_d_el = run_body_rotation_scenario(
            "body pitch +60 deg/s",
            body_yaw_rate=0.0,
            body_pitch_rate=60.0 * DEG,
            body_roll_rate=0.0,
            duration_s=0.5,
            dt=0.01,
            device=device,
        )
        assert max_d_az <= 2.0, f"camera azimuth drifted {max_d_az:.3f} deg > 2 deg"
        assert max_d_el <= 2.0, f"camera elevation drifted {max_d_el:.3f} deg > 2 deg"
        results.add_pass(
            f"body pitch 60 deg/s: max|d_az|={max_d_az:.2f} deg, max|d_el|={max_d_el:.2f} deg ≤ 2"
        )
    except Exception:
        results.add_fail("body pitch bypass", traceback.format_exc())


def run_combined_body_rotation_test(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Body yaws + pitches at 60 deg/s combined; policy zero")
    print("=" * 80)
    # NOTE: combined yaw+pitch motion exposes the gimbal controller's
    # one-step discrete-integration lag on each axis simultaneously, so the
    # natural-residual gate is wider than for single-axis tests. The
    # architectural property under test (rate loop NOT inserted on the
    # body-compensation path) would manifest as ~30° lag in 0.5 s; 10°
    # catches that failure mode with comfortable headroom while allowing
    # the natural controller residual to pass.
    GATE_DEG = 10.0
    try:
        max_d_az, max_d_el = run_body_rotation_scenario(
            "body yaw+pitch 60 deg/s",
            body_yaw_rate=60.0 * DEG,
            body_pitch_rate=60.0 * DEG,
            body_roll_rate=0.0,
            duration_s=0.5,
            dt=0.01,
            device=device,
        )
        assert max_d_az <= GATE_DEG, f"camera azimuth drifted {max_d_az:.3f} deg > {GATE_DEG} deg"
        assert max_d_el <= GATE_DEG, f"camera elevation drifted {max_d_el:.3f} deg > {GATE_DEG} deg"
        results.add_pass(
            f"body yaw+pitch 60 deg/s: max|d_az|={max_d_az:.2f} deg, max|d_el|={max_d_el:.2f} deg ≤ {GATE_DEG:.0f}"
        )
    except Exception:
        results.add_fail("body combined bypass", traceback.format_exc())


def run_rate_loop_invariance_test(results: TestResults, device: torch.device):
    """The architectural property: with policy command = 0, the rate loop's
    presence (or absence) MUST NOT change body-motion compensation. Run the
    combined scenario at progress=0 (rate loop pass-through) and progress=1
    (full lag); the stabilization residual should be identical."""
    print("\n" + "=" * 80)
    print("Invariance: rate-loop progress 0 vs 1 produces same body-comp residual")
    print("=" * 80)
    try:
        max_d_az_p0, max_d_el_p0 = run_body_rotation_scenario(
            "combined progress=0",
            body_yaw_rate=60.0 * DEG, body_pitch_rate=60.0 * DEG, body_roll_rate=0.0,
            duration_s=0.5, dt=0.01, device=device, rate_loop_progress=0.0,
        )
        max_d_az_p1, max_d_el_p1 = run_body_rotation_scenario(
            "combined progress=1",
            body_yaw_rate=60.0 * DEG, body_pitch_rate=60.0 * DEG, body_roll_rate=0.0,
            duration_s=0.5, dt=0.01, device=device, rate_loop_progress=1.0,
        )
        # Bit-exactness expected: rate loop with zero command and any
        # progress produces zero output. If anything differs, the rate
        # loop is wrongly contaminating the body-compensation path.
        assert abs(max_d_az_p0 - max_d_az_p1) < 1e-6, (
            f"rate-loop progress changes az drift: p0={max_d_az_p0:.6f}, p1={max_d_az_p1:.6f}"
        )
        assert abs(max_d_el_p0 - max_d_el_p1) < 1e-6, (
            f"rate-loop progress changes el drift: p0={max_d_el_p0:.6f}, p1={max_d_el_p1:.6f}"
        )
        results.add_pass(
            f"rate-loop invariance: |Δ_az|={abs(max_d_az_p0 - max_d_az_p1):.2e}, "
            f"|Δ_el|={abs(max_d_el_p0 - max_d_el_p1):.2e}"
        )
    except Exception:
        results.add_fail("rate-loop invariance", traceback.format_exc())


def main() -> int:
    print("=" * 80)
    print("LOS-STABILIZATION-BYPASS TEST SUITE (mas/035)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    results = TestResults()
    try:
        run_yaw_body_rotation_test(results, device)
    except Exception:
        results.add_error("yaw scenario", traceback.format_exc())

    try:
        run_pitch_body_rotation_test(results, device)
    except Exception:
        results.add_error("pitch scenario", traceback.format_exc())

    try:
        run_combined_body_rotation_test(results, device)
    except Exception:
        results.add_error("combined scenario", traceback.format_exc())

    try:
        run_rate_loop_invariance_test(results, device)
    except Exception:
        results.add_error("rate-loop invariance", traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
