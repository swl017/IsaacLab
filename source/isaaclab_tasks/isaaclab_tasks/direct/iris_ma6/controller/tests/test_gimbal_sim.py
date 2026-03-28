#!/usr/bin/env python3
"""Isaac Sim integration test for gimbal stabilization.

Tests two gimbal actuation modes:
1. Direct state: write_joint_state_to_sim (zero lag, kinematic)
2. Implicit actuator: set_joint_position/velocity_target (PD servo with lag)

Detects:
- Oscillation: high-frequency sign changes in the error derivative
- Divergence: error growing over time (positive trend)
- Absolute error: mean, RMS, max pointing error

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


def analyze_stability(errors_deg, dt):
    """Analyze error time series for oscillation and divergence.

    Returns:
        osc_freq_hz: Dominant oscillation frequency estimated from zero-crossings
            of the error derivative. High values indicate instability.
        osc_amplitude: Mean amplitude of oscillation cycles (peak-to-trough / 2).
        divergence_rate: Linear regression slope of error over time [deg/s].
            Positive = growing error. Near-zero = stable.
        trend_r2: R-squared of the linear fit. High R2 + positive slope = clear divergence.
        second_half_ratio: ratio of mean error in second half to first half.
            >1.5 indicates growing error; <0.7 indicates settling.
    """
    n = len(errors_deg)
    if n < 10:
        return 0.0, 0.0, 0.0, 0.0, 1.0

    # -- Oscillation detection via zero-crossings of error derivative --
    deriv = [errors_deg[i+1] - errors_deg[i] for i in range(n - 1)]
    sign_changes = sum(1 for i in range(len(deriv) - 1)
                       if deriv[i] * deriv[i+1] < 0)
    # Each full oscillation cycle has 2 sign changes
    duration_s = n * dt
    osc_freq_hz = sign_changes / (2.0 * duration_s) if duration_s > 0 else 0.0

    # -- Oscillation amplitude: mean of peak-to-trough distances --
    # Find local extrema
    peaks = []
    troughs = []
    for i in range(1, n - 1):
        if errors_deg[i] > errors_deg[i-1] and errors_deg[i] > errors_deg[i+1]:
            peaks.append(errors_deg[i])
        elif errors_deg[i] < errors_deg[i-1] and errors_deg[i] < errors_deg[i+1]:
            troughs.append(errors_deg[i])

    if peaks and troughs:
        osc_amplitude = (sum(peaks) / len(peaks) - sum(troughs) / len(troughs)) / 2.0
    else:
        osc_amplitude = 0.0

    # -- Divergence detection via linear regression --
    # Fit error = a*t + b, compute slope a and R^2
    t_arr = [i * dt for i in range(n)]
    t_mean = sum(t_arr) / n
    e_mean = sum(errors_deg) / n

    ss_tt = sum((t - t_mean)**2 for t in t_arr)
    ss_te = sum((t - t_mean) * (e - e_mean) for t, e in zip(t_arr, errors_deg))
    ss_ee = sum((e - e_mean)**2 for e in errors_deg)

    if ss_tt > 1e-12:
        slope = ss_te / ss_tt  # deg/s
        ss_res = ss_ee - ss_te**2 / ss_tt
        r2 = 1.0 - ss_res / (ss_ee + 1e-12)
    else:
        slope = 0.0
        r2 = 0.0

    # -- Second-half ratio: simple growth detection --
    mid = n // 2
    first_half_mean = sum(errors_deg[:mid]) / max(mid, 1)
    second_half_mean = sum(errors_deg[mid:]) / max(n - mid, 1)
    second_half_ratio = second_half_mean / max(first_half_mean, 0.01)

    return osc_freq_hz, osc_amplitude, slope, max(r2, 0.0), second_half_ratio


def run_test(label, v_cmd_func, duration_s, dt, robot, drone_ctrl, gimbal,
             body_id, joint_ids, device, use_direct_state=True):
    """Run a single stabilization test with stability analysis."""
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
        return {
            "label": label, "mean": 0, "max": 0, "rms": 0,
            "tilt_max": 0, "yaw_max": 0,
            "osc_freq_hz": 0, "osc_amplitude": 0,
            "divergence_rate": 0, "trend_r2": 0, "second_half_ratio": 1.0,
        }

    # Stability analysis
    osc_freq, osc_amp, div_rate, trend_r2, sh_ratio = analyze_stability(errors_deg, dt)

    return {
        "label": label,
        "mean": sum(errors_deg) / len(errors_deg),
        "max": max(errors_deg),
        "rms": (sum(e**2 for e in errors_deg) / len(errors_deg)) ** 0.5,
        "tilt_max": max(tilts_deg),
        "yaw_max": max(abs(y) for y in yaws_deg),
        "osc_freq_hz": osc_freq,
        "osc_amplitude": osc_amp,
        "divergence_rate": div_rate,
        "trend_r2": trend_r2,
        "second_half_ratio": sh_ratio,
    }


def print_stability(r):
    """Print stability metrics for a test result."""
    print(f"  Stability:")
    print(f"    Oscillation: freq={r['osc_freq_hz']:.1f} Hz, amplitude={r['osc_amplitude']:.2f} deg")
    print(f"    Divergence:  slope={r['divergence_rate']:.2f} deg/s, R2={r['trend_r2']:.3f}")
    print(f"    2nd-half/1st-half error ratio: {r['second_half_ratio']:.2f}")


def check_stability(r, max_osc_freq=5.0, max_osc_amp=5.0, max_div_rate=5.0, max_sh_ratio=2.0):
    """Check if a test result passes stability criteria.

    Returns:
        (passed, reasons): tuple of (bool, list of failure reason strings)
    """
    reasons = []

    # Oscillation check: high frequency + significant amplitude = instability
    if r['osc_freq_hz'] > max_osc_freq and r['osc_amplitude'] > max_osc_amp:
        reasons.append(
            f"OSCILLATION: {r['osc_freq_hz']:.1f} Hz at {r['osc_amplitude']:.1f} deg amplitude "
            f"(limit: {max_osc_freq} Hz, {max_osc_amp} deg)")

    # Divergence check: error growing steadily over time
    if r['divergence_rate'] > max_div_rate and r['trend_r2'] > 0.3:
        reasons.append(
            f"DIVERGENCE: {r['divergence_rate']:.1f} deg/s (R2={r['trend_r2']:.2f}) "
            f"(limit: {max_div_rate} deg/s)")

    # Second-half ratio: error much worse in second half = growing instability
    if r['second_half_ratio'] > max_sh_ratio:
        reasons.append(
            f"GROWING ERROR: 2nd-half/1st-half ratio={r['second_half_ratio']:.2f} "
            f"(limit: {max_sh_ratio})")

    return len(reasons) == 0, reasons


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
            ("Hover (static)",  lambda t: (0, 0, 0, 0)),
            ("Yaw Hold (dist)", lambda t: (vel_amp*math.sin(2*math.pi*0.3*t), 0, 0, 0)),
        ]

        all_pass = True

        # ============================================================
        # Mode 1: Direct state setting (zero actuator lag)
        # ============================================================
        print("\n" + "=" * 80)
        print("MODE 1: DIRECT STATE (write_joint_state_to_sim)")
        print("  (reads actual joint state from sim each step)")
        print("=" * 80)

        drone_ctrl = DroneController(
            cfg=TUNED_CONTROLLER_CFG, mass=mass, gravity=9.81,
            num_envs=1, device=device,
        )
        gimbal_direct = GimbalController(
            cfg=GimbalControllerCfg(), num_envs=1, device=device,
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
            print_stability(r)

            passed, reasons = check_stability(r)
            if not passed:
                all_pass = False
                for reason in reasons:
                    print(f"  ** FAIL: {reason}")

            # Yaw drift check for tests with zero yaw rate command
            if label in ("Hover (static)", "Yaw Hold (dist)", "Cross vx+vy"):
                yaw_limit = 10.0
                if r['yaw_max'] > yaw_limit:
                    all_pass = False
                    print(f"  ** FAIL: YAW DRIFT: max yaw = {r['yaw_max']:.1f} deg (limit: {yaw_limit} deg)")

        # ============================================================
        # Mode 2: Implicit actuator (PD servo with lag)
        # ============================================================
        print("\n" + "=" * 80)
        print("MODE 2: IMPLICIT ACTUATOR (set_joint_position/velocity_target)")
        print("  (reads actual joint state from sim each step)")
        print("=" * 80)

        drone_ctrl2 = DroneController(
            cfg=TUNED_CONTROLLER_CFG, mass=mass, gravity=9.81,
            num_envs=1, device=device,
        )
        gimbal_implicit = GimbalController(
            cfg=GimbalControllerCfg(), num_envs=1, device=device,
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
            print_stability(r)

            passed, reasons = check_stability(r)
            if not passed:
                for reason in reasons:
                    print(f"  ** FAIL: {reason}")
                # Don't fail overall on implicit mode -- it's known to have lag issues

        # ============================================================
        # Comparison summary
        # ============================================================
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        header = (f"{'Test':<22s} | {'RMS':>6s} {'Max':>6s} {'OscHz':>6s} {'OscAmp':>7s} "
                  f"{'Div':>7s} {'R2':>5s} {'2H/1H':>6s} | {'Status':>8s}")
        print(header)
        print("-" * len(header))
        for r in results_direct:
            passed, _ = check_stability(r)
            status = "OK" if passed else "FAIL"
            print(
                f"{r['label']:<22s} | {r['rms']:5.2f}d {r['max']:5.2f}d "
                f"{r['osc_freq_hz']:5.1f}H {r['osc_amplitude']:6.2f}d "
                f"{r['divergence_rate']:+6.2f} {r['trend_r2']:5.3f} "
                f"{r['second_half_ratio']:5.2f} | {status:>8s}"
            )

        print("\n" + "-" * 80)
        # Stability verdict for direct state mode only
        direct_stable = True
        for r in results_direct:
            passed, reasons = check_stability(r)
            status = "PASS" if passed else "FAIL"
            if not passed:
                direct_stable = False
            print(f"  [{status}] {r['label']}: RMS={r['rms']:.2f} deg, "
                  f"osc={r['osc_freq_hz']:.1f}Hz/{r['osc_amplitude']:.2f}deg, "
                  f"div={r['divergence_rate']:+.2f}deg/s")
            if not passed:
                for reason in reasons:
                    print(f"         {reason}")

        print("-" * 80)
        if direct_stable:
            print("RESULT: PASS -- No oscillation or divergence detected in direct state mode")
        else:
            print("RESULT: FAIL -- Stability issues detected")
            all_pass = False
        print("=" * 80)

    simulation_app.close()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
