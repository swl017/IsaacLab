#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-in-the-loop validation for the SIYI rate-loop wrapper (mas/035).

Closes the loop the unit tests do NOT:
    GimbalRateLoop.step  →  GimbalController.compute_control
                                       │
                                       ▼
                           set_joint_position_target
                                       │
                                       ▼
                           Isaac Sim asset PD (k=2e3, c=1e2)
                                       │
                                       ▼
                              robot.data.joint_pos  (read back)

For each amplitude u ∈ {0.1, 0.25, 0.5, 0.75, 1.0}, this script:
  1. Spawns iris_gimbal3 in Isaac Sim with the mas/035 reverted-stiff PD.
  2. Holds the body at world identity via `write_root_pose_to_sim` each
     step (no thrust loop) so the LOS-stabilization path is dormant and
     the recorded joint-rate is purely the user-cmd path through the
     rate loop + asset PD.
  3. Issues a step rate command at t = 0.5 s (after a 0.5 s settle).
  4. Records joint angle for 1.5 s post-step, computes joint angular rate
     by finite difference, fits the first-order time constant.
  5. Compares sim (τ, w_ss) against measured rate_step_summary.csv.

Saves per-amplitude rate-trace plots to `plots_mas035/` next to this file.
Pass criteria match `test_gimbal_rate_loop.py`: |Δτ|/τ ≤ 10 %, |Δw_ss|/w_ss ≤ 5 %.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Physics-in-the-loop SIYI rate-loop validation (mas/035)"
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--save-plots", action="store_true", default=True)
parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
parser.add_argument("--test-verbose", action="store_true")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import csv
import math
import traceback
from datetime import datetime
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context

from isaaclab_assets import IRIS_GIMBAL3_CFG
from isaaclab_tasks.direct.iris_ma6.controller import DroneController, DroneControllerCfg
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_controller import YAW_JOINT_OFFSET


VERBOSE = args_cli.test_verbose
DEG = math.pi / 180.0
RATE_STEP_SUMMARY_CSV = Path(
    "/home/usrg/mas/src/gimbal_controller/scripts/gimbal_rate_step_followspeed_tune/"
    "rate_step_summary.csv"
)
PLOTS_DIR = Path(__file__).parent / "plots_mas035"

# Try to import matplotlib — plots are optional, sim-only validation runs anyway.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------------------------------------------------- helpers


def load_measured_summary() -> dict:
    """Returns {(axis, u_cmd): (w_ss_deg_s, w_peak_deg_s, rise_time_s, latency_s)}."""
    if not RATE_STEP_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing measurement CSV: {RATE_STEP_SUMMARY_CSV}")
    out = {}
    with RATE_STEP_SUMMARY_CSV.open() as f:
        for row in csv.DictReader(f):
            out[(row["axis"], float(row["u_cmd"]))] = (
                float(row["w_ss_deg_s"]),
                float(row["w_peak_deg_s"]),
                float(row["rise_time_s"]),
                float(row["latency_s"]),
            )
    return out


def fit_first_order_tau(times, rates, target):
    """Time at which `rates` first crosses 63.2 % of `target`."""
    threshold = 0.632 * target
    if target > 0:
        crossed = [r >= threshold for r in rates]
    else:
        crossed = [r <= threshold for r in rates]
    for i, c in enumerate(crossed):
        if c:
            return times[i]
    return float("inf")


def run_step_response(
    axis: str,
    u_norm: float,
    sim,
    robot: Articulation,
    drone_ctrl: DroneController,
    body_id,
    joint_ids: dict,
    device: torch.device,
    dt: float,
    settle_s: float = 0.5,
    record_s: float = 1.5,
):
    """Drive a step rate command on the chosen axis, record joint trace.

    The drone body is pinned to world identity at every step so the
    GimbalController's LOS stabilization is a no-op and the recorded
    joint trace reflects ONLY the user-rate-cmd path through the rate
    loop + asset PD. Returns (times[T], joint_angle_deg[T]) — the joint
    rate is computed downstream by finite difference.
    """
    drone_ctrl.reset()
    # Force rate loop fully active (curriculum is otherwise gated)
    drone_ctrl.gimbal_rate_loop.set_progress(1.0)

    # Pin body root pose at world origin
    root_pos = torch.tensor([[0.0, 0.0, 5.0]], device=device)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    root_vel = torch.zeros((1, 6), device=device)

    gimbal_joint_id_list = [
        joint_ids["pitch"], joint_ids["yaw"], joint_ids["roll"]
    ]

    settle_steps = int(settle_s / dt)
    record_steps = int(record_s / dt)

    times = []
    joint_angle_deg = []   # the axis being driven, in deg
    cmd_active = []        # bool: is the step command currently issued
    sim_dt_decimated = dt * 4  # decimation=4 in iris_ma6_test_cfg

    def one_step(yaw_norm: float, pitch_norm: float):
        # Pin body to world identity each step (no thrust dynamics / drift)
        robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        robot.write_root_velocity_to_sim(root_vel)

        gimbal_jp = torch.stack([
            robot.data.joint_pos[:, joint_ids["pitch"]],
            robot.data.joint_pos[:, joint_ids["yaw"]],
            robot.data.joint_pos[:, joint_ids["roll"]],
        ], dim=-1)

        F, tau, gp, gv, _ = drone_ctrl.step_policy(
            v_cmd=torch.zeros((1, 3), device=device),
            yaw_rate_cmd=torch.zeros(1, device=device),
            gimbal_yaw_rate_cmd=torch.tensor([yaw_norm], device=device),
            gimbal_pitch_rate_cmd=torch.tensor([pitch_norm], device=device),
            zoom_rate_cmd=torch.zeros(1, device=device),
            q_body=robot.data.root_quat_w,
            v_body=robot.data.root_lin_vel_w,
            omega_body=robot.data.root_ang_vel_b,
            sim_dt=sim_dt_decimated,
            gimbal_joint_positions=gimbal_jp,
            physics_dt=dt,
        )
        # Apply zero external force (body is pinned, so we don't need thrust)
        robot.set_external_force_and_torque(
            forces=torch.zeros((1, 1, 3), device=device),
            torques=torch.zeros((1, 1, 3), device=device),
            body_ids=body_id,
        )

        g_yaw, g_roll, g_pitch = gp
        g_yaw_v, g_roll_v, g_pitch_v = gv
        pos_cmd = torch.stack(
            [g_pitch, g_yaw + YAW_JOINT_OFFSET, g_roll], dim=-1
        )
        vel_cmd = torch.stack([g_pitch_v, g_yaw_v, g_roll_v], dim=-1)
        robot.set_joint_position_target(target=pos_cmd, joint_ids=gimbal_joint_id_list)
        robot.set_joint_velocity_target(target=vel_cmd, joint_ids=gimbal_joint_id_list)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ---- settle phase: zero rate command, hold pose ----
    for _ in range(settle_steps):
        one_step(0.0, 0.0)

    # ---- step phase: drive rate cmd on chosen axis ----
    yaw_cmd_step = u_norm if axis == "yaw" else 0.0
    pitch_cmd_step = u_norm if axis == "pitch" else 0.0

    for k in range(record_steps):
        one_step(yaw_cmd_step, pitch_cmd_step)
        times.append(k * dt)
        if axis == "yaw":
            j = float(robot.data.joint_pos[0, joint_ids["yaw"]].item()) - YAW_JOINT_OFFSET
        else:
            j = float(robot.data.joint_pos[0, joint_ids["pitch"]].item())
        joint_angle_deg.append(math.degrees(j))
        cmd_active.append(True)

    return times, joint_angle_deg, cmd_active


def analyze_response(
    times: list[float],
    angle_deg: list[float],
    u_norm: float,
    measured: tuple,
    cfg_max_rate_radps: float,
    axis: str,
):
    """Fit τ + steady-state from a recorded angle trace.

    The angle trace starts at the moment the step command is issued.
    Joint rate is the finite difference of angle wrt time (deg/s).

    Sign convention: rate_step_summary.csv reports unsigned `w_ss_deg_s`
    that increases with `u_cmd` per axis-internal direction. The IL
    pitch convention is opposite (policy +pitch_rate_cmd → joint pitch
    decreases, +elevation_world). We compare on |w_ss| so we are
    convention-agnostic.

    Steady-state window: 200 ms centered ~4τ after the step (t=0.30 s
    to t=0.50 s) — far enough out that τ has converged but before the
    pitch joint hits its ±45° limit at high amplitudes.
    """
    n = len(times)
    if n < 5:
        return {"valid": False}
    dt = times[1] - times[0]
    rates_deg = [
        (angle_deg[i + 1] - angle_deg[i]) / dt for i in range(n - 1)
    ]
    rate_times = [(times[i] + times[i + 1]) * 0.5 for i in range(n - 1)]

    # Steady-state window: post-τ, pre-limit. 0.30 s to 0.50 s post-step.
    win_start = 0.30
    win_end = 0.50
    win_idx = [i for i, t in enumerate(rate_times) if win_start <= t <= win_end]
    if not win_idx:
        return {"valid": False}
    w_ss_deg = sum(rates_deg[i] for i in win_idx) / len(win_idx)

    # τ from 63.2 % crossing of the steady-state value (signed match)
    tau_sim = fit_first_order_tau(rate_times, rates_deg, w_ss_deg)

    measured_w_ss = measured[0]
    measured_tau_default = 0.0995 if axis == "yaw" else 0.0954
    # Compare on |w_ss| — sign convention differs between IL pitch and
    # the bench measurement (see docstring).
    rel_w_ss = abs(abs(w_ss_deg) - abs(measured_w_ss)) / max(abs(measured_w_ss), 1e-9)

    return {
        "valid": True,
        "rate_times": rate_times,
        "rates_deg": rates_deg,
        "w_ss_deg_sim": w_ss_deg,
        "w_ss_deg_meas": measured_w_ss,
        "rel_w_ss": rel_w_ss,
        "tau_sim": tau_sim,
        "tau_meas": measured_tau_default,
    }


def plot_axis_summary(axis: str, results: list[dict]):
    """Per-axis time-series overlay of sim vs measured first-order math model.

    Three traces per amplitude:
      - solid : sim joint rate (finite-difference of robot.data.joint_pos)
      - dotted: analytical first-order model y(t) = ω_ss · (1 − exp(−t/τ))
                with τ from rate_model.json and ω_ss from rate_step_summary.csv
                (sign-flipped on pitch to match the IL convention)
      - dashed: measured steady-state target (horizontal asymptote)
    """
    if not HAS_MPL:
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    # Per-axis τ from rate_model.json
    tau_axis = 0.0995 if axis == "yaw" else 0.0954
    # IL pitch convention is opposite to bench measurement — flip the
    # math-model sign on pitch to overlay cleanly on the sim trace.
    sign = +1.0 if axis == "yaw" else -1.0

    for r, c in zip(results, colors):
        if not r["analysis"]["valid"]:
            continue
        a = r["analysis"]
        # Sim trace (solid)
        ax.plot(
            a["rate_times"], a["rates_deg"],
            color=c, linewidth=1.4,
            label=f"u={r['u']:.2f}: sim ω_ss={a['w_ss_deg_sim']:+.1f}°/s "
                  f"(meas {a['w_ss_deg_meas']:+.1f}°/s)",
        )
        # Analytical first-order overlay (dotted)
        w_ss_target = sign * a["w_ss_deg_meas"]
        model_t = [t for t in a["rate_times"]]
        model_rate = [
            w_ss_target * (1.0 - math.exp(-t / tau_axis)) for t in model_t
        ]
        ax.plot(
            model_t, model_rate,
            color=c, linewidth=1.0, linestyle=":",
            alpha=0.85,
        )
        # Steady-state asymptote (dashed)
        ax.axhline(
            w_ss_target,
            color=c, linewidth=0.5, linestyle="--", alpha=0.4,
        )
    ax.set_xlabel("time since step [s]")
    ax.set_ylabel(f"{axis} joint angular rate [deg/s]")
    ax.set_title(
        f"mas/035 physics-in-the-loop {axis} rate response\n"
        f"(solid=sim, dotted=first-order model τ={tau_axis:.4f}s, "
        f"dashed=meas ω_ss asymptote)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out_path = PLOTS_DIR / f"rate_response_{axis}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → wrote {out_path}")


# ---------------------------------------------------------------- TestResults


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []

    def add_pass(self, name): self.passed.append(name); print(f"  ✓ {name}")
    def add_fail(self, name, err):
        self.failed.append((name, err))
        print(f"  ✗ {name}")
        for line in (err or "").split("\n")[:4]:
            print(f"    {line}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 80)
        print("PHYSICS RATE-LOOP TEST SUMMARY (mas/035)")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        if self.failed:
            for n, e in self.failed:
                print(f"  ✗ {n}: {e.splitlines()[0] if e else ''}")
        return len(self.failed) == 0


# ------------------------------------------------------------------- main


def main() -> int:
    print("=" * 80)
    print("PHYSICS-IN-THE-LOOP RATE-LOOP TEST (mas/035)")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Plots dir: {PLOTS_DIR}")

    measured = load_measured_summary()

    sim_dt = 0.01
    results_total = TestResults()

    with build_simulation_context(
        dt=sim_dt, gravity_enabled=True, device=device_str,
        add_ground_plane=True, add_lighting=True,
    ) as sim:
        robot_cfg = copy.deepcopy(IRIS_GIMBAL3_CFG)
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # body pinned via write_root_pose; keep gravity off for stability
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

        # Use the iris_ma6 default DroneControllerCfg — picks up the mas/035
        # GimbalRateLoopCfg defaults via the `gimbal_rate_loop` field.
        ctrl_cfg = DroneControllerCfg()
        drone_ctrl = DroneController(
            cfg=ctrl_cfg, mass=mass, gravity=9.81, num_envs=1, device=device,
        )

        amplitudes = [0.1, 0.25, 0.5, 0.75, 1.0]
        all_results = {"yaw": [], "pitch": []}

        for axis in ("yaw", "pitch"):
            print("\n" + "-" * 80)
            print(f"Axis: {axis}")
            print("-" * 80)
            for u in amplitudes:
                print(f"\n  step u={u:.2f}", flush=True)
                try:
                    times, angle, _ = run_step_response(
                        axis=axis, u_norm=u, sim=sim, robot=robot,
                        drone_ctrl=drone_ctrl, body_id=body_id,
                        joint_ids=joint_ids, device=device, dt=sim_dt,
                    )
                    cfg_max_rate_radps = ctrl_cfg.gimbal.max_gimbal_rate
                    analysis = analyze_response(
                        times, angle, u_norm=u, measured=measured[(axis, u)],
                        cfg_max_rate_radps=cfg_max_rate_radps, axis=axis,
                    )
                    all_results[axis].append({
                        "u": u, "times": times, "angle": angle,
                        "analysis": analysis, "measured": measured[(axis, u)],
                    })
                    a = analysis
                    if not a["valid"]:
                        results_total.add_fail(
                            f"{axis} u={u}: analysis invalid",
                            "trace too short or NaN"
                        )
                        continue
                    print(
                        f"    sim:    ω_ss={a['w_ss_deg_sim']:+8.2f} deg/s  τ={a['tau_sim']:.4f} s"
                    )
                    print(
                        f"    meas:   ω_ss={a['w_ss_deg_meas']:+8.2f} deg/s  "
                        f"(rise_time {measured[(axis, u)][2]:.3f} s)"
                    )
                    rel_ss = a["rel_w_ss"]
                    if rel_ss <= 0.05:
                        results_total.add_pass(
                            f"{axis} u={u:.2f}: ω_ss rel error "
                            f"{rel_ss * 100:.1f}% ≤ 5%"
                        )
                    else:
                        results_total.add_fail(
                            f"{axis} u={u:.2f}: ω_ss rel error "
                            f"{rel_ss * 100:.1f}% > 5%",
                            f"sim {a['w_ss_deg_sim']:.2f} vs meas {a['w_ss_deg_meas']:.2f}",
                        )
                except Exception:
                    results_total.add_fail(
                        f"{axis} u={u:.2f}: exception", traceback.format_exc()
                    )

        # ---- Plots ----
        if args_cli.save_plots and HAS_MPL:
            for axis in ("yaw", "pitch"):
                plot_axis_summary(axis, all_results[axis])
        elif args_cli.save_plots and not HAS_MPL:
            print("  (matplotlib not available — skipping plots)")

    sim_app_close = getattr(sim, "_app", None)
    if sim_app_close is not None:
        try:
            simulation_app.close()
        except Exception:
            pass

    return 0 if results_total.print_summary() else 1


if __name__ == "__main__":
    sys.exit(main())
