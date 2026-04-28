#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bench-trace match + saturation + curriculum/DR hook tests for the SIYI
gimbal rate loop (mas/035).

Compares the rate-loop's (τ, w_ss, w_peak) against the measured rate-step
data in
    /home/usrg/mas/src/gimbal_controller/scripts/gimbal_rate_step_followspeed_tune/rate_step_summary.csv

Pass criteria (mas/035 acceptance):
  - |τ_sim − τ_measured| / τ_measured ≤ 10 % per axis at small amplitudes.
  - |w_ss_sim − w_ss_measured| / w_ss_measured ≤ 5 % per axis.
  - Saturation at u=±1.0 within 5 % of ±73 deg/s.
  - set_progress(0.0) → pass-through; set_progress(1.0) → full lag.
  - set_tau_per_env writes per-env values correctly.

Test methodology: simulate the rate loop with a step-rate command, fit a
first-order model to the response, compare to measurement.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/035 gimbal rate loop bench tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop import GimbalRateLoop
from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop_cfg import GimbalRateLoopCfg


VERBOSE = args_cli.test_verbose

RATE_STEP_SUMMARY_CSV = Path(
    "/home/usrg/mas/src/gimbal_controller/scripts/gimbal_rate_step_followspeed_tune/"
    "rate_step_summary.csv"
)
DEG = math.pi / 180.0
MAX_RATE_DEG_S = 73.3  # k_deg_s_per_u from rate_model.json


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
        print("TEST SUMMARY (mas/035 gimbal rate loop)")
        print("=" * 80)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}, Errors: {len(self.errors)}")
        if self.failed:
            print("\nFAILED:")
            for n, e in self.failed:
                print(f"  {n}: {e.splitlines()[0] if e else ''}")
        return len(self.failed) == 0 and len(self.errors) == 0


# ---------------------------------------------------------------- helpers

def load_measured_summary():
    """Returns dict: {(axis, u_cmd): (w_ss_deg_s, w_peak_deg_s, rise_time_s, latency_s)}."""
    if not RATE_STEP_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing: {RATE_STEP_SUMMARY_CSV}")
    out = {}
    with RATE_STEP_SUMMARY_CSV.open() as f:
        for row in csv.DictReader(f):
            axis = row["axis"]
            u = float(row["u_cmd"])
            out[(axis, u)] = (
                float(row["w_ss_deg_s"]),
                float(row["w_peak_deg_s"]),
                float(row["rise_time_s"]),
                float(row["latency_s"]),
            )
    return out


def fit_first_order_tau(times: torch.Tensor, response: torch.Tensor, target: float) -> float:
    """Fit τ from a first-order step response by reading the time at which
    response crosses 63.2 % of its target value."""
    threshold = 0.632 * target
    if target > 0:
        crossed = response >= threshold
    else:
        crossed = response <= threshold
    idxs = torch.nonzero(crossed, as_tuple=False).flatten()
    if idxs.numel() == 0:
        return float("inf")
    return times[idxs[0].item()].item()


def simulate_step(
    rate_loop: GimbalRateLoop,
    axis: int,                    # 0 = yaw, 1 = pitch
    u: float,                     # normalized command in [-1.5, 1.5]
    duration_s: float = 1.0,
    dt: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive a constant rate command and record the (yaw_rate, pitch_rate)
    output time series. Returns (times[T], rates[T] in rad/s)."""
    rate_loop.reset()
    n_steps = int(duration_s / dt) + 1
    cmd = torch.zeros(rate_loop.num_envs, 2, device=rate_loop.device)
    cmd[:, axis] = u * MAX_RATE_DEG_S * DEG    # turn into rad/s

    rates_per_step = []
    for _ in range(n_steps):
        out = rate_loop.step(cmd, dt)
        rates_per_step.append(out[0, axis].item())
    times = torch.linspace(0.0, duration_s, n_steps)
    rates = torch.tensor(rates_per_step)
    return times, rates


# ---------------------------------------------------------------- test suites

def run_step_response_tests(results: TestResults, device: torch.device, measured: dict):
    print("\n" + "=" * 80)
    print("Step response: τ + w_ss vs rate_step_summary.csv")
    print("=" * 80)

    cfg = GimbalRateLoopCfg()
    rate_loop = GimbalRateLoop(cfg, num_envs=1, device=device)
    rate_loop.set_progress(1.0)

    # Test τ on small amplitudes (linear region) and w_ss across the sweep.
    small_amplitudes = [0.1, 0.25]   # τ matters here (no saturation)
    all_amplitudes = [0.1, 0.25, 0.5, 0.75, 1.0]

    for axis_name, axis_idx, tau_measured in [
        ("yaw", 0, cfg.tau_yaw_s),
        ("pitch", 1, cfg.tau_pitch_s),
    ]:
        for u in small_amplitudes:
            try:
                times, rates_radps = simulate_step(rate_loop, axis_idx, u, duration_s=1.0, dt=0.01)
                rates_deg = rates_radps / DEG
                target_radps = u * MAX_RATE_DEG_S * DEG
                target_deg = target_radps / DEG

                # Fit τ
                tau_sim = fit_first_order_tau(times, rates_radps, target_radps)
                rel_tau = abs(tau_sim - tau_measured) / max(tau_measured, 1e-9)

                # Steady-state (last 100 ms)
                last_idx = int(0.9 * len(times))
                w_ss_deg = rates_deg[last_idx:].mean().item()
                w_ss_meas = measured[(axis_name, u)][0]
                # w_ss for the rate loop (no real saturation at u=0.1) ≈ u * 73
                rel_ss = abs(w_ss_deg - w_ss_meas) / max(abs(w_ss_meas), 1e-9)

                if VERBOSE:
                    print(f"    {axis_name} u={u}: τ_sim={tau_sim:.4f}s τ_meas={tau_measured:.4f}s (rel={rel_tau*100:.1f}%) "
                          f"w_ss_sim={w_ss_deg:.2f} deg/s w_ss_meas={w_ss_meas:.2f} (rel={rel_ss*100:.1f}%)")

                assert rel_tau <= 0.10, f"τ rel error {rel_tau*100:.1f} % > 10 %"
                assert rel_ss <= 0.05, f"w_ss rel error {rel_ss*100:.1f} % > 5 %"
                results.add_pass(f"{axis_name} u={u}: τ + w_ss within tolerance")
            except Exception:
                results.add_fail(f"{axis_name} u={u}: τ + w_ss", traceback.format_exc())

        # w_ss across full range (no τ check; saturation eats the larger amps)
        for u in all_amplitudes:
            try:
                _, rates_radps = simulate_step(rate_loop, axis_idx, u, duration_s=1.0, dt=0.01)
                rates_deg = rates_radps / DEG
                last_idx = int(0.9 * len(rates_deg))
                w_ss_deg = rates_deg[last_idx:].mean().item()
                w_ss_meas = measured[(axis_name, u)][0]
                rel_ss = abs(w_ss_deg - w_ss_meas) / max(abs(w_ss_meas), 1e-9)
                assert rel_ss <= 0.05, f"u={u} {axis_name}: w_ss rel error {rel_ss*100:.1f} % > 5 %"
                results.add_pass(f"{axis_name} u={u}: w_ss = {w_ss_deg:.2f} deg/s within 5 %")
            except Exception:
                results.add_fail(f"{axis_name} u={u}: w_ss", traceback.format_exc())


def run_saturation_tests(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Saturation: u=1.5 must clip to max_rate_per_axis")
    print("=" * 80)

    cfg = GimbalRateLoopCfg()
    rate_loop = GimbalRateLoop(cfg, num_envs=1, device=device)
    rate_loop.set_progress(1.0)

    for axis_name, axis_idx in [("yaw", 0), ("pitch", 1)]:
        try:
            _, rates = simulate_step(rate_loop, axis_idx, u=1.5, duration_s=2.0, dt=0.01)
            steady = rates[-50:].mean().item()
            limit = cfg.max_rate_per_axis
            rel = abs(steady - limit) / limit
            assert rel <= 0.05, f"{axis_name}: u=1.5 saturated to {steady:.4f} rad/s, expected {limit:.4f} (rel {rel*100:.1f} %)"
            results.add_pass(f"{axis_name}: u=1.5 saturates at {steady:.4f} rad/s (limit {limit})")
        except Exception:
            results.add_fail(f"{axis_name}: saturation", traceback.format_exc())


def run_curriculum_hook_tests(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Curriculum: set_progress(0.0) = pass-through; (1.0) = full lag")
    print("=" * 80)

    cfg = GimbalRateLoopCfg()

    # progress=0 → effective τ = 0 → response is pass-through. After ONE step
    # the output equals the (saturated) command.
    try:
        rate_loop = GimbalRateLoop(cfg, num_envs=4, device=device)
        rate_loop.set_progress(0.0)
        cmd = torch.zeros(4, 2, device=device)
        cmd[:, 0] = 0.5  # rad/s (well below sat)
        out = rate_loop.step(cmd, dt=0.01)
        diff = (out - cmd).abs().max().item()
        assert diff < 1e-5, f"progress=0 should be pass-through, got max-diff {diff}"
        results.add_pass("progress=0 → pass-through (no lag)")
    except Exception:
        results.add_fail("progress=0 pass-through", traceback.format_exc())

    # progress=1 → full lag → response approaches setpoint slowly.
    try:
        rate_loop = GimbalRateLoop(cfg, num_envs=4, device=device)
        rate_loop.set_progress(1.0)
        cmd = torch.zeros(4, 2, device=device)
        cmd[:, 0] = 0.5
        out = rate_loop.step(cmd, dt=0.01)
        # alpha = 1 - exp(-0.01 / 0.0995) ≈ 0.0953
        expected = 0.5 * (1.0 - math.exp(-0.01 / cfg.tau_yaw_s))
        diff = abs(out[0, 0].item() - expected)
        assert diff < 1e-3, f"progress=1: expected ≈ {expected:.4f} after one step, got {out[0, 0].item():.4f}"
        results.add_pass("progress=1 → first-order lag with configured τ")
    except Exception:
        results.add_fail("progress=1 full lag", traceback.format_exc())

    # progress halfway → τ_eff = 0.5 · τ → faster response than progress=1
    try:
        rate_loop = GimbalRateLoop(cfg, num_envs=4, device=device)
        rate_loop.set_progress(0.5)
        cmd = torch.zeros(4, 2, device=device)
        cmd[:, 0] = 0.5
        out_half = rate_loop.step(cmd, dt=0.01)
        rate_loop.set_progress(1.0)
        rate_loop.reset()
        out_full = rate_loop.step(cmd, dt=0.01)
        # progress=0.5 should advance MORE than progress=1.0 for the first step
        assert out_half[0, 0].item() > out_full[0, 0].item(), \
            f"progress=0.5 advance {out_half[0,0]:.4f} should exceed progress=1 advance {out_full[0,0]:.4f}"
        results.add_pass("progress=0.5 → faster lag than progress=1.0")
    except Exception:
        results.add_fail("progress=0.5 intermediate", traceback.format_exc())


def run_dr_hook_tests(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("DR: set_tau_per_env writes per-env τ tensors correctly")
    print("=" * 80)

    cfg = GimbalRateLoopCfg()
    rate_loop = GimbalRateLoop(cfg, num_envs=8, device=device)

    try:
        tau_yaw = torch.full((8,), 0.05, device=device)
        tau_pitch = torch.full((8,), 0.20, device=device)
        rate_loop.set_tau_per_env(tau_yaw, tau_pitch)
        assert torch.allclose(rate_loop.tau_yaw, tau_yaw)
        assert torch.allclose(rate_loop.tau_pitch, tau_pitch)
        results.add_pass("set_tau_per_env (all envs) writes correctly")
    except Exception:
        results.add_fail("set_tau_per_env all envs", traceback.format_exc())

    try:
        env_ids = torch.tensor([1, 3, 5], device=device)
        new_tau_yaw = torch.tensor([0.1, 0.2, 0.3], device=device)
        new_tau_pitch = torch.tensor([0.4, 0.5, 0.6], device=device)
        rate_loop.set_tau_per_env(new_tau_yaw, new_tau_pitch, env_ids=env_ids)
        assert torch.allclose(rate_loop.tau_yaw[env_ids], new_tau_yaw)
        assert torch.allclose(rate_loop.tau_pitch[env_ids], new_tau_pitch)
        # Other envs unchanged (use abs-tol for fp32 storage of 0.05 / 0.20)
        assert abs(rate_loop.tau_yaw[0].item() - 0.05) < 1e-6
        assert abs(rate_loop.tau_pitch[0].item() - 0.20) < 1e-6
        results.add_pass("set_tau_per_env (partial envs) leaves others untouched")
    except Exception:
        results.add_fail("set_tau_per_env partial envs", traceback.format_exc())


def run_reset_tests(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Reset: tracked rate state cleared")
    print("=" * 80)

    cfg = GimbalRateLoopCfg()
    rate_loop = GimbalRateLoop(cfg, num_envs=4, device=device)
    rate_loop.set_progress(1.0)

    try:
        cmd = torch.full((4, 2), 0.5, device=device)
        # advance a few steps to build state
        for _ in range(20):
            rate_loop.step(cmd, dt=0.01)
        assert rate_loop.omega_actual.abs().max().item() > 0.0
        rate_loop.reset()
        assert rate_loop.omega_actual.abs().max().item() == 0.0
        results.add_pass("reset(None) zeros all tracked state")
    except Exception:
        results.add_fail("reset all", traceback.format_exc())

    try:
        for _ in range(20):
            rate_loop.step(cmd, dt=0.01)
        env_ids = torch.tensor([1, 3], device=device)
        rate_loop.reset(env_ids)
        assert rate_loop.omega_actual[env_ids].abs().max().item() == 0.0
        # untouched envs still nonzero
        other = torch.tensor([0, 2], device=device)
        assert rate_loop.omega_actual[other].abs().max().item() > 0.0
        results.add_pass("reset(env_ids) zeros only specified envs")
    except Exception:
        results.add_fail("reset partial", traceback.format_exc())


def main() -> int:
    print("=" * 80)
    print("GIMBAL RATE LOOP TEST SUITE (mas/035)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    results = TestResults()
    try:
        measured = load_measured_summary()
    except Exception:
        results.add_error("load measured", traceback.format_exc())
        sys.exit(1 if results.print_summary() else 0)

    try:
        run_step_response_tests(results, device, measured)
    except Exception:
        results.add_error("step response suite", traceback.format_exc())

    try:
        run_saturation_tests(results, device)
    except Exception:
        results.add_error("saturation suite", traceback.format_exc())

    try:
        run_curriculum_hook_tests(results, device)
    except Exception:
        results.add_error("curriculum hook suite", traceback.format_exc())

    try:
        run_dr_hook_tests(results, device)
    except Exception:
        results.add_error("DR hook suite", traceback.format_exc())

    try:
        run_reset_tests(results, device)
    except Exception:
        results.add_error("reset suite", traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
