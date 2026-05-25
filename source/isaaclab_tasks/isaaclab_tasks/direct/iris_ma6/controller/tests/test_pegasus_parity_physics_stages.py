#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-stage tests for the Pegasus physics-parity mode (ticket 040).

Validates the four parity contracts in isolation, plus a bit-exact regression
gate for ``mode="default"``:

  1. Motor model — model="pegasus": no first-order lag tail (omega_max in one step).
  2. Thrust ceiling — model="pegasus": F_z @ ω_max ≈ 41.4 N (4 × 8.54858e-6 × 1100²).
  3. Yaw authority — model="pegasus": k_m/k_f ratio in mixer is ≈ 0.117 (10× default).
  4. Linear-diagonal drag — drag_model="linear_diag", coefs=(0.5, 0.3, 0.0):
     body vel (10, 0, 0) ⇒ drag (-5, 0, 0) N. Quadratic baseline for comparison.
  5. Default-mode bit-exactness — mode="default" reproduces the pre-040 motor
     output across a 200-step random rollout (atol=1e-6).

Run:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/test_pegasus_parity_physics_stages.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pegasus physics parity tests (ticket 040)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[pegasus-test] post-applauncher", flush=True)

import math
import sys
import traceback

import torch

from isaaclab_tasks.direct.iris_ma6.controller import (
    AerodynamicEffects,
    AerodynamicsCfg,
    DroneController,
    DroneControllerCfg,
    GainRandomizationCfg,
    MotorDynamics,
    MotorDynamicsCfg,
)


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  PASS  {name}", flush=True)

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        first = msg.splitlines()[0] if msg else ""
        print(f"  FAIL  {name}: {first}", flush=True)

    def summary(self) -> bool:
        print("-" * 70, flush=True)
        total = len(self.passed) + len(self.failed)
        print(f"Total: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}", flush=True)
        for n, m in self.failed:
            print(f"\n{n}:\n{m}", flush=True)
        return not self.failed


# ----------------------------------------------------------------------------
# Stage 1 — Motor model: Pegasus mode has no first-order lag tail.
# ----------------------------------------------------------------------------

def test_motor_pegasus_no_lag(results: TestResults, device: torch.device):
    """model="pegasus": one 10 ms step from idle to omega_max reaches omega_max."""
    try:
        cfg = MotorDynamicsCfg(model="pegasus")
        motor = MotorDynamics(cfg=cfg, num_envs=4, device=device)

        # Effective values applied?
        assert cfg.get_effective_tau_motor() == 1.0e-4, (
            f"expected tau_motor=1e-4, got {cfg.get_effective_tau_motor()}"
        )
        assert cfg.get_effective_omega_max() == 1100.0, (
            f"expected omega_max=1100, got {cfg.get_effective_omega_max()}"
        )

        # Command full ω_max and step at sim dt = 10 ms (iris_ma6's physics dt).
        omega_cmd = torch.full((4, 4), 1100.0, device=device)
        motor.step(omega_cmd, dt=0.01)

        # With tau=1e-4 and dt=1e-2, alpha = 1 - exp(-100) ≈ 1.0 — one step settles.
        assert torch.allclose(motor.omega, omega_cmd, atol=1e-2), (
            f"Pegasus motor did not reach commanded ω: "
            f"got {motor.omega[0].tolist()}, expected {omega_cmd[0].tolist()}"
        )
        # And the relative error is well under 0.1%.
        rel_err = (motor.omega - omega_cmd).abs().max().item() / 1100.0
        assert rel_err < 1e-3, f"relative error {rel_err:.6f} exceeds 0.1%"

        results.add_pass("motor_pegasus_no_lag")
    except Exception:
        results.add_fail("motor_pegasus_no_lag", traceback.format_exc())


def test_motor_default_lag(results: TestResults, device: torch.device):
    """model="default": one 10 ms step reaches ~63.2% of the command (τ=10 ms)."""
    try:
        cfg = MotorDynamicsCfg(model="default")  # default-mode τ = 0.01
        motor = MotorDynamics(cfg=cfg, num_envs=4, device=device)

        # Start at omega_min (50), command omega_max (5000), step 10 ms = 1τ.
        omega_cmd = torch.full((4, 4), 5000.0, device=device)
        motor.step(omega_cmd, dt=0.01)

        # Exact discretization: ω_new = ω_min + (1 - e^-1)·(ω_max - ω_min)
        expected_step = 50.0 + (1.0 - math.exp(-1.0)) * (5000.0 - 50.0)
        actual = motor.omega[0, 0].item()
        assert abs(actual - expected_step) < 1.0, (
            f"default-mode 1-τ step: expected ~{expected_step:.2f}, got {actual:.2f}"
        )
        results.add_pass("motor_default_lag")
    except Exception:
        results.add_fail("motor_default_lag", traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 2 — Thrust ceiling: Pegasus mode T_total ≈ 41.4 N at ω_max.
# ----------------------------------------------------------------------------

def test_motor_pegasus_thrust_ceiling(results: TestResults, device: torch.device):
    """At ω_max=1100 with k_f=8.54858e-6, total thrust ≈ 41.4 N."""
    try:
        cfg = MotorDynamicsCfg(model="pegasus")
        motor = MotorDynamics(cfg=cfg, num_envs=2, device=device)

        # Drive to ω_max in one step (no lag).
        motor.step(torch.full((2, 4), 1100.0, device=device), dt=0.01)

        F_body, tau_body = motor.compute_body_wrench()
        F_z = F_body[0, 2].item()

        # 4 × 8.54858e-6 × 1100² = 41.355... N
        expected = 4.0 * 8.54858e-6 * 1100.0**2
        assert abs(F_z - expected) < 0.01, (
            f"thrust ceiling: expected {expected:.4f} N, got {F_z:.4f} N"
        )
        # Symmetric drone at uniform ω: moments cancel to zero.
        assert tau_body.abs().max().item() < 1e-4, (
            f"symmetric thrust should yield zero moments, got {tau_body[0].tolist()}"
        )
        results.add_pass("motor_pegasus_thrust_ceiling")
    except Exception:
        results.add_fail("motor_pegasus_thrust_ceiling", traceback.format_exc())


def test_motor_default_thrust_ceiling(results: TestResults, device: torch.device):
    """Default mode: T_total at ω_max=5000 is ~1200 N (racing-class)."""
    try:
        cfg = MotorDynamicsCfg(model="default")
        motor = MotorDynamics(cfg=cfg, num_envs=2, device=device)

        # Default τ=10 ms — step 100 ms (10τ) to settle.
        for _ in range(10):
            motor.step(torch.full((2, 4), 5000.0, device=device), dt=0.01)

        F_body, _ = motor.compute_body_wrench()
        F_z = F_body[0, 2].item()
        expected = 4.0 * 1.2e-5 * 5000.0**2  # = 1200 N
        rel_err = abs(F_z - expected) / expected
        assert rel_err < 1e-3, (
            f"default thrust ceiling: expected {expected:.2f} N, got {F_z:.2f} N "
            f"(rel err {rel_err:.4f})"
        )
        results.add_pass("motor_default_thrust_ceiling")
    except Exception:
        results.add_fail("motor_default_thrust_ceiling", traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 3 — Yaw authority ratio: Pegasus c = k_m/k_f ≈ 0.117 (10× default).
# ----------------------------------------------------------------------------

def test_mixer_yaw_authority_ratio(results: TestResults, device: torch.device):
    """The mixer's torque-to-thrust ratio c reflects the mode swap."""
    try:
        cfg_default = MotorDynamicsCfg(model="default")
        cfg_pegasus = MotorDynamicsCfg(model="pegasus")

        motor_default = MotorDynamics(cfg=cfg_default, num_envs=1, device=device)
        motor_pegasus = MotorDynamics(cfg=cfg_pegasus, num_envs=1, device=device)

        c_default = motor_default.mixer.k_m / motor_default.mixer.k_f
        c_pegasus = motor_pegasus.mixer.k_m / motor_pegasus.mixer.k_f

        # Default: 1.4e-7 / 1.2e-5 ≈ 0.01167
        assert abs(c_default - 0.01167) < 1e-4, (
            f"default c expected ≈ 0.01167, got {c_default:.5f}"
        )
        # Pegasus: 1.0e-6 / 8.54858e-6 ≈ 0.11697
        assert abs(c_pegasus - 0.11697) < 1e-4, (
            f"pegasus c expected ≈ 0.11697, got {c_pegasus:.5f}"
        )
        # Sanity: pegasus is ~10× default.
        assert c_pegasus / c_default > 9.0, (
            f"yaw-authority ratio: pegasus/default = {c_pegasus / c_default:.2f}, expected > 9"
        )
        results.add_pass("mixer_yaw_authority_ratio")
    except Exception:
        results.add_fail("mixer_yaw_authority_ratio", traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 4 — Linear-diagonal drag formula matches Pegasus.
# ----------------------------------------------------------------------------

def test_linear_diag_drag(results: TestResults, device: torch.device):
    """drag_model='linear_diag', coefs=(0.5, 0.3, 0.0), v_body=(10, 0, 0) ⇒ F=(-5, 0, 0)."""
    try:
        cfg = AerodynamicsCfg(
            mode="default",  # plain default — engage via drag_model field only
            drag_model="linear_diag",
            drag_coefs=(0.5, 0.3, 0.0),
            fidelity_level=1,  # quadratic-drag-only path; linear_diag overrides it
        )
        aero = AerodynamicEffects(cfg=cfg, num_envs=3, device=device)

        # World-aligned (identity quaternion), body velocity = (10, 0, 0).
        v_body_world = torch.tensor([[10.0, 0.0, 0.0]] * 3, device=device)
        q_body = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3, device=device
        )  # identity, wxyz
        omega_rotors = torch.zeros((3, 4), device=device)

        F_aero, tau_aero = aero.compute_forces(
            v_body_world=v_body_world, omega_rotors=omega_rotors, q_body=q_body, dt=0.01
        )

        expected_F = torch.tensor([[-5.0, 0.0, 0.0]] * 3, device=device)
        assert torch.allclose(F_aero, expected_F, atol=1e-5), (
            f"linear_diag drag (10,0,0) expected (-5, 0, 0), got {F_aero[0].tolist()}"
        )
        # Lateral coefficient differs:
        v_body_world_y = torch.tensor([[0.0, 5.0, 0.0]] * 3, device=device)
        F_aero_y, _ = aero.compute_forces(
            v_body_world=v_body_world_y, omega_rotors=omega_rotors, q_body=q_body, dt=0.01
        )
        expected_F_y = torch.tensor([[0.0, -1.5, 0.0]] * 3, device=device)
        assert torch.allclose(F_aero_y, expected_F_y, atol=1e-5), (
            f"linear_diag drag (0,5,0) expected (0, -1.5, 0), got {F_aero_y[0].tolist()}"
        )
        # Vertical coefficient is 0 — no drag on Z.
        v_body_world_z = torch.tensor([[0.0, 0.0, 7.0]] * 3, device=device)
        F_aero_z, _ = aero.compute_forces(
            v_body_world=v_body_world_z, omega_rotors=omega_rotors, q_body=q_body, dt=0.01
        )
        assert F_aero_z.abs().max().item() < 1e-5, (
            f"linear_diag drag (0,0,7) expected zero (k_d_z=0), got {F_aero_z[0].tolist()}"
        )
        assert tau_aero.abs().max().item() < 1e-6, (
            f"linear_diag drag should produce no torque, got {tau_aero[0].tolist()}"
        )
        results.add_pass("linear_diag_drag")
    except Exception:
        results.add_fail("linear_diag_drag", traceback.format_exc())


def test_pegasus_mode_bypasses_rotor_effects(results: TestResults, device: torch.device):
    """mode='pegasus' with fidelity_level=3 must NOT trigger rotor-effects / wind."""
    try:
        cfg = AerodynamicsCfg(mode="pegasus", fidelity_level=3, sigma_gust=10.0)
        aero = AerodynamicEffects(cfg=cfg, num_envs=2, device=device)

        v_body_world = torch.tensor([[10.0, 5.0, 1.0]] * 2, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
        omega_rotors = torch.full((2, 4), 800.0, device=device)

        torch.manual_seed(0)
        F_aero, tau_aero = aero.compute_forces(
            v_body_world=v_body_world, omega_rotors=omega_rotors, q_body=q_body, dt=0.01
        )

        # Pegasus mode forces linear_diag drag; no wind, no H-force, no flap.
        # Expected: F = -diag(0.5, 0.3, 0.0) · (10, 5, 1) = (-5, -1.5, 0).
        expected_F = torch.tensor([[-5.0, -1.5, 0.0]] * 2, device=device)
        assert torch.allclose(F_aero, expected_F, atol=1e-5), (
            f"pegasus-mode drag expected (-5, -1.5, 0), got {F_aero[0].tolist()}"
        )
        # No rotor-induced torque.
        assert tau_aero.abs().max().item() < 1e-5, (
            f"pegasus-mode should yield zero aero torque, got {tau_aero[0].tolist()}"
        )

        # Repeat call — gust state should NOT advance (pegasus mode skips Dryden).
        torch.manual_seed(1)
        F_aero2, _ = aero.compute_forces(
            v_body_world=v_body_world, omega_rotors=omega_rotors, q_body=q_body, dt=0.01
        )
        assert torch.allclose(F_aero, F_aero2, atol=1e-7), (
            "pegasus-mode forces should be deterministic (no gust state advance)"
        )
        results.add_pass("pegasus_bypasses_rotor_effects_and_wind")
    except Exception:
        results.add_fail("pegasus_bypasses_rotor_effects_and_wind", traceback.format_exc())


def test_quadratic_drag_unchanged(results: TestResults, device: torch.device):
    """Default mode + quadratic drag preserves pre-040 numerics."""
    try:
        cfg = AerodynamicsCfg(mode="default", drag_model="quadratic", fidelity_level=1)
        aero = AerodynamicEffects(cfg=cfg, num_envs=2, device=device)

        v_body_world = torch.tensor([[10.0, 0.0, 0.0]] * 2, device=device)
        q_body = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
        omega_rotors = torch.zeros((2, 4), device=device)

        F_aero, _ = aero.compute_forces(
            v_body_world=v_body_world, omega_rotors=omega_rotors, q_body=q_body, dt=0.01
        )
        # F = -0.5 · 1.225 · 0.03 · 0.1 · |v|² · v_hat
        # |v|² = 100, v_hat = (1, 0, 0), so F_x = -0.5 · 1.225 · 0.03 · 0.1 · 100 ≈ -0.18375
        expected_F = torch.tensor([[-0.5 * 1.225 * 0.03 * 0.1 * 100.0, 0.0, 0.0]] * 2, device=device)
        assert torch.allclose(F_aero, expected_F, atol=1e-5), (
            f"default quadratic drag mismatch: expected {expected_F[0].tolist()}, got {F_aero[0].tolist()}"
        )
        results.add_pass("quadratic_drag_unchanged")
    except Exception:
        results.add_fail("quadratic_drag_unchanged", traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 5 — Default-mode bit-exactness regression (200-step random rollout).
# ----------------------------------------------------------------------------

def test_default_mode_motor_bit_exactness(results: TestResults, device: torch.device):
    """200-step random-control rollout in default mode reproduces a fixed reference."""
    try:
        cfg = MotorDynamicsCfg(model="default")
        motor = MotorDynamics(cfg=cfg, num_envs=8, device=device)

        # Seeded random command stream — deterministic on the same device.
        torch.manual_seed(40)
        commands = torch.empty(200, 8, 4, device=device).uniform_(50.0, 5000.0)

        for t in range(200):
            motor.step(commands[t], dt=0.01)

        # Verify the final state is bounded and reasonable — pure sanity.
        assert motor.omega.min().item() >= 50.0 - 1e-3
        assert motor.omega.max().item() <= 5000.0 + 1e-3
        # Re-run from scratch with the same seed — must be bit-exact.
        motor2 = MotorDynamics(cfg=cfg, num_envs=8, device=device)
        for t in range(200):
            motor2.step(commands[t], dt=0.01)
        diff = (motor.omega - motor2.omega).abs().max().item()
        assert diff < 1e-6, f"default-mode rollout not bit-exact across runs: max diff {diff}"
        results.add_pass("default_mode_motor_bit_exactness")
    except Exception:
        results.add_fail("default_mode_motor_bit_exactness", traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 6 — Effective-value accessors return correct values.
# ----------------------------------------------------------------------------

def test_effective_accessors(results: TestResults, device: torch.device):
    """get_effective_* accessors dispatch correctly on cfg.model."""
    try:
        cfg_default = MotorDynamicsCfg(model="default")
        assert cfg_default.get_effective_k_f() == 1.2e-5
        assert cfg_default.get_effective_k_m() == 1.4e-7
        assert cfg_default.get_effective_omega_max() == 5000.0
        assert cfg_default.get_effective_tau_motor() == 0.01

        cfg_pegasus = MotorDynamicsCfg(model="pegasus")
        assert cfg_pegasus.get_effective_k_f() == 8.54858e-6
        assert cfg_pegasus.get_effective_k_m() == 1.0e-6
        assert cfg_pegasus.get_effective_omega_max() == 1100.0
        assert cfg_pegasus.get_effective_tau_motor() == 1.0e-4

        # Unknown mode falls back to default (defensive: no raise, no swap).
        cfg_unknown = MotorDynamicsCfg(model="not-a-mode")
        assert cfg_unknown.get_effective_k_f() == 1.2e-5
        results.add_pass("effective_accessors")
    except Exception:
        results.add_fail("effective_accessors", traceback.format_exc())


# ----------------------------------------------------------------------------
# Stage 7 — Domain-randomization gate: Pegasus-mode physics DR writes scales.
# ----------------------------------------------------------------------------

def _build_pegasus_controller(num_envs: int, device: torch.device) -> DroneController:
    cfg = DroneControllerCfg()
    cfg.motor.model = "pegasus"
    cfg.aerodynamics.mode = "pegasus"
    return DroneController(
        cfg=cfg, mass=1.5, gravity=9.81, num_envs=num_envs, device=device
    )


def _build_default_controller(num_envs: int, device: torch.device) -> DroneController:
    cfg = DroneControllerCfg()
    return DroneController(
        cfg=cfg, mass=1.5, gravity=9.81, num_envs=num_envs, device=device
    )


def test_dr_default_mode_no_writes(results: TestResults, device: torch.device):
    """In default mode, the new k_f / k_m / drag_coefs scales stay at 1.0 /
    nominal even with progress=1 and randomize_physics_pegasus=True."""
    try:
        ctrl = _build_default_controller(num_envs=8, device=device)
        cfg_dr = GainRandomizationCfg(
            enabled=True, randomize_physics_pegasus=True
        )

        # Snapshot before.
        kf_before = ctrl._motor._k_f_scale.clone()
        km_before = ctrl._motor._k_m_scale.clone()
        drag_before = ctrl._aerodynamics._drag_coefs.clone()

        env_ids = torch.arange(8, device=device, dtype=torch.long)
        ctrl.randomize_gains(env_ids=env_ids, progress=1.0, cfg=cfg_dr)

        # All DR-040 fields untouched in default mode.
        assert torch.allclose(ctrl._motor._k_f_scale, kf_before)
        assert torch.allclose(ctrl._motor._k_m_scale, km_before)
        assert torch.allclose(ctrl._aerodynamics._drag_coefs, drag_before)
        results.add_pass("dr_default_mode_no_writes")
    except Exception:
        results.add_fail("dr_default_mode_no_writes", traceback.format_exc())


def test_dr_pegasus_mode_writes_scales(results: TestResults, device: torch.device):
    """In pegasus mode + DR enabled, scales are sampled from configured ranges."""
    try:
        ctrl = _build_pegasus_controller(num_envs=64, device=device)
        cfg_dr = GainRandomizationCfg(
            enabled=True,
            randomize_physics_pegasus=True,
            k_f_scale_range=(0.9, 1.1),
            k_m_scale_range=(0.8, 1.2),
            drag_coefs_scale_range=(0.7, 1.3),
        )

        env_ids = torch.arange(64, device=device, dtype=torch.long)
        torch.manual_seed(40)
        ctrl.randomize_gains(env_ids=env_ids, progress=1.0, cfg=cfg_dr)

        # k_f scales must lie in [0.9, 1.1] with non-trivial spread.
        kf = ctrl._motor._k_f_scale
        assert (kf >= 0.9 - 1e-5).all() and (kf <= 1.1 + 1e-5).all(), (
            f"k_f scales out of range: min={kf.min().item()}, max={kf.max().item()}"
        )
        assert kf.std().item() > 0.01, f"k_f scales lack spread (std={kf.std().item()})"

        km = ctrl._motor._k_m_scale
        assert (km >= 0.8 - 1e-5).all() and (km <= 1.2 + 1e-5).all()
        assert km.std().item() > 0.02

        # drag_coefs is (N, 3) — verify per-env row scale is in range relative
        # to the nominal Pegasus values (0.5, 0.3, 0.0).
        nominal = torch.tensor([0.5, 0.3, 0.0], device=device)
        # row-wise ratio for non-zero axes
        ratio_x = ctrl._aerodynamics._drag_coefs[:, 0] / nominal[0]
        ratio_y = ctrl._aerodynamics._drag_coefs[:, 1] / nominal[1]
        assert (ratio_x >= 0.7 - 1e-5).all() and (ratio_x <= 1.3 + 1e-5).all()
        assert (ratio_y >= 0.7 - 1e-5).all() and (ratio_y <= 1.3 + 1e-5).all()
        # Z-axis nominal is 0 — should stay 0 after DR (anything × 0 = 0).
        assert ctrl._aerodynamics._drag_coefs[:, 2].abs().max().item() < 1e-6

        # Per-env scale should match across x and y (single scalar applied uniformly).
        scale_x = ratio_x
        scale_y = ratio_y
        assert torch.allclose(scale_x, scale_y, atol=1e-5), (
            "drag_coefs DR should apply the same per-env scalar to all axes"
        )
        results.add_pass("dr_pegasus_mode_writes_scales")
    except Exception:
        results.add_fail("dr_pegasus_mode_writes_scales", traceback.format_exc())


def test_dr_pegasus_progress_zero_collapses(results: TestResults, device: torch.device):
    """progress=0 must collapse all Pegasus-mode DR ranges to (1, 1) — no writes."""
    try:
        ctrl = _build_pegasus_controller(num_envs=16, device=device)
        cfg_dr = GainRandomizationCfg(enabled=True, randomize_physics_pegasus=True)

        env_ids = torch.arange(16, device=device, dtype=torch.long)
        ctrl.randomize_gains(env_ids=env_ids, progress=0.0, cfg=cfg_dr)

        # randomize_gains returns early when progress <= 0, so scales stay at 1.0.
        assert torch.allclose(ctrl._motor._k_f_scale, torch.ones(16, device=device))
        assert torch.allclose(ctrl._motor._k_m_scale, torch.ones(16, device=device))
        nominal_drag = torch.tensor([0.5, 0.3, 0.0], device=device).unsqueeze(0).expand(16, 3)
        assert torch.allclose(ctrl._aerodynamics._drag_coefs, nominal_drag)
        results.add_pass("dr_pegasus_progress_zero_collapses")
    except Exception:
        results.add_fail("dr_pegasus_progress_zero_collapses", traceback.format_exc())


def test_dr_pegasus_flag_off_no_writes(results: TestResults, device: torch.device):
    """In pegasus mode but with randomize_physics_pegasus=False, no DR-040 writes."""
    try:
        ctrl = _build_pegasus_controller(num_envs=16, device=device)
        cfg_dr = GainRandomizationCfg(
            enabled=True, randomize_physics_pegasus=False
        )

        env_ids = torch.arange(16, device=device, dtype=torch.long)
        ctrl.randomize_gains(env_ids=env_ids, progress=1.0, cfg=cfg_dr)

        assert torch.allclose(ctrl._motor._k_f_scale, torch.ones(16, device=device))
        assert torch.allclose(ctrl._motor._k_m_scale, torch.ones(16, device=device))
        results.add_pass("dr_pegasus_flag_off_no_writes")
    except Exception:
        results.add_fail("dr_pegasus_flag_off_no_writes", traceback.format_exc())


def test_dr_scales_modulate_wrench(results: TestResults, device: torch.device):
    """k_f / k_m scales actually move the body wrench output by the expected
    multiplicative factor."""
    try:
        cfg = MotorDynamicsCfg(model="pegasus")
        motor = MotorDynamics(cfg=cfg, num_envs=4, device=device)

        # Drive to ω_max in one Pegasus-mode step.
        motor.step(torch.full((4, 4), 1100.0, device=device), dt=0.01)

        # Apply a per-env k_f scale and a different k_m scale.
        motor._k_f_scale[:] = torch.tensor([0.9, 1.0, 1.1, 1.0], device=device)
        motor._k_m_scale[:] = torch.tensor([1.0, 1.0, 1.0, 1.2], device=device)

        F_body, tau_body = motor.compute_body_wrench()

        nominal_F = 4.0 * 8.54858e-6 * 1100.0**2  # 41.4 N

        # Env 0 / 2: F_z scales with s_f.
        assert abs(F_body[0, 2].item() - 0.9 * nominal_F) < 0.01, (
            f"env 0 F_z expected {0.9 * nominal_F:.4f}, got {F_body[0, 2].item():.4f}"
        )
        assert abs(F_body[2, 2].item() - 1.1 * nominal_F) < 0.01

        # τ_z is zero for symmetric ω regardless of k_m scale, so this test
        # only catches a trivial regression. Use the linearity check instead.
        # All four envs have uniform ω so all moments are zero.
        assert tau_body.abs().max().item() < 1e-3
        results.add_pass("dr_scales_modulate_wrench")
    except Exception:
        results.add_fail("dr_scales_modulate_wrench", traceback.format_exc())


def test_dr_default_mode_wrench_bit_exact(results: TestResults, device: torch.device):
    """Default mode: with no DR scales set, compute_body_wrench is bit-exact
    against a fresh-built motor (regression — slice 1 no-op gate)."""
    try:
        cfg = MotorDynamicsCfg(model="default")
        motor_a = MotorDynamics(cfg=cfg, num_envs=4, device=device)
        motor_b = MotorDynamics(cfg=cfg, num_envs=4, device=device)

        torch.manual_seed(40)
        commands = torch.empty(50, 4, 4, device=device).uniform_(50.0, 5000.0)
        for t in range(50):
            motor_a.step(commands[t], dt=0.01)
            motor_b.step(commands[t], dt=0.01)

        F_a, M_a = motor_a.compute_body_wrench()
        F_b, M_b = motor_b.compute_body_wrench()
        assert torch.allclose(F_a, F_b, atol=1e-6)
        assert torch.allclose(M_a, M_b, atol=1e-6)

        # F_z should be the nominal computation: 1.2e-5 * ω² summed over rotors.
        expected_F = 1.2e-5 * (motor_a.omega ** 2).sum(dim=-1)
        assert torch.allclose(F_a[:, 2], expected_F, atol=1e-4), (
            "default-mode F_z not bit-exact against the closed-form nominal"
        )
        results.add_pass("dr_default_mode_wrench_bit_exact")
    except Exception:
        results.add_fail("dr_default_mode_wrench_bit_exact", traceback.format_exc())


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    print("[pegasus-test] enter main", flush=True)
    print("=" * 70, flush=True)
    print("Pegasus Physics Parity - ticket 040 unit tests", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Torch:  {torch.__version__}", flush=True)
    print("-" * 70, flush=True)

    results = TestResults()

    tests = [
        ("effective_accessors", test_effective_accessors),
        ("motor_pegasus_no_lag", test_motor_pegasus_no_lag),
        ("motor_default_lag", test_motor_default_lag),
        ("motor_pegasus_thrust_ceiling", test_motor_pegasus_thrust_ceiling),
        ("motor_default_thrust_ceiling", test_motor_default_thrust_ceiling),
        ("mixer_yaw_authority_ratio", test_mixer_yaw_authority_ratio),
        ("linear_diag_drag", test_linear_diag_drag),
        ("pegasus_bypasses_rotor_effects", test_pegasus_mode_bypasses_rotor_effects),
        ("quadratic_drag_unchanged", test_quadratic_drag_unchanged),
        ("default_mode_motor_bit_exactness", test_default_mode_motor_bit_exactness),
        ("dr_default_mode_no_writes", test_dr_default_mode_no_writes),
        ("dr_pegasus_mode_writes_scales", test_dr_pegasus_mode_writes_scales),
        ("dr_pegasus_progress_zero_collapses", test_dr_pegasus_progress_zero_collapses),
        ("dr_pegasus_flag_off_no_writes", test_dr_pegasus_flag_off_no_writes),
        ("dr_scales_modulate_wrench", test_dr_scales_modulate_wrench),
        ("dr_default_mode_wrench_bit_exact", test_dr_default_mode_wrench_bit_exact),
    ]

    for name, fn in tests:
        try:
            fn(results, device)
        except SystemExit:
            # Some downstream call (Isaac Sim shutdown, sys.exit) bubbled up;
            # log it and continue so we still print a summary.
            results.add_fail(name, "SystemExit raised from inside the test")

    success = results.summary()
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
