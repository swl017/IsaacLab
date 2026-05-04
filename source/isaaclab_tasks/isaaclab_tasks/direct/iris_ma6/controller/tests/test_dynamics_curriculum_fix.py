#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the iris_ma6 dynamics-randomization curriculum fixes.

Covers:
- Bug 4: ``DroneController._nominal_gains`` is captured at construction
  time from the configured cfg values, not lazily on first
  ``randomize_gains`` call (which previously snapshotted whatever the
  env had written into ``_zoom._tau_zoom`` for curriculum gating).
- Range tightening: ``zoom_scale_range = (0.5, 2.0)`` and
  ``max_lin_vel_scale_range = (0.8, 1.2)`` defaults.
- ``randomize_gains`` no longer accepts ``tau_zoom_nominal``: per-env
  tau_zoom is sampled around the configured cfg.tau_zoom only.

Run via:
    ./isaaclab.sh -p source/.../controller/tests/test_dynamics_curriculum_fix.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run dynamics-curriculum fix tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback

import torch

from isaaclab_tasks.direct.iris_ma6.controller import (
    DroneController,
    DroneControllerCfg,
    GainRandomizationCfg,
)


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  PASS  {name}")

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        first = msg.splitlines()[0] if msg else ""
        print(f"  FAIL  {name}: {first}")

    def summary(self) -> bool:
        print("-" * 70)
        print(f"Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        for n, m in self.failed:
            print(f"\n{n}:\n{m}")
        return not self.failed


def _make_controller(num_envs: int, device: torch.device) -> DroneController:
    cfg = DroneControllerCfg()
    return DroneController(
        cfg=cfg, mass=1.5, gravity=9.81, num_envs=num_envs, device=device
    )


def test_nominal_gains_captured_at_construction(results: TestResults, device):
    """`_nominal_gains` exists immediately after __init__ and matches cfg."""
    try:
        ctrl = _make_controller(num_envs=8, device=device)
        assert hasattr(ctrl, "_nominal_gains"), "missing _nominal_gains attribute"
        nom = ctrl._nominal_gains
        for key in [
            "Kp_vel", "Ki_vel", "Kp_att",
            "Kp_rate", "Ki_rate", "Kd_rate",
            "tau_motor", "tau_zoom", "max_zoom_rate",
        ]:
            assert key in nom, f"missing key {key} in _nominal_gains"
        results.add_pass("Nominal gains exist immediately after __init__")
    except Exception:
        results.add_fail(
            "Nominal gains exist immediately after __init__", traceback.format_exc()
        )


def test_nominal_tau_zoom_matches_cfg(results: TestResults, device):
    """The captured tau_zoom nominal must equal cfg.zoom.tau_zoom for all envs.

    Before the fix, the lazy capture would have grabbed whatever was in
    ``_zoom._tau_zoom`` at first randomize_gains call (typically 1e-4 from
    env-side curriculum gating).
    """
    try:
        ctrl = _make_controller(num_envs=16, device=device)
        cfg_tau = ctrl.cfg.zoom.tau_zoom
        captured = ctrl._nominal_gains["tau_zoom"]
        assert captured.shape == (16,), f"expected (16,), got {tuple(captured.shape)}"
        assert torch.allclose(
            captured, torch.full_like(captured, cfg_tau)
        ), f"nominal tau_zoom = {captured.unique()}, expected {cfg_tau}"
        results.add_pass("Nominal tau_zoom equals cfg.zoom.tau_zoom for all envs")
    except Exception:
        results.add_fail(
            "Nominal tau_zoom equals cfg.zoom.tau_zoom for all envs",
            traceback.format_exc(),
        )


def test_progress_zero_is_identity(results: TestResults, device):
    """At progress=0, randomize_gains is a no-op (returns early)."""
    try:
        ctrl = _make_controller(num_envs=8, device=device)
        gain_cfg = GainRandomizationCfg(enabled=True)
        # Snapshot live gains
        tau_zoom_before = ctrl._zoom._tau_zoom.clone()
        kp_vel_before = ctrl._velocity._Kp_vel.clone()
        kp_rate_before = ctrl._rate._Kp_rate.clone()
        tau_motor_before = ctrl._motor._tau_motor.clone()

        ctrl.randomize_gains(
            env_ids=torch.arange(8, device=device),
            progress=0.0,
            cfg=gain_cfg,
        )

        assert torch.equal(ctrl._zoom._tau_zoom, tau_zoom_before)
        assert torch.equal(ctrl._velocity._Kp_vel, kp_vel_before)
        assert torch.equal(ctrl._rate._Kp_rate, kp_rate_before)
        assert torch.equal(ctrl._motor._tau_motor, tau_motor_before)
        results.add_pass("progress=0 is identity (no gain mutation)")
    except Exception:
        results.add_fail(
            "progress=0 is identity (no gain mutation)", traceback.format_exc()
        )


def test_full_progress_zoom_range_around_nominal(results: TestResults, device):
    """At progress=1 with zoom_scale_range=(0.5, 2.0), per-env tau_zoom must
    fall in [cfg.tau_zoom * 0.5, cfg.tau_zoom * 2.0]."""
    try:
        torch.manual_seed(0)
        N = 1024
        ctrl = _make_controller(num_envs=N, device=device)
        cfg_tau = ctrl.cfg.zoom.tau_zoom
        gain_cfg = GainRandomizationCfg(
            enabled=True, zoom_scale_range=(0.5, 2.0)
        )
        ctrl.randomize_gains(
            env_ids=torch.arange(N, device=device),
            progress=1.0,
            cfg=gain_cfg,
        )
        tau = ctrl._zoom._tau_zoom
        lo = cfg_tau * 0.5
        hi = cfg_tau * 2.0
        assert tau.min().item() >= lo - 1e-7, (
            f"min tau_zoom {tau.min().item()} < {lo}"
        )
        assert tau.max().item() <= hi + 1e-7, (
            f"max tau_zoom {tau.max().item()} > {hi}"
        )
        # Sanity: range is actually exercised (not stuck at nominal)
        assert tau.std().item() > 0.0, "tau_zoom did not vary across envs"
        # And the nominal snapshot is still untouched
        assert torch.allclose(
            ctrl._nominal_gains["tau_zoom"],
            torch.full_like(ctrl._nominal_gains["tau_zoom"], cfg_tau),
        ), "_nominal_gains['tau_zoom'] mutated after randomize_gains"
        results.add_pass(
            f"Full-progress tau_zoom in [{lo:.3f}, {hi:.3f}], nominal unchanged"
        )
    except Exception:
        results.add_fail(
            "Full-progress tau_zoom range", traceback.format_exc()
        )


def test_partial_progress_only_targets_env_ids(results: TestResults, device):
    """Randomizing env_ids=[0,1,2] must NOT touch other envs' gains."""
    try:
        torch.manual_seed(1)
        N = 16
        ctrl = _make_controller(num_envs=N, device=device)
        gain_cfg = GainRandomizationCfg(enabled=True)
        # Capture untouched envs' starting gains (these are post-init nominal copies)
        kp_vel_before = ctrl._velocity._Kp_vel.clone()
        tau_zoom_before = ctrl._zoom._tau_zoom.clone()

        ctrl.randomize_gains(
            env_ids=torch.tensor([0, 1, 2], device=device),
            progress=1.0,
            cfg=gain_cfg,
        )

        # Untouched rows must be exactly equal
        assert torch.equal(
            ctrl._velocity._Kp_vel[3:], kp_vel_before[3:]
        ), "Untouched envs had Kp_vel modified"
        assert torch.equal(
            ctrl._zoom._tau_zoom[3:], tau_zoom_before[3:]
        ), "Untouched envs had tau_zoom modified"
        results.add_pass("Partial randomization confines mutation to env_ids")
    except Exception:
        results.add_fail(
            "Partial randomization confines mutation to env_ids",
            traceback.format_exc(),
        )


def test_default_gain_cfg_ranges(results: TestResults, _device):
    """The new defaults (per the curriculum fix plan) are tighter than the
    pre-fix wide ranges intended to compensate for the broken curriculum."""
    try:
        cfg = GainRandomizationCfg()
        assert cfg.max_lin_vel_scale_range == (0.8, 1.2), (
            f"max_lin_vel_scale_range default is {cfg.max_lin_vel_scale_range}, "
            f"expected (0.8, 1.2)"
        )
        assert cfg.zoom_scale_range == (0.5, 2.0), (
            f"zoom_scale_range default is {cfg.zoom_scale_range}, "
            f"expected (0.5, 2.0)"
        )
        results.add_pass("GainRandomizationCfg defaults match the fix plan")
    except Exception:
        results.add_fail(
            "GainRandomizationCfg defaults match the fix plan",
            traceback.format_exc(),
        )


def test_randomize_gains_signature_no_tau_zoom_nominal(results: TestResults, device):
    """Calling randomize_gains with the legacy tau_zoom_nominal kwarg must fail."""
    try:
        ctrl = _make_controller(num_envs=4, device=device)
        gain_cfg = GainRandomizationCfg(enabled=True)
        try:
            ctrl.randomize_gains(
                env_ids=torch.arange(4, device=device),
                progress=1.0,
                cfg=gain_cfg,
                tau_zoom_nominal=0.5,  # type: ignore[call-arg]
            )
            raise AssertionError(
                "randomize_gains accepted legacy tau_zoom_nominal kwarg"
            )
        except TypeError:
            results.add_pass(
                "randomize_gains rejects legacy tau_zoom_nominal kwarg"
            )
    except Exception:
        results.add_fail(
            "randomize_gains signature check", traceback.format_exc()
        )


def main() -> int:
    print("=" * 70)
    print("DYNAMICS CURRICULUM FIX TESTS")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = TestResults()
    for fn in [
        test_nominal_gains_captured_at_construction,
        test_nominal_tau_zoom_matches_cfg,
        test_progress_zero_is_identity,
        test_full_progress_zoom_range_around_nominal,
        test_partial_progress_only_targets_env_ids,
        test_default_gain_cfg_ranges,
        test_randomize_gains_signature_no_tau_zoom_nominal,
    ]:
        try:
            fn(results, device)
        except Exception:
            results.add_fail(fn.__name__, traceback.format_exc())

    ok = results.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
