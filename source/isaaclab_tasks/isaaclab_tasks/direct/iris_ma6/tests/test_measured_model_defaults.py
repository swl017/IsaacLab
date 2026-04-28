#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ticket mas/029 — measured camera/gimbal/latency models in iris_ma6.

Covers:
  1. SIYI zoom curve `compute_z_eff(zoom_cmd)` matches the bench-fit
     `zoom_curve.json` over the trustworthy domain [1.0, 5.0] and clamps
     correctly outside it.
  2. `CameraRandomizationCfg.focal_length_range` is centered on the measured
     1x mrcal calibration (`fx = 1053.04 ± 26.49 px`) within the expected
     ±3σ envelope, and `fov_scale_range` has been narrowed from the prior
     (0.5, 1.0) now that optical zoom is handled by the zoom curve.
  3. `delay_system_params.ego_detection_latency_mean/std` matches the
     measured mid-regime fit from phase7_mid_latency.csv.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="mas/029 measured-model defaults tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback
from datetime import datetime

import torch


VERBOSE = args_cli.test_verbose


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
        for line in error.split("\n")[:5]:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        print(f"    {error.splitlines()[-1] if error else ''}")

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY (mas/029 measured-model defaults)")
        print("=" * 80)
        print(f"Total Tests: {total}")
        if total > 0:
            print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
            print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
            print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)")
        if self.failed:
            print("\nFAILED TESTS:")
            for name, err in self.failed:
                print(f"  {name}: {err.splitlines()[0] if err else ''}")
        return len(self.failed) == 0 and len(self.errors) == 0


# Truth values from the measurement artifacts referenced by the ticket.
ZOOM_CURVE_LOOKUP = [
    # (cmd, z_eff) — selected entries from zoom_curve.json's lookup_table.
    (1.0, 1.0),
    (2.0, 1.198427),
    (3.0, 1.518043),
    (4.0, 2.032863),
    (5.0, 2.862127),
]

# 1x mrcal intrinsics — see datasets/camera_calibration/2026-04-17/1x/intrinsics_summary.json
MEASURED_FX_1X_MEAN = 1053.044591
MEASURED_FX_1X_STD = 26.49069557730026
MEASURED_IMAGE_WIDTH = 1920
MEASURED_IMAGE_HEIGHT = 1080

# Glass-to-topic detection latency = upstream A8/RTSP transit (~245 ms,
# mas/031 QR bench) + YOLO inference (mas/021 §C2 / phase7). std combines
# in quadrature: sqrt(σ_transit^2 + σ_yolo^2) ≈ sqrt(17^2 + 13^2) ≈ 21 ms
# for the mid YOLO regime currently shipped as the baseline.
MEASURED_DETECTION_LATENCY_E2E_MEAN_S = 0.31
MEASURED_DETECTION_LATENCY_E2E_STD_S = 0.021


def run_zoom_curve_tests(results: TestResults, device: torch.device):
    print("\n" + "=" * 80)
    print("Testing SIYI zoom curve (compute_z_eff)")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.controller.zoom_controller import (
        ZOOM_CURVE_A,
        ZOOM_CURVE_B,
        ZOOM_CURVE_CMD_MAX,
        compute_z_eff,
    )

    # Constants are the bench fit
    try:
        assert abs(ZOOM_CURVE_A - 0.32489) < 1e-6
        assert abs(ZOOM_CURVE_B - 0.4767) < 1e-6
        assert ZOOM_CURVE_CMD_MAX == 5.0
        results.add_pass("Curve constants match zoom_curve.json fit (a, b, cmd_max)")
    except Exception:
        results.add_fail("Curve constants match zoom_curve.json fit", traceback.format_exc())

    # Lookup-table parity over the trust region
    try:
        cmds = torch.tensor([cmd for cmd, _ in ZOOM_CURVE_LOOKUP], device=device)
        expected = torch.tensor([z for _, z in ZOOM_CURVE_LOOKUP], device=device)
        actual = compute_z_eff(cmds)
        if VERBOSE:
            for c, a, e in zip(cmds.tolist(), actual.tolist(), expected.tolist()):
                print(f"    cmd={c:.1f}  z_eff={a:.6f}  expected={e:.6f}")
        assert torch.allclose(actual, expected, atol=1e-3), \
            f"max diff = {(actual - expected).abs().max().item():.4e}"
        results.add_pass("Lookup-table parity for cmd in {1, 2, 3, 4, 5}")
    except Exception:
        results.add_fail("Lookup-table parity", traceback.format_exc())

    # Identity at cmd = 1
    try:
        z = compute_z_eff(torch.tensor([1.0], device=device))
        assert torch.allclose(z, torch.ones_like(z))
        results.add_pass("z_eff(1.0) == 1.0 (identity at base zoom)")
    except Exception:
        results.add_fail("z_eff(1.0) == 1.0", traceback.format_exc())

    # Clamping above ZOOM_CURVE_CMD_MAX
    try:
        z_at_max = compute_z_eff(torch.tensor([5.0], device=device))
        z_above = compute_z_eff(torch.tensor([6.0, 10.0], device=device))
        assert torch.allclose(z_above, z_at_max.expand_as(z_above), atol=1e-6)
        results.add_pass("Clamp above 5.0 (untrusted calibration region)")
    except Exception:
        results.add_fail("Clamp above 5.0", traceback.format_exc())

    # Clamping below 1.0
    try:
        z_at_one = compute_z_eff(torch.tensor([1.0], device=device))
        z_below = compute_z_eff(torch.tensor([0.5, 0.0, -1.0], device=device))
        assert torch.allclose(z_below, z_at_one.expand_as(z_below), atol=1e-6)
        results.add_pass("Clamp below 1.0 (no widening below 1x)")
    except Exception:
        results.add_fail("Clamp below 1.0", traceback.format_exc())

    # Strict monotonicity over the trust region
    try:
        cmd = torch.linspace(1.0, 5.0, 41, device=device)
        z = compute_z_eff(cmd)
        diffs = z[1:] - z[:-1]
        assert torch.all(diffs > 0), "z_eff must be strictly increasing in cmd"
        results.add_pass("Strict monotonicity over [1.0, 5.0]")
    except Exception:
        results.add_fail("Strict monotonicity", traceback.format_exc())

    # Sublinearity at the upper end (cmd=5 maps to ~2.86, not 5)
    try:
        z5 = compute_z_eff(torch.tensor([5.0], device=device)).item()
        assert 2.5 < z5 < 3.0, f"z_eff(5) = {z5:.3f} should be in (2.5, 3.0)"
        results.add_pass(f"Sublinear: z_eff(5) = {z5:.3f}, not 5.0")
    except Exception:
        results.add_fail("Sublinearity at cmd=5", traceback.format_exc())


def run_camera_dr_tests(results: TestResults):
    print("\n" + "=" * 80)
    print("Testing CameraRandomizationCfg defaults vs measured intrinsics")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.domain_randomization.domain_randomization_cfg import (
        CameraRandomizationCfg,
    )

    cfg = CameraRandomizationCfg()

    # focal_length_range covers the measured 1x mean
    try:
        lo, hi = cfg.focal_length_range
        assert lo <= MEASURED_FX_1X_MEAN <= hi, \
            f"measured 1x fx mean ({MEASURED_FX_1X_MEAN}) not in range ({lo}, {hi})"
        # Range should be at least 2σ wide (so DR has room to sample)
        assert (hi - lo) >= 2.0 * MEASURED_FX_1X_STD, \
            f"range width {hi-lo:.1f} < 2σ ({2*MEASURED_FX_1X_STD:.1f})"
        # And not absurdly wide (e.g. >10σ each side suggests range was not updated)
        center = 0.5 * (lo + hi)
        assert abs(center - MEASURED_FX_1X_MEAN) < 5.0 * MEASURED_FX_1X_STD, \
            f"range not centered on measured 1x: center={center:.1f}, measured={MEASURED_FX_1X_MEAN:.1f}"
        results.add_pass(f"focal_length_range {cfg.focal_length_range} centered on measured 1x")
    except Exception:
        results.add_fail("focal_length_range centered on measured 1x", traceback.format_exc())

    # fov_scale_range is narrower than the prior (0.5, 1.0)
    try:
        lo, hi = cfg.fov_scale_range
        assert hi == 1.0, f"upper bound should remain 1.0 (full FOV), got {hi}"
        assert lo > 0.5, \
            f"fov_scale_range lower bound should exceed 0.5 now that zoom is handled by z_eff (got {lo})"
        results.add_pass(f"fov_scale_range {cfg.fov_scale_range} narrowed (zoom no longer double-counted)")
    except Exception:
        results.add_fail("fov_scale_range narrowed", traceback.format_exc())

    # TiledCameraCfg resolution + (focal_length, horizontal_aperture) realize
    # the measured 1x fx within ±1 px. This is the regression that closes the
    # prior gap where focal_length_range was updated but the camera spawn
    # never repointed to the calibrated 1920×1080 / fx≈1053 setup.
    try:
        from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import IrisMA6TestEnvCfg

        cam = IrisMA6TestEnvCfg().camera
        assert cam.width == MEASURED_IMAGE_WIDTH, \
            f"camera.width = {cam.width}, expected {MEASURED_IMAGE_WIDTH} (mrcal 1x calibration)"
        assert cam.height == MEASURED_IMAGE_HEIGHT, \
            f"camera.height = {cam.height}, expected {MEASURED_IMAGE_HEIGHT} (mrcal 1x calibration)"
        fx_realized = (cam.spawn.focal_length / cam.spawn.horizontal_aperture) * cam.width
        assert abs(fx_realized - MEASURED_FX_1X_MEAN) < 1.0, (
            f"realized fx = {fx_realized:.4f} px deviates from measured 1x mean "
            f"{MEASURED_FX_1X_MEAN:.4f} px by > 1 px"
        )
        results.add_pass(
            f"TiledCameraCfg realizes fx = {fx_realized:.2f} px @ "
            f"{cam.width}×{cam.height} (target {MEASURED_FX_1X_MEAN:.2f})"
        )
    except Exception:
        results.add_fail("TiledCameraCfg realizes measured 1x fx", traceback.format_exc())


def run_latency_dr_tests(results: TestResults):
    print("\n" + "=" * 80)
    print("Testing detection latency defaults vs measured glass-to-topic E2E")
    print("=" * 80)

    from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import IrisMA6TestEnvCfg

    cfg = IrisMA6TestEnvCfg()
    params = cfg.delay_system_params

    try:
        assert abs(params.ego_detection_latency_mean - MEASURED_DETECTION_LATENCY_E2E_MEAN_S) < 1e-6, \
            f"ego_detection_latency_mean = {params.ego_detection_latency_mean}, " \
            f"expected {MEASURED_DETECTION_LATENCY_E2E_MEAN_S} (glass-to-topic E2E)"
        assert abs(params.ego_detection_latency_std - MEASURED_DETECTION_LATENCY_E2E_STD_S) < 1e-6, \
            f"ego_detection_latency_std = {params.ego_detection_latency_std}, " \
            f"expected {MEASURED_DETECTION_LATENCY_E2E_STD_S} (glass-to-topic E2E)"
        results.add_pass(
            f"ego_detection_latency = {params.ego_detection_latency_mean*1000:.0f} ± "
            f"{params.ego_detection_latency_std*1000:.0f} ms (glass-to-topic E2E)"
        )
    except Exception:
        results.add_fail("ego_detection_latency matches glass-to-topic E2E", traceback.format_exc())


def run_gimbal_pd_tests(results: TestResults):
    print("\n" + "=" * 80)
    print("Testing rate-loop τ + reverted joint PD (mas/035)")
    print("=" * 80)

    from isaaclab_assets import IRIS_GIMBAL3_CFG
    from isaaclab_tasks.direct.iris_ma6.controller.gimbal_rate_loop_cfg import GimbalRateLoopCfg

    # mas/035: rate loop owns the user-command dynamics; the joint PD
    # reverts to stiff position tracking.
    try:
        rl_cfg = GimbalRateLoopCfg()
        assert abs(rl_cfg.tau_yaw_s - 0.0995) < 1e-6, f"tau_yaw_s={rl_cfg.tau_yaw_s}"
        assert abs(rl_cfg.tau_pitch_s - 0.0954) < 1e-6, f"tau_pitch_s={rl_cfg.tau_pitch_s}"
        assert abs(rl_cfg.max_rate_per_axis - 1.28) < 1e-6, \
            f"max_rate_per_axis={rl_cfg.max_rate_per_axis}"
        results.add_pass(
            f"GimbalRateLoopCfg: τ_yaw={rl_cfg.tau_yaw_s}, τ_pitch={rl_cfg.tau_pitch_s}, "
            f"max_rate={rl_cfg.max_rate_per_axis} rad/s (rate_model.json)"
        )
    except Exception:
        results.add_fail("GimbalRateLoopCfg matches measured", traceback.format_exc())

    actuators = IRIS_GIMBAL3_CFG.actuators

    def _scalar_pd(value, name: str) -> float:
        # ImplicitActuatorCfg.stiffness/damping may be float or per-joint dict.
        # iris_gimbal3 uses scalar for the active gimbal axes; assert that
        # and unwrap.
        if isinstance(value, dict):
            vals = list(value.values())
            assert len(set(vals)) == 1, f"{name}: per-joint values differ: {value}"
            return float(vals[0])
        assert value is not None, f"{name}: stiffness/damping is None"
        return float(value)

    try:
        for axis_name in ("roll", "pitch", "yaw"):
            act = actuators[axis_name]
            stiff = _scalar_pd(act.stiffness, f"{axis_name}.stiffness")
            damp = _scalar_pd(act.damping, f"{axis_name}.damping")
            assert stiff == 2e3, f"{axis_name}: stiffness={stiff}, expected 2e3"
            assert damp == 1e2, f"{axis_name}: damping={damp}, expected 1e2"
        results.add_pass(
            "iris_gimbal3 joint PD reverted to stiff position tracking (k=2e3, c=1e2)"
        )
    except Exception:
        results.add_fail("iris_gimbal3 PD reverted", traceback.format_exc())


def main() -> int:
    print("=" * 80)
    print("MEASURED-MODEL DEFAULTS TEST SUITE (ticket mas/029)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    try:
        run_zoom_curve_tests(results, device)
    except Exception:
        results.add_error("zoom curve suite", traceback.format_exc())

    try:
        run_camera_dr_tests(results)
    except Exception:
        results.add_error("camera DR suite", traceback.format_exc())

    try:
        run_latency_dr_tests(results)
    except Exception:
        results.add_error("latency DR suite", traceback.format_exc())

    try:
        run_gimbal_pd_tests(results)
    except Exception:
        results.add_error("gimbal PD suite", traceback.format_exc())

    success = results.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
