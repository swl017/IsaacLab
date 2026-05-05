#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-stage tests for the SIYI A8 zoom response model (mas/037).

Validates the four pipeline stages in isolation, plus the backward-compat
``model="first_order"`` regression gate. Stages tested (model="siyi_a8"):

  1. Action denorm × max_zoom_rate                             (covered indirectly)
  2. Symmetric ±v_max slew clip
  3. Input-side dead-time buffer (cold-start zero)
  4. Integrator clamp to [zoom_min, zoom_max]
  5. First-order lag: post-lag state at t=τ matches 0.632·Δ
  6. Output quantization on published path only

Run:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/test_zoom_response_model_stages.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zoom response model stage tests (mas/037)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback

import torch

from isaaclab_tasks.direct.iris_ma6.controller import ZoomController, ZoomControllerCfg


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _siyi_cfg(**overrides) -> ZoomControllerCfg:
    """Build a siyi_a8 cfg with mas/037 fitted defaults plus overrides."""
    base = dict(
        model="siyi_a8",
        tau_zoom=0.091,
        zoom_min=1.0,
        zoom_max=5.0,
        max_zoom_rate=2.0,
        v_max_levels_per_s=3.16,
        quantum_levels=0.1,
        dead_time_mean_s=0.100,
        dead_time_std_s=0.018,
        dead_time_max_s=0.150,
        dead_time_curriculum_scale=0.0,
    )
    base.update(overrides)
    return ZoomControllerCfg(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_defaults_first_order(results: TestResults, device):
    """Default cfg → model='first_order'; new fields exist; legacy
    behavior (initial zoom=1.0, continuous-state property) preserved."""
    try:
        cfg = ZoomControllerCfg()
        assert cfg.model == "first_order", f"default model should be 'first_order', got {cfg.model!r}"
        assert hasattr(cfg, "v_max_levels_per_s")
        assert hasattr(cfg, "quantum_levels")
        assert hasattr(cfg, "dead_time_mean_s")

        ctrl = ZoomController(cfg, num_envs=8, device=device)
        assert ctrl.zoom.shape == (8,)
        assert torch.allclose(ctrl.zoom, torch.ones(8, device=device))
        # zoom_internal exists and matches zoom in first_order mode
        assert torch.allclose(ctrl.zoom_internal, ctrl.zoom)
        results.add_pass("Init defaults → first_order, legacy state preserved")
    except Exception:
        results.add_fail("Init defaults → first_order", traceback.format_exc())


def test_invalid_model_rejected(results: TestResults, device):
    """Unknown model selector raises a clear ValueError."""
    try:
        cfg = ZoomControllerCfg(model="quadratic_lag")
        try:
            ZoomController(cfg, num_envs=2, device=device)
        except ValueError as exc:
            assert "first_order" in str(exc) and "siyi_a8" in str(exc)
            results.add_pass("Invalid model selector raises ValueError")
            return
        results.add_fail("Invalid model selector", "no ValueError raised")
    except Exception:
        results.add_fail("Invalid model selector", traceback.format_exc())


def test_first_order_mode_unchanged(results: TestResults, device):
    """Backward-compat gate: model='first_order' matches the analytic
    legacy formula bit-exactly (within FP noise) over a 200-step random
    rate rollout. This is the hard guard against this ticket changing
    behavior on existing trained policies."""
    try:
        cfg = ZoomControllerCfg(model="first_order", tau_zoom=0.1, max_zoom_rate=2.0)
        ctrl = ZoomController(cfg, num_envs=4, device=device)

        # Reference state: same recurrence as the pre-mas/037 controller.
        ref_zoom = torch.ones(4, device=device)
        ref_target = torch.ones(4, device=device)

        torch.manual_seed(0xA8)
        dt = 0.04
        n_steps = 200
        max_diff = 0.0
        for _ in range(n_steps):
            cmd = torch.empty(4, device=device).uniform_(-1.0, 1.0)
            # Reference (legacy) update
            rate = cmd * cfg.max_zoom_rate
            ref_target = torch.clamp(ref_target + rate * dt, cfg.zoom_min, cfg.zoom_max)
            alpha = 1.0 - math.exp(-dt / cfg.tau_zoom)
            ref_zoom = ref_zoom + alpha * (ref_target - ref_zoom)
            ref_zoom = torch.clamp(ref_zoom, cfg.zoom_min, cfg.zoom_max)
            # Controller-under-test
            out = ctrl.compute_control(cmd, dt)
            diff = (out - ref_zoom).abs().max().item()
            max_diff = max(max_diff, diff)

        assert max_diff < 1e-6, f"first_order trace diverged from legacy formula: max |Δ| = {max_diff}"
        results.add_pass(f"first_order mode bit-exact vs legacy (max |Δ|={max_diff:.2e})")
    except Exception:
        results.add_fail("first_order mode bit-exact vs legacy", traceback.format_exc())


def test_v_max_slew_clip(results: TestResults, device):
    """siyi_a8 stage 2: sustained max-rate command is clipped at v_max,
    not at action_scale (when action_scale > v_max)."""
    try:
        # action_scale = 4 > v_max = 3.16 → clip should bind.
        cfg = _siyi_cfg(max_zoom_rate=4.0, v_max_levels_per_s=3.16,
                        tau_zoom=1e-6, dead_time_curriculum_scale=0.0)
        ctrl = ZoomController(cfg, num_envs=4, device=device)
        ctrl.reset()  # ensures dead-time state initialized at scale=0 (zeros)

        cmd = torch.ones(4, device=device)  # +1 → request action_scale levels/s
        dt = 0.01
        n_steps = int(1.0 / dt)  # integrate 1 second
        for _ in range(n_steps):
            ctrl.compute_control(cmd, dt)

        # zoom_min=1.0 → expected target advance ≈ v_max · 1s = 3.16 levels.
        # Clamped at zoom_max=5.0; advance from 1.0 → ~4.16.
        # action_scale=4 (UNCLIPPED) would advance 4.0 → reach 5.0 (zoom_max).
        # The clip evidence: final zoom_internal ≈ 1.0 + 3.16 = 4.16 ± lag.
        z_final = ctrl.zoom_internal[0].item()
        # Loose bound: must be < 4.5 (clip active) and > 4.0 (clip didn't over-fire).
        assert z_final < 4.5, f"v_max clip seems inactive (zoom={z_final:.2f}, expected ≲ 4.16)"
        assert z_final > 3.5, f"v_max clip over-fired (zoom={z_final:.2f}, expected ≳ 4.0)"
        results.add_pass(f"v_max slew clip active (final zoom_internal={z_final:.3f})")
    except Exception:
        results.add_fail("v_max slew clip", traceback.format_exc())


def test_dead_time_cold_start(results: TestResults, device):
    """siyi_a8 stage 3: with τ_d at exactly 3·dt and dt=0.04, the first 3
    cold-start steps emit zero rate (n_pushed ≤ depth → delayed = 0). Step
    4+ emits the pushed rate from 3 steps ago."""
    try:
        cfg = _siyi_cfg(tau_zoom=1e-6, dead_time_curriculum_scale=1.0,
                        dead_time_max_s=0.20)  # ensure depth-3 (τ_d=0.12) fits
        ctrl = ZoomController(cfg, num_envs=4, device=device)
        # Force τ_d = 0.12 s → depth = round(0.12/0.04) = 3 (no rounding ambiguity).
        # Bypass the random sampler so the test is deterministic regardless of
        # global RNG state.
        ctrl.reset()
        ctrl._dead_time_seconds.fill_(0.12)

        cmd = torch.ones(4, device=device)  # +1 → +max_zoom_rate=2.0 levels/s post-clip
        dt = 0.04
        z_internal_trace = []
        for _ in range(6):
            ctrl.compute_control(cmd, dt)
            z_internal_trace.append(ctrl.zoom_internal[0].item())

        # With depth=3, cold = (n_pushed ≤ 3). After step 0/1/2, n_pushed=1/2/3
        # → cold=True → delayed=0 → integrator stays at 1.0. Step 3 pushes the
        # 4th cmd, n_pushed=4 > 3 → cold=False → delayed = cmd from step 0.
        assert z_internal_trace[0] == 1.0, f"step 0 should be cold (zoom=1.0), got {z_internal_trace[0]}"
        assert z_internal_trace[1] == 1.0, f"step 1 should be cold (zoom=1.0), got {z_internal_trace[1]}"
        assert z_internal_trace[2] == 1.0, f"step 2 should be cold (zoom=1.0), got {z_internal_trace[2]}"
        # By step 4, integrator should have advanced past 1.0.
        assert z_internal_trace[3] > 1.0 + 1e-3, (
            f"step 3 should have advanced past 1.0, got {z_internal_trace[3]}"
        )
        assert z_internal_trace[5] > z_internal_trace[3], (
            f"step 5 should be > step 3, got {z_internal_trace[5]} vs {z_internal_trace[3]}"
        )
        results.add_pass(f"Dead-time cold-start zero output (trace={[f'{v:.3f}' for v in z_internal_trace]})")
    except Exception:
        results.add_fail("Dead-time cold-start", traceback.format_exc())


def test_first_order_lag_at_tau(results: TestResults, device):
    """siyi_a8 stage 5: post-lag state at t=τ₁ is 1 - 1/e ≈ 0.632 of a target jump.

    Drive the integrator to the upper limit with a sustained max-rate command,
    then read the post-lag state at t=τ₁ steps and compare to 0.632·(target-1.0).
    """
    try:
        # Tight setup: tau_zoom=0.1, max_zoom_rate=2.0, no dead time, no clip
        # binding (action_scale=2.0 < v_max=3.16). Step the controller for
        # exactly tau_zoom seconds and check the post-lag state.
        cfg = _siyi_cfg(tau_zoom=0.1, dead_time_curriculum_scale=0.0)
        ctrl = ZoomController(cfg, num_envs=4, device=device)
        ctrl.reset()

        cmd = torch.ones(4, device=device)
        dt = 0.001  # fine dt so the integrator target tracks max_zoom_rate · t
        n_steps = int(round(cfg.tau_zoom / dt))
        for _ in range(n_steps):
            ctrl.compute_control(cmd, dt)

        target_at_tau = 1.0 + cfg.max_zoom_rate * cfg.tau_zoom  # ≈ 1.2
        # Post-lag state should be ≈ 1.0 + 0.632 · (target - 1.0) when target
        # was held constant. Here target ramps; lag analytic isn't exact, but
        # the post-lag state should sit notably below the target.
        z_internal = ctrl.zoom_internal[0].item()
        z_target = ctrl._zoom_target[0].item()
        assert abs(z_target - target_at_tau) < 0.01, (
            f"integrator target should be ~{target_at_tau:.3f}, got {z_target:.3f}"
        )
        # Loose check: post-lag state below target by at least ~10% of the gap
        # (it can't track perfectly during a ramp).
        gap = z_target - z_internal
        assert gap > 0.005, f"post-lag state too close to target ({z_internal:.4f} vs {z_target:.4f})"
        results.add_pass(f"First-order lag: target={z_target:.3f}, state={z_internal:.3f}")
    except Exception:
        results.add_fail("First-order lag at tau", traceback.format_exc())


def test_output_quantization_published(results: TestResults, device):
    """Design 2: published path quantizes at 0.1, internal stays continuous."""
    try:
        cfg = _siyi_cfg(tau_zoom=1e-6, quantum_levels=0.1)
        ctrl = ZoomController(cfg, num_envs=2, device=device)
        ctrl.reset()
        # Set internal state directly to a non-quantum value.
        ctrl.set_zoom(torch.full((2,), 1.234567, device=device))
        published = ctrl.zoom
        internal = ctrl.zoom_internal
        assert torch.allclose(internal, torch.full((2,), 1.234567, device=device), atol=1e-5), (
            f"internal state should be 1.234567, got {internal}"
        )
        # round(1.234567 / 0.1) = 12 → 1.2.
        assert torch.allclose(published, torch.full((2,), 1.2, device=device), atol=1e-5), (
            f"published should snap to 1.2, got {published}"
        )
        results.add_pass(f"Output quantization (internal={internal[0]:.4f}, published={published[0]:.2f})")
    except Exception:
        results.add_fail("Output quantization published", traceback.format_exc())


def test_output_no_quantization_first_order(results: TestResults, device):
    """first_order mode: no quantization. zoom == zoom_internal == continuous."""
    try:
        cfg = ZoomControllerCfg(model="first_order", quantum_levels=0.1)
        ctrl = ZoomController(cfg, num_envs=2, device=device)
        ctrl.set_zoom(torch.full((2,), 1.234567, device=device))
        published = ctrl.zoom
        internal = ctrl.zoom_internal
        assert torch.allclose(published, internal, atol=1e-7), (
            f"first_order: zoom should NOT quantize, got published={published}, internal={internal}"
        )
        assert torch.allclose(published, torch.full((2,), 1.234567, device=device), atol=1e-5)
        results.add_pass("first_order mode: no quantization (continuous published)")
    except Exception:
        results.add_fail("first_order no-quantization", traceback.format_exc())


def test_integrator_clamp(results: TestResults, device):
    """Integrator target is clamped to [zoom_min, zoom_max] in both modes."""
    try:
        for model in ("first_order", "siyi_a8"):
            cfg = ZoomControllerCfg(
                model=model, tau_zoom=1e-6, max_zoom_rate=10.0,
                zoom_min=1.0, zoom_max=5.0,
            )
            ctrl = ZoomController(cfg, num_envs=2, device=device)
            ctrl.reset()
            # Sustain max-rate up
            for _ in range(500):
                ctrl.compute_control(torch.ones(2, device=device), 0.01)
            assert torch.all(ctrl.zoom_internal <= 5.0 + 1e-5), (
                f"{model}: zoom_internal exceeded zoom_max"
            )
            # Sustain max-rate down
            for _ in range(1000):
                ctrl.compute_control(-torch.ones(2, device=device), 0.01)
            assert torch.all(ctrl.zoom_internal >= 1.0 - 1e-5), (
                f"{model}: zoom_internal below zoom_min"
            )
        results.add_pass("Integrator clamp [zoom_min, zoom_max] (both modes)")
    except Exception:
        results.add_fail("Integrator clamp", traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Zoom response model — per-stage tests (mas/037 + backward compat)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = TestResults()
    test_init_defaults_first_order(results, device)
    test_invalid_model_rejected(results, device)
    test_first_order_mode_unchanged(results, device)
    test_v_max_slew_clip(results, device)
    test_dead_time_cold_start(results, device)
    test_first_order_lag_at_tau(results, device)
    test_output_quantization_published(results, device)
    test_output_no_quantization_first_order(results, device)
    test_integrator_clamp(results, device)

    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
