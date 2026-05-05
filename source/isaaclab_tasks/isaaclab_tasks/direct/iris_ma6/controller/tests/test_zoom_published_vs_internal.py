#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locks in mas/037 design 2: published path quantizes, internal does not.

The integrator's internal state must remain continuous so sub-quantum
momentum is preserved. Quantizing the integrator state would corrupt slow
slewing and produce jitter that does not exist on hardware (mas/037 Tip 1).

Tests:
- Slow rate command produces a monotonically increasing continuous
  ``zoom_internal`` (no quantization on internal state).
- The published ``zoom`` snaps to 0.1-level boundaries.
- Across the staircase transitions, the published level moves 1.0 → 1.1
  → 1.2 monotonically, never rebinning back.
- ``set_zoom`` with a non-quantum value writes the continuous internal
  state exactly; published reads back the snapped quantum.

Run:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/controller/tests/test_zoom_published_vs_internal.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zoom published-vs-internal contract tests (mas/037)")
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


def _siyi_cfg(**overrides) -> ZoomControllerCfg:
    base = dict(
        model="siyi_a8",
        tau_zoom=1e-6,                # disable lag → integrator state visible directly
        max_zoom_rate=1.0,            # action_scale=1 simplifies the unit math
        v_max_levels_per_s=10.0,      # take v_max out of the picture
        quantum_levels=0.1,
        dead_time_mean_s=0.0,
        dead_time_std_s=0.0,
        dead_time_max_s=0.0,
        dead_time_curriculum_scale=0.0,
    )
    base.update(overrides)
    return ZoomControllerCfg(**base)


def test_slow_rate_internal_continuous(results: TestResults, device):
    """Slow rate cmd → zoom_internal is continuous; zoom is quantized."""
    try:
        cfg = _siyi_cfg()
        ctrl = ZoomController(cfg, num_envs=1, device=device)
        ctrl.reset()

        cmd = torch.tensor([0.05], device=device)  # +0.05 levels/s (very slow)
        dt = 0.04
        n_steps = 80  # 80 · 0.04 · 0.05 = 0.16 levels total → crosses 1.0→1.1→1.2

        internal_trace, published_trace = [], []
        for _ in range(n_steps):
            ctrl.compute_control(cmd, dt)
            internal_trace.append(ctrl.zoom_internal[0].item())
            published_trace.append(ctrl.zoom[0].item())

        # 1) Internal monotonic increase (continuous).
        for i in range(1, n_steps):
            assert internal_trace[i] >= internal_trace[i - 1] - 1e-7, (
                f"internal not monotonic at step {i}: "
                f"{internal_trace[i-1]:.6f} → {internal_trace[i]:.6f}"
            )
        # 2) Internal increment per step ~= cmd · dt = 0.002 (continuous proof).
        deltas = [internal_trace[i] - internal_trace[i - 1] for i in range(1, n_steps)]
        max_delta = max(deltas)
        assert max_delta < 0.005, (
            f"internal increment {max_delta:.5f} too large; not continuous"
        )
        # 3) Published values snap to 0.1: each must be in {1.0, 1.1, 1.2, ...}.
        for v in published_trace:
            ratio = round(v / 0.1)
            snapped = ratio * 0.1
            assert abs(v - snapped) < 1e-5, f"published value {v} not on 0.1 grid"
        # 4) Published is monotonic non-decreasing across the staircase.
        for i in range(1, n_steps):
            assert published_trace[i] >= published_trace[i - 1] - 1e-6, (
                f"published rebinned downward at step {i}: "
                f"{published_trace[i-1]:.2f} → {published_trace[i]:.2f}"
            )
        # 5) Published transitions 1.0 → 1.1 → 1.2 are observed in the trace.
        unique_levels = sorted(set(round(v, 2) for v in published_trace))
        assert 1.0 in unique_levels and 1.1 in unique_levels and 1.2 in unique_levels, (
            f"expected published trace to cover {{1.0, 1.1, 1.2}}, saw {unique_levels}"
        )
        results.add_pass(
            f"Slow ramp: internal continuous (max Δ={max_delta:.5f}), "
            f"published staircase {unique_levels}"
        )
    except Exception:
        results.add_fail("Slow rate internal continuous", traceback.format_exc())


def test_set_zoom_writes_continuous(results: TestResults, device):
    """set_zoom(non-quantum) writes the continuous state; published snaps."""
    try:
        cfg = _siyi_cfg()
        ctrl = ZoomController(cfg, num_envs=3, device=device)
        # Distinct non-quantum values per env.
        values = torch.tensor([1.234567, 2.789, 3.555], device=device)
        ctrl.set_zoom(values)
        # Internal: continuous, exact (within FP).
        assert torch.allclose(ctrl.zoom_internal, values, atol=1e-5), (
            f"internal should equal set value; got {ctrl.zoom_internal}"
        )
        # Published: snapped to 0.1.
        expected_pub = torch.tensor([1.2, 2.8, 3.6], device=device)
        assert torch.allclose(ctrl.zoom, expected_pub, atol=1e-5), (
            f"published should snap; got {ctrl.zoom}, expected {expected_pub}"
        )
        results.add_pass("set_zoom writes continuous internal; published snaps")
    except Exception:
        results.add_fail("set_zoom continuous", traceback.format_exc())


def test_first_order_mode_no_split(results: TestResults, device):
    """In first_order mode, zoom == zoom_internal at all times — no quantization."""
    try:
        cfg = ZoomControllerCfg(model="first_order", tau_zoom=0.05, max_zoom_rate=1.0)
        ctrl = ZoomController(cfg, num_envs=2, device=device)
        ctrl.reset()
        torch.manual_seed(0x1234)
        for _ in range(30):
            cmd = torch.empty(2, device=device).uniform_(-1.0, 1.0)
            ctrl.compute_control(cmd, dt=0.04)
            assert torch.allclose(ctrl.zoom, ctrl.zoom_internal, atol=1e-7), (
                f"first_order: zoom and zoom_internal should be identical; "
                f"got zoom={ctrl.zoom}, internal={ctrl.zoom_internal}"
            )
        results.add_pass("first_order: zoom == zoom_internal throughout rollout")
    except Exception:
        results.add_fail("first_order no split", traceback.format_exc())


def main():
    print("=" * 70)
    print("Zoom published-vs-internal contract (mas/037 design 2)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = TestResults()
    test_slow_rate_internal_continuous(results, device)
    test_set_zoom_writes_continuous(results, device)
    test_first_order_mode_no_split(results, device)

    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
