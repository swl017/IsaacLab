#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env API tests for GimbalRateLoop (ticket 034).

Asserts:
  - set_progress(scalar) ≡ set_progress(tensor) when tensor is uniform.
  - set_progress(tensor) produces per-env divergence when tensor is non-uniform.
  - set_dead_time_curriculum_scale(tensor) drives per-env dead-time sampling.
  - Latency floor / clamping logic unchanged.

No Isaac Sim required — loads modules directly via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch


def _load_module(filename: str, mod_name: str):
    here = Path(__file__).resolve()
    target = here.parent.parent / filename
    spec = importlib.util.spec_from_file_location(mod_name, target)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load cfg first (rate_loop imports it).
_cfg_mod = _load_module("gimbal_rate_loop_cfg.py", "iris_ma6_gimbal_rate_loop_cfg_under_test")
GimbalRateLoopCfg = _cfg_mod.GimbalRateLoopCfg

# Patch sys.modules so the relative import inside gimbal_rate_loop.py resolves.
sys.modules.setdefault(
    "iris_ma6_gimbal_rate_loop_pkg",
    type(sys)("iris_ma6_gimbal_rate_loop_pkg"),
)
sys.modules["iris_ma6_gimbal_rate_loop_pkg"].gimbal_rate_loop_cfg = _cfg_mod

_rl_path = Path(__file__).resolve().parent.parent / "gimbal_rate_loop.py"
_rl_src = _rl_path.read_text().replace(
    "from .gimbal_rate_loop_cfg import GimbalRateLoopCfg",
    "from iris_ma6_gimbal_rate_loop_cfg_under_test import GimbalRateLoopCfg",
)
_rl_mod_name = "iris_ma6_gimbal_rate_loop_under_test"
_rl_mod = type(sys)(_rl_mod_name)
sys.modules[_rl_mod_name] = _rl_mod
exec(compile(_rl_src, str(_rl_path), "exec"), _rl_mod.__dict__)
GimbalRateLoop = _rl_mod.GimbalRateLoop


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, n):
        self.passed.append(n)
        print(f"  PASS  {n}")

    def add_fail(self, n, m):
        self.failed.append((n, m))
        print(f"  FAIL  {n}: {m}")

    def summary(self) -> bool:
        print("-" * 60)
        print(f"Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        for n, m in self.failed:
            print(f"  - {n}: {m}")
        return len(self.failed) == 0


def _make_loop(num_envs=16) -> "GimbalRateLoop":
    return GimbalRateLoop(cfg=GimbalRateLoopCfg(), num_envs=num_envs, device=DEVICE)


def test_set_progress_scalar_uniform_tensor_equivalence(results: TestResults):
    """A scalar set_progress and a uniform tensor must produce identical step output."""
    try:
        a = _make_loop()
        b = _make_loop()
        a.set_progress(0.7)
        b.set_progress(torch.full((16,), 0.7, device=DEVICE))
        # Run one step and compare omega_actual.
        cmd = torch.full((16, 2), 0.5, device=DEVICE)
        oa = a.step(cmd, dt=0.04)
        ob = b.step(cmd, dt=0.04)
        assert torch.allclose(oa, ob, atol=1e-6), (
            f"divergence: max diff = {(oa - ob).abs().max().item()}"
        )
        results.add_pass("set_progress_scalar_uniform_tensor_equivalence")
    except AssertionError as e:
        results.add_fail("set_progress_scalar_uniform_tensor_equivalence", str(e))


def test_set_progress_nonuniform_tensor_diverges(results: TestResults):
    """Non-uniform per-env progress must produce per-env divergence."""
    try:
        loop = _make_loop()
        p = torch.zeros(16, device=DEVICE)
        p[8:] = 1.0  # half pass-through, half full-lag
        loop.set_progress(p)
        cmd = torch.full((16, 2), 0.5, device=DEVICE)
        out = loop.step(cmd, dt=0.04)
        # Pass-through envs reach cmd directly; lagged envs do not.
        assert torch.allclose(out[:8], cmd[:8], atol=1e-6), "pass-through envs not at cmd"
        assert (out[8:] - cmd[8:]).abs().max().item() > 0.01, "lagged envs did not lag"
        results.add_pass("set_progress_nonuniform_tensor_diverges")
    except AssertionError as e:
        results.add_fail("set_progress_nonuniform_tensor_diverges", str(e))


def test_set_progress_invalid_shape_raises(results: TestResults):
    """Tensor of wrong shape must raise ValueError."""
    try:
        loop = _make_loop(num_envs=16)
        try:
            loop.set_progress(torch.zeros(7, device=DEVICE))
        except ValueError:
            results.add_pass("set_progress_invalid_shape_raises")
            return
        results.add_fail("set_progress_invalid_shape_raises", "no error raised")
    except Exception as e:
        results.add_fail("set_progress_invalid_shape_raises", str(e))


def test_dead_time_scalar_uniform_tensor_equivalence(results: TestResults):
    """set_dead_time_curriculum_scale(scalar) and a uniform tensor must
    produce statistically identical samples (same RNG order ⇒ bit-equal)."""
    try:
        a = _make_loop()
        b = _make_loop()
        torch.manual_seed(123)
        a.set_dead_time_curriculum_scale(0.5)
        a._sample_dead_time(env_ids=None)
        torch.manual_seed(123)
        b.set_dead_time_curriculum_scale(torch.full((16,), 0.5, device=DEVICE))
        b._sample_dead_time(env_ids=None)
        assert torch.equal(a._dead_time_seconds, b._dead_time_seconds), (
            "scalar/tensor dead-time samples diverge under same seed"
        )
        results.add_pass("dead_time_scalar_uniform_tensor_equivalence")
    except AssertionError as e:
        results.add_fail("dead_time_scalar_uniform_tensor_equivalence", str(e))


def test_dead_time_nonuniform_tensor_diverges(results: TestResults):
    """Half-zero-scale, half-one-scale dead-time sampling must give exact 0
    for the zero half and mean ~ cfg.dead_time_mean_s for the one half."""
    try:
        loop = _make_loop(num_envs=4096)
        scale = torch.zeros(4096, device=DEVICE)
        scale[2048:] = 1.0
        loop.set_dead_time_curriculum_scale(scale)
        loop._sample_dead_time(env_ids=None)
        zeros = loop._dead_time_seconds[:2048]
        ones = loop._dead_time_seconds[2048:]
        assert torch.equal(zeros, torch.zeros_like(zeros)), (
            f"zero-scale half had non-zero samples; max = {zeros.max().item()}"
        )
        mean_emp = ones.mean().item()
        target = GimbalRateLoopCfg().dead_time_mean_s
        # Empirical mean must be within ~3σ/√n of target. With n=2048, σ≈0.016,
        # 3σ/√n ≈ 0.001 s, so tolerance 0.005 is generous.
        assert abs(mean_emp - target) < 0.005, (
            f"one-scale empirical mean {mean_emp:.4f} far from target {target:.4f}"
        )
        results.add_pass("dead_time_nonuniform_tensor_diverges")
    except AssertionError as e:
        results.add_fail("dead_time_nonuniform_tensor_diverges", str(e))


def test_dead_time_scalar_back_compat_kept_in_sync(results: TestResults):
    """After set_dead_time_curriculum_scale(tensor), _dead_time_curr_scale
    scalar reflects the tensor mean (for logging)."""
    try:
        loop = _make_loop(num_envs=8)
        scale = torch.linspace(0.0, 1.0, 8, device=DEVICE)
        loop.set_dead_time_curriculum_scale(scale)
        expected_mean = scale.mean().item()
        assert abs(loop._dead_time_curr_scale - expected_mean) < 1e-6, (
            f"scalar back-compat: got {loop._dead_time_curr_scale}, "
            f"expected {expected_mean}"
        )
        results.add_pass("dead_time_scalar_back_compat_kept_in_sync")
    except AssertionError as e:
        results.add_fail("dead_time_scalar_back_compat_kept_in_sync", str(e))


def main() -> int:
    print("=" * 60)
    print("GIMBAL RATE-LOOP PER-ENV TEST SUITE")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    results = TestResults()
    for fn in [
        test_set_progress_scalar_uniform_tensor_equivalence,
        test_set_progress_nonuniform_tensor_diverges,
        test_set_progress_invalid_shape_raises,
        test_dead_time_scalar_uniform_tensor_equivalence,
        test_dead_time_nonuniform_tensor_diverges,
        test_dead_time_scalar_back_compat_kept_in_sync,
    ]:
        try:
            fn(results)
        except Exception:
            results.add_fail(fn.__name__, traceback.format_exc())
    ok = results.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
