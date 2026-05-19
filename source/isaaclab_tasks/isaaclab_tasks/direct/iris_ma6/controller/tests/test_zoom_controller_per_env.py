#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env API tests for ZoomController dead-time curriculum (ticket 034).

Same shape as test_gimbal_rate_loop_per_env.py: scalar/tensor equivalence,
non-uniform divergence, scalar back-compat sync.

No Isaac Sim required.
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


_cfg_mod = _load_module("zoom_controller_cfg.py", "iris_ma6_zoom_controller_cfg_under_test")
ZoomControllerCfg = _cfg_mod.ZoomControllerCfg

_zc_path = Path(__file__).resolve().parent.parent / "zoom_controller.py"
_zc_src = _zc_path.read_text().replace(
    "from .zoom_controller_cfg import ZoomControllerCfg",
    "from iris_ma6_zoom_controller_cfg_under_test import ZoomControllerCfg",
)
_zc_mod_name = "iris_ma6_zoom_controller_under_test"
_zc_mod = type(sys)(_zc_mod_name)
sys.modules[_zc_mod_name] = _zc_mod
exec(compile(_zc_src, str(_zc_path), "exec"), _zc_mod.__dict__)
ZoomController = _zc_mod.ZoomController


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


def _make_zc(num_envs: int = 16) -> "ZoomController":
    cfg = ZoomControllerCfg(model="siyi_a8")
    return ZoomController(cfg=cfg, num_envs=num_envs, device=DEVICE)


def test_scalar_uniform_tensor_equivalence(results: TestResults):
    """A scalar set + a uniform tensor must produce identical samples
    under the same seed."""
    try:
        a = _make_zc()
        b = _make_zc()
        torch.manual_seed(42)
        a.set_dead_time_curriculum_scale(0.7)
        a._sample_dead_time(env_ids=None)
        torch.manual_seed(42)
        b.set_dead_time_curriculum_scale(torch.full((16,), 0.7, device=DEVICE))
        b._sample_dead_time(env_ids=None)
        assert torch.equal(a._dead_time_seconds, b._dead_time_seconds), (
            f"max diff = {(a._dead_time_seconds - b._dead_time_seconds).abs().max().item()}"
        )
        results.add_pass("scalar_uniform_tensor_equivalence")
    except AssertionError as e:
        results.add_fail("scalar_uniform_tensor_equivalence", str(e))


def test_nonuniform_tensor_diverges(results: TestResults):
    """Half-zero, half-one scale → zero half deterministic 0, one half
    distributed around the configured mean."""
    try:
        zc = _make_zc(num_envs=4096)
        scale = torch.zeros(4096, device=DEVICE)
        scale[2048:] = 1.0
        zc.set_dead_time_curriculum_scale(scale)
        zc._sample_dead_time(env_ids=None)
        zeros = zc._dead_time_seconds[:2048]
        ones = zc._dead_time_seconds[2048:]
        assert torch.equal(zeros, torch.zeros_like(zeros)), (
            f"zero-scale half had non-zero; max = {zeros.max().item()}"
        )
        mean_emp = ones.mean().item()
        target = ZoomControllerCfg(model="siyi_a8").dead_time_mean_s
        assert abs(mean_emp - target) < 0.01, (
            f"one-scale empirical mean {mean_emp:.4f} far from target {target:.4f}"
        )
        results.add_pass("nonuniform_tensor_diverges")
    except AssertionError as e:
        results.add_fail("nonuniform_tensor_diverges", str(e))


def test_invalid_shape_raises(results: TestResults):
    """Tensor of wrong length must raise ValueError."""
    try:
        zc = _make_zc(num_envs=16)
        try:
            zc.set_dead_time_curriculum_scale(torch.zeros(7, device=DEVICE))
        except ValueError:
            results.add_pass("invalid_shape_raises")
            return
        results.add_fail("invalid_shape_raises", "no error raised")
    except Exception as e:
        results.add_fail("invalid_shape_raises", str(e))


def test_scalar_back_compat_kept_in_sync(results: TestResults):
    """Tensor input must update the scalar `_dead_time_curr_scale` to its mean."""
    try:
        zc = _make_zc(num_envs=8)
        scale = torch.linspace(0.0, 1.0, 8, device=DEVICE)
        zc.set_dead_time_curriculum_scale(scale)
        expected = scale.mean().item()
        assert abs(zc._dead_time_curr_scale - expected) < 1e-6, (
            f"got {zc._dead_time_curr_scale}, expected {expected}"
        )
        results.add_pass("scalar_back_compat_kept_in_sync")
    except AssertionError as e:
        results.add_fail("scalar_back_compat_kept_in_sync", str(e))


def test_non_siyi_a8_is_noop(results: TestResults):
    """When model != 'siyi_a8' the setter is a no-op regardless of input."""
    try:
        cfg = ZoomControllerCfg(model="first_order")
        zc = ZoomController(cfg=cfg, num_envs=16, device=DEVICE)
        # default per-env tensor is whatever __init__ wrote; setter should
        # leave it untouched.
        snapshot = zc._dead_time_curr_scale_per_env.clone()
        zc.set_dead_time_curriculum_scale(torch.full((16,), 0.9, device=DEVICE))
        assert torch.equal(zc._dead_time_curr_scale_per_env, snapshot), (
            "non-siyi_a8 setter modified per-env scale"
        )
        results.add_pass("non_siyi_a8_is_noop")
    except AssertionError as e:
        results.add_fail("non_siyi_a8_is_noop", str(e))


def main() -> int:
    print("=" * 60)
    print("ZOOM CONTROLLER PER-ENV TEST SUITE")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    results = TestResults()
    for fn in [
        test_scalar_uniform_tensor_equivalence,
        test_nonuniform_tensor_diverges,
        test_invalid_shape_raises,
        test_scalar_back_compat_kept_in_sync,
        test_non_siyi_a8_is_noop,
    ]:
        try:
            fn(results)
        except Exception:
            results.add_fail(fn.__name__, traceback.format_exc())
    ok = results.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
