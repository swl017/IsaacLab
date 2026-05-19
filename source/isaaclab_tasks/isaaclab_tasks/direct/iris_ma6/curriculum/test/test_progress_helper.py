#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for curriculum.progress_helper.

Run with:
    python source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/curriculum/test/test_progress_helper.py

No Isaac Sim / AppLauncher required — pure torch.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch


def _load_progress_helper():
    """Load progress_helper.py directly, same pattern as test_curriculum_cfg.py."""
    here = Path(__file__).resolve()
    mod_path = here.parent.parent / "progress_helper.py"
    spec = importlib.util.spec_from_file_location(
        "iris_ma6_progress_helper_under_test", mod_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["iris_ma6_progress_helper_under_test"] = module
    spec.loader.exec_module(module)
    return module.sample_per_env_progress


sample_per_env_progress = _load_progress_helper()


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  PASS  {name}")

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        print(f"  FAIL  {name}: {msg}")

    def summary(self) -> bool:
        print("-" * 60)
        print(f"Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        if self.failed:
            for n, m in self.failed:
                print(f"  - {n}: {m}")
        return len(self.failed) == 0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_support_includes_zero_at_p_one(results: TestResults):
    """At p=1, with enough samples we expect to see values close to 0."""
    try:
        out = sample_per_env_progress(1.0, num_envs=4096, num_agents=2, device=DEVICE)
        assert out.shape == (4096, 2), f"expected (4096, 2), got {tuple(out.shape)}"
        # With 8192 i.i.d. samples on [0, 1], P(min < 0.01) ≈ 1 - 0.99**8192 ≈ 1.0
        assert out.min().item() < 0.01, (
            f"expected min < 0.01 (anti-forgetting support at 0); got {out.min().item()}"
        )
        results.add_pass("support_includes_zero_at_p_one")
    except AssertionError as e:
        results.add_fail("support_includes_zero_at_p_one", str(e))


def test_support_does_not_exceed_global_progress(results: TestResults):
    """Output must be clamped to [0, p]."""
    for p in (0.1, 0.5, 0.9):
        try:
            out = sample_per_env_progress(p, num_envs=2048, num_agents=3, device=DEVICE)
            mx = out.max().item()
            mn = out.min().item()
            assert mx <= p + 1e-6, f"max {mx} exceeds p={p}"
            assert mn >= 0.0, f"min {mn} below 0"
            results.add_pass(f"support_does_not_exceed_global_progress (p={p})")
        except AssertionError as e:
            results.add_fail(f"support_does_not_exceed_global_progress (p={p})", str(e))


def test_p_zero_yields_zeros(results: TestResults):
    """At p=0, every sample must be exactly 0."""
    try:
        out = sample_per_env_progress(0.0, num_envs=64, num_agents=2, device=DEVICE)
        assert torch.equal(out, torch.zeros_like(out)), (
            f"non-zero entries at p=0: {out.nonzero().tolist()[:5]}"
        )
        results.add_pass("p_zero_yields_zeros")
    except AssertionError as e:
        results.add_fail("p_zero_yields_zeros", str(e))


def test_env_ids_subset_returns_subset_shape(results: TestResults):
    """When env_ids is passed, output rows must match env_ids.numel()."""
    try:
        env_ids = torch.arange(7, device=DEVICE)
        out = sample_per_env_progress(
            0.5, num_envs=1024, num_agents=2, device=DEVICE, env_ids=env_ids
        )
        assert out.shape == (7, 2), f"expected (7, 2), got {tuple(out.shape)}"
        results.add_pass("env_ids_subset_returns_subset_shape")
    except AssertionError as e:
        results.add_fail("env_ids_subset_returns_subset_shape", str(e))


def test_generator_reproducible(results: TestResults):
    """Two calls with the same seeded Generator must produce identical output."""
    try:
        g1 = torch.Generator(device=DEVICE)
        g1.manual_seed(42)
        g2 = torch.Generator(device=DEVICE)
        g2.manual_seed(42)
        a = sample_per_env_progress(0.7, 128, 2, DEVICE, generator=g1)
        b = sample_per_env_progress(0.7, 128, 2, DEVICE, generator=g2)
        assert torch.equal(a, b), "outputs differ under identical seeds"
        results.add_pass("generator_reproducible")
    except AssertionError as e:
        results.add_fail("generator_reproducible", str(e))


def test_anti_forgetting_invariant_across_axes(results: TestResults):
    """For every p in a sweep, the empirical support must contain the easy regime.

    Defined as: at least 1% of samples are below 0.1 * p (i.e. the lower
    decile of the curriculum is meaningfully populated). Property test —
    if this fails, the sampler is broken.
    """
    try:
        for p in (0.1, 0.3, 0.5, 0.7, 1.0):
            out = sample_per_env_progress(p, num_envs=4096, num_agents=2, device=DEVICE)
            below_decile_fraction = (out < 0.1 * p).float().mean().item()
            # For Uniform(0, p), P(x < 0.1 * p) = 0.1. With 8192 samples,
            # 0.1 ± 0.007 (1 std). 0.05 floor is very loose.
            assert below_decile_fraction > 0.05, (
                f"at p={p}, only {below_decile_fraction*100:.2f}% below 0.1*p; "
                "anti-forgetting invariant violated"
            )
        results.add_pass("anti_forgetting_invariant_across_axes")
    except AssertionError as e:
        results.add_fail("anti_forgetting_invariant_across_axes", str(e))


def main() -> int:
    print("=" * 60)
    print("PROGRESS_HELPER TEST SUITE")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    results = TestResults()
    for fn in [
        test_support_includes_zero_at_p_one,
        test_support_does_not_exceed_global_progress,
        test_p_zero_yields_zeros,
        test_env_ids_subset_returns_subset_shape,
        test_generator_reproducible,
        test_anti_forgetting_invariant_across_axes,
    ]:
        try:
            fn(results)
        except Exception:
            results.add_fail(fn.__name__, traceback.format_exc())
    ok = results.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
