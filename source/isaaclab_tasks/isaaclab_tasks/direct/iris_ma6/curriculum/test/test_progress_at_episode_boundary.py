#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test the per-env step-at-episode-boundary semantics for reward-gating
curriculum signals (progress_coord, progress_safety).

This test does not require Isaac Sim. It loads CurriculumCfg directly and
emulates the env's _reset_idx write pattern on a synthetic per-env tensor,
exercising the same arithmetic the env runs.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch


def _load_curriculum_cfg():
    here = Path(__file__).resolve()
    cfg_path = here.parent.parent / "curriculum_cfg.py"
    mod_name = "iris_ma6_curriculum_cfg_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, cfg_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.CurriculumCfg


CurriculumCfg = _load_curriculum_cfg()


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
            print(f"\n{n}:\n{m}")
        return not self.failed


def _linear_progress(start: int, end: int, current: int) -> float:
    """Mirror env's _linear_progress so the test exercises the same arithmetic."""
    if end <= start:
        return 1.0
    x = (current - start) / (end - start)
    return 0.0 if x < 0 else (1.0 if x > 1 else float(x))


def _emulate_reset(progress_safety: torch.Tensor,
                   env_ids: torch.Tensor,
                   current_step: int,
                   cfg: CurriculumCfg) -> None:
    """Mirror the env's per-env progress_safety write in _reset_idx."""
    progress_safety[env_ids] = _linear_progress(
        cfg.safety_start_step, cfg.safety_end_step, current_step
    )


def _emulate_coord_reset(progress_coord: torch.Tensor,
                         env_ids: torch.Tensor,
                         current_step: int,
                         cfg: CurriculumCfg) -> None:
    """Mirror the env's per-env progress_coord step-at-boundary write."""
    coord_active = float(current_step >= cfg.coordination_start_step)
    progress_coord[env_ids] = coord_active


def test_progress_safety_per_env_distribution(results: TestResults):
    """Different envs that reset at different global steps within the
    safety ramp must capture different per-env progress values."""
    try:
        cfg = CurriculumCfg()
        N = 8
        device = torch.device("cpu")
        progress = torch.zeros(N, device=device)

        # Walk the safety ramp window; reset env i at step that lands at
        # progress = i / (N - 1) of the way through the ramp.
        ramp = cfg.safety_end_step - cfg.safety_start_step
        for i in range(N):
            step = cfg.safety_start_step + (ramp * i) // (N - 1)
            _emulate_reset(progress, torch.tensor([i]), step, cfg)

        # Boundary checks
        assert progress[0].item() == 0.0, f"first env: {progress[0]}"
        assert progress[-1].item() == 1.0, f"last env: {progress[-1]}"

        # Monotone non-decreasing
        deltas = progress[1:] - progress[:-1]
        assert (deltas >= -1e-9).all(), f"not monotone: {deltas}"

        # Spread (envs are at different points of the ramp)
        assert progress.std().item() > 0.1, (
            f"std too small ({progress.std().item():.3f}) — envs converged"
        )

        results.add_pass(
            "Per-env progress_safety captures different ramp values"
        )
    except Exception:
        results.add_fail(
            "Per-env progress_safety captures different ramp values",
            traceback.format_exc(),
        )


def test_progress_safety_held_within_episode(results: TestResults):
    """Once an env's progress_safety is set in _reset_idx, mid-episode
    advances of common_step_counter must NOT change it. Verified by:
    write at step T, advance step counter, never call _reset_idx for that
    env, and confirm the value is unchanged."""
    try:
        cfg = CurriculumCfg()
        device = torch.device("cpu")
        progress = torch.zeros(4, device=device)

        # Reset env 0 mid-ramp at step T_a
        T_a = cfg.safety_start_step + (cfg.safety_end_step - cfg.safety_start_step) // 4
        _emulate_reset(progress, torch.tensor([0]), T_a, cfg)
        v_at_a = progress[0].item()

        # Reset env 1 later at step T_b > T_a, but env 0 is still in its episode.
        T_b = cfg.safety_start_step + 3 * (cfg.safety_end_step - cfg.safety_start_step) // 4
        _emulate_reset(progress, torch.tensor([1]), T_b, cfg)

        # Reset env 2 after the ramp ends.
        T_c = cfg.safety_end_step + 1000
        _emulate_reset(progress, torch.tensor([2]), T_c, cfg)

        # Env 0 still holds its T_a value (no in-episode change)
        assert progress[0].item() == v_at_a, (
            f"env 0 drifted: was {v_at_a}, now {progress[0].item()}"
        )
        # Envs reflect their own reset time
        assert progress[1].item() > v_at_a, (
            f"env 1 (later reset) should have higher progress: "
            f"{progress[1].item()} vs {v_at_a}"
        )
        assert progress[2].item() == 1.0, f"env 2: {progress[2]}"
        # Env 3 never reset → still 0
        assert progress[3].item() == 0.0, f"env 3: {progress[3]}"

        results.add_pass(
            "progress_safety held constant within episode (no mid-episode drift)"
        )
    except Exception:
        results.add_fail(
            "progress_safety held constant within episode", traceback.format_exc()
        )


def test_progress_coord_step_at_episode_boundary(results: TestResults):
    """progress_coord must be 0 for envs that reset before
    coordination_start_step and 1 for envs that reset at/after."""
    try:
        cfg = CurriculumCfg()
        device = torch.device("cpu")
        progress = torch.zeros(3, device=device)

        # Env 0 resets just before coord activation
        _emulate_coord_reset(progress, torch.tensor([0]),
                             cfg.coordination_start_step - 1, cfg)
        # Env 1 resets exactly at coord_start
        _emulate_coord_reset(progress, torch.tensor([1]),
                             cfg.coordination_start_step, cfg)
        # Env 2 resets well after
        _emulate_coord_reset(progress, torch.tensor([2]),
                             cfg.coordination_start_step + 5000, cfg)

        assert progress[0].item() == 0.0, f"env 0: {progress[0]}"
        assert progress[1].item() == 1.0, f"env 1 (at boundary): {progress[1]}"
        assert progress[2].item() == 1.0, f"env 2: {progress[2]}"
        results.add_pass("progress_coord steps cleanly at coord_start_step")
    except Exception:
        results.add_fail(
            "progress_coord steps cleanly at coord_start_step",
            traceback.format_exc(),
        )


def test_progress_safety_pre_ramp_is_zero(results: TestResults):
    """Envs reset before safety_start_step must have progress_safety == 0."""
    try:
        cfg = CurriculumCfg()
        device = torch.device("cpu")
        progress = torch.zeros(4, device=device)
        _emulate_reset(progress, torch.arange(4),
                       cfg.safety_start_step - 5000, cfg)
        assert (progress == 0.0).all(), f"got {progress.tolist()}"
        results.add_pass("Pre-ramp progress_safety == 0 for all envs")
    except Exception:
        results.add_fail(
            "Pre-ramp progress_safety == 0 for all envs", traceback.format_exc()
        )


def main() -> int:
    print("=" * 60)
    print("PROGRESS-AT-EPISODE-BOUNDARY TESTS (Phase A)")
    print("=" * 60)
    results = TestResults()
    for fn in [
        test_progress_safety_per_env_distribution,
        test_progress_safety_held_within_episode,
        test_progress_coord_step_at_episode_boundary,
        test_progress_safety_pre_ramp_is_zero,
    ]:
        try:
            fn(results)
        except Exception:
            results.add_fail(fn.__name__, traceback.format_exc())
    ok = results.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
