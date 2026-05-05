#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-config tests for CurriculumCfg. No Isaac Sim required.

Run with:
    python -m isaaclab_tasks.direct.iris_ma6.curriculum.test.test_curriculum_cfg
or
    python source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/curriculum/test/test_curriculum_cfg.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def _load_curriculum_cfg():
    """Load curriculum_cfg.py directly from disk.

    Going through ``isaaclab_tasks.direct.iris_ma6.curriculum.curriculum_cfg``
    triggers ``isaaclab_tasks/__init__.py`` -> ``isaaclab.envs`` -> ``isaacsim``
    which in turn requires AppLauncher to be initialized. Since this test is
    purely a config-shape check, we sidestep the package import.

    The module is registered in ``sys.modules`` BEFORE exec because
    ``@configclass`` -> ``dataclass`` looks up ``cls.__module__`` to validate
    forward refs.
    """
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


def test_phase_ordering(results: TestResults):
    """Dynamics ramp must finish before observation-corruption phases begin.

    The whole point of the repositioning is that the policy adapts to
    randomized dynamics under clean observations BEFORE noise/delay/dropout
    are layered on top.
    """
    cfg = CurriculumCfg()

    # Basic flight skills before dynamics
    try:
        assert cfg.agent_velocity_end_step <= cfg.dynamics_start_step, (
            f"agent_velocity must end before dynamics starts: "
            f"agent_velocity_end={cfg.agent_velocity_end_step}, "
            f"dynamics_start={cfg.dynamics_start_step}"
        )
        results.add_pass("agent_velocity_end <= dynamics_start")
    except AssertionError as e:
        results.add_fail("agent_velocity_end <= dynamics_start", str(e))

    try:
        assert cfg.safety_end_step <= cfg.dynamics_start_step, (
            f"safety must end before dynamics starts: "
            f"safety_end={cfg.safety_end_step}, "
            f"dynamics_start={cfg.dynamics_start_step}"
        )
        results.add_pass("safety_end <= dynamics_start")
    except AssertionError as e:
        results.add_fail("safety_end <= dynamics_start", str(e))

    # Dynamics finishes before observation corruption
    try:
        assert cfg.dynamics_end_step <= cfg.noise_start_step, (
            f"dynamics must end at/before noise starts: "
            f"dynamics_end={cfg.dynamics_end_step}, "
            f"noise_start={cfg.noise_start_step}"
        )
        results.add_pass("dynamics_end <= noise_start")
    except AssertionError as e:
        results.add_fail("dynamics_end <= noise_start", str(e))


def test_well_formed_phase_windows(results: TestResults):
    """Every (start, end) pair must have start <= end."""
    cfg = CurriculumCfg()
    pairs = [
        ("tracking", cfg.tracking_start_step, cfg.tracking_end_step),
        ("agent_velocity", cfg.agent_velocity_start_step, cfg.agent_velocity_end_step),
        ("noise", cfg.noise_start_step, cfg.noise_end_step),
        ("fixed_delay", cfg.fixed_delay_start_step, cfg.fixed_delay_end_step),
        ("random_delay", cfg.random_delay_start_step, cfg.random_delay_end_step),
        ("dropout", cfg.dropout_start_step, cfg.dropout_end_step),
        ("burst_dropout", cfg.burst_dropout_start_step, cfg.burst_dropout_end_step),
        ("coordination", cfg.coordination_start_step, cfg.coordination_end_step),
        ("safety", cfg.safety_start_step, cfg.safety_end_step),
        ("moving_target", cfg.moving_target_start_step, cfg.moving_target_end_step),
        ("dynamics", cfg.dynamics_start_step, cfg.dynamics_end_step),
        ("gimbal_dead_time", cfg.gimbal_dead_time_start_step, cfg.gimbal_dead_time_end_step),
    ]
    for name, s, e in pairs:
        try:
            assert s <= e, f"{name}: start={s} > end={e}"
            results.add_pass(f"phase {name} well-formed (start <= end)")
        except AssertionError as ex:
            results.add_fail(f"phase {name} well-formed", str(ex))


def test_coordination_is_step_at_boundary(results: TestResults):
    """coordination_start_step must equal coordination_end_step under
    step-at-episode-boundary semantics. The env code reads
    coordination_start_step only; coordination_end_step is retained for
    backward-compat but must not introduce a ramp window."""
    cfg = CurriculumCfg()
    try:
        assert cfg.coordination_start_step == cfg.coordination_end_step, (
            f"coord must be a step: start={cfg.coordination_start_step}, "
            f"end={cfg.coordination_end_step}"
        )
        results.add_pass("coordination is step (start == end)")
    except AssertionError as ex:
        results.add_fail("coordination is step (start == end)", str(ex))

    # Step is post-bootstrap (after the first ~20k where pair_valid plateaus)
    # and pre-noise so observability corruption doesn't onset before the
    # triangulation gradient signal exists.
    try:
        assert cfg.coordination_start_step >= 20_000, (
            f"coord step at {cfg.coordination_start_step} is mid-bootstrap"
        )
        assert cfg.coordination_start_step <= cfg.noise_start_step, (
            f"coord step at {cfg.coordination_start_step} starts after noise "
            f"({cfg.noise_start_step}) — triangulation should activate before "
            f"observation corruption"
        )
        results.add_pass(
            f"coord step {cfg.coordination_start_step} between bootstrap and noise"
        )
    except AssertionError as ex:
        results.add_fail("coord step between bootstrap and noise", str(ex))


def test_dynamics_progress_monotone(results: TestResults):
    """get_dynamics_progress must be 0 before start, 1 after end, and
    monotonically non-decreasing in between."""
    cfg = CurriculumCfg()

    try:
        assert cfg.get_dynamics_progress(cfg.dynamics_start_step - 1) == 0.0
        assert cfg.get_dynamics_progress(cfg.dynamics_start_step) == 0.0
        assert cfg.get_dynamics_progress(cfg.dynamics_end_step) == 1.0
        assert cfg.get_dynamics_progress(cfg.dynamics_end_step + 1) == 1.0
        results.add_pass("dynamics progress: 0 before start, 1 after end")
    except AssertionError as ex:
        results.add_fail("dynamics progress endpoints", str(ex))

    # Mid-ramp: progress should be in (0, 1) and monotone
    try:
        prev = -1.0
        ok = True
        for step in range(
            cfg.dynamics_start_step,
            cfg.dynamics_end_step + 1,
            max(1, (cfg.dynamics_end_step - cfg.dynamics_start_step) // 8),
        ):
            p = cfg.get_dynamics_progress(step)
            if p < prev - 1e-9:
                ok = False
                break
            prev = p
        assert ok, "progress not monotone"
        results.add_pass("dynamics progress monotone non-decreasing")
    except AssertionError as ex:
        results.add_fail("dynamics progress monotone", str(ex))


def main() -> int:
    print("=" * 60)
    print("CURRICULUM CFG TEST SUITE")
    print("=" * 60)
    results = TestResults()
    for fn in [
        test_phase_ordering,
        test_well_formed_phase_windows,
        test_coordination_is_step_at_boundary,
        test_dynamics_progress_monotone,
    ]:
        try:
            fn(results)
        except Exception:
            results.add_fail(fn.__name__, traceback.format_exc())
    ok = results.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
