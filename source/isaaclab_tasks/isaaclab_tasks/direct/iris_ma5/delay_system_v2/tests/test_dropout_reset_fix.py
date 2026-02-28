#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for dropout reset fix (Bug 1): seed_dropout_held_data().

Background
----------
In hold-last-value dropout semantics, when a packet is dropped the pipeline
returns the previously-held value rather than the current input. This is the
physically correct model — a receiver's buffer always holds the last received
packet.

The bug: after ``pipeline.reset(env_ids)``, ``dropout_held_data[env_ids]``
is set to 0.0 (not None). When 100% dropout fires on the next ``process()``
call, it returns 0.0 instead of a valid initial state. This causes the "other
agent" to appear at position (0,0,0) in episode 2+, breaking triangulation.

The fix: ``DelaySystemV2.seed_dropout_held_data(env_ids)`` is called from
``_initialize_pipelines_from_gt_states(env_ids)`` after storing the initial
GT states to the DataBus. This seeds ``dropout_held_data[env_ids]`` with the
real initial GT snapshot so that hold-last-value correctly holds valid data
from the start of every episode.

Tests
-----
Group 1: DelayPipeline unit tests
  1. dropout_held_data is None at construction
  2. First process() call returns fresh data and initializes dropout_held_data
  3. 100% dropout freezes output at step-1 data
  4. reset() sets dropout_held_data to 0.0 (not None)
  5. Post-reset 100% dropout returns zeros [BUG DEMONSTRATION]

Group 2: DelaySystemV2.seed_dropout_held_data()
  6. seed_dropout_held_data() skips pipelines where dropout_held_data is None
  7. seed_dropout_held_data() updates dropout_held_data from DataBus contents
  8. Post-reset + seed: 100% dropout returns seeded GT data, not zeros [FIX VERIFICATION]

Group 3: MultiAgentDelaySystemV2 integration
  9. Episode-2 other-agent body position comes from initial_gt_states, not zeros
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test dropout reset fix (seed_dropout_held_data)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import torch

# Isaac Sim suppresses Python print() via its carb logging system.
# Use sys.stderr.write() for test output — stderr reaches the terminal.
# Also write to a results file in the test directory for record-keeping.
_RESULTS_FILE = Path(__file__).parent / "test_dropout_reset_fix_results.txt"


def _log(msg: str = ""):
    """Write to stderr (visible in terminal) and to a results file."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    with open(_RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


from isaaclab_tasks.direct.iris_ma5.delay_system_v2 import (
    AgentStates,
    DelayPipeline,
    DelaySystemV2,
    DelaySystemCfgV2,
    FieldDelayCfg,
    MultiAgentDelaySystemV2,
    MultiAgentDelaySystemV2Cfg,
)


# ---------------------------------------------------------------------------
# Test result tracker (same pattern as run_tests.py)
# ---------------------------------------------------------------------------

class TestResults:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        _log(f"  ✓ {name}")

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        _log(f"  ✗ {name}")
        for line in error.split("\n")[:6]:
            _log(f"    {line}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        _log("\n" + "=" * 80)
        _log("TEST SUMMARY")
        _log("=" * 80)
        _log(f"Total:  {total}")
        _log(f"Passed: {len(self.passed)}")
        _log(f"Failed: {len(self.failed)}")
        if self.failed:
            _log("\nFAILED TESTS:")
            for name, err in self.failed:
                _log(f"  {name}: {err.split(chr(10))[0]}")
        _log("=" * 80)
        return len(self.failed) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pipeline(device: torch.device, dropout_prob: float = 1.0, num_envs: int = 8, field_dim: int = 3) -> DelayPipeline:
    """Create a pipeline with only dropout enabled (no lag/staleness/latency)."""
    cfg = FieldDelayCfg(
        first_order_lag_enabled=False,
        staleness_enabled=False,
        latency_enabled=False,
        dropout_enabled=True,
        dropout_prob=dropout_prob,
    )
    return DelayPipeline(cfg=cfg, num_envs=num_envs, field_dim=field_dim, dt=0.01, device=device)


def make_delay_system(device: torch.device, dropout_prob: float = 1.0, num_envs: int = 8, field_dim: int = 3) -> DelaySystemV2:
    """Create a single-field DelaySystemV2 with only dropout enabled."""
    field_cfg = FieldDelayCfg(
        first_order_lag_enabled=False,
        staleness_enabled=False,
        latency_enabled=False,
        dropout_enabled=True,
        dropout_prob=dropout_prob,
    )
    cfg = DelaySystemCfgV2(
        dt=0.01,
        use_enhanced_mode=True,
        field_configs={"pos": field_cfg},
    )
    return DelaySystemV2(cfg=cfg, num_envs=num_envs, device=str(device), field_dims={"pos": field_dim})


def setup_ma_system(device: torch.device, num_envs: int = 16) -> MultiAgentDelaySystemV2:
    """Create a MultiAgentDelaySystemV2 with 100% combined comm+detection dropout."""
    cfg = MultiAgentDelaySystemV2Cfg(
        dt=0.01,
        enable_noise=False,
        # Combined dropout: 0.5 + 0.5 = 1.0 → 100% on _delay_noisy_other
        detection_dropout_rate=0.5,
        inter_agent_comm_dropout_rate=0.5,
        # Disable staleness/latency for simplicity
        detection_fps_mean=1000.0,
        inter_agent_comm_fps_mean=1000.0,
        detection_latency_mean=0.0,
        detection_latency_std=0.0,
        inter_agent_comm_latency_mean=0.0,
        inter_agent_comm_latency_std=0.0,
        motion_time_constant=0.0,
        orientation_time_constant=0.0,
        joint_time_constant=0.0,
    )
    possible_agents = ["drone_0", "drone_1"]
    system = MultiAgentDelaySystemV2(
        cfg=cfg,
        possible_agents=possible_agents,
        num_envs=num_envs,
        num_joints_per_agent={a: 3 for a in possible_agents},
        num_targets_per_agent={a: 1 for a in possible_agents},
        device=device,
    )
    # Camera config required for derived field computation
    offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
    offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    for agent_id in possible_agents:
        system.set_camera_configs(
            agent_id,
            width=640, height=480,
            focal_length=24.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.0,
            offset_position_b=offset_pos,
            offset_rotation_b=offset_rot,
        )
    return system


def step_ma_system(system: MultiAgentDelaySystemV2, num_envs: int, device: torch.device,
                    position_value: float = 5.0):
    """Run one step of the MA system with fixed positions."""
    possible_agents = system._possible_agents
    system.update_time()
    for agent_id in possible_agents:
        system.update_gt_states(
            agent_id=agent_id,
            body_position_w=torch.full((num_envs, 3), position_value, device=device),
            body_orientation_w=torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4),
            body_linear_velocity_w=torch.zeros(num_envs, 3, device=device),
            body_angular_velocity_w=torch.zeros(num_envs, 3, device=device),
            body_linear_acceleration_w=torch.zeros(num_envs, 3, device=device),
            body_combined_angular_velocity_w=torch.zeros(num_envs, 3, device=device),
            joint_positions_b=torch.zeros(num_envs, 3, device=device),
            zoom_level=torch.ones(num_envs, device=device),
        )
        system.update_detections(
            agent_id=agent_id,
            bboxes_2d_gt=torch.tensor([[[320., 240., 50., 50.]]], device=device).expand(num_envs, 1, 4),
        )


# ---------------------------------------------------------------------------
# Group 1: DelayPipeline unit tests
# ---------------------------------------------------------------------------

def run_pipeline_lifecycle_tests(results: TestResults, device: torch.device):
    _log("\n" + "=" * 80)
    _log("Group 1: DelayPipeline dropout lifecycle")
    _log("=" * 80)

    N, D = 8, 3
    t = torch.zeros(N, device=device)
    real_data = torch.full((N, D), 5.0, device=device)
    data2 = torch.full((N, D), 99.0, device=device)

    # Test 1: dropout_held_data is None at construction
    try:
        pipeline = make_pipeline(device)
        assert pipeline.dropout_held_data is None, "Expected None at construction"
        results.add_pass("dropout_held_data is None at construction")
    except Exception:
        results.add_fail("dropout_held_data is None at construction", traceback.format_exc())

    # Test 2: First process() returns fresh data and initializes dropout_held_data
    try:
        pipeline = make_pipeline(device)
        result = pipeline.process(real_data, t)
        assert torch.allclose(result, real_data), f"Expected real_data (5.0), got {result[0]}"
        assert pipeline.dropout_held_data is not None, "dropout_held_data should be initialized"
        assert torch.allclose(pipeline.dropout_held_data, real_data), "dropout_held_data should equal real_data"
        results.add_pass("First process() returns fresh data, initializes dropout_held_data")
    except Exception:
        results.add_fail("First process() returns fresh data, initializes dropout_held_data", traceback.format_exc())

    # Test 3: 100% dropout freezes at step-1 data
    try:
        pipeline = make_pipeline(device)
        pipeline.process(real_data, t)          # step 1 — initializes to real_data
        result2 = pipeline.process(data2, t)    # step 2 — 100% dropout → should return real_data
        assert torch.allclose(result2, real_data), \
            f"100% dropout should return step-1 data (5.0), got {result2[0]}"
        assert torch.allclose(pipeline.dropout_held_data, real_data), \
            "dropout_held_data should remain at step-1 value after 100% dropout"
        results.add_pass("100% dropout freezes output at step-1 data")
    except Exception:
        results.add_fail("100% dropout freezes output at step-1 data", traceback.format_exc())

    # Test 4: reset() sets dropout_held_data to 0.0 (not None)
    try:
        pipeline = make_pipeline(device)
        pipeline.process(real_data, t)                          # initialize
        env_ids = torch.arange(N, device=device)
        pipeline.reset(env_ids)
        assert pipeline.dropout_held_data is not None, "dropout_held_data must not become None after reset"
        assert torch.allclose(pipeline.dropout_held_data, torch.zeros(N, D, device=device)), \
            f"Expected 0.0 after reset, got {pipeline.dropout_held_data[0]}"
        results.add_pass("reset() sets dropout_held_data to 0.0 (not None)")
    except Exception:
        results.add_fail("reset() sets dropout_held_data to 0.0 (not None)", traceback.format_exc())

    # Test 5: Post-reset 100% dropout returns zeros [BUG DEMONSTRATION]
    # This verifies the bug exists: without seed_dropout_held_data,
    # the first process() call after reset returns 0.0 at 100% dropout.
    try:
        pipeline = make_pipeline(device)
        pipeline.process(real_data, t)           # episode 1: initialize dropout_held_data = real_data
        env_ids = torch.arange(N, device=device)
        pipeline.reset(env_ids)                  # reset: dropout_held_data = 0.0
        result3 = pipeline.process(data2, t)     # episode 2, no seed — returns 0.0 (BUG)
        assert torch.allclose(result3, torch.zeros(N, D, device=device)), \
            f"Bug: expected 0.0 from unreseeded pipeline, got {result3[0]}"
        results.add_pass("Post-reset 100% dropout returns zeros [bug confirmed]")
    except Exception:
        results.add_fail("Post-reset 100% dropout returns zeros [bug confirmed]", traceback.format_exc())


# ---------------------------------------------------------------------------
# Group 2: DelaySystemV2.seed_dropout_held_data()
# ---------------------------------------------------------------------------

def run_seed_held_data_tests(results: TestResults, device: torch.device):
    _log("\n" + "=" * 80)
    _log("Group 2: DelaySystemV2.seed_dropout_held_data()")
    _log("=" * 80)

    N, D = 8, 3

    # Test 6: seed_dropout_held_data() skips pipelines where dropout_held_data is None
    try:
        ds = make_delay_system(device)
        env_ids = torch.arange(N, device=device)
        # Never called process() — dropout_held_data is None
        ds.seed_dropout_held_data(env_ids)      # should not raise, should be a no-op
        pipeline = ds._noisy_pipelines["pos"]
        assert pipeline.dropout_held_data is None, \
            "seed_dropout_held_data should leave None pipeline untouched"
        results.add_pass("seed_dropout_held_data() skips None pipelines (no-op)")
    except Exception:
        results.add_fail("seed_dropout_held_data() skips None pipelines (no-op)", traceback.format_exc())

    # Test 7: seed_dropout_held_data() updates dropout_held_data from DataBus
    try:
        ds = make_delay_system(device)
        env_ids = torch.arange(N, device=device)
        gt_data = torch.full((N, D), 7.0, device=device)

        # Episode 1: initialize dropout_held_data via first process() call
        ds.store("pos", torch.full((N, D), 1.0, device=device))
        ds.get_delayed_noisy("pos")              # initializes dropout_held_data = 1.0

        # Reset: dropout_held_data = 0.0
        ds.reset(env_ids)

        # Store GT data to DataBus (simulating _initialize_pipelines_from_gt_states)
        ds.store("pos", gt_data)

        # Seed: should copy DataBus contents to dropout_held_data
        ds.seed_dropout_held_data(env_ids)

        pipeline = ds._noisy_pipelines["pos"]
        assert pipeline.dropout_held_data is not None
        # No noise in this config, so get_noisy == gt_data
        assert torch.allclose(pipeline.dropout_held_data, gt_data), \
            f"Expected 7.0 after seed, got {pipeline.dropout_held_data[0]}"
        results.add_pass("seed_dropout_held_data() updates dropout_held_data from DataBus")
    except Exception:
        results.add_fail("seed_dropout_held_data() updates dropout_held_data from DataBus", traceback.format_exc())

    # Test 8: Post-reset + seed: 100% dropout returns seeded GT data [FIX VERIFICATION]
    try:
        ds = make_delay_system(device)
        env_ids = torch.arange(N, device=device)
        gt_data = torch.full((N, D), 7.0, device=device)
        new_data = torch.full((N, D), 99.0, device=device)

        # Episode 1: initialize dropout_held_data
        ds.store("pos", torch.full((N, D), 1.0, device=device))
        ds.get_delayed_noisy("pos")

        # Reset + re-store GT (as _initialize_pipelines_from_gt_states does)
        ds.reset(env_ids)
        ds.store("pos", gt_data)

        # Fix: seed dropout_held_data with GT data
        ds.seed_dropout_held_data(env_ids)

        # Episode 2, step 1: 100% dropout → should return GT data, not zeros
        ds.store("pos", new_data)
        result = ds.get_delayed_noisy("pos")
        assert torch.allclose(result, gt_data), \
            f"After seed, 100% dropout should return GT data (7.0), got {result[0]}"
        results.add_pass("Post-reset + seed: 100% dropout returns GT data, not zeros [fix verified]")
    except Exception:
        results.add_fail("Post-reset + seed: 100% dropout returns GT data, not zeros [fix verified]", traceback.format_exc())

    # Test 8b: Partial reset — seed only affects specified env_ids, others unchanged
    try:
        ds = make_delay_system(device, num_envs=16)
        reset_ids = torch.arange(8, device=device)     # only first 8

        initial_data = torch.full((16, D), 1.0, device=device)
        gt_data_partial = torch.full((16, D), 7.0, device=device)

        # Episode 1: initialize all envs
        ds.store("pos", initial_data)
        ds.get_delayed_noisy("pos")

        # Reset only first 8 envs
        ds.reset(reset_ids)

        # Store GT for all envs, then seed only the reset envs
        ds.store("pos", gt_data_partial)
        ds.seed_dropout_held_data(reset_ids)

        pipeline = ds._noisy_pipelines["pos"]
        # Reset envs should have GT data (7.0)
        assert torch.allclose(pipeline.dropout_held_data[:8], gt_data_partial[:8]), \
            f"Reset envs should have GT data 7.0, got {pipeline.dropout_held_data[0]}"
        # Non-reset envs should still have step-1 data (1.0)
        assert torch.allclose(pipeline.dropout_held_data[8:], initial_data[8:]), \
            f"Non-reset envs should retain 1.0, got {pipeline.dropout_held_data[8]}"
        results.add_pass("seed_dropout_held_data() only affects specified env_ids")
    except Exception:
        results.add_fail("seed_dropout_held_data() only affects specified env_ids", traceback.format_exc())


# ---------------------------------------------------------------------------
# Group 3: MultiAgentDelaySystemV2 integration
# ---------------------------------------------------------------------------

def run_multi_episode_integration_test(results: TestResults, device: torch.device):
    _log("\n" + "=" * 80)
    _log("Group 3: MultiAgentDelaySystemV2 multi-episode integration")
    _log("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1"]

    # Test 9: Episode-2 other-agent body position comes from initial_gt_states, not zeros
    try:
        system = setup_ma_system(device, num_envs)

        # Enable 100% combined dropout (0.5 detection + 0.5 comm = 1.0 on _delay_noisy_other)
        system.set_dropout_enabled(True, progress=1.0)
        system.set_delay_mode("random", progress=1.0)

        # ---- Episode 1: run a few steps at position 5.0 ----
        for _ in range(5):
            step_ma_system(system, num_envs, device, position_value=5.0)

        # Verify episode 1: other agent appears at step-1 frozen position (5.0)
        all_states = system.get_all_states_for_observations("drone_0")
        other_pos_ep1 = all_states["drone_1"].data.body_position_w
        # Step-1 data should be non-zero (5.0)
        assert not torch.allclose(other_pos_ep1, torch.zeros(num_envs, 3, device=device)), \
            "Episode 1: other agent position should be non-zero (frozen at step-1 = 5.0)"

        # ---- Reset all envs with real spawn positions (initial_gt_states) ----
        env_ids = torch.arange(num_envs, device=device)
        spawn_pos = torch.full((num_envs, 3), 3.0, device=device)
        spawn_quat = torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4)

        # Build initial_gt_states mimicking iris_ma_env5._build_initial_gt_states_for_reset()
        initial_gt_states = {}
        for agent_id in possible_agents:
            state = AgentStates(num_envs=num_envs, num_joints=3, num_targets=1, device=device)
            state.data.body_position_w[:] = spawn_pos
            state.data.body_orientation_w[:] = spawn_quat
            initial_gt_states[agent_id] = state

        system.reset(env_ids=env_ids, initial_gt_states=initial_gt_states)

        # ---- Episode 2, step 1: update with new position (8.0) ----
        step_ma_system(system, num_envs, device, position_value=8.0)

        # With fix: other agent position should NOT be all-zeros.
        # With 100% dropout, it should be frozen at the seeded initial state (spawn_pos = 3.0),
        # not at zeros.
        all_states_ep2 = system.get_all_states_for_observations("drone_0")
        other_pos_ep2 = all_states_ep2["drone_1"].data.body_position_w

        assert not torch.allclose(other_pos_ep2, torch.zeros(num_envs, 3, device=device)), \
            f"Episode 2: other agent should NOT be zeros (fix failed). Got: {other_pos_ep2[0]}"

        results.add_pass("Episode 2 other-agent position is non-zero after reset (fix verified)")

    except Exception:
        results.add_fail("Episode 2 other-agent position is non-zero after reset (fix verified)", traceback.format_exc())

    # Test 10: Without the fix (manual regression test using DelayPipeline directly)
    # Verify that the old behavior (reset → zeros → process returns zeros) is what
    # seed_dropout_held_data() specifically prevents.
    try:
        N, D = 8, 3
        t = torch.zeros(N, device=device)
        pipeline = make_pipeline(device, dropout_prob=1.0, num_envs=N, field_dim=D)

        real_data = torch.full((N, D), 5.0, device=device)
        gt_data = torch.full((N, D), 3.0, device=device)
        new_data = torch.full((N, D), 8.0, device=device)
        env_ids = torch.arange(N, device=device)

        # Episode 1
        pipeline.process(real_data, t)

        # Reset WITHOUT seed → dropout_held_data = 0.0
        pipeline.reset(env_ids)
        result_no_fix = pipeline.process(new_data, t)
        assert torch.allclose(result_no_fix, torch.zeros(N, D, device=device)), \
            "Without fix: should return zeros"

        # Now replay with seed_held_data
        pipeline2 = make_pipeline(device, dropout_prob=1.0, num_envs=N, field_dim=D)
        pipeline2.process(real_data, t)
        pipeline2.reset(env_ids)
        # Manually seed (simulating seed_dropout_held_data)
        pipeline2.dropout_held_data[env_ids] = gt_data[env_ids]
        result_with_fix = pipeline2.process(new_data, t)
        assert torch.allclose(result_with_fix, gt_data), \
            f"With fix: should return GT data (3.0), got {result_with_fix[0]}"

        results.add_pass("Regression: seed prevents zeros, holds GT data across episode reset")
    except Exception:
        results.add_fail("Regression: seed prevents zeros, holds GT data across episode reset", traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Clear results file from previous runs
    _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_FILE.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"\nDevice: {device}")

    results = TestResults()

    run_pipeline_lifecycle_tests(results, device)
    run_seed_held_data_tests(results, device)
    run_multi_episode_integration_test(results, device)

    success = results.print_summary()
    return success


if __name__ == "__main__":
    success = main()
    simulation_app.close()
    exit(0 if success else 1)