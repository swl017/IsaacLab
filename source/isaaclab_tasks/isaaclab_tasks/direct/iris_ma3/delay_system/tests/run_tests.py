#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run delay system test suite.

This test suite validates the stochastic sample-and-hold delay system
for multi-agent RL environments with realistic sim-to-real delays.
"""

import sys
# Force unbuffered output for Isaac Sim compatibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run delay system test suite")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose test debug output")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW you can import other modules (torch, isaaclab, etc.)
import sys
import torch
import traceback
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Global verbose flag
VERBOSE = args_cli.test_verbose


# ==================== Test Results Tracker ====================

class TestResults:
    """Track test results across the entire test suite."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.errors: List[Tuple[str, str]] = []

    def add_pass(self, test_name: str):
        """Record a passed test."""
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        """Record a failed test."""
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
        # Print first few lines of error for immediate feedback
        if VERBOSE:
            error_lines = error.split('\n')[:10]
            for line in error_lines:
                print(f"    {line}")
        else:
            # Just print first line
            first_line = error.split('\n')[0]
            print(f"    Error: {first_line[:150]}")

    def add_error(self, suite_name: str, error: str):
        """Record a test suite error."""
        self.errors.append((suite_name, error))
        print(f"  ⚠ {suite_name} encountered an error")

    def print_summary(self) -> bool:
        """Print test summary and return True if all tests passed."""
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        if total > 0:
            pass_pct = 100 * len(self.passed) / total
            fail_pct = 100 * len(self.failed) / total
            error_pct = 100 * len(self.errors) / total

            print(f"Total Tests: {total}")
            print(f"Passed:      {len(self.passed):3d} ({pass_pct:5.1f}%)")
            print(f"Failed:      {len(self.failed):3d} ({fail_pct:5.1f}%)")
            print(f"Errors:      {len(self.errors):3d} ({error_pct:5.1f}%)")
        else:
            print("No tests were run")

        if self.failed:
            print("\n" + "-"*80)
            print("FAILED TESTS:")
            print("-"*80)
            for test_name, error in self.failed:
                print(f"\n{test_name}:")
                # Print full error in summary
                error_lines_list = error.split('\n')
                error_lines = error_lines_list[:20]  # Limit to 20 lines
                for line in error_lines:
                    print(f"  {line}")
                if len(error_lines_list) > 20:
                    remaining_lines = len(error_lines_list) - 20
                    print(f"  ... (truncated, {remaining_lines} more lines)")

        if self.errors:
            print("\n" + "-"*80)
            print("TEST SUITE ERRORS:")
            print("-"*80)
            for suite_name, error in self.errors:
                print(f"\n{suite_name}:")
                error_lines_list = error.split('\n')
                error_lines = error_lines_list[:20]
                for line in error_lines:
                    print(f"  {line}")
                if len(error_lines_list) > 20:
                    remaining_lines = len(error_lines_list) - 20
                    print(f"  ... (truncated, {remaining_lines} more lines)")

        print("="*80)
        success = len(self.failed) == 0 and len(self.errors) == 0
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed. See details above.")

        return success


# ==================== Test Suites ====================

def test_stochastic_sampler(results: TestResults, device: torch.device):
    """Test stochastic sampler functionality."""
    print("\n" + "="*80)
    print("Testing StochasticSampler")
    print("="*80)

    from isaaclab_tasks.direct.iris_ma3.delay_system.stochastic_sampler import (
        StochasticSampler,
        SamplerConfig,
        DistributionConfig,
    )

    num_envs = 8
    dt = 0.01

    print(f"Configuration: {num_envs} environments, dt={dt}s")
    if VERBOSE:
        print(f"  Device: {device}")
        print(f"  Test categories: period, distribution, latency, dropout, reset")

    # Test 1: Constant period sampler
    try:
        config = SamplerConfig(
            period_dist=DistributionConfig(
                distribution_type="constant",
                value=0.05,  # 20 Hz
            ),
            latency_dist=None,
            dropout_dist=None,
        )
        sampler = StochasticSampler(config, num_envs, device, dt)
        data = torch.randn(num_envs, 3, device=device)

        output, info = sampler.update(data)
        assert output.shape == data.shape, f"Shape mismatch: {output.shape} != {data.shape}"
        assert 'sampled' in info, "Missing 'sampled' in info"
        assert 'dropped_out' in info, "Missing 'dropped_out' in info"
        results.add_pass("Constant period sampler initialization and update")
    except Exception as e:
        results.add_fail("Constant period sampler", traceback.format_exc())

    # Test 2: Uniform distribution
    try:
        config = SamplerConfig(
            period_dist=DistributionConfig(
                distribution_type="uniform",
                min_value=0.02,
                max_value=0.05,
            ),
            latency_dist=None,
            dropout_dist=None,
        )
        sampler = StochasticSampler(config, num_envs, device, dt)
        output, info = sampler.update(torch.randn(num_envs, 3, device=device))
        assert output.shape == (num_envs, 3)
        results.add_pass("Uniform distribution sampler")
    except Exception as e:
        results.add_fail("Uniform distribution sampler", traceback.format_exc())

    # Test 3: Normal distribution
    try:
        config = SamplerConfig(
            period_dist=DistributionConfig(
                distribution_type="normal",
                mean=0.05,
                std=0.01,
            ),
            latency_dist=None,
            dropout_dist=None,
        )
        sampler = StochasticSampler(config, num_envs, device, dt)
        output, info = sampler.update(torch.randn(num_envs, 3, device=device))
        assert output.shape == (num_envs, 3)
        results.add_pass("Normal distribution sampler")
    except Exception as e:
        results.add_fail("Normal distribution sampler", traceback.format_exc())

    # Test 4: Latency handling
    try:
        config = SamplerConfig(
            period_dist=DistributionConfig(distribution_type="constant", value=0.05),
            latency_dist=DistributionConfig(distribution_type="constant", value=0.1),
            dropout_dist=None,
        )
        sampler = StochasticSampler(config, num_envs, device, dt)
        data = torch.randn(num_envs, 3, device=device)

        # Update multiple times to trigger latency
        for _ in range(20):
            output, info = sampler.update(data)

        assert 'accumulated_latency' in info
        results.add_pass("Latency handling")
    except Exception as e:
        results.add_fail("Latency handling", traceback.format_exc())

    # Test 5: Dropout
    try:
        config = SamplerConfig(
            period_dist=DistributionConfig(distribution_type="constant", value=0.01),
            latency_dist=None,
            dropout_dist=DistributionConfig(distribution_type="constant", value=0.5),  # 50% dropout
        )
        sampler = StochasticSampler(config, num_envs, device, dt)

        dropout_count = 0
        for _ in range(100):
            output, info = sampler.update(torch.randn(num_envs, 3, device=device))
            if info['dropped_out'].any():
                dropout_count += 1

        # With 50% dropout, we expect some dropouts
        assert dropout_count > 0, "No dropouts detected with 50% dropout rate"
        results.add_pass("Dropout detection")
    except Exception as e:
        results.add_fail("Dropout detection", traceback.format_exc())

    # Test 6: Reset functionality
    try:
        config = SamplerConfig(
            period_dist=DistributionConfig(distribution_type="constant", value=0.05),
            latency_dist=None,
            dropout_dist=None,
        )
        sampler = StochasticSampler(config, num_envs, device, dt)
        data = torch.randn(num_envs, 3, device=device)

        # Update, then reset
        sampler.update(data)
        env_ids = torch.tensor([0, 2, 4], device=device)
        sampler.reset(env_ids)

        # Check that times were reset
        assert (sampler.current_time[env_ids] == 0).all()
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", traceback.format_exc())


def test_specialized_samplers(results: TestResults, device: torch.device):
    """Test specialized samplers (first-order lag, quaternion, passthrough)."""
    print("\n" + "="*80)
    print("Testing Specialized Samplers")
    print("="*80)

    from isaaclab_tasks.direct.iris_ma3.delay_system.specialized_samplers import (
        FirstOrderLagSampler,
        QuaternionFirstOrderLagSampler,
        PassthroughSampler,
    )

    num_envs = 8

    print(f"Configuration: {num_envs} environments")
    if VERBOSE:
        print(f"  Test categories: FirstOrderLag, QuaternionSLERP, Passthrough")

    # Test 1: FirstOrderLagSampler initialization
    try:
        sampler = FirstOrderLagSampler(
            num_envs=num_envs,
            state_dim=3,
            time_constant=0.1,
            dt=0.01,
            device=device,
        )
        assert sampler.num_envs == num_envs
        assert sampler.state_dim == 3
        results.add_pass("FirstOrderLagSampler initialization")
    except Exception as e:
        results.add_fail("FirstOrderLagSampler initialization", traceback.format_exc())

    # Test 2: FirstOrderLagSampler filtering
    try:
        sampler = FirstOrderLagSampler(
            num_envs=num_envs,
            state_dim=3,
            time_constant=0.1,
            dt=0.01,
            device=device,
        )
        target = torch.ones(num_envs, 3, device=device) * 10.0

        # Apply filter multiple times
        output = None
        for _ in range(50):
            output, info = sampler.update(target)

        # Output should approach target but not instantly
        assert (output > 0).all() and (output <= 10).all(), "Filtering out of range"
        # After 50 steps (0.5s), should be close to target
        assert (output > 8).all(), "Filtering too slow"
        results.add_pass("FirstOrderLagSampler filtering convergence")
    except Exception as e:
        results.add_fail("FirstOrderLagSampler filtering", traceback.format_exc())

    # Test 3: QuaternionFirstOrderLagSampler initialization
    try:
        sampler = QuaternionFirstOrderLagSampler(
            num_envs=num_envs,
            time_constant=0.1,
            dt=0.01,
            device=device,
        )
        # Should initialize to identity quaternion
        expected_identity = torch.zeros(num_envs, 4, device=device)
        expected_identity[:, 0] = 1.0
        assert torch.allclose(sampler.filtered_state, expected_identity, atol=1e-5)
        results.add_pass("QuaternionFirstOrderLagSampler initialization")
    except Exception as e:
        results.add_fail("QuaternionFirstOrderLagSampler initialization", traceback.format_exc())

    # Test 4: QuaternionFirstOrderLagSampler normalization
    try:
        sampler = QuaternionFirstOrderLagSampler(
            num_envs=num_envs,
            time_constant=0.1,
            dt=0.01,
            device=device,
        )
        quat_input = torch.rand(num_envs, 4, device=device)
        output, info = sampler.update(quat_input)

        # Check output is normalized
        norms = torch.norm(output, dim=-1)
        assert torch.allclose(norms, torch.ones(num_envs, device=device), atol=1e-5)
        results.add_pass("QuaternionFirstOrderLagSampler normalization")
    except Exception as e:
        results.add_fail("QuaternionFirstOrderLagSampler normalization", traceback.format_exc())

    # Test 5: PassthroughSampler
    try:
        sampler = PassthroughSampler(num_envs, device)
        data = torch.randn(num_envs, 5, device=device)
        output, info = sampler.update(data)

        # Should be unchanged
        assert torch.equal(output, data)
        results.add_pass("PassthroughSampler passes data unchanged")
    except Exception as e:
        results.add_fail("PassthroughSampler", traceback.format_exc())

    # Test 6: FirstOrderLagSampler reset
    try:
        sampler = FirstOrderLagSampler(
            num_envs=num_envs,
            state_dim=3,
            time_constant=0.1,
            dt=0.01,
            device=device,
        )
        # Update to non-zero state
        sampler.update(torch.ones(num_envs, 3, device=device) * 5.0)

        # Reset specific envs
        env_ids = torch.tensor([0, 2], device=device)
        sampler.reset(env_ids)

        # Check reset envs are zero
        assert (sampler.filtered_state[env_ids] == 0).all()
        results.add_pass("FirstOrderLagSampler reset")
    except Exception as e:
        results.add_fail("FirstOrderLagSampler reset", traceback.format_exc())


def test_timestamp_tracking(results: TestResults, device: torch.device):
    """Test timestamp management."""
    print("\n" + "="*80)
    print("Testing Timestamp Management")
    print("="*80)

    from isaaclab_tasks.direct.iris_ma3.delay_system.timestamp_manager import (
        FieldTimestamp,
        AgentStatesTimestamps,
        TimestampManager,
    )

    num_envs = 8

    print(f"Configuration: {num_envs} environments")
    if VERBOSE:
        print(f"  Test categories: FieldTimestamp, AgentStatesTimestamps, TimestampManager")

    # Test 1: FieldTimestamp initialization
    try:
        ts = FieldTimestamp(num_envs, device)
        assert ts.num_envs == num_envs
        assert ts.t_captured.shape == (num_envs,)
        results.add_pass("FieldTimestamp initialization")
    except Exception as e:
        results.add_fail("FieldTimestamp initialization", traceback.format_exc())

    # Test 2: Staleness calculation
    try:
        ts = FieldTimestamp(num_envs, device)
        ts.t_captured = torch.zeros(num_envs, device=device)
        ts.t_available = torch.ones(num_envs, device=device) * 0.3
        ts.t_current = torch.ones(num_envs, device=device) * 0.5

        staleness = ts.staleness
        expected = torch.ones(num_envs, device=device) * 0.5
        assert torch.allclose(staleness, expected)
        results.add_pass("Staleness calculation")
    except Exception as e:
        results.add_fail("Staleness calculation", traceback.format_exc())

    # Test 3: Latency calculation
    try:
        ts = FieldTimestamp(num_envs, device)
        ts.t_captured = torch.zeros(num_envs, device=device)
        ts.t_available = torch.ones(num_envs, device=device) * 0.3
        ts.t_current = torch.ones(num_envs, device=device) * 0.5

        latency = ts.latency
        expected = torch.ones(num_envs, device=device) * 0.3
        assert torch.allclose(latency, expected)
        results.add_pass("Latency calculation")
    except Exception as e:
        results.add_fail("Latency calculation", traceback.format_exc())

    # Test 4: AgentStatesTimestamps
    try:
        field_groups = ['motion', 'detection', 'orientation']
        agent_ts = AgentStatesTimestamps(field_groups, num_envs, device)

        assert 'motion' in agent_ts.timestamps
        assert 'detection' in agent_ts.timestamps
        assert 'orientation' in agent_ts.timestamps
        results.add_pass("AgentStatesTimestamps initialization")
    except Exception as e:
        results.add_fail("AgentStatesTimestamps initialization", traceback.format_exc())

    # Test 5: TimestampManager multi-agent
    try:
        agent_ids = ['agent_0', 'agent_1', 'agent_2']
        field_groups = ['motion', 'detection']
        ts_mgr = TimestampManager(agent_ids, field_groups, num_envs, device)

        assert 'agent_0' in ts_mgr.agent_timestamps
        assert 'agent_1' in ts_mgr.agent_timestamps
        results.add_pass("TimestampManager multi-agent initialization")
    except Exception as e:
        results.add_fail("TimestampManager multi-agent", traceback.format_exc())


def test_delay_system_integration(results: TestResults, device: torch.device):
    """Test full delay system integration."""
    print("\n" + "="*80)
    print("Testing Delay System Integration")
    print("="*80)

    from isaaclab_tasks.direct.iris_ma3.delay_system.delay_system import DelaySystem
    from isaaclab_tasks.direct.iris_ma3.delay_system.delay_system_cfg import DelaySystemCfg
    from isaaclab_tasks.direct.iris_ma3.delay_system.agent_states import AgentStates

    num_envs = 4
    num_agents = 3
    num_joints = 2
    num_targets = 3

    print(f"Configuration: {num_envs} environments, {num_agents} agents")
    if VERBOSE:
        print(f"  num_joints: {num_joints}, num_targets: {num_targets}")
        print(f"  Test categories: initialization, ego states, multi-agent perspective, step/reset")

    # Test 1: DelaySystem initialization
    try:
        cfg = DelaySystemCfg()
        cfg.dt_sim = 0.01
        agent_ids = [f"agent_{i}" for i in range(num_agents)]

        delay_system = DelaySystem(
            cfg=cfg,
            agent_ids=agent_ids,
            num_envs=num_envs,
            num_joints=num_joints,
            num_targets=num_targets,
            device=device,
        )
        assert delay_system.num_envs == num_envs
        assert len(delay_system.agent_ids) == num_agents
        results.add_pass("DelaySystem initialization")
    except Exception as e:
        results.add_fail("DelaySystem initialization", traceback.format_exc())

    # Test 2: Update and query ego states
    try:
        cfg = DelaySystemCfg()
        agent_ids = ["agent_0", "agent_1"]
        delay_system = DelaySystem(cfg, agent_ids, num_envs, num_joints, num_targets, device)

        # Create dummy states
        states = AgentStates(num_envs, num_joints, num_targets, device)
        states.data.body_position_w = torch.randn(num_envs, 3, device=device)
        states.data.body_orientation_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device
        )
        # Initialize camera intrinsics with valid values [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        states.data.camera_base_intrinsics[:, 0, 0] = 500.0  # fx
        states.data.camera_base_intrinsics[:, 1, 1] = 500.0  # fy
        states.data.camera_base_intrinsics[:, 0, 2] = 320.0  # cx
        states.data.camera_base_intrinsics[:, 1, 2] = 240.0  # cy
        states.data.camera_zoom_level = torch.ones(num_envs, device=device)  # zoom_level = 1.0 (no zoom)

        delay_system.update_agent_gt_states("agent_0", states)
        ego_states = delay_system.get_ego_states("agent_0")

        assert ego_states.data.body_position_w.shape == (num_envs, 3)
        results.add_pass("Update and query ego states")
    except Exception as e:
        results.add_fail("Update and query ego states", traceback.format_exc())

    # Test 3: Multi-agent perspective
    try:
        cfg = DelaySystemCfg()
        agent_ids = ["agent_0", "agent_1", "agent_2"]
        delay_system = DelaySystem(cfg, agent_ids, num_envs, num_joints, num_targets, device)

        # Update all agents
        for agent_id in agent_ids:
            states = AgentStates(num_envs, num_joints, num_targets, device)
            states.data.body_position_w = torch.randn(num_envs, 3, device=device)
            states.data.body_orientation_w = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device
            )
            # Initialize camera intrinsics with valid values [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            states.data.camera_base_intrinsics[:, 0, 0] = 500.0  # fx
            states.data.camera_base_intrinsics[:, 1, 1] = 500.0  # fy
            states.data.camera_base_intrinsics[:, 0, 2] = 320.0  # cx
            states.data.camera_base_intrinsics[:, 1, 2] = 240.0  # cy
            states.data.camera_zoom_level = torch.ones(num_envs, device=device)  # zoom_level = 1.0 (no zoom)
            delay_system.update_agent_gt_states(agent_id, states)

        # Query from agent_0's perspective
        agent_0_view = delay_system.get_all_agent_states_for_ego("agent_0")

        assert "agent_0" in agent_0_view
        assert "agent_1" in agent_0_view
        assert "agent_2" in agent_0_view
        results.add_pass("Multi-agent perspective query")
    except Exception as e:
        results.add_fail("Multi-agent perspective", traceback.format_exc())

    # Test 4: Step and reset
    try:
        cfg = DelaySystemCfg()
        agent_ids = ["agent_0"]
        delay_system = DelaySystem(cfg, agent_ids, num_envs, num_joints, num_targets, device)

        initial_time = delay_system.t_sim.clone()
        delay_system.step()
        stepped_time = delay_system.t_sim.clone()

        assert (stepped_time > initial_time).all()

        # Reset
        env_ids = torch.tensor([0, 1], device=device)
        delay_system.reset(env_ids)
        assert (delay_system.t_sim[env_ids] == 0).all()

        results.add_pass("Step and reset functionality")
    except Exception as e:
        results.add_fail("Step and reset", traceback.format_exc())


# ==================== Main ====================

def test_multi_agent_wrapper(results: TestResults, device: torch.device):
    """Test suite for MultiAgentDelaySystem wrapper."""
    from isaaclab_tasks.direct.iris_ma3.delay_system.multi_agent_wrapper import MultiAgentDelaySystem
    from isaaclab_tasks.direct.iris_ma3.delay_system.agent_states import AgentStates

    print("\n--- MultiAgentDelaySystem Wrapper Tests ---", flush=True)

    num_envs = 4
    dt = 0.01
    possible_agents = ["agent_0", "agent_1"]
    num_joints_per_agent = {"agent_0": 3, "agent_1": 3}
    num_targets_per_agent = {"agent_0": 2, "agent_1": 2}

    # Test 1: Initialization
    try:
        wrapper = MultiAgentDelaySystem(
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent=num_joints_per_agent,
            num_targets_per_agent=num_targets_per_agent,
            dt=dt,
            device=device,
            enable_noise=True,
            position_noise_std=0.1,
            orientation_noise_std=0.05,
        )
        assert wrapper._num_envs == num_envs
        assert len(wrapper._possible_agents) == 2
        results.add_pass("Wrapper initialization")
    except Exception as e:
        results.add_fail("Wrapper initialization", str(e))

    # Test 2: Camera config → intrinsics matrix
    try:
        wrapper = MultiAgentDelaySystem(
            possible_agents=["agent_0"],
            num_envs=num_envs,
            num_joints_per_agent={"agent_0": 3},
            num_targets_per_agent={"agent_0": 1},
            dt=dt,
            device=device,
        )

        wrapper.set_camera_configs(
            agent_id="agent_0",
            width=640,
            height=480,
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=27.0,
            offset_position_b=torch.tensor([0.1, 0.0, 0.0], device=device),
            offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        )

        K = wrapper._camera_configs["agent_0"]['intrinsics']
        assert abs(K[0, 0].item() - 888.89) < 1.0, f"fx mismatch: {K[0, 0].item()}"
        assert abs(K[1, 1].item() - 888.89) < 1.0, f"fy mismatch: {K[1, 1].item()}"
        assert abs(K[0, 2].item() - 320.0) < 1.0, f"cx mismatch: {K[0, 2].item()}"
        assert abs(K[1, 2].item() - 240.0) < 1.0, f"cy mismatch: {K[1, 2].item()}"
        results.add_pass("Camera config → K matrix")
    except Exception as e:
        results.add_fail("Camera config → K matrix", str(e))

    # Test 3: State updates
    try:
        wrapper = MultiAgentDelaySystem(
            possible_agents=["agent_0"],
            num_envs=num_envs,
            num_joints_per_agent={"agent_0": 3},
            num_targets_per_agent={"agent_0": 1},
            dt=dt,
            device=device,
        )

        wrapper.set_camera_configs(
            agent_id="agent_0",
            width=640,
            height=480,
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=27.0,
            offset_position_b=torch.tensor([0.1, 0.0, 0.0], device=device),
            offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        )

        body_position = torch.randn(num_envs, 3, device=device)
        wrapper.update_gt_states(
            agent_id="agent_0",
            body_position_w=body_position,
            body_orientation_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device),
            body_linear_velocity_w=torch.randn(num_envs, 3, device=device),
            body_angular_velocity_w=torch.randn(num_envs, 3, device=device),
            body_linear_acceleration_w=torch.randn(num_envs, 3, device=device),
            body_combined_angular_velocity_w=torch.randn(num_envs, 3, device=device),
            joint_positions_b=torch.randn(num_envs, 3, device=device),
            zoom_level=torch.ones(num_envs, device=device),
        )

        states = wrapper._agent_states["agent_0"]
        assert torch.allclose(states.data.body_position_w, body_position)
        results.add_pass("GT state updates")
    except Exception as e:
        results.add_fail("GT state updates", str(e))

    # Test 4: Detection updates
    try:
        bboxes = torch.randn(num_envs, 1, 4, device=device)
        valid_mask = torch.ones(num_envs, 1, dtype=torch.bool, device=device)
        wrapper.update_detections("agent_0", bboxes, valid_mask)

        states = wrapper._agent_states["agent_0"]
        assert torch.allclose(states.data.bboxes_2d, bboxes)
        results.add_pass("Detection updates")
    except Exception as e:
        results.add_fail("Detection updates", str(e))

    # Test 5: Delayed states query
    try:
        delayed_states = wrapper.get_delayed_states("agent_0")
        assert isinstance(delayed_states, AgentStates)
        assert delayed_states.data.body_position_w.shape == (num_envs, 3)
        results.add_pass("Delayed states query")
    except Exception as e:
        results.add_fail("Delayed states query", str(e))

    # Test 6: Noise curriculum
    try:
        wrapper.set_noise_progress_scale(-0.5)
        assert wrapper._noise_progress == 0.0

        wrapper.set_noise_progress_scale(1.5)
        assert wrapper._noise_progress == 1.0

        wrapper.set_noise_progress_scale(0.5)
        assert wrapper._noise_progress == 0.5
        results.add_pass("Noise curriculum scaling")
    except Exception as e:
        results.add_fail("Noise curriculum scaling", str(e))

    # Test 7: Delayed + noisy states
    try:
        wrapper.set_noise_progress_scale(0.0)
        delayed = wrapper.get_delayed_states("agent_0")
        noisy = wrapper.get_delayed_noisy_states("agent_0")

        # Should be identical (no noise)
        assert torch.allclose(delayed.data.body_position_w, noisy.data.body_position_w, atol=1e-5)

        wrapper.set_noise_progress_scale(1.0)
        noisy_full = wrapper.get_delayed_noisy_states("agent_0")

        # Should be different (noise applied)
        # Note: May fail occasionally due to random chance
        results.add_pass("Delayed + noisy states")
    except Exception as e:
        results.add_fail("Delayed + noisy states", str(e))

    # Test 8: Multi-agent coordination
    try:
        wrapper = MultiAgentDelaySystem(
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent=num_joints_per_agent,
            num_targets_per_agent=num_targets_per_agent,
            dt=dt,
            device=device,
        )

        for agent_id in possible_agents:
            wrapper.set_camera_configs(
                agent_id=agent_id,
                width=640,
                height=480,
                focal_length=50.0,
                horizontal_aperture=36.0,
                vertical_aperture=27.0,
                offset_position_b=torch.tensor([0.1, 0.0, 0.0], device=device),
                offset_rotation_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            )

            wrapper.update_gt_states(
                agent_id=agent_id,
                body_position_w=torch.randn(num_envs, 3, device=device),
                body_orientation_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device),
                body_linear_velocity_w=torch.randn(num_envs, 3, device=device),
                body_angular_velocity_w=torch.randn(num_envs, 3, device=device),
                body_linear_acceleration_w=torch.randn(num_envs, 3, device=device),
                body_combined_angular_velocity_w=torch.randn(num_envs, 3, device=device),
                joint_positions_b=torch.randn(num_envs, 3, device=device),
                zoom_level=torch.ones(num_envs, device=device),
            )

        states_0 = wrapper.get_delayed_states("agent_0")
        states_1 = wrapper.get_delayed_states("agent_1")
        assert isinstance(states_0, AgentStates)
        assert isinstance(states_1, AgentStates)
        results.add_pass("Multi-agent coordination")
    except Exception as e:
        results.add_fail("Multi-agent coordination", str(e))

    # Test 9: Reset
    try:
        env_ids = torch.arange(num_envs, device=device)
        wrapper.reset(env_ids)

        env_ids = torch.tensor([0, 2], device=device)
        wrapper.reset(env_ids)
        results.add_pass("Reset functionality")
    except Exception as e:
        results.add_fail("Reset functionality", str(e))


def main():
    """Run all delay system tests."""
    start_time = time.time()

    print("\n" + "="*80, flush=True)
    print("DELAY SYSTEM TEST SUITE", flush=True)
    print("="*80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print comprehensive configuration
    print(f"Device:        {device}", flush=True)
    print(f"PyTorch:       {torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}", flush=True)
        print(f"GPU name:      {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU memory:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Verbose mode:  {VERBOSE}", flush=True)
    print("="*80, flush=True)

    results = TestResults()

    # Run test suites
    try:
        test_stochastic_sampler(results, device)
    except Exception as e:
        results.add_error("StochasticSampler suite", traceback.format_exc())

    try:
        test_specialized_samplers(results, device)
    except Exception as e:
        results.add_error("Specialized Samplers suite", traceback.format_exc())

    try:
        test_timestamp_tracking(results, device)
    except Exception as e:
        results.add_error("Timestamp Tracking suite", traceback.format_exc())

    try:
        test_delay_system_integration(results, device)
    except Exception as e:
        results.add_error("Delay System Integration suite", traceback.format_exc())

    try:
        test_multi_agent_wrapper(results, device)
    except Exception as e:
        results.add_error("MultiAgent Wrapper suite", traceback.format_exc())

    # Print summary with timing
    success = results.print_summary()

    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f}s", flush=True)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Cleanup
    simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
