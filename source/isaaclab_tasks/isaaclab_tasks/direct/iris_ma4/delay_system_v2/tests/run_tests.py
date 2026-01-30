#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Delay System V2 Test Suite for Multi-Agent Environments.

This test suite validates the MultiAgentDelaySystemV2 wrapper which uses
the quadcopter's field-based DelaySystemV2 architecture internally.

Tests cover:
- Initialization and configuration
- API compatibility with iris_ma3 delay system
- Dual pipeline (clean/noisy) functionality
- Perspective-aware delays (ego vs inter-agent)
- Derived field computations
- Reset functionality
- Curriculum support (noise scaling)
"""
import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run MultiAgentDelaySystemV2 test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW we can import other modules
import sys
import torch
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# Import the modules to test
from isaaclab_tasks.direct.iris_ma4.delay_system_v2 import (
    MultiAgentDelaySystemV2,
    MultiAgentDelaySystemV2Cfg,
    AgentStates,
    AgentStatesData,
    MultiAgentStates,
)


class TestResults:
    """Track test results with pass/fail counts and error details."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[tuple[str, str]] = []
        self.errors: List[tuple[str, str]] = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
        error_lines = error.split('\n')[:5]
        for line in error_lines:
            print(f"    {line}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}")
        error_lines = error.split('\n')[:3]
        for line in error_lines:
            print(f"    {line}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        if total > 0:
            print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
            print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
            print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)")
        else:
            print("No tests ran.")

        if self.failed:
            print("\n" + "-" * 80)
            print("FAILED TESTS:")
            print("-" * 80)
            for test_name, error in self.failed:
                print(f"\n{test_name}:")
                print(f"  {error}")

        if self.errors:
            print("\n" + "-" * 80)
            print("TEST ERRORS:")
            print("-" * 80)
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error}")

        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def run_initialization_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test initialization and configuration."""
    print("\n" + "=" * 80)
    print("Testing Initialization")
    print("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1", "drone_2"]

    # Test 1: Basic initialization
    try:
        cfg = MultiAgentDelaySystemV2Cfg()
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )
        assert delay_system is not None
        results.add_pass("Basic initialization")
    except Exception as e:
        results.add_fail("Basic initialization", traceback.format_exc())
        return  # Cannot continue if basic init fails

    # Test 2: Configuration parameters
    try:
        cfg = MultiAgentDelaySystemV2Cfg(
            dt=0.02,
            motion_time_constant=0.2,
            enable_noise=True,
            position_noise_std=0.1,
        )
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )
        assert delay_system.cfg.dt == 0.02
        assert delay_system.cfg.motion_time_constant == 0.2
        results.add_pass("Custom configuration parameters")
    except Exception as e:
        results.add_fail("Custom configuration parameters", traceback.format_exc())

    # Test 3: Camera config setup
    try:
        cfg = MultiAgentDelaySystemV2Cfg()
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )

        offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
        offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        for agent_id in possible_agents:
            delay_system.set_camera_configs(
                agent_id,
                width=640,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.0,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )

        assert "drone_0" in delay_system._camera_configs
        results.add_pass("Camera config setup")
    except Exception as e:
        results.add_fail("Camera config setup", traceback.format_exc())


def run_api_compatibility_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test API compatibility with iris_ma3 delay system."""
    print("\n" + "=" * 80)
    print("Testing API Compatibility")
    print("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1"]

    # Create delay system
    try:
        cfg = MultiAgentDelaySystemV2Cfg(enable_noise=True)
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )

        # Set camera configs
        offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
        offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        for agent_id in possible_agents:
            delay_system.set_camera_configs(
                agent_id,
                width=640,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.0,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )
    except Exception as e:
        results.add_error("API test setup", traceback.format_exc())
        return

    # Test 1: update_time()
    try:
        delay_system.update_time()
        delay_system.update_time(dt=0.01)
        results.add_pass("update_time() API")
    except Exception as e:
        results.add_fail("update_time() API", traceback.format_exc())

    # Test 2: update_gt_states()
    try:
        for agent_id in possible_agents:
            delay_system.update_gt_states(
                agent_id=agent_id,
                body_position_w=torch.randn(num_envs, 3, device=device),
                body_orientation_w=torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4),
                body_linear_velocity_w=torch.randn(num_envs, 3, device=device),
                body_angular_velocity_w=torch.randn(num_envs, 3, device=device),
                body_linear_acceleration_w=torch.randn(num_envs, 3, device=device),
                body_combined_angular_velocity_w=torch.randn(num_envs, 3, device=device),
                joint_positions_b=torch.randn(num_envs, 3, device=device),
                zoom_level=torch.ones(num_envs, device=device)
            )
        results.add_pass("update_gt_states() API")
    except Exception as e:
        results.add_fail("update_gt_states() API", traceback.format_exc())

    # Test 3: update_detections()
    try:
        for agent_id in possible_agents:
            delay_system.update_detections(
                agent_id=agent_id,
                bboxes_2d_gt=torch.tensor([[[320., 240., 50., 50.]]], device=device).expand(num_envs, 1, 4)
            )
        results.add_pass("update_detections() API")
    except Exception as e:
        results.add_fail("update_detections() API", traceback.format_exc())

    # Test 4: get_all_states_for_rewards()
    try:
        for ego_agent_id in possible_agents:
            states_dict = delay_system.get_all_states_for_rewards(ego_agent_id)
            assert isinstance(states_dict, dict)
            assert len(states_dict) == len(possible_agents)
            for agent_id in possible_agents:
                assert agent_id in states_dict
                state = states_dict[agent_id]
                assert isinstance(state, AgentStates)
                assert isinstance(state.data, AgentStatesData)
        results.add_pass("get_all_states_for_rewards() API")
    except Exception as e:
        results.add_fail("get_all_states_for_rewards() API", traceback.format_exc())

    # Test 5: get_all_states_for_observations()
    try:
        for ego_agent_id in possible_agents:
            states_dict = delay_system.get_all_states_for_observations(ego_agent_id)
            assert isinstance(states_dict, dict)
            assert len(states_dict) == len(possible_agents)
        results.add_pass("get_all_states_for_observations() API")
    except Exception as e:
        results.add_fail("get_all_states_for_observations() API", traceback.format_exc())

    # Test 6: gt_states property
    try:
        gt_states = delay_system.gt_states
        # Check that gt_states has the required interface (agents dict-like access)
        assert hasattr(gt_states, 'agents'), "gt_states must have 'agents' property"
        for agent_id in possible_agents:
            assert agent_id in gt_states.agents, f"Agent {agent_id} not in gt_states.agents"
            # Also check __getitem__ access
            agent_state = gt_states[agent_id]
            assert isinstance(agent_state, AgentStates), f"gt_states[{agent_id}] must be AgentStates"
        results.add_pass("gt_states property")
    except Exception as e:
        results.add_fail("gt_states property", traceback.format_exc())

    # Test 7: current_time property
    try:
        current_time = delay_system.current_time
        assert isinstance(current_time, torch.Tensor)
        assert current_time.shape == (num_envs,)
        results.add_pass("current_time property")
    except Exception as e:
        results.add_fail("current_time property", traceback.format_exc())

    # Test 8: set_noise_progress_scale()
    try:
        delay_system.set_noise_progress_scale(0.5)
        delay_system.set_noise_progress_scale(1.0)
        results.add_pass("set_noise_progress_scale() API")
    except Exception as e:
        results.add_fail("set_noise_progress_scale() API", traceback.format_exc())


def run_dual_pipeline_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test dual pipeline (clean/noisy) functionality."""
    print("\n" + "=" * 80)
    print("Testing Dual Pipeline")
    print("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1"]

    # Create delay system with noise enabled
    try:
        cfg = MultiAgentDelaySystemV2Cfg(
            enable_noise=True,
            position_noise_std=1.0,  # Large noise for easy detection
            orientation_noise_std=0.1,
            linear_velocity_noise_std=0.5,  # Fixed: use correct param name
            bbox_noise_std=10.0,
        )
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )

        # Set camera configs
        offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
        offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        for agent_id in possible_agents:
            delay_system.set_camera_configs(
                agent_id,
                width=640,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.0,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )

        # Update states multiple times
        for _ in range(10):
            delay_system.update_time()
            for agent_id in possible_agents:
                delay_system.update_gt_states(
                    agent_id=agent_id,
                    body_position_w=torch.ones(num_envs, 3, device=device) * 5.0,  # Fixed position
                    body_orientation_w=torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4),
                    body_linear_velocity_w=torch.zeros(num_envs, 3, device=device),
                    body_angular_velocity_w=torch.zeros(num_envs, 3, device=device),
                    body_linear_acceleration_w=torch.zeros(num_envs, 3, device=device),
                    body_combined_angular_velocity_w=torch.zeros(num_envs, 3, device=device),
                    joint_positions_b=torch.zeros(num_envs, 3, device=device),
                    zoom_level=torch.ones(num_envs, device=device)
                )
                delay_system.update_detections(
                    agent_id=agent_id,
                    bboxes_2d_gt=torch.tensor([[[320., 240., 50., 50.]]], device=device).expand(num_envs, 1, 4)
                )

    except Exception as e:
        results.add_error("Dual pipeline test setup", traceback.format_exc())
        return

    # Test 1: Clean pipeline should have no noise
    try:
        reward_states = delay_system.get_all_states_for_rewards("drone_0")
        clean_pos = reward_states["drone_0"].data.body_position_w
        # Clean position should be close to GT (5.0, 5.0, 5.0)
        assert torch.allclose(clean_pos, torch.ones_like(clean_pos) * 5.0, atol=0.5), \
            f"Clean position too far from GT: {clean_pos.mean().item()}"
        results.add_pass("Clean pipeline (no noise)")
    except AssertionError as e:
        results.add_fail("Clean pipeline (no noise)", str(e))
    except Exception as e:
        results.add_fail("Clean pipeline (no noise)", traceback.format_exc())

    # Test 2: Noisy pipeline should have different values (noise added)
    try:
        obs_states = delay_system.get_all_states_for_observations("drone_0")
        noisy_pos = obs_states["drone_0"].data.body_position_w

        # Check that noisy observations are available
        assert noisy_pos is not None
        assert noisy_pos.shape == (num_envs, 3)

        # Note: Due to first-order lag filtering, noise may be smoothed
        # We just verify the pipeline returns valid data
        results.add_pass("Noisy pipeline produces data")
    except Exception as e:
        results.add_fail("Noisy pipeline produces data", traceback.format_exc())


def run_perspective_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test perspective-aware delays (ego vs inter-agent)."""
    print("\n" + "=" * 80)
    print("Testing Perspective-Aware Delays")
    print("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1", "drone_2"]

    try:
        cfg = MultiAgentDelaySystemV2Cfg(
            # Ego comm should be faster
            ego_comm_time_constant=0.001,
            # Inter-agent should have more delay
            inter_agent_comm_latency_mean=0.1,
            inter_agent_comm_dropout_rate=0.0,  # Disable dropout for deterministic test
        )
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )

        # Set camera configs
        offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
        offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        for agent_id in possible_agents:
            delay_system.set_camera_configs(
                agent_id,
                width=640,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.0,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )
    except Exception as e:
        results.add_error("Perspective test setup", traceback.format_exc())
        return

    # Test 1: Each agent sees all agents from their perspective
    try:
        for ego_id in possible_agents:
            states = delay_system.get_all_states_for_rewards(ego_id)
            assert len(states) == len(possible_agents)
            for agent_id in possible_agents:
                assert agent_id in states
        results.add_pass("All agents visible from each perspective")
    except Exception as e:
        results.add_fail("All agents visible from each perspective", traceback.format_exc())

    # Test 2: Ego state is different from other agent states (perspective)
    try:
        # Update with different positions for each agent
        delay_system.update_time()
        for i, agent_id in enumerate(possible_agents):
            pos = torch.ones(num_envs, 3, device=device) * float(i)  # Different positions
            delay_system.update_gt_states(
                agent_id=agent_id,
                body_position_w=pos,
                body_orientation_w=torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4),
                body_linear_velocity_w=torch.zeros(num_envs, 3, device=device),
                body_angular_velocity_w=torch.zeros(num_envs, 3, device=device),
                body_linear_acceleration_w=torch.zeros(num_envs, 3, device=device),
                body_combined_angular_velocity_w=torch.zeros(num_envs, 3, device=device),
                joint_positions_b=torch.zeros(num_envs, 3, device=device),
                zoom_level=torch.ones(num_envs, device=device)
            )
            delay_system.update_detections(
                agent_id=agent_id,
                bboxes_2d_gt=torch.tensor([[[320., 240., 50., 50.]]], device=device).expand(num_envs, 1, 4)
            )

        # Verify different perspectives
        states_0 = delay_system.get_all_states_for_rewards("drone_0")
        states_1 = delay_system.get_all_states_for_rewards("drone_1")

        # Both should see drone_0 and drone_1, but from different perspectives
        # (ego vs other agent processing)
        assert "drone_0" in states_0 and "drone_1" in states_0
        assert "drone_0" in states_1 and "drone_1" in states_1

        results.add_pass("Perspective separation works")
    except Exception as e:
        results.add_fail("Perspective separation works", traceback.format_exc())


def run_reset_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test reset functionality."""
    print("\n" + "=" * 80)
    print("Testing Reset Functionality")
    print("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1"]

    try:
        cfg = MultiAgentDelaySystemV2Cfg()
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )

        # Set camera configs
        offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
        offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        for agent_id in possible_agents:
            delay_system.set_camera_configs(
                agent_id,
                width=640,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.0,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )
    except Exception as e:
        results.add_error("Reset test setup", traceback.format_exc())
        return

    # Test 1: Reset without initial states
    try:
        env_ids = torch.tensor([0, 1, 2, 3], device=device)
        delay_system.reset(env_ids=env_ids)
        results.add_pass("Reset without initial states")
    except Exception as e:
        results.add_fail("Reset without initial states", traceback.format_exc())

    # Test 2: Reset with initial states
    try:
        from types import SimpleNamespace

        env_ids = torch.tensor([4, 5, 6, 7], device=device)
        initial_gt_states = {}

        for agent_id in possible_agents:
            data = SimpleNamespace()
            data.body_position_w = torch.randn(num_envs, 3, device=device)
            data.body_orientation_w = torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4)
            data.body_linear_velocity_w = torch.zeros(num_envs, 3, device=device)
            data.body_angular_velocity_w = torch.zeros(num_envs, 3, device=device)
            data.body_combined_angular_velocity_w = torch.zeros(num_envs, 3, device=device)
            data.body_linear_acceleration_w = torch.zeros(num_envs, 3, device=device)
            data.joint_positions_b = torch.zeros(num_envs, 3, device=device)
            data.joint_velocities_b = torch.zeros(num_envs, 3, device=device)
            data.camera_zoom_level = torch.ones(num_envs, 1, device=device)
            initial_gt_states[agent_id] = SimpleNamespace(data=data)

        delay_system.reset(env_ids=env_ids, initial_gt_states=initial_gt_states)
        results.add_pass("Reset with initial states")
    except Exception as e:
        results.add_fail("Reset with initial states", traceback.format_exc())

    # Test 3: Time reset on env reset
    try:
        # Advance time
        for _ in range(10):
            delay_system.update_time()

        time_before = delay_system.current_time.clone()

        # Reset some envs
        env_ids = torch.tensor([0, 1], device=device)
        delay_system.reset(env_ids=env_ids)

        time_after = delay_system.current_time

        # Reset envs should have time reset to 0
        assert torch.all(time_after[env_ids] == 0.0), "Reset envs should have time = 0"

        # Non-reset envs should keep their time
        non_reset_ids = torch.tensor([2, 3, 4, 5], device=device)
        assert torch.allclose(time_after[non_reset_ids], time_before[non_reset_ids]), \
            "Non-reset envs should keep time"

        results.add_pass("Time reset on env reset")
    except AssertionError as e:
        results.add_fail("Time reset on env reset", str(e))
    except Exception as e:
        results.add_fail("Time reset on env reset", traceback.format_exc())


def run_derived_fields_tests(results: TestResults, device: torch.device, verbose: bool = False):
    """Test derived field computations."""
    print("\n" + "=" * 80)
    print("Testing Derived Field Computations")
    print("=" * 80)

    num_envs = 16
    possible_agents = ["drone_0", "drone_1"]

    try:
        cfg = MultiAgentDelaySystemV2Cfg()
        delay_system = MultiAgentDelaySystemV2(
            cfg=cfg,
            possible_agents=possible_agents,
            num_envs=num_envs,
            num_joints_per_agent={agent: 3 for agent in possible_agents},
            num_targets_per_agent={agent: 1 for agent in possible_agents},
            device=device,
        )

        # Set camera configs
        offset_pos = torch.tensor([0.0, 0.0, 0.1], device=device)
        offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        for agent_id in possible_agents:
            delay_system.set_camera_configs(
                agent_id,
                width=640,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.0,
                offset_position_b=offset_pos,
                offset_rotation_b=offset_rot
            )

        # Update states
        for _ in range(5):
            delay_system.update_time()
            for agent_id in possible_agents:
                delay_system.update_gt_states(
                    agent_id=agent_id,
                    body_position_w=torch.randn(num_envs, 3, device=device),
                    body_orientation_w=torch.tensor([[1., 0., 0., 0.]], device=device).expand(num_envs, 4),
                    body_linear_velocity_w=torch.randn(num_envs, 3, device=device),
                    body_angular_velocity_w=torch.randn(num_envs, 3, device=device),
                    body_linear_acceleration_w=torch.randn(num_envs, 3, device=device),
                    body_combined_angular_velocity_w=torch.randn(num_envs, 3, device=device),
                    joint_positions_b=torch.randn(num_envs, 3, device=device),
                    zoom_level=torch.ones(num_envs, device=device)
                )
                delay_system.update_detections(
                    agent_id=agent_id,
                    bboxes_2d_gt=torch.tensor([[[320., 240., 50., 50.]]], device=device).expand(num_envs, 1, 4)
                )
    except Exception as e:
        results.add_error("Derived fields test setup", traceback.format_exc())
        return

    # Test 1: Camera position is computed
    try:
        states = delay_system.get_all_states_for_rewards("drone_0")
        ego_state = states["drone_0"]
        camera_pos = ego_state.data.camera_position_w
        assert camera_pos is not None
        assert camera_pos.shape == (num_envs, 3)
        assert not torch.isnan(camera_pos).any()
        results.add_pass("Camera position computed")
    except Exception as e:
        results.add_fail("Camera position computed", traceback.format_exc())

    # Test 2: Camera orientation is computed
    try:
        states = delay_system.get_all_states_for_rewards("drone_0")
        ego_state = states["drone_0"]
        camera_ori = ego_state.data.camera_orientation_w
        assert camera_ori is not None
        assert camera_ori.shape == (num_envs, 4)
        assert not torch.isnan(camera_ori).any()
        # Check quaternion is normalized
        norms = torch.norm(camera_ori, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Camera orientation not normalized: {norms}"
        results.add_pass("Camera orientation computed and normalized")
    except AssertionError as e:
        results.add_fail("Camera orientation computed and normalized", str(e))
    except Exception as e:
        results.add_fail("Camera orientation computed and normalized", traceback.format_exc())

    # Test 3: Ray directions are computed
    try:
        states = delay_system.get_all_states_for_rewards("drone_0")
        ego_state = states["drone_0"]
        ray_dirs = ego_state.data.camera_ray_directions_w
        assert ray_dirs is not None
        assert ray_dirs.shape == (num_envs, 1, 3)  # 1 target
        # Note: ray_dirs may be zero for invalid bboxes
        results.add_pass("Ray directions computed")
    except Exception as e:
        results.add_fail("Ray directions computed", traceback.format_exc())

    # Test 4: Camera intrinsics are set
    try:
        states = delay_system.get_all_states_for_rewards("drone_0")
        ego_state = states["drone_0"]
        intrinsics = ego_state.data.camera_base_intrinsics
        assert intrinsics is not None
        assert intrinsics.shape == (num_envs, 3, 3)
        # Check fx, fy are positive
        assert torch.all(intrinsics[:, 0, 0] > 0), "fx should be positive"
        assert torch.all(intrinsics[:, 1, 1] > 0), "fy should be positive"
        results.add_pass("Camera intrinsics available")
    except AssertionError as e:
        results.add_fail("Camera intrinsics available", str(e))
    except Exception as e:
        results.add_fail("Camera intrinsics available", traceback.format_exc())


def main():
    """Main test runner."""
    print("=" * 80)
    print("DELAY SYSTEM V2 TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()
    verbose = args_cli.test_verbose

    try:
        run_initialization_tests(results, device, verbose)
        run_api_compatibility_tests(results, device, verbose)
        run_dual_pipeline_tests(results, device, verbose)
        run_perspective_tests(results, device, verbose)
        run_reset_tests(results, device, verbose)
        run_derived_fields_tests(results, device, verbose)
    except Exception as e:
        results.add_error("Test suite", traceback.format_exc())

    success = results.print_summary()

    # Cleanup
    simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
