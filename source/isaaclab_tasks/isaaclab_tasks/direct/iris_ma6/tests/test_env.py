#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple test script for iris_ma6 test environment."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test iris_ma6 environment")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def log(msg):
    """Print with flush for immediate output."""
    print(msg, flush=True)


log("=" * 80)
log("IRIS MA6 TEST ENVIRONMENT VERIFICATION")
log("=" * 80)

try:
    # Create the environment configuration
    log("\n[1] Creating environment...")
    task_name = 'Isaac-Iris-MA6-Direct-Test-v0'
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)
    env = gym.make(task_name, cfg=env_cfg)
    log(f"    Environment created successfully!")
    log(f"    Possible agents: {env.unwrapped.cfg.possible_agents}")
    log(f"    Num envs: {env.unwrapped.num_envs}")
    log(f"    Device: {env.unwrapped.device}")

    # Get initial observations
    log("\n[2] Resetting environment...")
    obs, info = env.reset()
    log(f"    Reset successful!")
    log(f"    Observations keys: {list(obs.keys())}")
    for agent_id, agent_obs in obs.items():
        log(f"      {agent_id}: shape={agent_obs.shape}, dtype={agent_obs.dtype}")

    # Set viewport to follow the first robot in env 0
    log("\n[2.5] Setting viewport to follow first drone...")
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf
        viewport = get_active_viewport()
        if viewport is not None:
            # Get the first robot's prim path
            unwrapped = env.unwrapped
            first_robot = unwrapped._robots["drone_0"]  # type: ignore[attr-defined]
            prim_path = first_robot.cfg.prim_path.replace("env_.*", "env_0")
            camera_path = f"{prim_path}/body/Camera"
            viewport.set_active_camera(Sdf.Path(camera_path))
            log(f"    Viewport set to follow: {camera_path}")
    except Exception as e:
        log(f"    Could not set viewport camera: {e}")

    # Take a step with random actions
    log("\n[3] Taking first step...")
    actions = {}
    for agent_id in env.unwrapped.cfg.possible_agents:
        actions[agent_id] = torch.rand(4, 7, device=env.unwrapped.device) * 2 - 1

    obs, rewards, terminated, truncated, info = env.step(actions)
    log(f"    Step completed!")
    log(f"    Rewards:")
    for agent_id, r in rewards.items():
        log(f"      {agent_id}: {r.mean().item():.4f}")

    # Run multiple steps to verify stability (100 steps = 4 seconds at 0.04s/step)
    log("\n[4] Running 100 steps (4 seconds) to verify stability...")
    for i in range(100):
        actions = {}
        for agent_id in env.unwrapped.cfg.possible_agents:
            actions[agent_id] = torch.rand(4, 7, device=env.unwrapped.device) * 2 - 1
        obs, rewards, terminated, truncated, info = env.step(actions)

    log(f"    100 steps completed successfully!")
    log(f"    Final rewards:")
    for agent_id, r in rewards.items():
        log(f"      {agent_id}: {r.mean().item():.4f}")

    # Check drone positions
    log("\n[5] Checking drone states...")
    for agent_id in env.unwrapped.cfg.possible_agents:
        robot = env.unwrapped._robots[agent_id]
        pos = robot.data.root_pos_w
        vel = robot.data.root_lin_vel_w
        log(f"    {agent_id}:")
        log(f"      Position: {pos.mean(dim=0).cpu().numpy().round(2)}")
        log(f"      Velocity: {vel.mean(dim=0).cpu().numpy().round(2)}")

    env.close()

    log("\n" + "=" * 80)
    log("TEST PASSED!")
    log("=" * 80)

except Exception as e:
    import traceback
    log(f"\nTEST FAILED!")
    log(f"Error: {e}")
    traceback.print_exc()
    sys.stdout.flush()

simulation_app.close()
