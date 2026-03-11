#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment-facing smoke test for bbox_raycaster_v2 integration in iris_ma6."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Frontend/API smoke test for iris_ma6 bbox_raycaster_v2")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    task_name = "Isaac-Iris-MA6-Direct-Test-v0"
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=2)
    env = gym.make(task_name, cfg=env_cfg)

    try:
        obs, _ = env.reset()
        expected_obs_dim = 18
        possible_agents = env.unwrapped.cfg.possible_agents
        assert_true(len(obs) == len(possible_agents), "observation agent count mismatch")

        for agent_id in possible_agents:
            agent_obs = obs[agent_id]
            assert_true(agent_obs.shape == (env.unwrapped.num_envs, expected_obs_dim), f"{agent_id} obs shape mismatch")
            bbox = agent_obs[:, 13:17]
            bbox_empty = agent_obs[:, 17]
            assert_true(torch.isfinite(agent_obs).all().item(), f"{agent_id} observation contains non-finite values")
            assert_true(torch.isfinite(bbox).all().item(), f"{agent_id} bbox contains non-finite values")
            assert_true(torch.all((bbox_empty == 0.0) | (bbox_empty == 1.0)).item(), f"{agent_id} bbox_empty must be binary")

        zero_actions = {
            agent_id: torch.zeros(env.unwrapped.num_envs, 7, device=env.unwrapped.device)
            for agent_id in possible_agents
        }
        obs, rewards, terminated, truncated, _ = env.step(zero_actions)

        for agent_id in possible_agents:
            agent_obs = obs[agent_id]
            bbox = agent_obs[:, 13:17]
            bbox_empty = agent_obs[:, 17]
            assert_true(agent_obs.shape == (env.unwrapped.num_envs, expected_obs_dim), f"{agent_id} step obs shape mismatch")
            assert_true(torch.isfinite(agent_obs).all().item(), f"{agent_id} step observation contains non-finite values")
            assert_true(torch.all((bbox_empty == 0.0) | (bbox_empty == 1.0)).item(), f"{agent_id} step bbox_empty must be binary")
            if torch.any(bbox_empty == 1.0):
                assert_true(torch.all(bbox[bbox_empty == 1.0] == 0.0).item(), f"{agent_id} empty bbox must be zero-filled")
            assert_true(rewards[agent_id].shape == (env.unwrapped.num_envs,), f"{agent_id} reward shape mismatch")
            assert_true(terminated[agent_id].shape == (env.unwrapped.num_envs,), f"{agent_id} terminated shape mismatch")
            assert_true(truncated[agent_id].shape == (env.unwrapped.num_envs,), f"{agent_id} truncated shape mismatch")

        print("bbox_raycaster_v2 frontend smoke test passed", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
