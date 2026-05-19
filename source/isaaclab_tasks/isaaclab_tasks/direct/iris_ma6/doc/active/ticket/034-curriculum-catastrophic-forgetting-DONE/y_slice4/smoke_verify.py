#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice 4 smoke verification — obs-side anti-forgetting at step=399k.

Pass criteria:
  - Wrapper holds per-(env, agent) state tensors with non-trivial spread.
  - At least one downstream noise pipeline observes per-env std variation.
  - Latency-floor invariant: every sampled latency step ≥ min_latency_steps.
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main() -> int:
    task = "Isaac-Iris-MA6-Direct-Test-v0"
    num_envs = 256
    step = 399000

    env_cfg = parse_env_cfg(task, num_envs=num_envs)
    env_cfg.use_debug_initial_step = True
    env_cfg.debug_initial_step = step
    env_cfg.seed = 0

    env = gym.make(task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()
    action = {
        aid: torch.zeros(num_envs, env.action_space(aid).shape[0], device=unwrapped.device)
        for aid in unwrapped.cfg.possible_agents
    }
    env.step(action)

    ds = unwrapped._delay_system  # MultiAgentDelaySystemWrapper

    noise_t = ds._noise_scale_per_env_agent.detach().cpu()
    dropout_t = ds._base_dropout_rate_per_env_agent.detach().cpu()
    progress_t = ds._progress_per_env_agent.detach().cpu()

    eff_noise = unwrapped._eff_progress_noise.detach().cpu()
    eff_dropout = unwrapped._eff_progress_dropout.detach().cpu()
    eff_delay = unwrapped._eff_progress_delay.detach().cpu()
    eff_burst = unwrapped._eff_progress_burst_dropout.detach().cpu()

    # Sample a latency pipeline to read step values
    sample_pipe = None
    for pipe in ds._delay_system._all_pipelines():
        if pipe._latency_sampler._values.numel() > 0:
            sample_pipe = pipe
            break

    lat_step_values = (
        sample_pipe._latency_sampler.step_values.detach().cpu()
        if sample_pipe is not None
        else torch.zeros(num_envs, dtype=torch.long)
    )

    min_latency_steps_cfg = unwrapped.cfg.delay_system_params.min_latency_steps

    stats = {
        "step": step,
        "num_envs": int(num_envs),
        "noise_scale_per_env_agent_min": float(noise_t.min().item()),
        "noise_scale_per_env_agent_max": float(noise_t.max().item()),
        "noise_scale_per_env_agent_mean": float(noise_t.mean().item()),
        "noise_scale_below_005": int((noise_t < 0.05).sum().item()),
        "noise_scale_above_095": int((noise_t > 0.95).sum().item()),
        "dropout_rate_per_env_agent_min": float(dropout_t.min().item()),
        "dropout_rate_per_env_agent_max": float(dropout_t.max().item()),
        "dropout_rate_per_env_agent_mean": float(dropout_t.mean().item()),
        "dropout_rate_below_001": int((dropout_t < 0.001).sum().item()),
        "progress_per_env_agent_min": float(progress_t.min().item()),
        "progress_per_env_agent_max": float(progress_t.max().item()),
        "progress_per_env_agent_mean": float(progress_t.mean().item()),
        "eff_progress_noise_range": [float(eff_noise.min().item()), float(eff_noise.max().item())],
        "eff_progress_dropout_range": [float(eff_dropout.min().item()), float(eff_dropout.max().item())],
        "eff_progress_delay_range": [float(eff_delay.min().item()), float(eff_delay.max().item())],
        "eff_progress_burst_range": [float(eff_burst.min().item()), float(eff_burst.max().item())],
        "latency_step_values_min": int(lat_step_values.min().item()),
        "latency_step_values_max": int(lat_step_values.max().item()),
        "latency_step_values_mean": float(lat_step_values.float().mean().item()),
        "min_latency_steps_cfg": int(min_latency_steps_cfg),
        "latency_floor_violations": int((lat_step_values < min_latency_steps_cfg).sum().item()),
    }

    print("=" * 60)
    print("SLICE 4 SMOKE VERIFICATION")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:46s} = {v}")
    print("=" * 60)

    out_path = Path(__file__).resolve().parent / "slice4_smoke.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")

    # Pass criteria
    passed = (
        # Per-(env, agent) state tensors span [near-0, near-1].
        stats["noise_scale_per_env_agent_min"] < 0.05
        and stats["noise_scale_per_env_agent_max"] > 0.5
        # Dropout spans low and high.
        and stats["dropout_rate_per_env_agent_min"] < 0.01
        and stats["dropout_rate_per_env_agent_max"] > 0.01
        # Latency-floor invariant.
        and stats["latency_floor_violations"] == 0
    )
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
