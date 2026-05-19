#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice 3 smoke verification — zoom dead-time anti-forgetting at step=399k.

Pass:
  - `zoom_controller._dead_time_curr_scale_per_env` spans [0, 1]
    (min < 0.05, max > 0.5 since at step 399k progress_zoom_dead_time=1).
  - `zoom_controller._dead_time_seconds` covers near-0 envs AND envs near
    the configured mean.
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

    zc = unwrapped._controller.zoom_controller
    is_siyi = zc.cfg.model == "siyi_a8"
    dead_time = zc._dead_time_seconds.detach().cpu()
    dead_scale = zc._dead_time_curr_scale_per_env.detach().cpu()
    eff_zdt = unwrapped._eff_progress_zoom_dead_time.detach().cpu()

    stats = {
        "step": step,
        "num_envs": int(num_envs),
        "zoom_model_is_siyi_a8": bool(is_siyi),
        "dead_time_seconds_min": float(dead_time.min().item()),
        "dead_time_seconds_max": float(dead_time.max().item()),
        "dead_time_seconds_mean": float(dead_time.mean().item()),
        "dead_time_seconds_below_001": int((dead_time < 0.001).sum().item()),
        "dead_time_seconds_above_005": int((dead_time > 0.05).sum().item()),
        "dead_time_curr_scale_per_env_min": float(dead_scale.min().item()),
        "dead_time_curr_scale_per_env_max": float(dead_scale.max().item()),
        "dead_time_curr_scale_per_env_mean": float(dead_scale.mean().item()),
        "eff_progress_zoom_dead_time_min": float(eff_zdt.min().item()),
        "eff_progress_zoom_dead_time_max": float(eff_zdt.max().item()),
        "eff_progress_zoom_dead_time_mean": float(eff_zdt.mean().item()),
    }

    print("=" * 60)
    print("SLICE 3 SMOKE VERIFICATION")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:46s} = {v}")
    print("=" * 60)

    out_path = Path(__file__).resolve().parent / "slice3_smoke.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")

    if is_siyi:
        passed = (
            stats["dead_time_curr_scale_per_env_min"] < 0.05
            and stats["dead_time_curr_scale_per_env_max"] > 0.5
            and stats["dead_time_seconds_min"] < 0.005
            and stats["dead_time_seconds_max"] > 0.05
        )
    else:
        # Non-siyi_a8 model: setter is a no-op. Only verify env wiring path
        # didn't crash; setter no-op means scale stays at its __init__ value.
        passed = True
    print(f"\nResult: {'PASS' if passed else 'FAIL'} (siyi_a8={is_siyi})")
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
