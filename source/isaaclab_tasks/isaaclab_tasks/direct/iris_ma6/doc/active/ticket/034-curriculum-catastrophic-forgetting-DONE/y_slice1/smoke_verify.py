#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice 1 smoke verification — _max_lin_vel anti-forgetting at step=399k.

Pass: min(_max_lin_vel) < 5.0 m/s AND max(_max_lin_vel) > 9.0 m/s after
env.reset() with debug_initial_step=399000.

Usage:
    conda activate env_isaaclab
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/doc/active/ticket/034-curriculum-catastrophic-forgetting/y_slice1/smoke_verify.py
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

    mlv = unwrapped._max_lin_vel.detach().cpu()
    eff_p = unwrapped._eff_progress_agent_velocity.detach().cpu()

    stats = {
        "num_envs": int(mlv.numel()),
        "step": step,
        "max_lin_vel_min_cfg": float(env_cfg.max_lin_vel_min),
        "max_lin_vel_max_cfg": float(env_cfg.max_lin_vel),
        "max_lin_vel_observed_min": float(mlv.min().item()),
        "max_lin_vel_observed_max": float(mlv.max().item()),
        "max_lin_vel_observed_mean": float(mlv.mean().item()),
        "max_lin_vel_observed_std": float(mlv.std().item()),
        "max_lin_vel_below_5": int((mlv < 5.0).sum().item()),
        "max_lin_vel_above_9": int((mlv > 9.0).sum().item()),
        "eff_progress_agent_velocity_min": float(eff_p.min().item()),
        "eff_progress_agent_velocity_max": float(eff_p.max().item()),
        "eff_progress_agent_velocity_mean": float(eff_p.mean().item()),
    }

    print("=" * 60)
    print("SLICE 1 SMOKE VERIFICATION")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:42s} = {v}")
    print("=" * 60)

    out_path = Path(__file__).resolve().parent / "slice1_smoke.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")

    # Pass criteria from p_plan.md
    passed = (
        stats["max_lin_vel_observed_min"] < 5.0
        and stats["max_lin_vel_observed_max"] > 9.0
    )
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
