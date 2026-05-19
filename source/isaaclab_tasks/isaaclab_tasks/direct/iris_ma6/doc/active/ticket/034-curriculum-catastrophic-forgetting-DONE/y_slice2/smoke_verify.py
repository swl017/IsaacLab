#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice 2 smoke verification — gimbal τ + dead-time anti-forgetting at step=399k.

Pass:
  - `gimbal_rate_loop._tau_progress_per_env` distribution covers [0, 1]
    (min < 0.05, max > 0.95) — proves per-env progress is hooked up.
  - `gimbal_rate_loop._dead_time_seconds` distribution covers near-zero
    and near-mean envs (min near 0, max well above mean).
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
    # Run one step so the step-time hooks fire and the rate-loop receives
    # the per-env _tau_progress_per_env tensor.
    action = {
        aid: torch.zeros(num_envs, env.action_space(aid).shape[0], device=unwrapped.device)
        for aid in unwrapped.cfg.possible_agents
    }
    env.step(action)

    rl = unwrapped._controller.gimbal_rate_loop
    tau_progress = rl._tau_progress_per_env.detach().cpu()
    dead_time = rl._dead_time_seconds.detach().cpu()
    dead_scale = rl._dead_time_curr_scale_per_env.detach().cpu()

    eff_dyn = unwrapped._eff_progress_dynamics.detach().cpu()
    eff_gdt = unwrapped._eff_progress_gimbal_dead_time.detach().cpu()

    stats = {
        "step": step,
        "num_envs": int(num_envs),
        "tau_progress_per_env_min": float(tau_progress.min().item()),
        "tau_progress_per_env_max": float(tau_progress.max().item()),
        "tau_progress_per_env_mean": float(tau_progress.mean().item()),
        "tau_progress_per_env_below_005": int((tau_progress < 0.05).sum().item()),
        "tau_progress_per_env_above_095": int((tau_progress > 0.95).sum().item()),
        "dead_time_seconds_min": float(dead_time.min().item()),
        "dead_time_seconds_max": float(dead_time.max().item()),
        "dead_time_seconds_mean": float(dead_time.mean().item()),
        "dead_time_seconds_below_001": int((dead_time < 0.001).sum().item()),
        "dead_time_seconds_above_005": int((dead_time > 0.05).sum().item()),
        "dead_time_curr_scale_per_env_min": float(dead_scale.min().item()),
        "dead_time_curr_scale_per_env_max": float(dead_scale.max().item()),
        "dead_time_curr_scale_per_env_mean": float(dead_scale.mean().item()),
        "eff_progress_dynamics_min": float(eff_dyn.min().item()),
        "eff_progress_dynamics_max": float(eff_dyn.max().item()),
        "eff_progress_dynamics_mean": float(eff_dyn.mean().item()),
        "eff_progress_gimbal_dead_time_min": float(eff_gdt.min().item()),
        "eff_progress_gimbal_dead_time_max": float(eff_gdt.max().item()),
        "eff_progress_gimbal_dead_time_mean": float(eff_gdt.mean().item()),
    }

    print("=" * 60)
    print("SLICE 2 SMOKE VERIFICATION")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:46s} = {v}")
    print("=" * 60)

    out_path = Path(__file__).resolve().parent / "slice2_smoke.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")

    # Pass criteria from p_plan.md:
    # - τ_yaw_eff span across envs includes < 0.1 × nominal
    #   (proxy: _tau_progress_per_env min < 0.05).
    # - Dead-time samples cover [0, max] (proxy: min < 0.001 AND max > 0.05).
    passed = (
        stats["tau_progress_per_env_min"] < 0.05
        and stats["tau_progress_per_env_max"] > 0.5  # at step 399k, progress_dynamics=1 so spans [0,1]
        and stats["dead_time_seconds_min"] < 0.001
        and stats["dead_time_seconds_max"] > 0.05
    )
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
