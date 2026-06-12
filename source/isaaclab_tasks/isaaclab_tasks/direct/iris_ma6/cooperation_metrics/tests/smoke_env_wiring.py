#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice-2 smoke test: confirm the ReacquisitionTracker is wired into the env and that
``Coop/*`` scalars land in extras["log"] at episode reset, with no exceptions.

    ./isaaclab.sh -p .../cooperation_metrics/tests/smoke_env_wiring.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Slice-2 env-wiring smoke test")
parser.add_argument("--task", default="Isaac-Iris-MA6-Direct-Test-v0")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = 0
    cfg.cooperation_metrics.enable = True
    cfg.episode_length_s = 2.0       # force a reset within the step budget
    cfg.use_flight_scene = False     # faster scene build

    env = gym.make(args_cli.task, cfg=cfg)
    uenv = env.unwrapped
    agents = uenv.possible_agents
    device = uenv.device

    def zero_actions():
        return {
            a: torch.zeros((uenv.num_envs, int(uenv.cfg.action_spaces[a])), device=device)
            for a in agents
        }

    env.reset()
    print(f"[smoke] tracker is None? {uenv._reacq_tracker is None}")

    found = {}
    for step in range(200):
        env.step(zero_actions())
        log = uenv.extras.get("log", {})
        coop = {k: v for k, v in log.items() if k.startswith("Coop/")}
        if coop:
            found = coop
            print(f"[smoke] Coop keys appeared at step {step}")
            break

    ok = len(found) > 0 and uenv._reacq_tracker is not None
    print(f"\nRESULT: {'PASS — Coop keys logged' if ok else 'FAIL — no Coop keys'}")
    for k in sorted(found):
        try:
            print(f"  {k} = {float(found[k]):.4f}")
        except Exception:
            print(f"  {k} = {found[k]}")

    env.close()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
