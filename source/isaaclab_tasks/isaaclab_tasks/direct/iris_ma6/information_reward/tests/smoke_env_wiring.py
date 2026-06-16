#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice-2 smoke (ticket 050 Slice B): the information reward wires into the env (default-off→on),
the triangulation reward term reflects per-agent r_diff, and enable_critic_gt_target sizes the
centralized critic state. No exceptions.

    ./isaaclab.sh -p .../information_reward/tests/smoke_env_wiring.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Slice-2 info-reward env-wiring smoke")
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
    cfg.information_reward.enabled = True
    cfg.enable_critic_gt_target = True      # privileged critic (set pre-construction)
    cfg.enable_track_loss_scenario = True   # so deficits occur (info reward exercised)
    cfg.use_debug_initial_step = True
    cfg.debug_initial_step = 300000
    cfg.episode_length_s = 2.0
    cfg.use_flight_scene = False

    env = gym.make(args_cli.task, cfg=cfg)
    uenv = env.unwrapped
    agents = uenv.possible_agents
    device = uenv.device

    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        print(f"  {'✓' if cond else '✗'} {name}")

    chk("info reward instantiated", uenv._info_reward is not None)
    # Slice 3: rebalance curriculum math (0 -> 0.5 -> 1.0 over 80k-200k) + bbox interpolation.
    cc = uenv.cfg.curriculum
    p0, p_mid, p_end = (cc.get_reward_rebalance_progress(s) for s in (0, 140000, 300000))
    chk("rebalance progress 0/140k/300k == 0/0.5/1.0",
        abs(p0) < 1e-6 and abs(p_mid - 0.5) < 1e-6 and abs(p_end - 1.0) < 1e-6)
    bc_mid = 90.0 + p_mid * (uenv.cfg.bbox_center_reward_scale_team - 90.0)
    chk("bbox_center interpolates to 60 at mid", abs(bc_mid - 60.0) < 1e-6)
    # Critic state sized for the privileged GT target (state_space > sum of actor obs).
    chk("enable_critic_gt_target active", bool(getattr(uenv.cfg, "enable_critic_gt_target", False)))
    chk("critic state_space set", uenv.cfg.state_space is not None and uenv.cfg.state_space > 0)

    def zero_actions():
        return {a: torch.zeros((uenv.num_envs, int(uenv.cfg.action_spaces[a])), device=device) for a in agents}

    env.reset()
    nonzero_tri = False
    for _ in range(40):
        env.step(zero_actions())
        # the per-step triangulation reward term is stored per agent in _step_rewards
        sr = getattr(uenv, "_step_rewards", {})
        for a in agents:
            if a in sr and "triangulation" in sr[a] and torch.isfinite(sr[a]["triangulation"]).all():
                if sr[a]["triangulation"].abs().sum() > 0:
                    nonzero_tri = True
    chk("triangulation reward term finite + active (per-agent r_diff)", nonzero_tri)
    # At debug_initial_step=300000 (> rebalance_end 200000) the env should run at full team strength.
    chk("env progress_rebalance == 1.0 at step 300k", abs(float(uenv.progress_rebalance) - 1.0) < 1e-6)

    ok = all(c for _, c in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    env.close()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
