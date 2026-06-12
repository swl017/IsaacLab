#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice-3 smoke test (ticket 050): the cooperation-trigger ceiling raise.

Part A (deterministic): the overlay raises ceilings and leaves floors untouched.
Part B (runtime): the flag-on env builds + steps without error and still logs Coop/* keys.

NOTE: the *gate* number (Coop/track_loss_event_rate >= 0.5/episode) is only meaningful with the
trained t047 policy actively tracking — zero-action agents do not reproduce hand-offs. That
gate run is launched separately (see the slice report). This smoke verifies the mechanism wiring.

    ./isaaclab.sh -p .../cooperation_metrics/tests/smoke_scenario_gate.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Slice-3 scenario-overlay smoke test")
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
from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import IrisMA6TestEnvCfg


def part_a_config_assertions():
    print("\n[Part A] Deterministic overlay assertions")
    base = IrisMA6TestEnvCfg()
    raised = IrisMA6TestEnvCfg()
    raised.apply_track_loss_scenario_overlay()
    s = raised.track_loss_scenario

    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        print(f"  {'✓' if cond else '✗'} {name}")

    # Ceilings raised
    chk("target_distance_max raised", raised.initial_states.target_distance_max == s.target_distance_max
        and raised.initial_states.target_distance_max != base.initial_states.target_distance_max)
    chk("zoom_initial_max_end raised", raised.initial_states.zoom_initial_max_end == s.zoom_initial_max_end)
    chk("cylinder_diameter_max raised", raised.initial_states.cylinder_diameter_max == s.cylinder_diameter_max)
    chk("target max_speed_end raised", raised.target_controller.max_speed_end == s.target_max_speed_end)
    chk("update_interval_min_end lowered", raised.target_controller.update_interval_min_end == s.target_update_interval_min_end)
    chk("dropout_start_step moved earlier", raised.curriculum.dropout_start_step == s.dropout_start_step)
    chk("dropout_prob raised", raised.delay_system_params.dropout_prob == s.dropout_prob)

    # Floors untouched
    chk("target_distance_min floor untouched",
        raised.initial_states.target_distance_min == base.initial_states.target_distance_min)
    chk("max_speed_start floor untouched",
        raised.target_controller.max_speed_start == base.target_controller.max_speed_start)
    chk("cylinder_diameter_min floor untouched",
        raised.initial_states.cylinder_diameter_min == base.initial_states.cylinder_diameter_min)
    chk("zoom_initial_min floor untouched",
        raised.initial_states.zoom_initial_min == base.initial_states.zoom_initial_min)

    # Ceiling strictly raised above the (t046-tightened) baseline
    chk("target_distance_max strictly raised above baseline",
        raised.initial_states.target_distance_max > base.initial_states.target_distance_max)
    chk("target max_speed_end strictly raised above baseline",
        raised.target_controller.max_speed_end > base.target_controller.max_speed_end)
    print(f"  (baseline target_distance_max={base.initial_states.target_distance_max}, "
          f"raised={raised.initial_states.target_distance_max})")

    return all(ok for _, ok in checks)


def part_b_env_runs():
    print("\n[Part B] Flag-on env builds + steps + logs Coop/*")
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = 0
    cfg.enable_track_loss_scenario = True        # overlay applied in env __init__ (before super)
    cfg.cooperation_metrics.enable = True
    cfg.cooperation_metrics.collect_episode_values = True   # eval buffer-stash path (Slice 4)
    cfg.use_debug_initial_step = True            # force full curriculum progress -> ceiling active
    cfg.debug_initial_step = 300000
    cfg.episode_length_s = 2.0
    cfg.use_flight_scene = False

    env = gym.make(args_cli.task, cfg=cfg)
    uenv = env.unwrapped
    agents = uenv.possible_agents
    device = uenv.device

    # Confirm the overlay actually took effect on the live cfg the env holds.
    live_dist = uenv.cfg.initial_states.target_distance_max
    live_speed = uenv.cfg.target_controller.max_speed_end
    print(f"  live target_distance_max = {live_dist} (expect 40.0)")
    print(f"  live target max_speed_end = {live_speed} (expect 8.0)")
    overlay_live = (live_dist == 40.0 and live_speed == 8.0)
    print(f"  {'✓' if overlay_live else '✗'} overlay applied to the live env cfg")

    def zero_actions():
        return {a: torch.zeros((uenv.num_envs, int(uenv.cfg.action_spaces[a])), device=device) for a in agents}

    env.reset()
    coop = {}
    for _ in range(200):
        env.step(zero_actions())
        log = uenv.extras.get("log", {})
        coop = {k: v for k, v in log.items() if k.startswith("Coop/")}
        if coop:
            break
    coop_ok = len(coop) > 0
    print(f"  {'✓' if coop_ok else '✗'} Coop/* keys logged under raised scenario")
    if "Coop/track_loss_event_rate" in coop:
        print(f"  (observed Coop/track_loss_event_rate = {float(coop['Coop/track_loss_event_rate']):.3f} "
              f"— zero-action agents; gate run uses the policy)")
    # Slice 4: the eval buffer-stash path populates as episodes complete.
    buf_len = len(getattr(uenv, "_reacq_episode_buffer", []))
    buf_ok = buf_len > 0
    print(f"  {'✓' if buf_ok else '✗'} eval episode-value buffer populated ({buf_len} batches)")
    env.close()
    return overlay_live and coop_ok and buf_ok


def main():
    a = part_a_config_assertions()
    b = part_b_env_runs()
    ok = a and b
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  (Part A config={a}, Part B runtime={b})")
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
