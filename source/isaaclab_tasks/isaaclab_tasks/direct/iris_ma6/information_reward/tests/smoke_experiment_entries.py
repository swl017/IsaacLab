#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slice-4 smoke (ticket 050 Slice B): the b050 experiment entries load and their env_overrides
resolve onto IrisMA6TestEnvCfg (no env build needed).

    ./isaaclab.sh -p .../information_reward/tests/smoke_experiment_entries.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Slice-4 b050 experiment-entry smoke")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

from isaaclab_tasks.direct.iris_ma6.experiments.experiment_registry import get_experiment, list_experiments
from isaaclab_tasks.direct.iris_ma6.experiments.env_overrides import apply_env_overrides
from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import IrisMA6TestEnvCfg


def main():
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        print(f"  {'✓' if cond else '✗'} {name}")

    names = list_experiments("T050B")
    chk("B050 group lists both entries", set(names) == {"t050b_team_reward", "t050b_baseline_no_info"})

    for name, info_on in [("t050b_team_reward", True), ("t050b_baseline_no_info", False)]:
        exp = get_experiment(name)
        cfg = IrisMA6TestEnvCfg()
        apply_env_overrides(cfg, exp.env_overrides)
        chk(f"{name}: information_reward.enabled == {info_on}", cfg.information_reward.enabled is info_on)
        chk(f"{name}: enable_track_loss_scenario", cfg.enable_track_loss_scenario is True)
        chk(f"{name}: cooperation_metrics.enable", cfg.cooperation_metrics.enable is True)
        # These three are read at env __init__/runtime (not __post_init__), so post-construction
        # override is correct — unlike enable_critic_gt_target which we deliberately omitted.

    ok = all(c for _, c in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
