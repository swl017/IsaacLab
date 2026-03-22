#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a full experiment suite (batch of named experiments) for iris_ma6.

Usage:
    # Preview commands without running
    ./isaaclab.sh -p .../experiments/run_suite.py --suite iros2026_must --dry_run

    # Run all experiments in the suite
    ./isaaclab.sh -p .../experiments/run_suite.py --suite iros2026_must

    # Override seeds and num_envs
    ./isaaclab.sh -p .../experiments/run_suite.py --suite iros2026_sweeps \\
        --seeds 42,123,456 --num_envs 2048

    # List available suites
    ./isaaclab.sh -p .../experiments/run_suite.py --list
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Run a full iris_ma6 experiment suite.")
parser.add_argument("--suite", type=str, default=None, help="Suite name from registry")
parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (overrides experiment defaults)")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs for all experiments")
parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
parser.add_argument("--list", action="store_true", help="List available suites and exit")
args = parser.parse_args()

# These imports don't need Isaac Sim
sys.path.insert(0, os.path.dirname(__file__))
from experiment_registry import get_suite, get_experiment, list_suites, list_experiments


def main():
    if args.list:
        print("\n=== Available Suites ===")
        for name in list_suites():
            suite = get_suite(name)
            print(f"  {name:25s} ({len(suite.experiments)} experiments)")
            for exp_name in suite.experiments:
                exp = get_experiment(exp_name)
                print(f"    - {exp_name:40s} [{exp.group}]")
        return

    if args.suite is None:
        parser.error("--suite is required (or use --list)")

    suite = get_suite(args.suite)
    print(f"\n[SUITE] Running: {suite.name}")
    print(f"[SUITE] Experiments: {len(suite.experiments)}")

    run_experiment_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment.py")
    isaaclab_sh = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "..", "..",
        "isaaclab.sh",
    )
    isaaclab_sh = os.path.normpath(isaaclab_sh)

    commands = []
    for exp_name in suite.experiments:
        exp = get_experiment(exp_name)

        # Skip eval-only experiments
        if exp_name == "baseline_greedy":
            print(f"[SUITE] Skipping {exp_name} (eval-only)")
            continue

        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else exp.seeds

        for seed in seeds:
            cmd = [
                isaaclab_sh, "-p", run_experiment_script,
                "--experiment", exp_name,
                "--seed", str(seed),
            ]
            if args.num_envs:
                cmd += ["--num_envs", str(args.num_envs)]

            commands.append((exp_name, seed, cmd))

    print(f"\n[SUITE] Total training runs: {len(commands)}")
    print("-" * 80)
    for exp_name, seed, cmd in commands:
        print(f"  {exp_name:40s} seed={seed}")
    print("-" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for _, _, cmd in commands:
            print(f"  {' '.join(cmd)}")
        print("\n[DRY RUN] No commands executed.")
        return

    results = []
    for i, (exp_name, seed, cmd) in enumerate(commands):
        print(f"\n{'='*80}")
        print(f"[SUITE] [{i+1}/{len(commands)}] Starting: {exp_name} seed={seed}")
        print(f"{'='*80}")
        result = subprocess.run(cmd, check=False)
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        results.append((exp_name, seed, status))
        print(f"[SUITE] [{i+1}/{len(commands)}] {exp_name} seed={seed}: {status}")

    print(f"\n{'='*80}")
    print(f"[SUITE] Summary for {suite.name}")
    print(f"{'='*80}")
    for exp_name, seed, status in results:
        print(f"  {exp_name:40s} seed={seed:5d}  {status}")

    failed = [r for r in results if "FAILED" in r[2]]
    if failed:
        print(f"\n[SUITE] {len(failed)}/{len(results)} runs FAILED")
    else:
        print(f"\n[SUITE] All {len(results)} runs completed successfully")


if __name__ == "__main__":
    main()
