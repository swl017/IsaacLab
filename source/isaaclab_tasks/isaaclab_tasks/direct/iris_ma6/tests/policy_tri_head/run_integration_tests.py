#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env-integration test runner for the policy tri head (ticket 031).

Launches Isaac Sim via AppLauncher (per CLAUDE.md template) and runs tests
that exercise the real iris_ma6 env + MAPPOWithAux trainer end-to-end.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/tests/policy_tri_head/run_integration_tests.py
    ./isaaclab.sh -p .../run_integration_tests.py --test-verbose
"""
import argparse

from isaaclab.app import AppLauncher

# AppLauncher must run before any other Isaac Lab / torch import.
parser = argparse.ArgumentParser(description="Run policy tri head env-integration tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose debug output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- post-launch imports ---
import os
import sys
import traceback
from datetime import datetime

import torch

# Make the trainer-side module (mappo_with_aux.py + train_mappo_with_aux_hydra.py) importable.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKRL_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "..", "..", "..", "..", "scripts", "reinforcement_learning", "skrl"))
if SKRL_DIR not in sys.path:
    sys.path.insert(0, SKRL_DIR)
# Make the local test modules importable as plain `import test_*`.
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}", flush=True)

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}", flush=True)
        for line in error.split("\n")[:6]:
            print(f"    {line}", flush=True)

    def add_error(self, name: str, error: str):
        self.errors.append((name, error))
        print(f"  ERROR {name}", flush=True)
        for line in error.split("\n")[:6]:
            print(f"    {line}", flush=True)

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 80)
        if total == 0:
            print("No tests ran.")
            return False
        print(f"Total: {total}  Passed: {len(self.passed)}  Failed: {len(self.failed)}  Errors: {len(self.errors)}")
        if self.failed:
            print("\n" + "-" * 80)
            print("FAILED:")
            for name, err in self.failed:
                print(f"\n  {name}:")
                for line in err.split("\n")[:25]:
                    print(f"    {line}")
        if self.errors:
            print("\n" + "-" * 80)
            print("ERRORS:")
            for name, err in self.errors:
                print(f"\n  {name}:")
                for line in err.split("\n")[:25]:
                    print(f"    {line}")
        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def _run_one(label, fn, results, *args, **kwargs):
    print(f"\n{'=' * 80}\nRunning: {label}\n{'=' * 80}", flush=True)
    try:
        fn(results, *args, **kwargs)
    except Exception:
        results.add_error(label, traceback.format_exc())


def main():
    print("=" * 80)
    print("POLICY TRI HEAD — ENV-INTEGRATION TEST SUITE")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    if device.type == "cuda":
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    results = TestResults()

    # Slice 2 tests — shared-env strategy:
    # Run smoke first (builds its env), keep it open, then reuse it for the
    # supervision contract tests. Avoids a second gym.make() inside the same
    # process, which deadlocks under Isaac Sim USD-cache contention when
    # other kit processes are running.
    from test_world_frame_supervision import run_supervision_tests
    from test_preflight_smoke import run_preflight_smoke

    smoke_env = None
    print(f"\n{'=' * 80}\nRunning: Preflight smoke (≥128 steps, aux_loss_scale=0.1)\n{'=' * 80}", flush=True)
    try:
        smoke_env = run_preflight_smoke(
            results, device, verbose=args_cli.test_verbose, return_env=True,
        )
    except Exception:
        results.add_error("Preflight smoke", traceback.format_exc())

    if smoke_env is not None:
        print(f"\n{'=' * 80}\nRunning: World-frame supervision (re-using smoke env)\n{'=' * 80}", flush=True)
        try:
            run_supervision_tests(
                results, device, verbose=args_cli.test_verbose, existing_env=smoke_env,
            )
        except Exception:
            results.add_error("World-frame supervision", traceback.format_exc())
        finally:
            try:
                smoke_env.close()
            except Exception:
                pass
    else:
        print("\n[SKIP] Smoke pipeline failed to construct; skipping supervision tests.", flush=True)
        results.add_error("World-frame supervision", "skipped — smoke pipeline did not return an env")

    success = results.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
    simulation_app.close()
