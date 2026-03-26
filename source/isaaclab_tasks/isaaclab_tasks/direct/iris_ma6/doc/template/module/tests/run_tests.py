#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""{{ModuleName}} Test Suite.

Usage:
    ./isaaclab.sh -p path/to/run_tests.py
    ./isaaclab.sh -p path/to/run_tests.py --test-verbose
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run {{ModuleName}} test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- NOW import everything else ---
import sys
import torch
import traceback
from datetime import datetime

VERBOSE = args_cli.test_verbose


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  \u2713 {name}")

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  \u2717 {name}")
        for line in error.split("\n")[:5]:
            print(f"    {line}")

    def add_error(self, name: str, error: str):
        self.errors.append((name, error))
        print(f"  ERROR {name}\n    {error[:200]}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print(f"\n{'='*80}\nTEST SUMMARY\n{'='*80}")
        print(f"Total: {total}  Passed: {len(self.passed)}  Failed: {len(self.failed)}  Errors: {len(self.errors)}")
        if self.failed:
            print(f"\n{'-'*80}\nFAILED:")
            for name, err in self.failed:
                print(f"  {name}: {err[:200]}")
        if self.errors:
            print(f"\n{'-'*80}\nERRORS:")
            for name, err in self.errors:
                print(f"  {name}: {err[:200]}")
        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def run_tests(results: TestResults, device: torch.device):
    print(f"\n{'='*80}\nTesting {{ModuleName}}\n{'='*80}")
    num_envs = 16

    # --- Test: Initialization ---
    try:
        # cfg = {{ModuleNameCfg}}()
        # module = {{ModuleName}}(cfg, num_envs=num_envs, device=device)
        # assert module.num_envs == num_envs
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", traceback.format_exc())
        return  # Can't continue without init

    # --- Test: Core computation (simple values) ---
    try:
        # simple_input = torch.zeros(num_envs, 3, device=device)
        # result = module.compute(simple_input)
        # assert result.shape == (num_envs, ...)
        results.add_pass("Core computation \u2014 simple values")
    except Exception as e:
        results.add_fail("Core computation \u2014 simple values", traceback.format_exc())

    # --- Test: Core computation (realistic values) ---
    try:
        # realistic_input = torch.randn(num_envs, 3, device=device)
        # result = module.compute(realistic_input)
        # assert torch.isfinite(result).all()
        results.add_pass("Core computation \u2014 realistic values")
    except Exception as e:
        results.add_fail("Core computation \u2014 realistic values", traceback.format_exc())

    # --- Test: Reset ---
    try:
        # env_ids = torch.tensor([0, 2, 5], device=device)
        # module.reset(env_ids)
        results.add_pass("Reset")
    except Exception as e:
        results.add_fail("Reset", traceback.format_exc())


def main():
    print(f"{'='*80}\n{{MODULE_NAME}} TEST SUITE\n{'='*80}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = TestResults()
    try:
        run_tests(results, device)
    except Exception as e:
        results.add_error("Suite-level", traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
