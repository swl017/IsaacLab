#!/usr/bin/env python3
"""Pure-PyTorch test runner for the policy tri head (ticket 031, slice 1+3).

These tests do NOT require Isaac Sim — they exercise model architecture and
loss arithmetic only. Env-integration tests live separately at
iris_ma6/tests/policy_tri_head/run_integration_tests.py.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/tests/policy_tri_head/run_tests.py
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/tests/policy_tri_head/run_tests.py --test-verbose
"""
import argparse
import os
import sys
import traceback
from datetime import datetime

import torch

# Make `mappo_with_aux` importable (its parent dir is the skrl/ folder).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SKRL_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SKRL_DIR not in sys.path:
    sys.path.insert(0, SKRL_DIR)


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        for line in error.split("\n")[:6]:
            print(f"    {line}")

    def add_error(self, name: str, error: str):
        self.errors.append((name, error))
        print(f"  ERROR {name}")
        for line in error.split("\n")[:6]:
            print(f"    {line}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
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
                for line in err.split("\n")[:20]:
                    print(f"    {line}")
        if self.errors:
            print("\n" + "-" * 80)
            print("ERRORS:")
            for name, err in self.errors:
                print(f"\n  {name}:")
                for line in err.split("\n")[:20]:
                    print(f"    {line}")
        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def _run_one(label, fn, results, *args, **kwargs):
    """Run a test-suite entry function, swallowing any uncaught exception
    into the results.errors bucket so a single broken module does not abort
    the whole runner."""
    print(f"\n{'=' * 80}\nRunning: {label}\n{'=' * 80}")
    try:
        fn(results, *args, **kwargs)
    except Exception:
        results.add_error(label, traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Run policy tri head pure-PyTorch tests.")
    parser.add_argument("--test-verbose", action="store_true", help="Verbose debug output.")
    args = parser.parse_args()

    print("=" * 80)
    print("POLICY TRI HEAD — PURE-PYTORCH TEST SUITE")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now():%Y-%m-%d %H:%M:%S}")

    results = TestResults()

    # Slice 1 tests
    from test_actor_with_tri_head_shape import run_shape_tests
    from test_no_skip_connection import run_no_skip_tests
    from test_input_parity import run_input_parity_tests
    from test_aux_loss_masking import run_masking_tests
    from test_invalid_triangulation_no_grad import run_no_grad_tests

    _run_one("Shape", run_shape_tests, results, device, args.test_verbose)
    _run_one("No skip connection", run_no_skip_tests, results, device, args.test_verbose)
    _run_one("Input parity", run_input_parity_tests, results, device, args.test_verbose)
    _run_one("Aux loss masking", run_masking_tests, results, device, args.test_verbose)
    _run_one("Invalid triangulation no-grad", run_no_grad_tests, results, device, args.test_verbose)

    # Slice 2 tests
    from test_memory_tensor_registration import run_registration_tests
    _run_one("Memory tensor registration", run_registration_tests, results, device, args.test_verbose)

    # Slice 3 tests
    from test_loss_scale_zero_regression import run_bit_exact_tests
    _run_one("Loss-scale-zero bit-exact regression", run_bit_exact_tests, results, device, args.test_verbose)

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    # Ensure local test modules are importable as plain `import test_*` (not as a package).
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    main()
