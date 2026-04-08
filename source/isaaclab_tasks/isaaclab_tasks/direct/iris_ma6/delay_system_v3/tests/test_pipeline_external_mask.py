#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DelayPipelineV3 with external burst dropout mask.

Verifies that:
1. Pipeline respects external mask (all-True → always hold, all-False → always pass)
2. Pipeline without external mask behaves identically to before (backward compat)
3. process() passes burst_dropout_mask through correctly
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run pipeline external mask tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import traceback
from datetime import datetime

from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_pipeline_v3 import DelayPipelineV3
from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
    DelayPipelineCfgV3,
    LatencyCfg,
    StalenessCfg,
    DropoutCfg,
    DistributionCfg,
)

VERBOSE = args_cli.test_verbose


class TestResults:
    def __init__(self):
        self.passed, self.failed, self.errors = [], [], []

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
        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def make_pipeline(num_envs, device, dropout_prob=0.5):
    """Create a simple pipeline with dropout enabled, minimal latency.

    Sets mode to 'fixed' so the pipeline runs through latency+dropout stages
    (mode='none' shortcuts past all stages).
    """
    cfg = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(type="constant", value=0.0),
            min_steps=0,
        ),
        staleness=StalenessCfg(enabled=False),
        dropout=DropoutCfg(enabled=True, probability=dropout_prob),
    )
    pipe = DelayPipelineV3(cfg, num_envs=num_envs, device=device, dt=0.04, data_shape=(3,))
    pipe.set_mode("fixed", progress=1.0)
    return pipe


def run_tests(results: TestResults, device: torch.device):
    print(f"\n{'='*80}\nTesting Pipeline External Mask\n{'='*80}")

    num_envs = 16

    # ==================================================================
    # Test 1: External mask all-True (always drop) → data frozen
    # ==================================================================
    try:
        pipe = make_pipeline(num_envs, device, dropout_prob=0.0)  # Internal prob=0 (no i.i.d. drop)
        t = torch.zeros(num_envs, device=device)

        # First advance: initialize
        data1 = torch.ones(num_envs, 3, device=device) * 1.0
        pipe.advance(data1, t, t)
        out1, _ = pipe.query(allow_dropout=True)

        # Second advance with external mask = all True (force drop)
        t2 = t + 0.04
        data2 = torch.ones(num_envs, 3, device=device) * 99.0
        mask_all_drop = torch.ones(num_envs, dtype=torch.bool, device=device)
        pipe.advance(data2, t2, t2, burst_dropout_mask=mask_all_drop)
        out2, _ = pipe.query(allow_dropout=True)

        # With all-drop mask, output should hold previous data (1.0), not new data (99.0)
        assert torch.allclose(out2, data1), (
            f"Expected held data {data1[0].tolist()}, got {out2[0].tolist()}"
        )
        results.add_pass("External mask all-True → data frozen")
    except Exception as e:
        results.add_fail("External mask all-True → data frozen", traceback.format_exc())

    # ==================================================================
    # Test 2: External mask all-False (never drop) → data passes through
    # ==================================================================
    try:
        pipe = make_pipeline(num_envs, device, dropout_prob=1.0)  # Internal prob=1.0 (always drop)
        t = torch.zeros(num_envs, device=device)

        # First advance: initialize
        data1 = torch.ones(num_envs, 3, device=device) * 1.0
        pipe.advance(data1, t, t)

        # Second advance with external mask = all False (force pass)
        t2 = t + 0.04
        data2 = torch.ones(num_envs, 3, device=device) * 42.0
        mask_no_drop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        pipe.advance(data2, t2, t2, burst_dropout_mask=mask_no_drop)
        out2, _ = pipe.query(allow_dropout=True)

        # With all-pass mask, output should be new data (42.0), overriding internal 100% drop
        assert torch.allclose(out2, data2), (
            f"Expected new data {data2[0].tolist()}, got {out2[0].tolist()}"
        )
        results.add_pass("External mask all-False → data passes (overrides internal)")
    except Exception as e:
        results.add_fail("External mask all-False → data passes (overrides internal)", traceback.format_exc())

    # ==================================================================
    # Test 3: No external mask → uses internal sampler (backward compat)
    # ==================================================================
    try:
        pipe = make_pipeline(num_envs, device, dropout_prob=0.0)  # 0% drop
        t = torch.zeros(num_envs, device=device)

        data1 = torch.ones(num_envs, 3, device=device) * 1.0
        pipe.advance(data1, t, t)

        t2 = t + 0.04
        data2 = torch.ones(num_envs, 3, device=device) * 77.0
        pipe.advance(data2, t2, t2)  # No burst_dropout_mask
        out2, _ = pipe.query(allow_dropout=True)

        # With internal 0% drop and no external mask, data should pass
        assert torch.allclose(out2, data2), (
            f"Expected new data {data2[0].tolist()}, got {out2[0].tolist()}"
        )
        results.add_pass("No external mask → backward compatible (internal sampler)")
    except Exception as e:
        results.add_fail("No external mask → backward compatible (internal sampler)", traceback.format_exc())

    # ==================================================================
    # Test 4: Pre-dropout cache unaffected by external mask
    # ==================================================================
    try:
        pipe = make_pipeline(num_envs, device, dropout_prob=0.0)
        t = torch.zeros(num_envs, device=device)

        data1 = torch.ones(num_envs, 3, device=device) * 1.0
        pipe.advance(data1, t, t)

        t2 = t + 0.04
        data2 = torch.ones(num_envs, 3, device=device) * 50.0
        mask_all_drop = torch.ones(num_envs, dtype=torch.bool, device=device)
        pipe.advance(data2, t2, t2, burst_dropout_mask=mask_all_drop)

        # Pre-dropout query should return new data regardless of mask
        out_no_drop, _ = pipe.query(allow_dropout=False)
        assert torch.allclose(out_no_drop, data2), (
            f"Pre-dropout should be {data2[0].tolist()}, got {out_no_drop[0].tolist()}"
        )
        results.add_pass("Pre-dropout cache unaffected by external mask")
    except Exception as e:
        results.add_fail("Pre-dropout cache unaffected by external mask", traceback.format_exc())

    # ==================================================================
    # Test 5: process() passes burst_dropout_mask through
    # ==================================================================
    try:
        pipe = make_pipeline(num_envs, device, dropout_prob=0.0)
        t = torch.zeros(num_envs, device=device)

        data1 = torch.ones(num_envs, 3, device=device) * 1.0
        pipe.process(data1, t, t)

        t2 = t + 0.04
        data2 = torch.ones(num_envs, 3, device=device) * 99.0
        mask_all_drop = torch.ones(num_envs, dtype=torch.bool, device=device)
        out2, _ = pipe.process(data2, t2, t2, burst_dropout_mask=mask_all_drop)

        assert torch.allclose(out2, data1), "process() should pass burst mask through to advance"
        results.add_pass("process() passes burst_dropout_mask correctly")
    except Exception as e:
        results.add_fail("process() passes burst_dropout_mask correctly", traceback.format_exc())


def main():
    print(f"{'='*80}\nPIPELINE EXTERNAL MASK TEST SUITE\n{'='*80}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Time: {datetime.now():%Y-%m-%d %H:%M:%S}")

    results = TestResults()
    try:
        run_tests(results, device)
    except Exception as e:
        results.add_error("Suite-level", traceback.format_exc())

    sys.exit(0 if results.print_summary() else 1)


if __name__ == "__main__":
    main()
