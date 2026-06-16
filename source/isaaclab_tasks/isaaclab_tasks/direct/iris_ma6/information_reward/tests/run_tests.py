#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test suite for information_reward.InformationReward (ticket 050, Slice B).

    ./isaaclab.sh -p source/.../iris_ma6/information_reward/tests/run_tests.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run information_reward test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import traceback
from datetime import datetime

from isaaclab_tasks.direct.iris_ma6.information_reward import (
    InformationReward,
    InformationRewardCfg,
)


class TestResults:
    def __init__(self):
        self.passed, self.failed = [], []

    def add_pass(self, n):
        self.passed.append(n)
        print(f"  ✓ {n}")

    def add_fail(self, n, e):
        self.failed.append((n, e))
        print(f"  ✗ {n}")
        for line in e.split("\n")[:6]:
            print(f"    {line}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*80}\nTotal: {total}  Passed: {len(self.passed)}  Failed: {len(self.failed)}\n{'='*80}")
        return not self.failed


def _cfg(**kw):
    base = dict(enabled=True, sigma_theta=0.01, sigma_prior=40.0, z_prior_scale=4.0,
                quality_const_c=10.0, in_plane_only=False, info_scale_max=8.0)
    base.update(kw)
    return InformationRewardCfg(**base)


def _inputs(cam_list, valid_list, device):
    """target at origin; bearings = normalize(target - cam)."""
    cam = torch.tensor([cam_list], dtype=torch.float32, device=device)   # [1, A, 3]
    target = torch.zeros(1, 3, device=device)
    d = torch.nn.functional.normalize(target.unsqueeze(1) - cam, dim=-1)  # [1, A, 3]
    valid = torch.tensor([valid_list], dtype=torch.bool, device=device)  # [1, A]
    return d, cam, target, valid


def run_tests(results, device):
    print(f"\n{'='*80}\nTesting InformationReward\n{'='*80}")

    # Reusable geometries (target at origin). perp = well-separated (bearings -x, -y);
    # collin = exactly parallel bearings (both -x) — the degenerate GDOP case.
    perp = [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]]
    collin = [[30.0, 0.0, 0.0], [60.0, 0.0, 0.0]]

    def quality(cam_list, valid_list):
        A = len(cam_list)
        ir = InformationReward(_cfg(), num_envs=1, num_agents=A, device=device)
        out = ir.compute(*_inputs(cam_list, valid_list, device))
        return out

    # --- Test 1: 0 valid bearings — defined, no NaN, r_diff all 0 ---
    try:
        out = quality(perp, [False, False])
        assert torch.isfinite(out["team_quality"]).all(), out["team_quality"]
        assert torch.allclose(out["r_diff"], torch.zeros_like(out["r_diff"])), out["r_diff"]
        results.add_pass("0 valid bearings: defined, r_diff=0")
    except Exception:
        results.add_fail("0 valid bearings: defined, r_diff=0", traceback.format_exc())

    # --- Test 2: 1 valid bearing — defined, quality(1) > quality(0), r_diff for valid only ---
    try:
        q0 = quality(perp, [False, False])["team_quality"].item()
        out1 = quality(perp, [True, False])
        q1 = out1["team_quality"].item()
        assert q1 > q0, f"q1={q1} q0={q0}"
        assert out1["r_diff"][0, 0].item() > 0, out1["r_diff"]
        assert out1["r_diff"][0, 1].item() == 0.0, out1["r_diff"]
        results.add_pass("1 valid bearing: quality rises, r_diff only for valid")
    except Exception:
        results.add_fail("1 valid bearing: quality rises, r_diff only for valid", traceback.format_exc())

    # --- Test 3: 2 well-separated bearings collapse Sigma (>> 1 bearing) ---
    try:
        q1 = quality(perp, [True, False])["team_quality"].item()
        out2 = quality(perp, [True, True])
        q2 = out2["team_quality"].item()
        assert q2 > 5.0 * q1, f"q2={q2} q1={q1} (expected large collapse)"
        assert (out2["r_diff"] > 0).all(), out2["r_diff"]
        results.add_pass("2 well-separated bearings collapse Sigma")
    except Exception:
        results.add_fail("2 well-separated bearings collapse Sigma", traceback.format_exc())

    # --- Test 4: 2 collinear bearings ~ 1 bearing (GDOP), << well-separated ---
    try:
        q1 = quality(perp, [True, False])["team_quality"].item()
        q2sep = quality(perp, [True, True])["team_quality"].item()
        q2col = quality(collin, [True, True])["team_quality"].item()
        assert q2col < 0.25 * q2sep, f"q2col={q2col} q2sep={q2sep}"
        assert abs(q2col - q1) / q1 < 0.5, f"collinear {q2col} should be ~1-bearing {q1}"
        results.add_pass("2 collinear bearings ~ 1 bearing (GDOP)")
    except Exception:
        results.add_fail("2 collinear bearings ~ 1 bearing (GDOP)", traceback.format_exc())

    # --- Test 5: r_diff ranks by marginal info (orthogonal contributor > redundant) ---
    try:
        # agent0,1 nearly redundant (both ~ -x bearing); agent2 orthogonal (~ -y) supplies x-info.
        cams = [[30.0, 0.0, 5.0], [32.0, 0.0, 5.0], [0.0, 30.0, 5.0]]
        ir = InformationReward(_cfg(), num_envs=1, num_agents=3, device=device)
        out = ir.compute(*_inputs(cams, [True, True, True], device))
        rd = out["r_diff"][0]
        assert rd[2].item() > rd[0].item(), f"orthogonal {rd[2]} should exceed redundant {rd[0]}"
        results.add_pass("r_diff ranks by marginal info (orthogonal > redundant)")
    except Exception:
        results.add_fail("r_diff ranks by marginal info (orthogonal > redundant)", traceback.format_exc())

    # --- Test 6: in_plane_only mode runs, finite, defined at 0/1/2 ---
    try:
        ir = InformationReward(_cfg(in_plane_only=True), num_envs=1, num_agents=2, device=device)
        for vlist in ([False, False], [True, False], [True, True]):
            out = ir.compute(*_inputs(perp, vlist, device))
            assert torch.isfinite(out["team_quality"]).all()
            assert torch.isfinite(out["r_diff"]).all()
        results.add_pass("in_plane_only mode: finite at 0/1/2 bearings")
    except Exception:
        results.add_fail("in_plane_only mode: finite at 0/1/2 bearings", traceback.format_exc())

    # --- Test 7: invalid mask ignores the bearing (r_diff=0, quality == absent) ---
    try:
        # agent1 has a real bearing but valid=False -> must equal the 1-valid-agent case.
        q_masked = quality(perp, [True, False])["team_quality"].item()
        out = quality(perp, [True, False])
        assert out["r_diff"][0, 1].item() == 0.0
        # Same geometry but agent1 valid=True changes quality (sanity: mask actually matters).
        q_both = quality(perp, [True, True])["team_quality"].item()
        assert q_both > q_masked
        results.add_pass("invalid mask ignores bearing")
    except Exception:
        results.add_fail("invalid mask ignores bearing", traceback.format_exc())

    # --- Test 8: batched multi-env, no NaN ---
    try:
        A = 3
        ir = InformationReward(_cfg(), num_envs=8, num_agents=A, device=device)
        bearings = torch.nn.functional.normalize(torch.randn(8, A, 3, device=device), dim=-1)
        cam = torch.randn(8, A, 3, device=device) * 20.0
        target = torch.zeros(8, 3, device=device)
        valid = torch.ones(8, A, dtype=torch.bool, device=device)
        out = ir.compute(bearings, cam, target, valid)
        assert torch.isfinite(out["team_quality"]).all() and torch.isfinite(out["r_diff"]).all()
        assert out["team_quality"].shape == (8,) and out["r_diff"].shape == (8, A)
        results.add_pass("batched multi-env, finite, correct shapes")
    except Exception:
        results.add_fail("batched multi-env, finite, correct shapes", traceback.format_exc())


    # --- Test 9: near-target / degenerate geometry does not produce a singular FIM ---
    try:
        ir = InformationReward(_cfg(), num_envs=4, num_agents=2, device=device)
        # agents essentially ON the target (r -> 0) => huge weights; the counterfactual must NOT
        # become singular (regression for the leave-one-out subtraction-cancellation bug).
        cam = torch.full((4, 2, 3), 1e-4, device=device)
        target = torch.zeros(4, 3, device=device)
        bearings = torch.nn.functional.normalize(torch.randn(4, 2, 3, device=device), dim=-1)
        valid = torch.ones(4, 2, dtype=torch.bool, device=device)
        out = ir.compute(bearings, cam, target, valid)
        assert torch.isfinite(out["team_quality"]).all(), out["team_quality"]
        assert torch.isfinite(out["r_diff"]).all(), out["r_diff"]
        results.add_pass("Near-target geometry: no singular FIM, finite outputs")
    except Exception:
        results.add_fail("Near-target geometry: no singular FIM, finite outputs", traceback.format_exc())


def main():
    print(f"{'='*80}\nINFORMATION_REWARD TEST SUITE\n{'='*80}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    results = TestResults()
    try:
        run_tests(results, device)
    except Exception:
        print(traceback.format_exc())
    ok = results.summary()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
