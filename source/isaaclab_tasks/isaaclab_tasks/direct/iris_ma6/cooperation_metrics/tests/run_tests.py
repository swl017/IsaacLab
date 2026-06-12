#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test suite for cooperation_metrics.ReacquisitionTracker (ticket 050, Slice A).

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cooperation_metrics/tests/run_tests.py
    ./isaaclab.sh -p .../run_tests.py --test-verbose
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run cooperation_metrics test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Verbose debug output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- NOW import everything else ---
import sys
import torch
import traceback
from datetime import datetime

from isaaclab_tasks.direct.iris_ma6.cooperation_metrics import (
    ReacquisitionTracker,
    ReacquisitionTrackerCfg,
)

VERBOSE = args_cli.test_verbose
STEP_DT = 0.1  # -> tau_min_s=0.2 == 2 steps, tau_hold_s=0.2 == 2 steps


class TestResults:
    def __init__(self):
        self.passed, self.failed, self.errors = [], [], []

    def add_pass(self, name):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def add_fail(self, name, error):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
        for line in error.split("\n")[:6]:
            print(f"    {line}")

    def add_error(self, name, error):
        self.errors.append((name, error))
        print(f"  ERROR {name}\n    {error[:300]}")

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print(f"\n{'='*80}\nTEST SUMMARY\n{'='*80}")
        print(
            f"Total: {total}  Passed: {len(self.passed)}  "
            f"Failed: {len(self.failed)}  Errors: {len(self.errors)}"
        )
        if self.failed:
            print(f"\n{'-'*80}\nFAILED:")
            for name, err in self.failed:
                print(f"  {name}: {err.splitlines()[-1] if err else ''}")
        print("=" * 80)
        return len(self.failed) == 0 and len(self.errors) == 0


def _make_tracker(device, num_envs=1, num_agents=2, **cfg_kwargs):
    cfg = ReacquisitionTrackerCfg(
        enable=True, tau_min_s=0.2, tau_hold_s=0.2, **cfg_kwargs
    )
    return ReacquisitionTracker(cfg, num_envs, num_agents, device, STEP_DT)


def _step(tracker, t, detected, device, *, age=None, rng=1000.0, fov=0.5, vmax=5.0, bearing=None):
    """Feed one step. `detected` is a bool tensor [N, A]. Default geometry keeps the gate open."""
    N, A = detected.shape
    bbox_age = torch.zeros(N, A, device=device) if age is None else age
    target_range = torch.full((N, A), rng, device=device)
    fov_eff_half = torch.full((N, A), fov, device=device)
    v_max = torch.full((N,), vmax, device=device)
    tracker.update(
        detected_nonempty=detected,
        bbox_age=bbox_age,
        target_range=target_range,
        fov_eff_half=fov_eff_half,
        v_max=v_max,
        agent_bearing_w=bearing,
        t=t,
    )


def _b(rows, device):
    return torch.tensor(rows, dtype=torch.bool, device=device)


def run_tests(results, device):
    print(f"\n{'='*80}\nTesting ReacquisitionTracker\n{'='*80}")
    print(f"Config: step_dt={STEP_DT}, tau_min=2 steps, tau_hold=2 steps")
    e0 = torch.tensor([0], device=device)

    # --- Test 1: gate at AoI=0 reduces to in-frame detection ---
    try:
        tr = _make_tracker(device)
        d, stale = tr._effective_track(
            _b([[True, False]], device),
            torch.zeros(1, 2, device=device),
            torch.full((1, 2), 1000.0, device=device),
            torch.full((1, 2), 0.5, device=device),
            torch.full((1,), 5.0, device=device),
        )
        assert d.tolist() == [[True, False]], f"d={d.tolist()}"
        assert stale.tolist() == [[False, False]], f"stale={stale.tolist()}"
        results.add_pass("Gate at AoI=0 == detection")
    except Exception:
        results.add_fail("Gate at AoI=0 == detection", traceback.format_exc())

    # --- Test 2: gate trips on staleness (nonempty but old) ---
    try:
        tr = _make_tracker(device)
        # reach = 5 * 200 = 1000 m; gate = 1000 * tan(0.01) ~= 10 m  -> trips
        d, stale = tr._effective_track(
            _b([[True, True]], device),
            torch.full((1, 2), 200.0, device=device),
            torch.full((1, 2), 1000.0, device=device),
            torch.full((1, 2), 0.01, device=device),
            torch.full((1,), 5.0, device=device),
        )
        assert d.tolist() == [[False, False]], f"d={d.tolist()}"
        assert stale.tolist() == [[True, True]], f"stale={stale.tolist()}"
        results.add_pass("Gate trips on staleness")
    except Exception:
        results.add_fail("Gate trips on staleness", traceback.format_exc())

    # --- Test 3: mid-loss deficit + successful re-acquisition ---
    try:
        tr = _make_tracker(device)
        seq = [[True, True], [False, True], [False, True], [False, True], [True, True], [True, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        s = tr.episode_summary(e0)
        assert abs(s["Coop/track_loss_event_rate"].item() - 1.0) < 1e-6, s["Coop/track_loss_event_rate"].item()
        assert abs(s["Coop/event_rate_fov"].item() - 1.0) < 1e-6, s["Coop/event_rate_fov"].item()
        assert abs(s["Coop/reacq_success_rate"].item() - 1.0) < 1e-6, s["Coop/reacq_success_rate"].item()
        # deficit lasted 3 steps (t1,t2,t3) before regain at t4 -> 0.3 s
        assert abs(s["Coop/time_to_reacq_mean"].item() - 0.3) < 1e-6, s["Coop/time_to_reacq_mean"].item()
        results.add_pass("Mid-loss deficit + successful re-acquisition")
    except Exception:
        results.add_fail("Mid-loss deficit + successful re-acquisition", traceback.format_exc())

    # --- Test 4: flicker shorter than tau_min is not counted ---
    try:
        tr = _make_tracker(device)
        seq = [[True, True], [False, True], [True, True], [True, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        s = tr.episode_summary(e0)
        assert s["Coop/track_loss_event_rate"].item() == 0.0, s["Coop/track_loss_event_rate"].item()
        assert s["Coop/reacq_success_rate"].item() == 0.0, s["Coop/reacq_success_rate"].item()
        results.add_pass("Sub-tau_min flicker not counted")
    except Exception:
        results.add_fail("Sub-tau_min flicker not counted", traceback.format_exc())

    # --- Test 5: cold deficit (present at episode start) tagged separately ---
    try:
        tr = _make_tracker(device)
        seq = [[False, True], [False, True], [True, True], [True, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        s = tr.episode_summary(e0)
        assert abs(s["Coop/cold_deficit_rate"].item() - 1.0) < 1e-6, s["Coop/cold_deficit_rate"].item()
        assert s["Coop/track_loss_event_rate"].item() == 0.0, s["Coop/track_loss_event_rate"].item()
        results.add_pass("Cold deficit tagged separately from mid-loss")
    except Exception:
        results.add_fail("Cold deficit tagged separately from mid-loss", traceback.format_exc())

    # --- Test 6: peer-assisted precondition (both lost == no event) ---
    try:
        tr = _make_tracker(device)
        seq = [[True, True], [False, False], [False, False], [True, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        s = tr.episode_summary(e0)
        assert s["Coop/track_loss_event_rate"].item() == 0.0, s["Coop/track_loss_event_rate"].item()
        assert s["Coop/cold_deficit_rate"].item() == 0.0, s["Coop/cold_deficit_rate"].item()
        results.add_pass("Both-lost is not a peer-assisted deficit")
    except Exception:
        results.add_fail("Both-lost is not a peer-assisted deficit", traceback.format_exc())

    # --- Test 7: dropout cause (nonempty but stale) ---
    try:
        tr = _make_tracker(device)
        big = torch.full((1, 2), 200.0, device=device)
        zero = torch.zeros(1, 2, device=device)
        _step(tr, 0.0, _b([[True, True]], device), device)
        # agent 0 nonempty-but-stale (age huge), agent 1 fresh
        age1 = torch.tensor([[200.0, 0.0]], device=device)
        _step(tr, 0.1, _b([[True, True]], device), device, age=age1)
        _step(tr, 0.2, _b([[True, True]], device), device, age=age1)
        s = tr.episode_summary(e0)
        assert abs(s["Coop/event_rate_dropout"].item() - 1.0) < 1e-6, s["Coop/event_rate_dropout"].item()
        assert abs(s["Coop/track_loss_event_rate"].item() - 1.0) < 1e-6, s["Coop/track_loss_event_rate"].item()
        results.add_pass("Dropout cause (stale, nonempty) tagged")
    except Exception:
        results.add_fail("Dropout cause (stale, nonempty) tagged", traceback.format_exc())

    # --- Test 8: idempotency within a step ---
    try:
        tr = _make_tracker(device)
        _step(tr, 0.0, _b([[True, True]], device), device)
        _step(tr, 0.1, _b([[False, True]], device), device)
        _step(tr, 0.1, _b([[False, True]], device), device)  # duplicate t -> no-op
        assert tr._ep_steps[0].item() == 2, tr._ep_steps[0].item()
        results.add_pass("Idempotent within a sim step")
    except Exception:
        results.add_fail("Idempotent within a sim step", traceback.format_exc())

    # --- Test 9: team-track maintenance fraction ---
    try:
        tr = _make_tracker(device)
        seq = [[True, True], [True, True], [False, True], [False, True]]  # 2/4 steps both track
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        s = tr.episode_summary(e0)
        assert abs(s["Coop/team_track_maintenance"].item() - 0.5) < 1e-6, s["Coop/team_track_maintenance"].item()
        results.add_pass("Team-track maintenance fraction")
    except Exception:
        results.add_fail("Team-track maintenance fraction", traceback.format_exc())

    # --- Test 10: reset clears state and accumulators ---
    try:
        tr = _make_tracker(device)
        for k, det in enumerate([[True, True], [False, True], [False, True]]):
            _step(tr, k * STEP_DT, _b([det], device), device)
        tr.reset(e0)
        assert tr._ep_steps[0].item() == 0
        assert tr._state[0].tolist() == [0, 0]
        assert tr._ep_cause_counts[0].sum().item() == 0
        s = tr.episode_summary(e0)
        assert s["Coop/track_loss_event_rate"].item() == 0.0
        results.add_pass("Reset clears state + accumulators")
    except Exception:
        results.add_fail("Reset clears state + accumulators", traceback.format_exc())

    # --- Test 11: failed hold (re-loss before tau_hold) is not a success ---
    try:
        tr = _make_tracker(device)
        # qualify deficit (2 steps), regain 1 step (not held), lose again
        seq = [[True, True], [False, True], [False, True], [True, True], [False, True], [False, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        s = tr.episode_summary(e0)
        # the original deficit qualified; recovery failed before tau_hold -> no success yet
        assert s["Coop/reacq_success_rate"].item() == 0.0, s["Coop/reacq_success_rate"].item()
        assert s["Coop/track_loss_event_rate"].item() >= 1.0, s["Coop/track_loss_event_rate"].item()
        results.add_pass("Failed hold is not a success")
    except Exception:
        results.add_fail("Failed hold is not a success", traceback.format_exc())

    # --- Test 12: GPU/CPU multi-env batch runs without shape errors ---
    try:
        tr = _make_tracker(device, num_envs=8, num_agents=3)
        det = torch.ones(8, 3, dtype=torch.bool, device=device)
        det[:, 0] = False  # agent 0 in deficit, peers hold
        for k in range(4):
            _step(tr, k * STEP_DT, det, device)
        s = tr.episode_summary(torch.arange(8, device=device))
        assert s["Coop/track_loss_event_rate"].dim() == 0
        results.add_pass("Multi-env / multi-agent batch")
    except Exception:
        results.add_fail("Multi-env / multi-agent batch", traceback.format_exc())

    # --- Test 13: bearing alignment reduces over ANY valid peer (max cosine) ---
    try:
        tr = _make_tracker(device, num_agents=3)
        # agent 0 bearing aligns with agent 2 (cos 1) but not agent 1 (cos 0)
        bearing = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], device=device
        )
        seq = [[True, True, True], [False, True, True], [False, True, True],
               [True, True, True], [True, True, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device, bearing=bearing)
        s = tr.episode_summary(e0)
        assert abs(s["Coop/reacq_success_rate"].item() - 1.0) < 1e-6, s["Coop/reacq_success_rate"].item()
        # best alignment to any valid peer == agent 2 -> cos 1.0
        assert abs(s["Coop/reacq_bearing_align_mean"].item() - 1.0) < 1e-5, s["Coop/reacq_bearing_align_mean"].item()
        results.add_pass("Bearing alignment = max over any valid peer")
    except Exception:
        results.add_fail("Bearing alignment = max over any valid peer", traceback.format_exc())

    # --- Test 14: episode_values returns per-env raw counts (eval aggregation) ---
    try:
        tr = _make_tracker(device)
        seq = [[True, True], [False, True], [False, True], [False, True], [True, True], [True, True]]
        for k, det in enumerate(seq):
            _step(tr, k * STEP_DT, _b([det], device), device)
        v = tr.episode_values(e0)
        assert v["mid_events"][0].item() == 1.0, v["mid_events"][0].item()
        assert v["qual_total"][0].item() == 1.0, v["qual_total"][0].item()
        assert v["success"][0].item() == 1.0, v["success"][0].item()
        assert v["reacq_step_sum"][0].item() == 3.0, v["reacq_step_sum"][0].item()  # 3 deficit steps
        assert v["fov"][0].item() == 1.0, v["fov"][0].item()
        results.add_pass("episode_values per-env raw counts")
    except Exception:
        results.add_fail("episode_values per-env raw counts", traceback.format_exc())


def main():
    print(f"{'='*80}\nCOOPERATION_METRICS TEST SUITE\n{'='*80}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Torch: {torch.__version__}  Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    results = TestResults()
    try:
        run_tests(results, device)
    except Exception:
        results.add_error("Suite-level", traceback.format_exc())
    ok = results.print_summary()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
