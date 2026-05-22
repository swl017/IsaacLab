#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ticket 039 — asymmetric PX4 z-velocity envelope.

Three test cases (single env build; flag toggled at runtime):
  1. cfg defaults — pre-build sanity (flag=False, z_up=3.0, z_dn=1.5).
  2. flag=False (symmetric, regression) — cmd_vel[..., 0:3] = action * _max_lin_vel.
  3. flag=True (asymmetric) — z action scaled by max_vel_z_up (>=0) or
     max_vel_z_dn (<0); xy unchanged.

A single Isaac Sim SimulationApp is shared across all tests (multiple
``gym.make`` calls in one process are not supported).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ticket 039 asymmetric z envelope tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import IrisMA6TestEnvCfg

TASK = "Isaac-Iris-MA6-Direct-Test-v0"


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}", flush=True)

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        print(f"  ✗ {name}", flush=True)
        for line in msg.split("\n")[:20]:
            print(f"      {line}", flush=True)

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 80, flush=True)
        print("TEST SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}", flush=True)
        if self.failed:
            print("\nFAILED:", flush=True)
            for n, m in self.failed:
                print(f"  {n}: {m.splitlines()[0] if m else ''}", flush=True)
        print("=" * 80, flush=True)
        return not self.failed


def _make_actions(unwrapped, vx: float, vy: float, vz: float) -> dict:
    actions = {}
    for agent_id in unwrapped.cfg.possible_agents:
        a = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
        a[:, 0] = vx
        a[:, 1] = vy
        a[:, 2] = vz
        actions[agent_id] = a
    return actions


def test_cfg_defaults(r: TestResults):
    """Default cfg has flag=False and PX4-matched z bounds."""
    try:
        cfg = IrisMA6TestEnvCfg()
        assert cfg.enable_asymmetric_z_envelope is False, \
            f"default flag should be False, got {cfg.enable_asymmetric_z_envelope}"
        assert cfg.max_vel_z_up == 3.0, f"max_vel_z_up={cfg.max_vel_z_up}"
        assert cfg.max_vel_z_dn == 1.5, f"max_vel_z_dn={cfg.max_vel_z_dn}"
        r.add_pass("cfg defaults: flag=False, z_up=3.0, z_dn=1.5")
    except Exception:
        r.add_fail("cfg_defaults", traceback.format_exc())


def test_symmetric_default(r: TestResults, unwrapped):
    """flag=False → cmd_vel[..., 0:3] = action[..., 0:3] * _max_lin_vel."""
    try:
        unwrapped.cfg.enable_asymmetric_z_envelope = False
        # Action values that would clip under asymmetric envelope (|vz|=1 → 10 m/s
        # symmetric vs ±[1.5, 3.0] asymmetric), exposing any cross-talk.
        actions = _make_actions(unwrapped, vx=0.5, vy=-0.3, vz=-1.0)
        unwrapped._pre_physics_step(actions)
        mlv = unwrapped._max_lin_vel  # (N,)
        for idx, _ in enumerate(unwrapped.cfg.possible_agents):
            got = unwrapped.cmd_vel[:, idx, :]
            assert torch.allclose(got[:, 0], 0.5 * mlv), \
                f"agent {idx}: vx mismatch"
            assert torch.allclose(got[:, 1], -0.3 * mlv), \
                f"agent {idx}: vy mismatch"
            assert torch.allclose(got[:, 2], -1.0 * mlv), \
                f"agent {idx}: vz mismatch (got {got[:, 2]}, expected {-1.0 * mlv})"
        r.add_pass("symmetric default: cmd_vel matches action * _max_lin_vel")
    except Exception:
        r.add_fail("symmetric_default", traceback.format_exc())


def test_asymmetric_z_clip(r: TestResults, unwrapped):
    """flag=True → z scaled by max_vel_z_up (>=0) or max_vel_z_dn (<0); xy unchanged."""
    try:
        cfg = unwrapped.cfg
        cfg.enable_asymmetric_z_envelope = True
        up = cfg.max_vel_z_up
        dn = cfg.max_vel_z_dn
        mlv = unwrapped._max_lin_vel

        # Full climb
        actions = _make_actions(unwrapped, vx=1.0, vy=0.0, vz=1.0)
        unwrapped._pre_physics_step(actions)
        for idx, _ in enumerate(cfg.possible_agents):
            got = unwrapped.cmd_vel[:, idx, :]
            assert torch.allclose(got[:, 0], mlv), \
                f"agent {idx}: xy still scaled by _max_lin_vel"
            assert torch.allclose(got[:, 2], torch.full_like(got[:, 2], up)), \
                f"agent {idx}: vz climb {got[:, 2]} != {up}"

        # Full descend
        actions = _make_actions(unwrapped, vx=0.0, vy=0.0, vz=-1.0)
        unwrapped._pre_physics_step(actions)
        for idx, _ in enumerate(cfg.possible_agents):
            got_vz = unwrapped.cmd_vel[:, idx, 2]
            assert torch.allclose(got_vz, torch.full_like(got_vz, -dn)), \
                f"agent {idx}: vz descend {got_vz} != {-dn}"

        # Mixed sign across envs: per-env sign-dependent gating
        actions = _make_actions(unwrapped, vx=0.0, vy=0.0, vz=0.0)
        for agent_id in cfg.possible_agents:
            z = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
            z[::2] = 0.5    # climb half-throttle
            z[1::2] = -0.5  # descend half-throttle
            actions[agent_id][:, 2] = z
        unwrapped._pre_physics_step(actions)
        for idx, _ in enumerate(cfg.possible_agents):
            got_vz = unwrapped.cmd_vel[:, idx, 2]
            expected = torch.zeros_like(got_vz)
            expected[::2] = 0.5 * up
            expected[1::2] = -0.5 * dn
            assert torch.allclose(got_vz, expected), \
                f"agent {idx}: mixed-sign vz {got_vz} vs {expected}"

        # Saturation: |action| <= 1 keeps cmd_vel in [-dn, +up]
        actions = _make_actions(unwrapped, vx=0.0, vy=0.0, vz=0.0)
        for agent_id in cfg.possible_agents:
            actions[agent_id][:, 2] = torch.empty(
                unwrapped.num_envs, device=unwrapped.device
            ).uniform_(-1.0, 1.0)
        unwrapped._pre_physics_step(actions)
        all_vz = unwrapped.cmd_vel[:, :, 2]
        assert (all_vz >= -dn - 1e-6).all(), f"min vz {all_vz.min().item()} < -{dn}"
        assert (all_vz <= up + 1e-6).all(), f"max vz {all_vz.max().item()} > {up}"

        r.add_pass(
            f"asymmetric clip: vz ∈ [-{dn}, +{up}] under uniform action; "
            f"sign-dependent scaling correct"
        )
    except Exception:
        r.add_fail("asymmetric_z_clip", traceback.format_exc())


def main():
    print("=" * 80, flush=True)
    print("TICKET 039 — ASYMMETRIC Z-VELOCITY ENVELOPE TEST SUITE", flush=True)
    print("=" * 80, flush=True)

    r = TestResults()

    print("\n--- Category 1 — cfg defaults (pre-build) ---", flush=True)
    test_cfg_defaults(r)

    # Single env build — Isaac Sim's SimulationApp is global per-process, so we
    # toggle cfg.enable_asymmetric_z_envelope at runtime instead of rebuilding.
    cfg = IrisMA6TestEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 42  # ticket-034 strict-seed assertion
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped

    try:
        print("\n--- Category 2 — symmetric default (regression) ---", flush=True)
        test_symmetric_default(r, unwrapped)

        print("\n--- Category 3 — asymmetric clip ---", flush=True)
        test_asymmetric_z_clip(r, unwrapped)
    finally:
        env.close()

    success = r.print_summary()
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
