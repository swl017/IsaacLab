#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ticket 046 — closer spawn + 2D target motion.

Covers Slice-1 acceptance:
  1. cfg defaults — TargetControllerCfg.enable_z_motion = True (backward-compat).
  2. cfg defaults — env-level spawn geometry lowered to (30, 15, 2).
  3. Bit-exact regression: enable_z_motion=True leaves v_cmd[:, 2] untouched
     by ``_apply_constraints`` (independent of the four velocity generators).
  4. z-clamp: enable_z_motion=False zeros v_cmd[:, 2] for every target across
     all behavior modes (linear / circular / approach / evade).
  5. Spawn geometry: at curriculum_progress=1.0, target placements respect the
     new ``target_distance_max`` (15 m) and ``target_height_offset_max`` (2 m).

A single Isaac Sim ``SimulationApp`` is shared across all tests (multiple
``gym.make`` calls in one process are not supported).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ticket 046 z-motion + spawn tests")
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
from isaaclab_tasks.direct.iris_ma6.target_controller import TargetControllerCfg

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
        for line in msg.split("\n")[:25]:
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


# ===========================================================================
# Category 1 — cfg defaults (no env)
# ===========================================================================


def test_target_controller_cfg_default(r: TestResults):
    """TargetControllerCfg.enable_z_motion defaults to True (backward-compat)."""
    try:
        cfg = TargetControllerCfg()
        assert hasattr(cfg, "enable_z_motion"), "TargetControllerCfg is missing enable_z_motion"
        assert cfg.enable_z_motion is True, (
            f"default enable_z_motion should be True (backward-compat), got {cfg.enable_z_motion}"
        )
        r.add_pass("TargetControllerCfg.enable_z_motion defaults to True")
    except Exception:
        r.add_fail("target_controller_cfg_default", traceback.format_exc())


def test_env_spawn_geometry_defaults(r: TestResults):
    """IrisMA6TestEnvCfg sets the lower spawn geometry defaults (t046)."""
    try:
        cfg = IrisMA6TestEnvCfg()
        assert cfg.initial_states.cylinder_diameter_max == 30.0, (
            f"cylinder_diameter_max={cfg.initial_states.cylinder_diameter_max} (expected 30.0)"
        )
        assert cfg.initial_states.target_distance_max == 15.0, (
            f"target_distance_max={cfg.initial_states.target_distance_max} (expected 15.0)"
        )
        assert cfg.initial_states.target_height_offset_max == 2.0, (
            f"target_height_offset_max={cfg.initial_states.target_height_offset_max} (expected 2.0)"
        )
        # min values unchanged (curriculum mechanism preserved).
        assert cfg.initial_states.cylinder_diameter_min == 20.0
        assert cfg.initial_states.target_distance_min == 10.0
        assert cfg.initial_states.target_height_offset_min == 0.0
        r.add_pass(
            "env cfg spawn geometry: diameter_max=30, distance_max=15, h_offset_max=2 (mins unchanged)"
        )
    except Exception:
        r.add_fail("env_spawn_geometry_defaults", traceback.format_exc())


# ===========================================================================
# Category 2 — z-motion clamp inside _apply_constraints
# ===========================================================================


def test_apply_constraints_z_passthrough(r: TestResults, unwrapped):
    """With enable_z_motion=True, _apply_constraints preserves the input v_cmd z
    (modulo geofence/altitude clamps, which only fire at the boundary)."""
    try:
        tc = unwrapped._target_controller
        assert tc is not None, "target controller not constructed; enable_target_controller=True required"
        # Snap to default (True).
        tc.cfg.enable_z_motion = True

        device = tc.device
        total = tc.total_targets

        # Place every target safely inside geofence (origin) and at a healthy
        # altitude (mid-band between min_altitude and max_altitude), so neither
        # the geofence nor the altitude clamp will fire and we can isolate the
        # z-motion flag's behavior.
        alt = 0.5 * (tc.cfg.min_altitude + tc.cfg.max_altitude)
        origins_local = tc._env_origins  # (N_env, 3)
        # Replicate origins to per-target then add (0, 0, alt) above each origin.
        pos_flat = (
            origins_local.unsqueeze(1)
            .expand(-1, tc.num_targets, -1)
            .reshape(-1, 3)
            .clone()
        )
        pos_flat[:, 2] = pos_flat[:, 2] + alt

        torch.manual_seed(7)
        v_in = torch.empty(total, 3, device=device).uniform_(-2.0, 2.0)
        v_in_z = v_in[:, 2].clone()

        v_out = tc._apply_constraints(v_in.clone(), pos_flat, curriculum_progress=0.5)
        # z must pass through unchanged when flag is True and altitude clamp idle.
        assert torch.equal(v_out[:, 2], v_in_z), (
            f"flag=True altered v_cmd[:, 2] without altitude clamp firing; "
            f"max |Δz|={(v_out[:, 2] - v_in_z).abs().max().item()}"
        )
        r.add_pass("enable_z_motion=True: _apply_constraints preserves v_cmd[:, 2] (no altitude clamp)")
    except Exception:
        r.add_fail("apply_constraints_z_passthrough", traceback.format_exc())


def test_apply_constraints_z_clamped(r: TestResults, unwrapped):
    """With enable_z_motion=False, _apply_constraints zeros v_cmd[:, 2] for all
    targets regardless of input magnitude or sign."""
    try:
        tc = unwrapped._target_controller
        tc.cfg.enable_z_motion = False
        try:
            device = tc.device
            total = tc.total_targets

            alt = 0.5 * (tc.cfg.min_altitude + tc.cfg.max_altitude)
            origins_local = tc._env_origins
            pos_flat = (
                origins_local.unsqueeze(1)
                .expand(-1, tc.num_targets, -1)
                .reshape(-1, 3)
                .clone()
            )
            pos_flat[:, 2] = pos_flat[:, 2] + alt

            torch.manual_seed(13)
            v_in = torch.empty(total, 3, device=device).uniform_(-5.0, 5.0)
            v_in_xy = v_in[:, :2].clone()

            v_out = tc._apply_constraints(v_in.clone(), pos_flat, curriculum_progress=0.5)
            # z exactly zero.
            assert torch.all(v_out[:, 2] == 0.0), (
                f"flag=False did not zero v_cmd[:, 2]; |z|_max={v_out[:, 2].abs().max().item()}"
            )
            # x/y untouched (no geofence violation in the test setup).
            assert torch.equal(v_out[:, :2], v_in_xy), "x/y components were modified unexpectedly"
            r.add_pass("enable_z_motion=False: v_cmd[:, 2] == 0 (x/y untouched)")
        finally:
            # Restore the default so subsequent tests start clean.
            tc.cfg.enable_z_motion = True
    except Exception:
        r.add_fail("apply_constraints_z_clamped", traceback.format_exc())


def test_apply_constraints_z_clamped_extreme_inputs(r: TestResults, unwrapped):
    """Cover all four behavior FSMs by feeding the kinds of (N, 3) velocity
    vectors each generator can emit (linear/circular/approach/evade): pure ±z
    pushes, near-zero, and large mixed values. ``_apply_constraints`` does not
    branch on the FSM mode, so the orchestrator-level clamp is unambiguously
    mode-agnostic."""
    try:
        tc = unwrapped._target_controller
        tc.cfg.enable_z_motion = False
        try:
            device = tc.device
            total = tc.total_targets

            alt = 0.5 * (tc.cfg.min_altitude + tc.cfg.max_altitude)
            origins_local = tc._env_origins
            pos_flat = (
                origins_local.unsqueeze(1)
                .expand(-1, tc.num_targets, -1)
                .reshape(-1, 3)
                .clone()
            )
            pos_flat[:, 2] = pos_flat[:, 2] + alt

            # 1) pure +z push (mimics linear/circular vertical preference).
            v = torch.zeros(total, 3, device=device)
            v[:, 2] = 3.0
            out = tc._apply_constraints(v.clone(), pos_flat, 0.5)
            assert torch.all(out[:, 2] == 0.0), "pure +z input not zeroed"

            # 2) pure -z push.
            v = torch.zeros(total, 3, device=device)
            v[:, 2] = -3.0
            out = tc._apply_constraints(v.clone(), pos_flat, 0.5)
            assert torch.all(out[:, 2] == 0.0), "pure -z input not zeroed"

            # 3) mixed approach-style (large xy + moderate z).
            v = torch.tensor([1.5, -1.5, 0.7], device=device).expand(total, 3).clone()
            xy_in = v[:, :2].clone()
            out = tc._apply_constraints(v.clone(), pos_flat, 0.5)
            assert torch.all(out[:, 2] == 0.0), "mixed approach-style input z not zeroed"
            assert torch.equal(out[:, :2], xy_in), "approach-style xy mutated"

            # 4) tiny z (numerical edge).
            v = torch.zeros(total, 3, device=device)
            v[:, 2] = 1e-9
            out = tc._apply_constraints(v.clone(), pos_flat, 0.5)
            assert torch.all(out[:, 2] == 0.0), "tiny +z not zeroed (exact == 0 required)"

            r.add_pass("z-clamp robust to ±z / mixed / near-zero inputs (mode-agnostic)")
        finally:
            tc.cfg.enable_z_motion = True
    except Exception:
        r.add_fail("apply_constraints_z_clamped_extreme_inputs", traceback.format_exc())


# ===========================================================================
# Category 3 — spawn geometry at curriculum_progress=1.0
# ===========================================================================


def test_spawn_geometry_at_full_progress(r: TestResults, unwrapped):
    """At progress=1.0 the InitialStates generator must respect the new
    cylinder_diameter_max, target_distance_max, and target_height_offset_max."""
    try:
        gen = unwrapped._initial_states
        assert gen is not None, "InitialStates not constructed; enable_initial_states_randomization=True required"
        cfg_is = unwrapped.cfg.initial_states

        env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        result = gen.generate(env_ids=env_ids, curriculum_progress=1.0)

        # 1) target distance from cylinder center in the xy plane.
        delta = result.target_positions[:, :2] - result.cylinder_centers[:, :2]
        dist_xy = torch.linalg.norm(delta, dim=-1)
        tol = 1e-5
        assert (dist_xy <= cfg_is.target_distance_max + tol).all(), (
            f"some target xy-distance exceeds target_distance_max={cfg_is.target_distance_max}: "
            f"max={dist_xy.max().item():.6f}"
        )
        assert (dist_xy >= cfg_is.target_distance_min - tol).all(), (
            f"some target xy-distance below target_distance_min={cfg_is.target_distance_min}: "
            f"min={dist_xy.min().item():.6f}"
        )

        # 2) target height offset envelope.
        z_off = result.target_positions[:, 2] - result.cylinder_centers[:, 2]
        assert (z_off <= cfg_is.target_height_offset_max + tol).all(), (
            f"target z-offset exceeds {cfg_is.target_height_offset_max}: max={z_off.max().item():.6f}"
        )
        assert (z_off >= cfg_is.target_height_offset_min - tol).all(), (
            f"target z-offset below {cfg_is.target_height_offset_min}: min={z_off.min().item():.6f}"
        )

        # 3) agent placements within the cylinder diameter.
        agent_xy = result.agent_positions[..., :2]
        center_xy = result.cylinder_centers[:, :2].unsqueeze(1)  # broadcast over agents
        agent_radius = torch.linalg.norm(agent_xy - center_xy, dim=-1)
        radius_max = 0.5 * cfg_is.cylinder_diameter_max
        assert (agent_radius <= radius_max + tol).all(), (
            f"some agent radius exceeds cylinder_diameter_max/2={radius_max}: "
            f"max={agent_radius.max().item():.6f}"
        )

        r.add_pass(
            f"spawn @progress=1.0: target dist xy ∈ [{dist_xy.min().item():.2f}, "
            f"{dist_xy.max().item():.2f}] ≤ {cfg_is.target_distance_max}, "
            f"|z_off| ≤ {z_off.abs().max().item():.2f} ≤ {cfg_is.target_height_offset_max}, "
            f"agent radius ≤ {agent_radius.max().item():.2f} ≤ {radius_max}"
        )
    except Exception:
        r.add_fail("spawn_geometry_at_full_progress", traceback.format_exc())


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 80, flush=True)
    print("TICKET 046 — CLOSER SPAWN + 2D TARGET MOTION TEST SUITE", flush=True)
    print("=" * 80, flush=True)

    r = TestResults()

    print("\n--- Category 1 — cfg defaults (no env) ---", flush=True)
    test_target_controller_cfg_default(r)
    test_env_spawn_geometry_defaults(r)

    cfg = IrisMA6TestEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 42
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped

    try:
        print("\n--- Category 2a — z-passthrough (enable_z_motion=True) ---", flush=True)
        test_apply_constraints_z_passthrough(r, unwrapped)

        print("\n--- Category 2b — z-clamp (enable_z_motion=False) ---", flush=True)
        test_apply_constraints_z_clamped(r, unwrapped)

        print("\n--- Category 2c — z-clamp robustness (mode-agnostic inputs) ---", flush=True)
        test_apply_constraints_z_clamped_extreme_inputs(r, unwrapped)

        print("\n--- Category 3 — spawn geometry @progress=1.0 ---", flush=True)
        test_spawn_geometry_at_full_progress(r, unwrapped)
    finally:
        env.close()

    success = r.print_summary()
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
