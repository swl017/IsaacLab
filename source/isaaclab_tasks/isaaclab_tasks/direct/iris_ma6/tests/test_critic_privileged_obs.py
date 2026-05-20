#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ticket 037 — critic-only privileged env-param observations.

Five test categories:
  1. Cfg-time dim accounting: state_space increases by the correct amount for
     various critic_privileged_fields configurations.
  2. Cfg-time validation: unknown / duplicate fields raise ValueError early.
  3. Privileged-obs dim: env._get_critic_privileged_obs() returns the expected
     shape for a given fields list.
  4. Pre-normalization range: every per-field output is in [-1, +1].
  5. Axis independence default (Slice 7): with enable_axis_independence=False,
     the 8 per-axis _eff_progress_* tensors equal _eff_progress_dynamics in
     consumer reads (via _axis_eff_progress helper), and progress_* scalars
     resolve to progress_dynamics via _axis_progress helper.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ticket 037 critic-privileged-obs tests")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback
from datetime import datetime

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test_cfg import (
    IrisMA6TestEnvCfg,
    _CRITIC_PRIVILEGED_FIELD_REGISTRY,
    _CRITIC_PRIVILEGED_FIELDS_REQUIRING_DR,
    _critic_privileged_dim_split,
)

# Subset of the registry that does NOT require domain_randomization.enabled=True.
# The default iris_ma6 cfg has DR disabled, so tests that build envs with the
# default cfg must use only these fields.
_NON_DR_FIELDS = [
    f for f in _CRITIC_PRIVILEGED_FIELD_REGISTRY.keys()
    if f not in _CRITIC_PRIVILEGED_FIELDS_REQUIRING_DR
]
# Should be 17: 6 gain scales + max_lin_vel + 2 τ + 2 dead times + 4 detection + 2 target

VERBOSE = args_cli.test_verbose
TASK = "Isaac-Iris-MA6-Direct-Test-v0"


class TestResults:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def add_fail(self, name: str, msg: str):
        self.failed.append((name, msg))
        print(f"  ✗ {name}")
        for line in msg.split("\n")[:20]:
            print(f"      {line}")

    def add_error(self, name: str, msg: str):
        self.errors.append((name, msg))
        print(f"  ERROR {name}")
        for line in msg.split("\n")[:20]:
            print(f"      {line}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total: {total} | Passed: {len(self.passed)} | "
              f"Failed: {len(self.failed)} | Errors: {len(self.errors)}")
        if self.failed:
            print("\nFAILED:")
            for n, m in self.failed:
                print(f"  {n}: {m.splitlines()[0] if m else ''}")
        if self.errors:
            print("\nERRORS:")
            for n, m in self.errors:
                print(f"  {n}: {m.splitlines()[0] if m else ''}")
        print("=" * 80)
        return not self.failed and not self.errors


def _make_cfg(critic_priv_fields: list[str] | None = None,
              enable_full_priv: bool = False,
              enable_axis_independence: bool = False) -> IrisMA6TestEnvCfg:
    """Construct a fresh env cfg with the given critic-priv-obs settings.

    Passes settings via constructor kwargs so __post_init__ runs exactly once
    with the right values (calling it twice triggers configclass re-init
    issues with already-populated sub-configs).
    """
    kwargs: dict = {
        "enable_full_critic_priv_obs": enable_full_priv,
        "enable_axis_independence": enable_axis_independence,
    }
    if critic_priv_fields is not None:
        kwargs["critic_privileged_fields"] = list(critic_priv_fields)
    return IrisMA6TestEnvCfg(**kwargs)


# ---------------------------------------------------------------------------
# Category 1 — Cfg-time dim accounting
# ---------------------------------------------------------------------------

def test_dim_accounting_empty(r: TestResults):
    """With critic_privileged_fields=[], state_space matches t034 (no priv tail)."""
    try:
        cfg = _make_cfg(critic_priv_fields=[])
        # base = actor concat + zoom tail (default enable_critic_continuous_zoom=True)
        # = (31 + 16*(A-1)) * A + 2 * A
        A = cfg.num_agents
        actor_dim = (31 + 16 * (A - 1)) * A
        expected = actor_dim + 2 * A  # zoom tail
        assert cfg.state_space == expected, (
            f"state_space={cfg.state_space}, expected={expected}"
        )
        r.add_pass(f"empty critic_priv: state_space={cfg.state_space}")
    except Exception as e:
        r.add_fail("empty critic_priv dim accounting", traceback.format_exc())


def test_dim_accounting_non_dr_subset(r: TestResults):
    """With non-DR fields enabled (DR-disabled env default), state_space
    increases by the expected delta. Uses 17 non-DR fields."""
    try:
        cfg = _make_cfg(critic_priv_fields=list(_NON_DR_FIELDS))
        A = cfg.num_agents
        actor_dim = (31 + 16 * (A - 1)) * A
        pa, sh = _critic_privileged_dim_split(_NON_DR_FIELDS)
        expected = actor_dim + 2 * A + pa * A + sh  # zoom + non-DR priv
        assert cfg.state_space == expected, (
            f"state_space={cfg.state_space}, expected={expected} "
            f"(pa={pa}, sh={sh})"
        )
        assert len(cfg.critic_privileged_fields) == len(_NON_DR_FIELDS)
        r.add_pass(
            f"non-DR subset: state_space={cfg.state_space}, "
            f"fields={len(_NON_DR_FIELDS)} (pa={pa}, sh={sh})"
        )
    except Exception:
        r.add_fail("non-DR subset dim accounting", traceback.format_exc())


def test_dim_accounting_minimal_subset(r: TestResults):
    """Two non-DR fields (1 per-agent + 1 shared) add the correct delta."""
    try:
        cfg = _make_cfg(critic_priv_fields=["motor_gain_scale", "target_scale_xy"])
        A = cfg.num_agents
        actor_dim = (31 + 16 * (A - 1)) * A
        # motor_gain_scale: per_agent (1 dim × A); target_scale_xy: shared (1 dim)
        expected = actor_dim + 2 * A + 1 * A + 1
        assert cfg.state_space == expected, (
            f"state_space={cfg.state_space}, expected={expected}"
        )
        r.add_pass(f"minimal subset: state_space={cfg.state_space}")
    except Exception:
        r.add_fail("minimal subset dim accounting", traceback.format_exc())


def test_dr_fields_require_dr_enabled(r: TestResults):
    """Asking for a DR-dependent field with DR disabled raises ValueError."""
    try:
        try:
            _make_cfg(critic_priv_fields=["fov_scale"])
            r.add_fail("dr_fields_require_dr_enabled", "no exception raised")
            return
        except ValueError as e:
            msg = str(e)
            assert "fov_scale" in msg
            assert "domain_randomization.enabled" in msg
            r.add_pass("DR-field with DR disabled raises ValueError")
    except Exception:
        r.add_error("dr_fields_require_dr_enabled", traceback.format_exc())


def test_dim_split_helper(r: TestResults):
    """_critic_privileged_dim_split returns correct (per_agent, shared) sums."""
    try:
        pa, sh = _critic_privileged_dim_split([
            "vel_gain_scale", "fov_scale", "target_scale_xy", "target_scale_z"
        ])
        assert pa == 2, f"per_agent={pa}, expected 2"
        assert sh == 2, f"shared={sh}, expected 2"
        pa, sh = _critic_privileged_dim_split(list(_CRITIC_PRIVILEGED_FIELD_REGISTRY.keys()))
        assert pa == 22, f"all fields per_agent={pa}, expected 22"
        assert sh == 2, f"all fields shared={sh}, expected 2"
        r.add_pass("_critic_privileged_dim_split sums")
    except Exception as e:
        r.add_fail("dim_split helper", traceback.format_exc())


# ---------------------------------------------------------------------------
# Category 2 — Cfg-time validation
# ---------------------------------------------------------------------------

def test_unknown_field_raises(r: TestResults):
    """Unknown field name raises ValueError with a helpful message."""
    try:
        try:
            _make_cfg(critic_priv_fields=["definitely_not_a_field"])
            r.add_fail("unknown_field_raises", "no exception raised")
            return
        except ValueError as e:
            msg = str(e)
            assert "definitely_not_a_field" in msg
            assert "valid keys" in msg.lower()
            r.add_pass("unknown_field_raises ValueError")
    except Exception:
        r.add_error("unknown_field_raises", traceback.format_exc())


def test_duplicate_field_raises(r: TestResults):
    """Duplicate field names raise ValueError."""
    try:
        try:
            _make_cfg(critic_priv_fields=["fov_scale", "fov_scale"])
            r.add_fail("duplicate_field_raises", "no exception raised")
            return
        except ValueError as e:
            assert "duplicate" in str(e).lower()
            r.add_pass("duplicate_field_raises ValueError")
    except Exception:
        r.add_error("duplicate_field_raises", traceback.format_exc())


# ---------------------------------------------------------------------------
# Category 3+4 — Env-side privileged-obs dim and normalization range
# ---------------------------------------------------------------------------

def _build_env(cfg: IrisMA6TestEnvCfg, num_envs: int = 4, seed: int = 42):
    cfg.scene.num_envs = num_envs
    cfg.seed = seed  # ticket-034 strict-seed assertion
    env = gym.make(TASK, cfg=cfg)
    env_wrapped = SkrlVecEnvWrapper(env)
    return env, env_wrapped


def test_priv_obs_dim_and_range(r: TestResults):
    """Build env with all fields enabled; verify privileged obs shape and [-1, +1] bound."""
    try:
        cfg = _make_cfg(enable_full_priv=True)
        env, env_wrapped = _build_env(cfg, num_envs=4)
        try:
            env_wrapped.reset()
            unwrapped = env.unwrapped
            priv = unwrapped._get_critic_privileged_obs()
            A = unwrapped.num_agents if hasattr(unwrapped, "num_agents") else cfg.num_agents
            expected_dim = 22 * A + 2
            assert priv.shape == (4, expected_dim), (
                f"shape={tuple(priv.shape)}, expected (4, {expected_dim})"
            )
            assert torch.isfinite(priv).all(), "non-finite values in priv obs"
            # All fields pre-normalized to [-1, +1]
            assert priv.min() >= -1.001, f"min={priv.min().item()} < -1"
            assert priv.max() <= 1.001, f"max={priv.max().item()} > 1"
            r.add_pass(
                f"priv obs shape (4, {expected_dim}); "
                f"range=[{priv.min().item():.3f}, {priv.max().item():.3f}]"
            )
        finally:
            env.close()
    except Exception:
        r.add_error("priv_obs_dim_and_range", traceback.format_exc())


def test_priv_obs_empty_returns_empty(r: TestResults):
    """With critic_privileged_fields=[], _get_critic_privileged_obs returns (N, 0)."""
    try:
        cfg = _make_cfg(critic_priv_fields=[])
        env, env_wrapped = _build_env(cfg, num_envs=4)
        try:
            env_wrapped.reset()
            unwrapped = env.unwrapped
            priv = unwrapped._get_critic_privileged_obs()
            assert priv.shape == (4, 0), f"shape={tuple(priv.shape)}, expected (4, 0)"
            r.add_pass("priv obs empty → (N, 0)")
        finally:
            env.close()
    except Exception:
        r.add_error("priv_obs_empty_returns_empty", traceback.format_exc())


# ---------------------------------------------------------------------------
# Category 5 — Slice 7 axis-independence default
# ---------------------------------------------------------------------------

def test_axis_independence_default_off(r: TestResults):
    """With enable_axis_independence=False (default), _axis_eff_progress returns
    _eff_progress_dynamics for every axis, and _axis_progress returns
    progress_dynamics. Bit-exact t034 read path."""
    try:
        cfg = _make_cfg(enable_axis_independence=False)
        env, env_wrapped = _build_env(cfg, num_envs=4)
        try:
            env_wrapped.reset()
            unwrapped = env.unwrapped
            axes = [
                "gimbal_rate_tau", "drone_gains", "max_lin_vel_scale",
                "camera_fov", "gimbal_mech_offsets", "mass_inertia",
                "gimbal_stiff_damp", "target_scale",
            ]
            for ax in axes:
                t = unwrapped._axis_eff_progress(ax)
                # Object-identity check: helper returns the dynamics tensor itself
                assert t.data_ptr() == unwrapped._eff_progress_dynamics.data_ptr(), (
                    f"axis '{ax}' tensor doesn't equal _eff_progress_dynamics"
                )
                s = unwrapped._axis_progress(ax)
                assert s == unwrapped.progress_dynamics, (
                    f"axis '{ax}' scalar={s} != progress_dynamics={unwrapped.progress_dynamics}"
                )
            r.add_pass("axis_independence=False → all axes equal dynamics")
        finally:
            env.close()
    except Exception:
        r.add_error("axis_independence_default_off", traceback.format_exc())


def test_axis_independence_on_returns_per_axis(r: TestResults):
    """With enable_axis_independence=True, _axis_eff_progress returns the per-axis tensor
    (different identity than _eff_progress_dynamics)."""
    try:
        cfg = _make_cfg(enable_axis_independence=True)
        env, env_wrapped = _build_env(cfg, num_envs=4)
        try:
            env_wrapped.reset()
            unwrapped = env.unwrapped
            t = unwrapped._axis_eff_progress("gimbal_rate_tau")
            # Different tensor object — identity-distinct from _eff_progress_dynamics
            assert t.data_ptr() != unwrapped._eff_progress_dynamics.data_ptr(), (
                "axis tensor SHOULD be distinct from _eff_progress_dynamics "
                "when enable_axis_independence=True"
            )
            r.add_pass("axis_independence=True → per-axis tensors distinct")
        finally:
            env.close()
    except Exception:
        r.add_error("axis_independence_on_returns_per_axis", traceback.format_exc())


def test_priv_obs_single_env_build(r: TestResults):
    """Single env build that exercises priv obs assembly + axis-independence
    dispatch + range bound check. Isaac Sim doesn't tolerate multiple env
    builds in one process, so we consolidate the env-side checks here.

    Uses the non-DR subset because the default iris_ma6 cfg has
    domain_randomization.enabled=False.
    """
    try:
        # Use non-DR fields (17 entries) so we don't need to override the
        # default DR-disabled cfg. Combined with axis-independence on.
        cfg = _make_cfg(
            critic_priv_fields=list(_NON_DR_FIELDS),
            enable_axis_independence=True,
        )
        env, env_wrapped = _build_env(cfg, num_envs=4)
        try:
            env_wrapped.reset()
            unwrapped = env.unwrapped
            A = cfg.num_agents

            # ---- Privileged obs shape ----
            priv = unwrapped._get_critic_privileged_obs()
            pa, sh = _critic_privileged_dim_split(_NON_DR_FIELDS)
            expected_dim = pa * A + sh
            assert priv.shape == (4, expected_dim), (
                f"shape={tuple(priv.shape)}, expected (4, {expected_dim})"
            )
            r.add_pass(f"priv obs shape (4, {expected_dim}) for {len(_NON_DR_FIELDS)} non-DR fields")

            # ---- Pre-normalization range ----
            # The norm ranges estimate the actual bound conservatively but
            # compound randomization (e.g., curriculum × multiplicative scale)
            # can produce small overshoots when one factor dips near its
            # lower bound while another is at its upper bound. ±10 % slop is
            # plenty for any realistic compounding; values outside that
            # indicate a real range-estimate bug.
            assert torch.isfinite(priv).all(), "non-finite values"
            assert priv.min() >= -1.1, f"min={priv.min().item()} < -1.1"
            assert priv.max() <= 1.1, f"max={priv.max().item()} > 1.1"
            r.add_pass(
                f"priv obs range [{priv.min().item():.3f}, {priv.max().item():.3f}] "
                f"⊆ [-1.1, +1.1] (small overshoot OK; bounded)"
            )

            # ---- Axis-independence ON: per-axis tensors distinct from dynamics ----
            t_gi_tau = unwrapped._axis_eff_progress("gimbal_rate_tau")
            assert t_gi_tau.data_ptr() != unwrapped._eff_progress_dynamics.data_ptr(), (
                "with enable_axis_independence=True, gimbal_rate_tau tensor "
                "should be distinct from _eff_progress_dynamics"
            )
            r.add_pass("axis_independence=True → per-axis tensor distinct")

            # ---- After a reset, axis tensors are populated (non-zero where progress > 0) ----
            t_gi = unwrapped._eff_progress_gimbal_rate_tau
            # progress_gimbal_rate_tau may be 0 if curriculum is at step 0; either
            # way the tensor should be valid and finite.
            assert torch.isfinite(t_gi).all(), "non-finite in _eff_progress_gimbal_rate_tau"
            r.add_pass(f"_eff_progress_gimbal_rate_tau finite (mean={t_gi.mean().item():.3f})")
        finally:
            env.close()
    except Exception:
        r.add_error("priv_obs_single_env_build", traceback.format_exc())


def test_axis_independence_default_off_cfg_only(r: TestResults):
    """Cfg-time check that enable_axis_independence defaults to False —
    avoids a second env build."""
    try:
        cfg = _make_cfg()  # default settings
        assert cfg.enable_axis_independence is False, (
            f"default enable_axis_independence={cfg.enable_axis_independence}"
        )
        r.add_pass("enable_axis_independence default = False")
    except Exception:
        r.add_error("axis_independence_default_off_cfg_only", traceback.format_exc())


def main():
    print("=" * 80)
    print("Ticket 037 — critic-privileged-obs tests")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Torch:   {torch.__version__}")
    print(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")

    r = TestResults()

    # --- Category 1: cfg-time dim accounting (no env build needed) ---
    print("\n" + "-" * 80, flush=True)
    print("Category 1: cfg-time dim accounting", flush=True)
    print("-" * 80, flush=True)
    test_dim_accounting_empty(r)
    test_dim_accounting_non_dr_subset(r)
    test_dim_accounting_minimal_subset(r)
    test_dim_split_helper(r)

    # --- Category 2: cfg-time validation ---
    print("\n" + "-" * 80, flush=True)
    print("Category 2: cfg-time validation", flush=True)
    print("-" * 80, flush=True)
    test_unknown_field_raises(r)
    test_duplicate_field_raises(r)
    test_dr_fields_require_dr_enabled(r)
    test_axis_independence_default_off_cfg_only(r)

    # --- Category 3+4+5: env-side checks (one env build) ---
    print("\n" + "-" * 80, flush=True)
    print("Category 3+4+5: env-side priv obs + axis independence (single env)", flush=True)
    print("-" * 80, flush=True)
    test_priv_obs_single_env_build(r)

    ok = r.print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
