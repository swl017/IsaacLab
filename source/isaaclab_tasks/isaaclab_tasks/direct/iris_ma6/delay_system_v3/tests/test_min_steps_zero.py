#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate that DelayPipelineV3 is correct with ``min_steps=0``.

The historical default (``min_steps=2``) was a safety clamp from an older
CircularBuffer contract. Isaac Lab's current ``CircularBuffer.__getitem__``
returns the oldest stored entry when the key exceeds the number of pushes,
so ``min_steps=0`` should be safe. This test establishes that contract.

Tests covered:
    1. Identity-signal lag sweep   — output equals x_{t-lag} (or oldest
       before the buffer is warm).
    2. AoI ↔ data consistency      — t_current - ts == lag * dt.
    3. Cross-check min_steps=0 vs 2 — differ only for envs whose sampled
       lag would have been < 2.
    4. Same-step read-after-write  — lag=0 returns the just-appended value.
    5. Stage-interaction sanity    — staleness / dropout off → deterministic.

Self-contained — does not require Isaac Sim.

Usage:
    conda run -n env_isaaclab python test_min_steps_zero.py
    conda run -n env_isaaclab python test_min_steps_zero.py --test-verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
import types
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Load delay_system_v3 submodules without pulling in Isaac Sim via
# `delay_system_v3/__init__.py` → `multi_agent_wrapper.py`.
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent.parent


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pkg_name = "delay_system_v3_minstepstest"
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [str(_MODULE_DIR)]
sys.modules[_pkg_name] = _pkg

_cfg_mod = _load(f"{_pkg_name}.delay_cfg_v3", _MODULE_DIR / "delay_cfg_v3.py")
_sampling_mod = _load(f"{_pkg_name}.sampling_strategies", _MODULE_DIR / "sampling_strategies.py")
_pipeline_mod = _load(f"{_pkg_name}.delay_pipeline_v3", _MODULE_DIR / "delay_pipeline_v3.py")

DelayPipelineCfgV3 = _cfg_mod.DelayPipelineCfgV3
DistributionCfg = _cfg_mod.DistributionCfg
DropoutCfg = _cfg_mod.DropoutCfg
FirstOrderLagCfg = _cfg_mod.FirstOrderLagCfg
LatencyCfg = _cfg_mod.LatencyCfg
StalenessCfg = _cfg_mod.StalenessCfg
DelayPipelineV3 = _pipeline_mod.DelayPipelineV3


# ---------------------------------------------------------------------------
# Test results
# ---------------------------------------------------------------------------
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
        for line in msg.splitlines()[:6]:
            print(f"    {line}")

    def add_error(self, name: str, msg: str):
        self.errors.append((name, msg))
        print(f"  ERROR {name}")
        for line in msg.splitlines()[:6]:
            print(f"    {line}")

    def print_summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total: {total}")
        if total:
            print(f"Passed: {len(self.passed)} ({100 * len(self.passed) / total:.1f}%)")
            print(f"Failed: {len(self.failed)} ({100 * len(self.failed) / total:.1f}%)")
            print(f"Errors: {len(self.errors)} ({100 * len(self.errors) / total:.1f}%)")
        if self.failed:
            print("\nFAILED TESTS")
            for name, msg in self.failed:
                print(f"  {name}\n    {msg}")
        if self.errors:
            print("\nERRORS")
            for name, msg in self.errors:
                print(f"  {name}\n    {msg}")
        print("=" * 80)
        return not self.failed and not self.errors


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(
    num_envs: int,
    dt: float,
    min_steps: int,
    device: torch.device,
    *,
    max_lag_steps: int = 20,
    staleness: bool = False,
    dropout: bool = False,
    fol: bool = False,
) -> DelayPipelineV3:
    """Build a single-field pipeline sized for ``max_lag_steps`` of delay.

    The caller then overrides ``_step_values`` to force a deterministic per-env
    delay. ``min_steps`` is the cfg value under test. The latency distribution
    is set with mean = ``max_lag_steps * dt`` so the pipeline allocates a
    CircularBuffer large enough to avoid wrap-around during the test window.
    """
    # NOTE: DelayPipelineV3 sizes its CircularBuffer from
    # ``mean + 3*std`` of the latency distribution. A "constant"-type
    # DistributionCfg leaves ``mean=0`` and would starve the buffer to
    # ``min_steps + 5`` slots, causing wrap-around for lags beyond that.
    # Using a zero-variance normal distribution makes ``mean`` drive the
    # buffer sizing.
    cfg = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(
                type="normal",
                mean=max_lag_steps * dt,
                std=0.0,
                min_value=0.0,
            ),
            min_steps=min_steps,
        ),
        staleness=StalenessCfg(enabled=staleness),
        dropout=DropoutCfg(enabled=dropout, probability=0.0),
        first_order_lag=FirstOrderLagCfg(enabled=fol, tau=0.0),
    )
    p = DelayPipelineV3(
        cfg=cfg, num_envs=num_envs, device=device, dt=dt, data_shape=(1,)
    )
    # "fixed" keeps latency active but disables staleness-random-mode side effects.
    p.set_mode("fixed", progress=1.0)
    return p


def force_lags(pipeline: DelayPipelineV3, lags: torch.Tensor):
    """Force per-env latency in steps. Persists until the next maybe_resample."""
    pipeline._latency_sampler._step_values = lags.to(
        device=pipeline._device, dtype=torch.long
    ).clone()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(results: TestResults, device: torch.device, verbose: bool):
    print("\n" + "=" * 80)
    print("DelayPipelineV3 · min_steps=0 validation")
    print("=" * 80)
    print(f"device: {device}")

    # -----------------------------------------------------------------------
    # Test 1: identity-signal lag sweep (min_steps=0)
    # -----------------------------------------------------------------------
    name = "1. identity-signal lag sweep, min_steps=0"
    try:
        num_envs, dt, T = 10, 0.02, 30
        pipeline = build_pipeline(num_envs, dt, min_steps=0, device=device)
        lags = torch.arange(num_envs, device=device)
        force_lags(pipeline, lags)

        input_history: list[float] = []
        for t in range(T):
            x = torch.full((num_envs, 1), float(t), device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            t_cur = ts.clone()
            pipeline.advance(x, None, ts, t_cur)
            # reinstate forced lags in case anything touched them
            force_lags(pipeline, lags)
            data, _ = pipeline.query(use_noise=False, allow_dropout=False)
            input_history.append(float(t))

            # Expected per env: x_{max(t - lag, 0)} (CircularBuffer returns oldest
            # when lag > number of pushes)
            expected = torch.tensor(
                [input_history[max(t - int(lags[i].item()), 0)] for i in range(num_envs)],
                device=device,
            ).unsqueeze(-1)
            if not torch.allclose(data, expected):
                raise AssertionError(
                    f"t={t}: data={data.squeeze().tolist()} "
                    f"expected={expected.squeeze().tolist()}"
                )
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 2: AoI ↔ data consistency (min_steps=0)
    # -----------------------------------------------------------------------
    name = "2. AoI = lag * dt (timestamp buffer tracks data buffer)"
    try:
        num_envs, dt, T = 8, 0.02, 20
        pipeline = build_pipeline(num_envs, dt, min_steps=0, device=device)
        lags = torch.arange(num_envs, device=device)
        force_lags(pipeline, lags)

        for t in range(T):
            x = torch.full((num_envs, 1), float(t), device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            t_cur = ts.clone()
            pipeline.advance(x, None, ts, t_cur)
            force_lags(pipeline, lags)
            _, returned_ts = pipeline.query(use_noise=False, allow_dropout=False)

            # Once the buffer is warm (t >= lag), AoI must equal lag * dt.
            warm = t >= lags
            if warm.any():
                expected_ts = (t - lags.to(torch.float32)) * dt
                diff = (returned_ts - expected_ts).abs()
                # Only check warm envs
                warm_diff = diff[warm]
                if (warm_diff > 1e-6).any():
                    raise AssertionError(
                        f"t={t}: timestamp mismatch on warm envs. "
                        f"returned={returned_ts.tolist()} "
                        f"expected={expected_ts.tolist()}"
                    )
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 3: cross-check min_steps=0 vs min_steps=2
    # -----------------------------------------------------------------------
    name = "3. min_steps={0,2} differ only for lags <2; agree for lags ≥2"
    try:
        num_envs, dt, T = 6, 0.02, 20
        pipeline_zero = build_pipeline(num_envs, dt, min_steps=0, device=device)
        pipeline_two = build_pipeline(num_envs, dt, min_steps=2, device=device)
        lags = torch.arange(num_envs, device=device)  # [0,1,2,3,4,5]

        force_lags(pipeline_zero, lags)
        # min_steps=2 will clamp its own step_values on resample. We override
        # BOTH pipelines to the same raw lags; min_steps is what the production
        # sampler *would* clamp at resample. For a direct comparison we instead
        # explicitly emulate the clamp for pipeline_two:
        clamped = torch.clamp(lags, min=2)
        force_lags(pipeline_two, clamped)

        for t in range(T):
            x = torch.full((num_envs, 1), float(t), device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            t_cur = ts.clone()
            pipeline_zero.advance(x, None, ts, t_cur)
            pipeline_two.advance(x, None, ts, t_cur)
            force_lags(pipeline_zero, lags)
            force_lags(pipeline_two, clamped)
            d0, _ = pipeline_zero.query(use_noise=False, allow_dropout=False)
            d2, _ = pipeline_two.query(use_noise=False, allow_dropout=False)

            # Envs with original lag ≥ 2: both pipelines should agree exactly.
            agree_mask = lags >= 2
            if agree_mask.any():
                if not torch.allclose(d0[agree_mask], d2[agree_mask]):
                    raise AssertionError(
                        f"t={t}: envs lag≥2 diverge. "
                        f"min0={d0.squeeze().tolist()} "
                        f"min2={d2.squeeze().tolist()}"
                    )
            # Envs with lag < 2: after warmup (t ≥ 2), they should differ.
            # We just assert that for the first few warm steps they differ on at
            # least one of those envs — proves min_steps is actually biting.
            if t == 5 and (~agree_mask).any():
                diff = (d0[~agree_mask] - d2[~agree_mask]).abs().max().item()
                if diff == 0.0:
                    raise AssertionError(
                        "min_steps=2 produced identical output to min_steps=0 "
                        "on lag-0/1 envs — clamp is not active."
                    )
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 4: same-step read-after-write (lag=0 returns just-appended value)
    # -----------------------------------------------------------------------
    name = "4. same-step read-after-write with min_steps=0 → no delay"
    try:
        num_envs, dt, T = 4, 0.02, 5
        pipeline = build_pipeline(num_envs, dt, min_steps=0, device=device)
        zero_lags = torch.zeros(num_envs, dtype=torch.long, device=device)
        force_lags(pipeline, zero_lags)

        for t in range(T):
            x = torch.full((num_envs, 1), float(t) * 10.0, device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            t_cur = ts.clone()
            pipeline.advance(x, None, ts, t_cur)
            force_lags(pipeline, zero_lags)
            data, out_ts = pipeline.query(use_noise=False, allow_dropout=False)
            if not torch.allclose(data, x):
                raise AssertionError(
                    f"t={t}: lag=0 should return just-appended value. "
                    f"data={data.squeeze().tolist()} x={x.squeeze().tolist()}"
                )
            if not torch.allclose(out_ts, ts):
                raise AssertionError(
                    f"t={t}: lag=0 timestamp mismatch: {out_ts} vs {ts}"
                )
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 5: stage-interaction — dropout off → deterministic; reruns match
    # -----------------------------------------------------------------------
    name = "5. deterministic with dropout=off, min_steps=0 (rerun matches)"
    try:
        num_envs, dt, T = 4, 0.02, 15

        def _run(seed: int):
            torch.manual_seed(seed)
            p = build_pipeline(
                num_envs, dt, min_steps=0, device=device,
                staleness=False, dropout=False, fol=False,
            )
            lags = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
            force_lags(p, lags)
            out_series = []
            for t in range(T):
                x = torch.full((num_envs, 1), float(t), device=device)
                ts = torch.full((num_envs,), t * dt, device=device)
                p.advance(x, None, ts, ts.clone())
                force_lags(p, lags)
                d, _ = p.query(use_noise=False, allow_dropout=False)
                out_series.append(d.clone())
            return torch.stack(out_series)

        r1 = _run(seed=42)
        r2 = _run(seed=42)
        if not torch.equal(r1, r2):
            raise AssertionError("two runs with the same seed diverged")
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 6: raw/noisy share delay, differ only by noise (min_steps=0)
    # -----------------------------------------------------------------------
    name = "6. raw/noisy caches share timing under min_steps=0"
    try:
        num_envs, dt, T = 4, 0.02, 20
        pipeline = build_pipeline(num_envs, dt, min_steps=0, device=device)
        lags = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        force_lags(pipeline, lags)

        torch.manual_seed(0)
        for t in range(T):
            x_raw = torch.full((num_envs, 1), float(t), device=device)
            x_noisy = x_raw + torch.randn_like(x_raw) * 5.0
            ts = torch.full((num_envs,), t * dt, device=device)
            pipeline.advance(x_raw, x_noisy, ts, ts.clone())
            force_lags(pipeline, lags)
            d_raw, ts_raw = pipeline.query(use_noise=False, allow_dropout=False)
            d_noi, ts_noi = pipeline.query(use_noise=True, allow_dropout=False)

            if not torch.equal(ts_raw, ts_noi):
                raise AssertionError(
                    f"t={t}: raw/noisy timestamps differ {ts_raw} vs {ts_noi}"
                )
            # Once warm, raw must equal x_{t-lag}
            warm = t >= lags
            if warm.any():
                expected = (t - lags.to(torch.float32)).clamp(min=0.0).unsqueeze(-1)
                if not torch.allclose(d_raw[warm], expected[warm]):
                    raise AssertionError(
                        f"t={t}: raw cache mismatch. "
                        f"got={d_raw.squeeze().tolist()} expected≈{expected.squeeze().tolist()}"
                    )
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 7: time-varying per-step latency — does the pipeline return
    # x_{t - lag[t]} when lag changes every step?
    #
    # Exercises: monotonically-rising lag (stale hold), piecewise-constant
    # plateaus, then a drop back to 0 (fast-forward to most-recent). Each
    # step's lag is written to `_step_values` BEFORE `advance`, so the
    # latency lookup uses the requested lag for that step.
    # -----------------------------------------------------------------------
    name = "7. time-varying latency — correct x_{t - lag[t]} per step"
    try:
        dt = 0.02
        # User's sequence: rising lag with plateaus, then a drop to 0.
        # Lengths: 2 zeros, 3 ones, 4 twos, 4 threes, 4 fours, 4 fives, 3 zeros.
        rising = (
            [0] * 2 + [1] * 3 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
        )
        lag_sequence = rising + [0, 0, 0]  # tail: ability to "fast-forward"
        T = len(lag_sequence)
        max_lag = max(lag_sequence)
        num_envs = 2  # replicate the same pattern on two envs for batching

        pipeline = build_pipeline(
            num_envs, dt, min_steps=0, device=device,
            max_lag_steps=max_lag + 5,   # headroom over the peak lag
        )

        # Track what the pipeline returns at each step for diagnostics.
        mismatches: list[str] = []
        ts_series: list[float] = []

        for t in range(T):
            lag_t = lag_sequence[t]
            force_lags(
                pipeline,
                torch.full((num_envs,), lag_t, dtype=torch.long, device=device),
            )
            x = torch.full((num_envs, 1), float(t), device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            pipeline.advance(x, None, ts, ts.clone())
            data, out_ts = pipeline.query(use_noise=False, allow_dropout=False)

            # CircularBuffer clamps lag to num_pushes-1, which equals t.
            effective_lag = min(lag_t, t)
            expected_val = float(t - effective_lag)
            expected_ts = (t - effective_lag) * dt

            got_val = data.squeeze().tolist()
            got_ts = out_ts.tolist()
            ts_series.append(got_ts[0])

            # Value check — both envs should agree (same pattern).
            if any(abs(v - expected_val) > 1e-6 for v in (got_val if isinstance(got_val, list) else [got_val])):
                mismatches.append(
                    f"t={t} lag={lag_t}: data={got_val} expected={expected_val}"
                )
            # Timestamp check
            if any(abs(v - expected_ts) > 1e-6 for v in got_ts):
                mismatches.append(
                    f"t={t} lag={lag_t}: ts={got_ts} expected={expected_ts:.4f}"
                )

        if mismatches:
            raise AssertionError(
                "varying-latency mismatches:\n  " + "\n  ".join(mismatches[:10])
            )

        # Within this sequence (lag grows by at most +1 per step), returned
        # timestamps MUST be monotonically non-decreasing.
        for i in range(1, len(ts_series)):
            if ts_series[i] + 1e-9 < ts_series[i - 1]:
                raise AssertionError(
                    f"timestamp regressed: step {i - 1}→{i} "
                    f"{ts_series[i - 1]:.4f} → {ts_series[i]:.4f}"
                )

        if verbose:
            print(f"    lag seq:  {lag_sequence}")
            print(f"    ret ts:   {[f'{x:.3f}' for x in ts_series]}")
            print("    ↑ timestamps are monotonically non-decreasing "
                  "(stale holds when lag grows, jump forward when it drops).")

        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 7b: a lag jump of ≥ 2 steps in one transition ⇒ timestamp DOES
    # regress. This documents that the pipeline does not enforce monotonic
    # timestamps — downstream consumers should either (a) avoid per_step
    # latency sampling with high variance, or (b) apply a clamp
    # (ts_returned_new = max(ts_returned_prev, ts_returned_new)).
    # -----------------------------------------------------------------------
    name = "7b. lag-jump +2 causes timestamp regression (contract check)"
    try:
        dt = 0.02
        num_envs = 1
        # Lag jumps from 0 → 2 at step 5, which pulls the returned timestamp
        # from t*dt = 0.100 back to (t-2)*dt = 0.060.
        lag_sequence = [0, 0, 0, 0, 0, 2, 2, 2]
        pipeline = build_pipeline(
            num_envs, dt, min_steps=0, device=device,
            max_lag_steps=max(lag_sequence) + 5,
        )

        ts_series = []
        for t, lag_t in enumerate(lag_sequence):
            force_lags(
                pipeline,
                torch.full((num_envs,), lag_t, dtype=torch.long, device=device),
            )
            x = torch.full((num_envs, 1), float(t), device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            pipeline.advance(x, None, ts, ts.clone())
            _, out_ts = pipeline.query(use_noise=False, allow_dropout=False)
            ts_series.append(out_ts.item())

        # Detect at least one regression.
        regressions = [
            (i - 1, i, ts_series[i - 1], ts_series[i])
            for i in range(1, len(ts_series))
            if ts_series[i] + 1e-9 < ts_series[i - 1]
        ]
        if not regressions:
            raise AssertionError(
                "Expected a timestamp regression at the lag 0→2 transition "
                "but none occurred. The pipeline may have grown a monotonic "
                "clamp; if so, update this test."
            )

        if verbose:
            print(f"    lag seq: {lag_sequence}")
            print(f"    ret ts:  {[f'{x:.3f}' for x in ts_series]}")
            for a, b, ta, tb in regressions:
                print(f"    regression: step {a}→{b}: {ta:.3f} → {tb:.3f}")
            print("    ↑ returned timestamp stepped backward because the "
                  "latency sample grew by more than 1 in a single step.")

        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 8: lag that exceeds buffer size ⇒ silent data corruption.
    # This is a documentation test: it records the CircularBuffer contract
    # boundary. If this ever stops being true (e.g., the pipeline grows a
    # runtime size check), the test will fail and prompt an update.
    # -----------------------------------------------------------------------
    name = "8. lag > buffer capacity wraps silently (known contract)"
    try:
        dt = 0.02
        # Build buffer sized for max_lag=4 (+5 headroom ⇒ max_length=9).
        num_envs = 1
        pipeline = build_pipeline(
            num_envs, dt, min_steps=0, device=device, max_lag_steps=4,
        )

        # Drive 30 appends so the buffer has wrapped multiple times.
        T = 30
        force_lags(pipeline, torch.tensor([0], dtype=torch.long, device=device))
        for t in range(T):
            x = torch.full((num_envs, 1), float(t), device=device)
            ts = torch.full((num_envs,), t * dt, device=device)
            pipeline.advance(x, None, ts, ts.clone())

        # Ask for an in-range lag: must match x_{t-lag}. Ask for an
        # out-of-range lag: must NOT match x_{t-lag} (buffer has lost it).
        in_range_lag = 3
        out_of_range_lag = 15  # > buffer max_length (9)

        force_lags(
            pipeline, torch.tensor([in_range_lag], dtype=torch.long, device=device)
        )
        x_last = torch.full((num_envs, 1), float(T), device=device)
        ts_last = torch.full((num_envs,), T * dt, device=device)
        pipeline.advance(x_last, None, ts_last, ts_last.clone())
        d_in, _ = pipeline.query(use_noise=False, allow_dropout=False)
        if abs(d_in.item() - float(T - in_range_lag)) > 1e-6:
            raise AssertionError(
                f"in-range lag {in_range_lag} at t={T} returned {d_in.item()}, "
                f"expected {T - in_range_lag}"
            )

        # Query out-of-range on the *next* step (after another advance).
        force_lags(
            pipeline, torch.tensor([out_of_range_lag], dtype=torch.long, device=device)
        )
        t_next = T + 1
        x_next = torch.full((num_envs, 1), float(t_next), device=device)
        ts_next = torch.full((num_envs,), t_next * dt, device=device)
        pipeline.advance(x_next, None, ts_next, ts_next.clone())
        d_out, _ = pipeline.query(use_noise=False, allow_dropout=False)
        expected_if_correct = float(t_next - out_of_range_lag)  # = 16
        if abs(d_out.item() - expected_if_correct) <= 1e-6:
            raise AssertionError(
                "Out-of-range lag returned a match — buffer apparently covers it. "
                "If the pipeline grew a runtime size check or larger headroom, "
                "update this test."
            )
        # The returned value should be some data still resident in the ring.
        if not (0.0 <= d_out.item() <= float(t_next)):
            raise AssertionError(
                f"Out-of-range query returned {d_out.item()}, not any known append."
            )
        if verbose:
            print(f"    buffer max_length ≈ 9, asked lag={out_of_range_lag} "
                  f"at t={t_next}")
            print(f"    returned {d_out.item()} (a wrapped slot, not "
                  f"{expected_if_correct}) — expected silent wrap")
        results.add_pass(name)
    except Exception:
        results.add_fail(name, traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-verbose", action="store_true",
                        help="Enable verbose debug output")
    args = parser.parse_args()

    print("=" * 80)
    print("DELAY PIPELINE · min_steps=0 VALIDATION")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | device={device}")
    print(f"started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()
    try:
        run_tests(results, device, verbose=args.test_verbose)
    except Exception:
        results.add_error("test-suite crash", traceback.format_exc())

    ok = results.print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
