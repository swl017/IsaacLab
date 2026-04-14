# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-cache regression tests for DelayPipelineV3.

Verifies ticket 029's structural fix for the shared-pipeline idempotency
bug: when reward and observation queries hit the same pipeline in the
same sim step, the second call must NOT overwrite (or be overwritten by)
the first call's cached payload. Both caches must be populated from a
single ``advance`` and the ``use_noise`` flag must select the matching
cache slot.

Invariants verified here:
1. Raw and noisy payloads share one delay realization (same latency,
   same staleness, same dropout). Timestamps match across use_noise.
2. When a noisy payload is provided, ``query(use_noise=True)`` returns it
   delayed and ``query(use_noise=False)`` returns the raw delayed — the
   two differ by the noise, not by a time slip.
3. When ``noisy_data=None`` (clean-only field — state field with no
   curriculum noise, or a test-only construction), both cache slots
   mirror the raw payload; obs query and reward query return the same
   data.
4. Idempotency: a second ``advance`` call at the same ``t_current`` with
   a different noisy payload does NOT change the caches — the first
   advance wins. This matches the "single advance per step" contract.
"""

from __future__ import annotations

import traceback

import torch

from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
    DelayPipelineCfgV3,
    DistributionCfg,
    DropoutCfg,
    FirstOrderLagCfg,
    LatencyCfg,
    SamplingCfg,
    StalenessCfg,
)
from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_pipeline_v3 import (
    DelayPipelineV3,
)


def _make_pipeline(
    num_envs: int,
    device: torch.device,
    *,
    dt: float = 0.04,
    data_shape=(3,),
    latency_steps: int = 2,
    dropout_prob: float = 0.0,
    staleness_enabled: bool = False,
    fol_enabled: bool = False,
) -> DelayPipelineV3:
    latency_s = latency_steps * dt
    cfg = DelayPipelineCfgV3(
        latency=LatencyCfg(
            enabled=True,
            distribution=DistributionCfg(type="constant", value=latency_s),
            min_steps=latency_steps,
        ),
        staleness=StalenessCfg(enabled=staleness_enabled),
        dropout=DropoutCfg(
            enabled=dropout_prob > 0.0,
            probability=dropout_prob,
            sampling=SamplingCfg(frequency="per_step"),
        ),
        first_order_lag=FirstOrderLagCfg(enabled=fol_enabled, tau=0.1 if fol_enabled else 0.0),
    )
    pipe = DelayPipelineV3(cfg, num_envs, device, dt, data_shape)
    pipe.set_mode("fixed", progress=1.0)
    return pipe


def run_dual_cache_tests(results, device: torch.device, verbose: bool = False):
    """Run dual-cache regression tests."""
    results.set_suite("Dual-Cache Regression (ticket 029)")

    num_envs = 16
    dt = 0.04
    data_shape = (3,)

    # -------------------------------------------------------------------
    # Test 1: raw and noisy cached separately under the SAME delay
    # -------------------------------------------------------------------
    # Advance with a raw/noisy pair. Reward and obs queries must:
    #   - return different *data* (noise is preserved)
    #   - return the SAME *timestamp* (single delay realization)
    try:
        pipe = _make_pipeline(num_envs, device, dt=dt, latency_steps=2)

        # Populate buffer with 3 steps so the 2-step lag pulls from known data
        t = torch.zeros(num_envs, device=device)
        for step in range(3):
            raw = torch.full((num_envs, 3), float(step), device=device)
            noisy = raw + torch.full((num_envs, 3), 10.0 * (step + 1), device=device)
            pipe.process(raw, noisy, t.clone(), t.clone())
            t = t + dt

        # After 3 advances at t={0,dt,2dt}, the 2-step-delayed cache should be
        # step=0. Reward query returns raw[0] = 0.0; obs query returns noisy[0] = 10.0.
        reward_data, reward_ts = pipe.query(use_noise=False, allow_dropout=False)
        obs_data, obs_ts = pipe.query(use_noise=True, allow_dropout=False)

        assert torch.allclose(reward_data, torch.zeros_like(reward_data)), (
            f"reward cache should hold raw step=0 (=0.0), got {reward_data[0]}"
        )
        assert torch.allclose(obs_data, torch.full_like(obs_data, 10.0)), (
            f"obs cache should hold noisy step=0 (=10.0), got {obs_data[0]}"
        )
        assert torch.allclose(reward_ts, obs_ts), (
            "reward_ts and obs_ts must match (shared delay realization) — "
            f"reward_ts={reward_ts[0].item()}, obs_ts={obs_ts[0].item()}"
        )
        results.add_pass("raw/noisy caches populated with shared delay")
    except Exception:
        results.add_fail(
            "raw/noisy caches populated with shared delay",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 2: idempotency guard — second advance same step is a no-op
    # -------------------------------------------------------------------
    # This is the key structural behavior. A second advance call at the same
    # t_current with a DIFFERENT noisy payload must not overwrite the cache.
    # The first call wins (reward-then-obs ordering in the env).
    try:
        pipe = _make_pipeline(num_envs, device, dt=dt, latency_steps=2)

        t = torch.zeros(num_envs, device=device)
        raw = torch.full((num_envs, 3), 1.0, device=device)
        noisy_first = torch.full((num_envs, 3), 2.0, device=device)
        noisy_second = torch.full((num_envs, 3), 999.0, device=device)

        # First advance at t=0 caches (raw, noisy_first)
        pipe.advance(raw, noisy_first, t.clone(), t.clone())
        out_noisy_after_first, _ = pipe.query(use_noise=True, allow_dropout=False)

        # Second advance at the SAME t=0 with different noisy → guard skips
        pipe.advance(raw, noisy_second, t.clone(), t.clone())
        out_noisy_after_second, _ = pipe.query(use_noise=True, allow_dropout=False)

        assert torch.allclose(out_noisy_after_first, out_noisy_after_second), (
            "second advance at same t_current must not change cached noisy data"
        )
        # Additional specificity: the cache must hold noisy_first's content
        # (after delay — with a fresh 2-step-lag buffer, all slots were
        # initialized with the first payload, so the lag read is noisy_first).
        assert torch.allclose(
            out_noisy_after_first,
            torch.full_like(out_noisy_after_first, 2.0),
        ), "first advance's noisy payload should be cached (not overwritten)"
        results.add_pass("idempotency guard ignores second advance this step")
    except Exception:
        results.add_fail(
            "idempotency guard ignores second advance this step",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 3: reward-first then obs-second — the 2be3 / ticket-020 scenario
    # -------------------------------------------------------------------
    # This reproduces the exact pattern that used to silently discard noise:
    # _get_rewards() advances with raw (use_noise=False), then
    # _get_observations() advances with noisy (use_noise=True). The second
    # call hits the guard; with dual-cache, the obs query still returns the
    # noisy payload because it was cached during the first advance.
    try:
        pipe = _make_pipeline(num_envs, device, dt=dt, latency_steps=2)

        t = torch.zeros(num_envs, device=device)

        for step in range(3):
            raw = torch.full((num_envs, 3), float(step), device=device)
            noisy = raw + torch.full((num_envs, 3), 100.0, device=device)

            # Simulate env-step ordering: reward call first (raw), obs call second (noisy).
            # In production UnifiedDelaySystem.get_delayed reads both payloads from
            # storage and feeds them into a single advance; here we reproduce that.
            pipe.advance(raw, noisy, t.clone(), t.clone())

            # Now query reward and obs in the order DirectRLEnv actually does.
            reward_val, reward_ts = pipe.query(use_noise=False, allow_dropout=False)
            obs_val, obs_ts = pipe.query(use_noise=True, allow_dropout=True)

            if step == 2:  # after warmup, latency is 2 steps → cache holds step=0
                assert torch.allclose(reward_val, torch.zeros_like(reward_val)), (
                    f"reward_val should be raw at step=0, got {reward_val[0]}"
                )
                assert torch.allclose(obs_val, torch.full_like(obs_val, 100.0)), (
                    f"obs_val should be noisy at step=0 (=100.0), got {obs_val[0]}"
                )
                assert torch.allclose(reward_ts, obs_ts), "timestamps must match"

            t = t + dt

        results.add_pass("reward-then-obs query ordering returns distinct payloads")
    except Exception:
        results.add_fail(
            "reward-then-obs query ordering returns distinct payloads",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 4: clean-only field (noisy_data=None) → obs mirrors raw
    # -------------------------------------------------------------------
    # Fields with no separate noisy payload collapse the dual-cache
    # trivially — obs query returns the same data as reward query.
    try:
        pipe = _make_pipeline(num_envs, device, dt=dt, latency_steps=2)

        t = torch.zeros(num_envs, device=device)
        for step in range(3):
            raw = torch.full((num_envs, 3), float(step * 7), device=device)
            pipe.advance(raw, None, t.clone(), t.clone())
            t = t + dt

        reward_val, reward_ts = pipe.query(use_noise=False, allow_dropout=False)
        obs_val, obs_ts = pipe.query(use_noise=True, allow_dropout=False)

        assert torch.allclose(reward_val, obs_val), (
            "clean-only field: obs cache must mirror raw cache"
        )
        assert torch.allclose(reward_ts, obs_ts), (
            "clean-only field: timestamps must match"
        )
        results.add_pass("clean-only field (noisy_data=None) mirrors raw into obs cache")
    except Exception:
        results.add_fail(
            "clean-only field (noisy_data=None) mirrors raw into obs cache",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 5: shared dropout mask — when a step drops, both payloads drop
    # -------------------------------------------------------------------
    # Verifies the dropout stage applies one mask to both raw and noisy.
    # Uses probability=1.0 so every step is dropped; the "held" buffers
    # should preserve raw and noisy jointly.
    try:
        pipe = _make_pipeline(
            num_envs, device, dt=dt,
            latency_steps=1,   # minimal latency so we isolate dropout behavior
            dropout_prob=1.0,
        )

        t = torch.zeros(num_envs, device=device)

        # Step 0: seed with a distinctive (raw, noisy) pair
        raw_seed = torch.full((num_envs, 3), 5.0, device=device)
        noisy_seed = torch.full((num_envs, 3), 55.0, device=device)
        pipe.advance(raw_seed, noisy_seed, t.clone(), t.clone())

        # Step 1: new payload, but dropout=1.0 should force hold
        t = t + dt
        raw_new = torch.full((num_envs, 3), 99.0, device=device)
        noisy_new = torch.full((num_envs, 3), 999.0, device=device)
        pipe.advance(raw_new, noisy_new, t.clone(), t.clone())

        # After-dropout reads should return the seed (held) for BOTH payloads.
        reward_held, _ = pipe.query(use_noise=False, allow_dropout=True)
        obs_held, _ = pipe.query(use_noise=True, allow_dropout=True)

        assert torch.allclose(reward_held, raw_seed), (
            f"dropout should hold raw seed, got {reward_held[0]}"
        )
        assert torch.allclose(obs_held, noisy_seed), (
            f"dropout should hold noisy seed, got {obs_held[0]}"
        )
        results.add_pass("shared dropout mask holds raw and noisy jointly")
    except Exception:
        results.add_fail(
            "shared dropout mask holds raw and noisy jointly",
            traceback.format_exc(),
        )

    # -------------------------------------------------------------------
    # Test 6: shape mismatch between raw and noisy raises
    # -------------------------------------------------------------------
    try:
        pipe = _make_pipeline(num_envs, device, dt=dt, latency_steps=2)

        t = torch.zeros(num_envs, device=device)
        raw = torch.zeros(num_envs, 3, device=device)
        bad_noisy = torch.zeros(num_envs, 4, device=device)  # wrong shape

        raised = False
        try:
            pipe.advance(raw, bad_noisy, t, t)
        except ValueError:
            raised = True
        assert raised, "shape mismatch should raise ValueError"
        results.add_pass("shape mismatch raises ValueError")
    except Exception:
        results.add_fail("shape mismatch raises ValueError", traceback.format_exc())
