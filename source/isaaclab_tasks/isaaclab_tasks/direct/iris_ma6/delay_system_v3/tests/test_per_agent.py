# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-agent randomization feature.

These tests verify:
1. Per-agent samplers create independent parameters per agent
2. Curriculum scaling works correctly
3. Distribution types (uniform, normal) function properly
4. Max values are respected
5. Reset resamples parameters correctly
6. Statistical independence between agents
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_tests import TestResults


def run_per_agent_randomization_tests(
    results: "TestResults", device: torch.device, verbose: bool = False
):
    """Run all per-agent randomization tests.

    Args:
        results: Test results tracker.
        device: Torch device.
        verbose: Enable verbose output.
    """
    results.set_suite("Per-Agent Randomization Tests")

    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.delay_cfg_v3 import (
        DistributionCfg,
        SamplingCfg,
    )
    from isaaclab_tasks.direct.iris_ma6.delay_system_v3.sampling_strategies import (
        PerAgentDistributionSampler,
        PerAgentParameterSampler,
        PerAgentLatencySampler,
        PerAgentStalenessSampler,
    )

    num_envs = 16
    num_agents = 3
    dt = 0.04

    # Test 1: PerAgentDistributionSampler initialization and shape
    try:
        cfg = DistributionCfg(type="uniform", mean=0.1, half_range=0.05)
        sampler = PerAgentDistributionSampler(cfg, num_envs, num_agents, device)

        values = sampler.sample()
        assert values.shape == (num_envs, num_agents), \
            f"Expected shape ({num_envs}, {num_agents}), got {values.shape}"

        # Check values are in expected range
        assert (values >= 0.05).all() and (values <= 0.15).all(), \
            f"Values out of range [0.05, 0.15]: min={values.min()}, max={values.max()}"

        results.add_pass("PerAgentDistributionSampler initialization")
    except Exception as e:
        results.add_fail("PerAgentDistributionSampler initialization", str(e))

    # Test 2: Per-agent scaled sampling
    try:
        cfg = DistributionCfg(type="constant", value=1.0)
        sampler = PerAgentDistributionSampler(cfg, num_envs, num_agents, device)

        # At scale=0.5, values should be 0.5
        values_half = sampler.sample_scaled(scale=0.5)
        assert values_half.shape == (num_envs, num_agents)
        assert torch.allclose(values_half, torch.full((num_envs, num_agents), 0.5, device=device))

        # At scale=1.0, values should be 1.0
        values_full = sampler.sample_scaled(scale=1.0)
        assert torch.allclose(values_full, torch.full((num_envs, num_agents), 1.0, device=device))

        results.add_pass("Per-agent scaled sampling")
    except Exception as e:
        results.add_fail("Per-agent scaled sampling", str(e))

    # Test 3: Per-agent selective sampling (env_ids, agent_ids)
    try:
        cfg = DistributionCfg(type="uniform", mean=0.5, half_range=0.3)
        sampler = PerAgentDistributionSampler(cfg, num_envs, num_agents, device)

        # Sample only for specific envs and agents
        env_ids = torch.tensor([0, 2, 5], device=device)
        agent_ids = torch.tensor([1], device=device)

        values = sampler.sample_scaled(scale=1.0, env_ids=env_ids, agent_ids=agent_ids)

        # Should return subset shape (len(env_ids), len(agent_ids))
        assert values.shape == (len(env_ids), len(agent_ids)), \
            f"Expected shape ({len(env_ids)}, {len(agent_ids)}), got {values.shape}"

        # Values should be in expected range
        assert (values >= 0.2).all() and (values <= 0.8).all()

        results.add_pass("Per-agent selective sampling")
    except Exception as e:
        results.add_fail("Per-agent selective sampling", str(e))

    # Test 4: PerAgentParameterSampler per-episode resampling
    try:
        cfg_dist = DistributionCfg(type="normal", mean=0.1, std=0.02)
        cfg_samp = SamplingCfg(frequency="per_episode")
        sampler = PerAgentParameterSampler(cfg_dist, cfg_samp, num_envs, num_agents, device)
        sampler.initialize()

        assert sampler.values.shape == (num_envs, num_agents)

        initial_values = sampler.values.clone()

        # Call maybe_resample with no reset - values shouldn't change
        sampler.maybe_resample(reset_env_ids=None)
        assert torch.allclose(sampler.values, initial_values)

        results.add_pass("PerAgentParameterSampler per-episode behavior")
    except Exception as e:
        results.add_fail("PerAgentParameterSampler per-episode behavior", str(e))

    # Test 5: PerAgentLatencySampler step conversion
    try:
        # Use 0.12s which gives 3 steps cleanly (0.12 / 0.04 = 3.0)
        cfg_dist = DistributionCfg(type="constant", value=0.12)
        cfg_samp = SamplingCfg(frequency="per_episode")
        sampler = PerAgentLatencySampler(
            cfg_dist, cfg_samp, num_envs, num_agents, device, dt=dt, min_steps=2
        )
        sampler.initialize()

        # Should have shape (num_envs, num_agents)
        assert sampler.step_values.shape == (num_envs, num_agents), \
            f"Expected shape ({num_envs}, {num_agents}), got {sampler.step_values.shape}"

        # 0.12s / 0.04s = 3 steps
        expected_steps = 3
        assert (sampler.step_values == expected_steps).all(), \
            f"Expected {expected_steps} steps, got {sampler.step_values}"

        results.add_pass("PerAgentLatencySampler step conversion")
    except Exception as e:
        results.add_fail("PerAgentLatencySampler step conversion", str(e))

    # Test 6: PerAgentLatencySampler min_steps enforcement
    try:
        cfg_dist = DistributionCfg(type="constant", value=0.02)  # 0.5 steps
        cfg_samp = SamplingCfg(frequency="per_episode")
        sampler = PerAgentLatencySampler(
            cfg_dist, cfg_samp, num_envs, num_agents, device, dt=dt, min_steps=2
        )
        sampler.initialize()

        # Should be clamped to min_steps=2
        assert (sampler.step_values >= 2).all(), \
            f"Expected >= 2 steps, got min={sampler.step_values.min()}"

        results.add_pass("PerAgentLatencySampler min_steps enforcement")
    except Exception as e:
        results.add_fail("PerAgentLatencySampler min_steps enforcement", str(e))

    # Test 7: PerAgentStalenessSampler FPS to period conversion
    try:
        cfg_dist = DistributionCfg(type="constant", value=20.0)  # 20 FPS
        cfg_samp = SamplingCfg(frequency="per_episode")
        sampler = PerAgentStalenessSampler(cfg_dist, cfg_samp, num_envs, num_agents, device)
        sampler.initialize()

        assert sampler.period_values.shape == (num_envs, num_agents)

        # Period should be 1/20 = 0.05s
        expected_period = 0.05
        assert torch.allclose(
            sampler.period_values,
            torch.full((num_envs, num_agents), expected_period, device=device)
        )

        results.add_pass("PerAgentStalenessSampler FPS to period")
    except Exception as e:
        results.add_fail("PerAgentStalenessSampler FPS to period", str(e))

    # Test 8: Statistical independence between agents
    try:
        cfg_dist = DistributionCfg(type="uniform", mean=0.5, half_range=0.3)
        cfg_samp = SamplingCfg(frequency="per_step")
        sampler = PerAgentParameterSampler(cfg_dist, cfg_samp, num_envs, num_agents, device)
        sampler.initialize()

        # Collect samples across multiple steps
        samples = []
        for _ in range(50):
            sampler.maybe_resample()
            samples.append(sampler.values.clone())

        all_samples = torch.stack(samples, dim=0)  # (50, num_envs, num_agents)

        # Compute correlation between agents
        # Agents should have low correlation since they're independent
        agent_0 = all_samples[:, :, 0].flatten()
        agent_1 = all_samples[:, :, 1].flatten()

        # Simple correlation check (Pearson)
        mean_0, mean_1 = agent_0.mean(), agent_1.mean()
        std_0, std_1 = agent_0.std(), agent_1.std()

        if std_0 > 0 and std_1 > 0:
            corr = ((agent_0 - mean_0) * (agent_1 - mean_1)).mean() / (std_0 * std_1)
            # Correlation should be low (close to 0)
            assert abs(corr) < 0.3, f"Agents should be independent, but correlation={corr:.3f}"

        results.add_pass("Statistical independence between agents")
    except Exception as e:
        results.add_fail("Statistical independence between agents", str(e))

    # Test 11: Curriculum scaling with per-agent sampler
    try:
        cfg_dist = DistributionCfg(type="constant", value=0.2)  # max delay
        cfg_samp = SamplingCfg(frequency="per_episode")
        sampler = PerAgentLatencySampler(
            cfg_dist, cfg_samp, num_envs, num_agents, device, dt=dt, min_steps=2
        )
        # Use set_scale() before initialize()
        sampler.set_scale(1.0)
        sampler.initialize()
        steps_full = sampler.step_values.clone()

        # Reinitialize at half progress
        sampler.set_scale(0.5)
        sampler.initialize()
        steps_half = sampler.step_values.clone()

        # At scale=0.5, effective delay is 0.1s = 2.5 steps -> 3 steps (rounded)
        # At scale=1.0, effective delay is 0.2s = 5 steps
        # Note: min_steps=2 applies

        if verbose:
            print(f"    Steps at scale=1.0: {steps_full[0]}")
            print(f"    Steps at scale=0.5: {steps_half[0]}")

        # Half progress should generally have fewer or equal steps
        assert (steps_half <= steps_full + 1).all(), \
            "Half progress should not have more steps than full progress"

        results.add_pass("Curriculum scaling with per-agent sampler")
    except Exception as e:
        results.add_fail("Curriculum scaling with per-agent sampler", str(e))

    # Test 12: Per-env-reset resampling
    try:
        cfg_dist = DistributionCfg(type="uniform", mean=0.5, half_range=0.3)
        cfg_samp = SamplingCfg(frequency="per_env_reset")
        sampler = PerAgentParameterSampler(cfg_dist, cfg_samp, num_envs, num_agents, device)
        sampler.initialize()

        initial_values = sampler.values.clone()

        # Reset only some envs
        reset_env_ids = torch.tensor([0, 3, 5], device=device)
        sampler.maybe_resample(reset_env_ids=reset_env_ids)

        # Non-reset envs should have same values
        non_reset_ids = [i for i in range(num_envs) if i not in [0, 3, 5]]
        for env_id in non_reset_ids:
            assert torch.allclose(sampler.values[env_id], initial_values[env_id]), \
                f"Non-reset env {env_id} should have unchanged values"

        results.add_pass("Per-env-reset resampling")
    except Exception as e:
        results.add_fail("Per-env-reset resampling", str(e))

    # Test 13: Normal distribution with clipping
    try:
        cfg_dist = DistributionCfg(
            type="normal", mean=0.5, std=0.2, min_value=0.0, max_value=1.0
        )
        sampler = PerAgentDistributionSampler(cfg_dist, num_envs, num_agents, device)

        # Sample many times and verify clipping
        for _ in range(20):
            values = sampler.sample()
            assert (values >= 0.0).all(), f"Min clipping failed: min={values.min()}"
            assert (values <= 1.0).all(), f"Max clipping failed: max={values.max()}"

        results.add_pass("Normal distribution with clipping")
    except Exception as e:
        results.add_fail("Normal distribution with clipping", str(e))
