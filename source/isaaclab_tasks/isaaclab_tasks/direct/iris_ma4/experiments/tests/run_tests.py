#!/usr/bin/env python3
"""Run experiment sub-module test suite.

Usage:
    ./isaaclab.sh -p .../experiments/tests/run_tests.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run experiments test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import traceback


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  PASS {test_name}")

    def add_fail(self, test_name: str, msg: str):
        self.failed.append((test_name, msg))
        print(f"  FAIL {test_name}: {msg}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  ERROR {test_name}: {error}")

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print(f"\n{'='*60}")
        print(f"Test Results: {len(self.passed)}/{total} passed")
        if self.failed:
            print(f"  Failed: {[f[0] for f in self.failed]}")
        if self.errors:
            print(f"  Errors: {[e[0] for e in self.errors]}")
        print(f"{'='*60}")
        return len(self.failed) == 0 and len(self.errors) == 0


def run_registry_tests(results: TestResults):
    """Test experiment registry operations."""
    print("\n" + "=" * 60)
    print("Testing Experiment Registry")
    print("=" * 60)

    from isaaclab_tasks.direct.iris_ma4.experiments import (
        get_experiment,
        list_experiments,
        list_suites,
        get_suite,
    )

    # Test: list experiments returns non-empty
    try:
        all_exps = list_experiments()
        assert len(all_exps) > 0, "No experiments registered"
        results.add_pass(f"list_experiments: {len(all_exps)} registered")
    except Exception as e:
        results.add_error("list_experiments", traceback.format_exc())

    # Test: get known experiments
    for name in ["a1_no_delay", "a2_rnn", "a3_clean_reward", "a4_analytical", "a4_angular_only", "baseline_gavin2024"]:
        try:
            exp = get_experiment(name)
            assert exp.name == name, f"Name mismatch: {exp.name} != {name}"
            assert exp.group != "", f"Group is empty for {name}"
            results.add_pass(f"get_experiment({name})")
        except Exception as e:
            results.add_error(f"get_experiment({name})", str(e))

    # Test: group filtering
    try:
        a1_exps = list_experiments(group="A1")
        assert len(a1_exps) >= 4, f"Expected >= 4 A1 experiments, got {len(a1_exps)}"
        results.add_pass(f"list_experiments(group='A1'): {len(a1_exps)}")
    except Exception as e:
        results.add_error("list_experiments_group", str(e))

    # Test: delay sweep experiments
    try:
        sweep_exps = list_experiments(group="A1_sweep")
        assert len(sweep_exps) == 5, f"Expected 5 delay sweep experiments, got {len(sweep_exps)}"
        results.add_pass(f"A1_sweep experiments: {len(sweep_exps)}")
    except Exception as e:
        results.add_error("delay_sweep_count", str(e))

    # Test: suites
    try:
        suites = list_suites()
        assert len(suites) >= 3, f"Expected >= 3 suites, got {len(suites)}"
        must_suite = get_suite("iros2026_must")
        assert len(must_suite.experiments) >= 13, f"Must suite should have >= 13 experiments"
        results.add_pass(f"suites: {len(suites)}, must_suite has {len(must_suite.experiments)} experiments")
    except Exception as e:
        results.add_error("suites", str(e))

    # Test: unknown experiment raises KeyError
    try:
        get_experiment("nonexistent_experiment")
        results.add_fail("unknown_experiment", "Should have raised KeyError")
    except KeyError:
        results.add_pass("unknown_experiment raises KeyError")
    except Exception as e:
        results.add_error("unknown_experiment", str(e))


def run_override_tests(results: TestResults):
    """Test config override application."""
    print("\n" + "=" * 60)
    print("Testing Config Overrides")
    print("=" * 60)

    from isaaclab_tasks.direct.iris_ma4.experiments import apply_env_overrides, apply_agent_overrides
    from isaaclab_tasks.direct.iris_ma4.iris_ma_env4_cfg import IrisMAEnvCfg

    # Test: flat env override
    try:
        cfg = IrisMAEnvCfg()
        original_reward_mode = cfg.reward_mode
        apply_env_overrides(cfg, {"reward_mode": "heuristic"})
        assert cfg.reward_mode == "heuristic", f"Expected 'heuristic', got '{cfg.reward_mode}'"
        results.add_pass("flat env override (reward_mode)")
    except Exception as e:
        results.add_error("flat_env_override", traceback.format_exc())

    # Test: nested env override
    try:
        cfg = IrisMAEnvCfg()
        original_step = cfg.curriculum.noise_start_step
        apply_env_overrides(cfg, {"curriculum.noise_start_step": 999999})
        assert cfg.curriculum.noise_start_step == 999999
        results.add_pass("nested env override (curriculum.noise_start_step)")
    except Exception as e:
        results.add_error("nested_env_override", traceback.format_exc())

    # Test: ablation flags
    try:
        cfg = IrisMAEnvCfg()
        assert cfg.use_noisy_rewards == False
        assert cfg.mask_aoi_in_obs == False
        apply_env_overrides(cfg, {"use_noisy_rewards": True, "mask_aoi_in_obs": True})
        assert cfg.use_noisy_rewards == True
        assert cfg.mask_aoi_in_obs == True
        results.add_pass("ablation flags (use_noisy_rewards, mask_aoi_in_obs)")
    except Exception as e:
        results.add_error("ablation_flags", traceback.format_exc())

    # Test: invalid path raises AttributeError
    try:
        cfg = IrisMAEnvCfg()
        apply_env_overrides(cfg, {"nonexistent.field": 42})
        results.add_fail("invalid_path", "Should have raised AttributeError")
    except AttributeError:
        results.add_pass("invalid env override path raises AttributeError")
    except Exception as e:
        results.add_error("invalid_path", str(e))

    # Test: agent dict override
    try:
        agent_cfg = {"models": {"policy": {"hidden_size": 64}}, "agent": {"learning_rate": 0.001}}
        apply_agent_overrides(agent_cfg, {"models.policy.hidden_size": 256})
        assert agent_cfg["models"]["policy"]["hidden_size"] == 256
        results.add_pass("agent dict override (models.policy.hidden_size)")
    except Exception as e:
        results.add_error("agent_override", traceback.format_exc())


def run_metrics_tests(results: TestResults, device: torch.device):
    """Test MetricTracker."""
    print("\n" + "=" * 60)
    print("Testing MetricTracker")
    print("=" * 60)

    from isaaclab_tasks.direct.iris_ma4.experiments.metrics import MetricTracker

    N = 16  # num_envs
    C = 2   # num_agents

    # Test: basic accumulation
    try:
        tracker = MetricTracker(num_envs=N, num_agents=C, device=device)

        # Simulate 10 steps
        for step in range(10):
            trace = torch.ones(N, 1, device=device) * (50.0 - step * 5)  # decreasing trace
            tri_pos = torch.zeros(N, 3, device=device)
            gt_pos = torch.ones(N, 3, device=device)  # RMSE = sqrt(3) ~ 1.73
            bbox_valid = torch.ones(N, C, device=device, dtype=torch.bool)
            collisions = torch.zeros(N, device=device)
            tri_valid = torch.ones(N, 1, device=device, dtype=torch.bool)

            tracker.step(trace, tri_pos, gt_pos, bbox_valid, collisions, tri_valid)

        # End all episodes
        all_envs = torch.arange(N, device=device)
        tracker.record_episode_end(all_envs)

        metrics = tracker.compute_final_metrics()
        assert metrics["num_episodes"] == N, f"Expected {N} episodes, got {metrics['num_episodes']}"
        assert metrics["visibility_ratio"] == 1.0, f"Expected visibility 1.0, got {metrics['visibility_ratio']}"
        assert metrics["collision_rate"] == 0.0, f"Expected collision rate 0.0, got {metrics['collision_rate']}"
        assert metrics["triangulation_rmse"] > 0, f"Expected positive RMSE"
        assert metrics["task_success_rate"] == 1.0, f"Expected success 1.0 (RMSE ~1.73 < 2.0)"
        results.add_pass("basic metric accumulation")
    except Exception as e:
        results.add_error("basic_accumulation", traceback.format_exc())

    # Test: no valid triangulation
    try:
        tracker = MetricTracker(num_envs=N, num_agents=C, device=device)

        for _ in range(5):
            tracker.step(
                trace_sigma=torch.zeros(N, 1, device=device),
                triangulated_pos=torch.zeros(N, 3, device=device),
                gt_target_pos=torch.ones(N, 3, device=device),
                bbox_valid_mask=torch.zeros(N, C, device=device, dtype=torch.bool),
                collision_flags=torch.zeros(N, device=device),
                tri_valid=torch.zeros(N, 1, device=device, dtype=torch.bool),
            )

        tracker.record_episode_end(torch.arange(N, device=device))
        metrics = tracker.compute_final_metrics()
        assert metrics["visibility_ratio"] == 0.0
        assert metrics["time_to_first_lock"] == -1.0, "No first lock when no valid tri"
        results.add_pass("no valid triangulation edge case")
    except Exception as e:
        results.add_error("no_valid_tri", traceback.format_exc())

    # Test: convergence detection
    try:
        tracker = MetricTracker(
            num_envs=N, num_agents=C, device=device,
            convergence_trace_threshold=10.0, first_lock_trace_threshold=50.0,
        )

        for step in range(20):
            trace_val = max(100.0 - step * 10, 1.0)
            tracker.step(
                trace_sigma=torch.full((N, 1), trace_val, device=device),
                triangulated_pos=torch.zeros(N, 3, device=device),
                gt_target_pos=torch.zeros(N, 3, device=device),
                bbox_valid_mask=torch.ones(N, C, device=device, dtype=torch.bool),
                collision_flags=torch.zeros(N, device=device),
                tri_valid=torch.ones(N, 1, device=device, dtype=torch.bool),
            )

        tracker.record_episode_end(torch.arange(N, device=device))
        metrics = tracker.compute_final_metrics()
        # First lock at step 6 (trace=40 < 50), convergence at step 10 (trace=0 < 10)
        assert metrics["time_to_first_lock"] > 0, "Should have first lock"
        assert metrics["convergence_speed"] > 0, "Should have converged"
        results.add_pass("convergence detection")
    except Exception as e:
        results.add_error("convergence", traceback.format_exc())


def run_mlp_model_tests(results: TestResults, device: torch.device):
    """Test MLP model forward pass."""
    print("\n" + "=" * 60)
    print("Testing MLP Models")
    print("=" * 60)

    import gymnasium as gym
    import numpy as np
    from isaaclab_tasks.direct.iris_ma4.experiments.models.mappo_mlp import (
        MAPPOMLPPolicy,
        MAPPOMLPValue,
    )

    obs_dim = 47
    act_dim = 7
    batch_size = 32
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    # Test: policy forward pass
    try:
        policy = MAPPOMLPPolicy(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            hidden_size=256,
            num_hidden_layers=3,
            num_envs=batch_size,
        )
        obs = torch.randn(batch_size, obs_dim, device=device)
        mean, log_std, outputs = policy.compute({"states": obs}, role="policy")
        assert mean.shape == (batch_size, act_dim), f"Policy output shape: {mean.shape}"
        assert log_std.shape == (act_dim,), f"Log std shape: {log_std.shape}"
        assert outputs == {}, f"MLP should return empty outputs dict"
        results.add_pass(f"MLP policy forward: input ({batch_size},{obs_dim}) -> ({batch_size},{act_dim})")
    except Exception as e:
        results.add_error("mlp_policy_forward", traceback.format_exc())

    # Test: value forward pass
    try:
        # Shared obs = concatenation of all agent obs
        shared_obs_dim = obs_dim * 2  # 2 agents
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(shared_obs_dim,), dtype=np.float32)
        value = MAPPOMLPValue(
            observation_space=shared_space,
            action_space=act_space,
            device=device,
            hidden_size=256,
            num_hidden_layers=3,
            num_envs=batch_size,
        )
        shared_obs = torch.randn(batch_size, shared_obs_dim, device=device)
        val, outputs = value.compute({"states": shared_obs}, role="value")
        assert val.shape == (batch_size, 1), f"Value output shape: {val.shape}"
        assert outputs == {}, f"MLP should return empty outputs dict"
        results.add_pass(f"MLP value forward: input ({batch_size},{shared_obs_dim}) -> ({batch_size},1)")
    except Exception as e:
        results.add_error("mlp_value_forward", traceback.format_exc())

    # Test: no RNN specification
    try:
        spec = policy.get_specification()
        assert spec == {}, f"MLP policy should have empty spec, got: {spec}"
        spec = value.get_specification()
        assert spec == {}, f"MLP value should have empty spec, got: {spec}"
        results.add_pass("MLP models have empty RNN specification")
    except Exception as e:
        results.add_error("mlp_no_rnn_spec", traceback.format_exc())

    # Test: 3D input handling (batch, seq=1, obs)
    try:
        obs_3d = torch.randn(batch_size, 1, obs_dim, device=device)
        mean, _, _ = policy.compute({"states": obs_3d}, role="policy")
        assert mean.shape == (batch_size, act_dim), f"3D input should flatten to 2D: {mean.shape}"
        results.add_pass("MLP policy handles 3D input (B, S=1, obs)")
    except Exception as e:
        results.add_error("mlp_3d_input", traceback.format_exc())


def run_aoi_indices_test(results: TestResults):
    """Test AoI index computation."""
    print("\n" + "=" * 60)
    print("Testing AoI Indices")
    print("=" * 60)

    from isaaclab_tasks.direct.iris_ma4.iris_ma_env4_cfg import IrisMAEnvCfg

    try:
        cfg = IrisMAEnvCfg()
        # 2-agent: AoI indices should be [21, 39, 40]
        C = len(cfg.possible_agents)
        assert C == 2, f"Expected 2 agents, got {C}"
        indices = [21] + list(range(26 + 13 * (C - 1), 26 + 15 * (C - 1)))
        assert indices == [21, 39, 40], f"Expected [21, 39, 40], got {indices}"
        results.add_pass("AoI indices for 2-agent: [21, 39, 40]")
    except Exception as e:
        results.add_error("aoi_indices_2agent", traceback.format_exc())

    # Verify obs dim consistency
    try:
        expected_dim = 26 + 15 * (C - 1) + 6
        assert expected_dim == 47, f"Expected 47D obs, got {expected_dim}"
        assert max(indices) < expected_dim, f"AoI index {max(indices)} exceeds obs dim {expected_dim}"
        results.add_pass("AoI indices within obs dim bounds")
    except Exception as e:
        results.add_error("aoi_indices_bounds", traceback.format_exc())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    results = TestResults()

    try:
        run_registry_tests(results)
        run_override_tests(results)
        run_metrics_tests(results, device)
        run_mlp_model_tests(results, device)
        run_aoi_indices_test(results)
    except Exception as e:
        print(f"\nUnhandled error: {traceback.format_exc()}")
        results.add_error("unhandled", str(e))

    success = results.summary()
    simulation_app.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
