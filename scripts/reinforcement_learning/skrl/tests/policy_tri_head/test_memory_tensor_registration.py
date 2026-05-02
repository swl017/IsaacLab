"""MAPPOWithAux must register `tri_target_position_w` and `tri_target_valid`
on every per-agent memory during init(). Skrl's `Memory.add_samples` silently
ignores unregistered keys — this test guards against that silent-failure mode.
"""
import traceback

import gymnasium as gym
import numpy as np
import torch

from skrl.memories.torch import RandomMemory

from mappo_with_aux import (
    MAPPO_WITH_AUX_DEFAULT_CONFIG,
    MAPPOWithAux,
    MAPPOWithAuxPolicy,
    MAPPOWithAuxValue,
)


def _build_minimal_agent(device, num_envs=2, num_agents=2, obs_dim=47, act_dim=7):
    """Construct a minimally-configured MAPPOWithAux suitable for init() to run.

    Uses RandomMemory and a per-agent RunningStandardScaler-free preprocessor
    setup so we don't need the full Hydra cfg pipeline.
    """
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    shared_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim * num_agents,), dtype=np.float32,
    )

    possible_agents = [f"agent_{i}" for i in range(num_agents)]
    obs_spaces = {a: obs_space for a in possible_agents}
    act_spaces = {a: act_space for a in possible_agents}
    shared_obs_spaces = {a: shared_obs_space for a in possible_agents}

    # Shared policy + value across agents (matches train_mappo_with_aux_hydra.create_models).
    shared_policy = MAPPOWithAuxPolicy(
        observation_space=obs_space, action_space=act_space, device=device,
        hidden_size=32, gru_num_layers=1, gru_hidden_size=32,
        num_envs=num_envs, sequence_length=8,
        tri_head_enabled=True, tri_head_hidden=(16,),
    ).to(device)
    shared_value = MAPPOWithAuxValue(
        observation_space=shared_obs_space, action_space=act_space, device=device,
        hidden_size=32, gru_num_layers=1, gru_hidden_size=32,
        num_envs=num_envs, sequence_length=8,
    ).to(device)

    models = {a: {"policy": shared_policy, "value": shared_value} for a in possible_agents}
    memories = {
        a: RandomMemory(memory_size=8, num_envs=num_envs, device=device)
        for a in possible_agents
    }

    cfg = MAPPO_WITH_AUX_DEFAULT_CONFIG.copy()
    # Disable the preprocessors that would otherwise need finalized kwargs.
    cfg["state_preprocessor"] = None
    cfg["shared_state_preprocessor"] = None
    cfg["value_preprocessor"] = None
    cfg["learning_rate_scheduler"] = None
    cfg["mixed_precision"] = False

    agent = MAPPOWithAux(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=obs_spaces,
        action_spaces=act_spaces,
        device=device,
        cfg=cfg,
        shared_observation_spaces=shared_obs_spaces,
    )
    return agent, memories, possible_agents


def run_registration_tests(results, device, verbose=False):
    # ---- Both tensors registered on every per-agent memory after init() ----
    try:
        agent, memories, possible_agents = _build_minimal_agent(device)
        agent.init(trainer_cfg=None)
        for uid in possible_agents:
            mem = memories[uid]
            tensor_keys = list(mem.tensors.keys())
            assert "tri_target_position_w" in tensor_keys, (
                f"agent {uid}: tri_target_position_w not registered. tensors_keys={tensor_keys}"
            )
            assert "tri_target_valid" in tensor_keys, (
                f"agent {uid}: tri_target_valid not registered. tensors_keys={tensor_keys}"
            )
        results.add_pass("tri_target_* tensors registered on every agent memory")
    except Exception:
        results.add_fail("tri_target_* tensors registered on every agent memory",
                         traceback.format_exc())

    # ---- Tensor shapes match the contract: (memory_size, num_envs, size) ----
    try:
        agent, memories, possible_agents = _build_minimal_agent(device)
        agent.init(trainer_cfg=None)
        mem = memories[possible_agents[0]]
        pos = mem.tensors["tri_target_position_w"]
        val = mem.tensors["tri_target_valid"]
        memory_size = mem.memory_size
        num_envs = mem.num_envs
        assert tuple(pos.shape) == (memory_size, num_envs, 3), \
            f"tri_target_position_w shape: expected ({memory_size}, {num_envs}, 3), got {tuple(pos.shape)}"
        assert tuple(val.shape) == (memory_size, num_envs, 1), \
            f"tri_target_valid shape: expected ({memory_size}, {num_envs}, 1), got {tuple(val.shape)}"
        assert pos.dtype == torch.float32
        assert val.dtype == torch.float32
        results.add_pass("tri_target_* tensor shapes and dtypes match contract")
    except Exception:
        results.add_fail("tri_target_* tensor shapes and dtypes match contract",
                         traceback.format_exc())

    # ---- Pre-write path used by record_transition writes at memory_index ----
    try:
        agent, memories, possible_agents = _build_minimal_agent(device, num_envs=2, num_agents=2)
        agent.init(trainer_cfg=None)

        # Construct a synthetic infos dict matching what the trainer wrapper produces.
        N = 2
        A = 2
        infos = {
            "tri_target_position_w": torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                    [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
                ], device=device,
            ),  # (N=2, A=2, 3)
            "tri_target_valid": torch.tensor(
                [[True, False], [False, True]], device=device,
            ),  # (N=2, A=2)
        }

        # Manually replicate the pre-write step from MAPPOWithAux.record_transition.
        # (We don't call record_transition itself because that requires a fully-wired
        # super().record_transition pipeline including value-net forward pass.)
        for agent_idx, uid in enumerate(possible_agents):
            mem = memories[uid]
            mi = mem.memory_index
            mem.tensors["tri_target_position_w"][mi].copy_(
                infos["tri_target_position_w"][:, agent_idx, :]
            )
            mem.tensors["tri_target_valid"][mi].copy_(
                infos["tri_target_valid"][:, agent_idx:agent_idx + 1].to(dtype=torch.float32)
            )

        # Verify the writes landed correctly per-agent.
        mem0 = memories["agent_0"]
        mem1 = memories["agent_1"]
        assert torch.allclose(
            mem0.tensors["tri_target_position_w"][0],
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device),
        ), "agent_0 position write incorrect"
        assert torch.allclose(
            mem1.tensors["tri_target_position_w"][0],
            torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device),
        ), "agent_1 position write incorrect"
        assert torch.allclose(
            mem0.tensors["tri_target_valid"][0],
            torch.tensor([[1.0], [0.0]], device=device),
        ), "agent_0 valid write incorrect"
        assert torch.allclose(
            mem1.tensors["tri_target_valid"][0],
            torch.tensor([[0.0], [1.0]], device=device),
        ), "agent_1 valid write incorrect"
        results.add_pass("Per-agent slice routing matches expected indexing")
    except Exception:
        results.add_fail("Per-agent slice routing matches expected indexing",
                         traceback.format_exc())
