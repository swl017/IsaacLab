"""End-to-end preflight smoke (ticket 031).

Runs ≥128 env steps with the full MAPPOWithAux pipeline and asserts:
  (a) no exception across the 128 steps,
  (b) no NaN/Inf in any model parameter or gradient after the gradient update,
  (c) tri_target_position_w / tri_target_valid tensors in memory contain
      non-zero entries (catches the silent-failure mode of Memory.add_samples
      ignoring unregistered keys),
  (d) gradient norm on policy_layer + value_layer is non-zero (PPO loss
      flowing); when aux_loss_scale>0 the tri_head grad MUST also be
      non-zero (catches "loss scale wired but _update doesn't actually
      compute aux_loss"); when aux_loss_scale==0 the tri_head grad MUST
      be exactly zero (the wired-but-off invariant).

Slice 3 default: aux_loss_scale=0.1, so the head trains. Run with
``aux_loss_scale=0.0`` to exercise the slice-2 invariant directly; the
bit-exact regression test covers the more rigorous version of that gate.
"""
import os
import traceback

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.envs import DirectMARLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed

from mappo_with_aux import (
    MAPPO_WITH_AUX_DEFAULT_CONFIG,
    MAPPOWithAux,
    MAPPOWithAuxPolicy,
    MAPPOWithAuxValue,
)


def _build_pipeline(device, num_envs=8, rollouts=8, mini_batches=2,
                    learning_epochs=2, aux_loss_scale=0.1):
    """Construct env + agent + memories with a tiny rollout buffer so an
    update fires within ~128 env steps.

    Args:
        aux_loss_scale: 0.1 by default (matches the slice-3 YAML default and
            exercises the aux loss path end-to-end). Set to 0.0 to verify the
            wired-but-off invariant in addition to the bit-exact gate.
    """
    set_seed(42)
    task = "Isaac-Iris-MA6-Direct-Test-v0"
    env_cfg = parse_env_cfg(task, device=str(device), num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    assert isinstance(env.unwrapped.unwrapped, DirectMARLEnv), \
        "test requires multi-agent env"
    possible_agents = env.possible_agents
    obs_spaces = env.observation_spaces
    act_spaces = env.action_spaces
    try:
        shared_obs_spaces = env.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(s.shape[0] for s in obs_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_obs_spaces = {a: shared_space for a in possible_agents}

    # Shared models with parameter-sharing across agents.
    shared_policy = MAPPOWithAuxPolicy(
        observation_space=obs_spaces[possible_agents[0]],
        action_space=act_spaces[possible_agents[0]],
        device=device,
        hidden_size=64, gru_num_layers=1, gru_hidden_size=64,
        num_envs=env.num_envs, sequence_length=8,
        tri_head_enabled=True, tri_head_hidden=(32,),
    ).to(device)
    shared_value = MAPPOWithAuxValue(
        observation_space=shared_obs_spaces[possible_agents[0]],
        action_space=act_spaces[possible_agents[0]],
        device=device,
        hidden_size=64, gru_num_layers=1, gru_hidden_size=64,
        num_envs=env.num_envs, sequence_length=8,
    ).to(device)

    models = {a: {"policy": shared_policy, "value": shared_value} for a in possible_agents}
    memories = {
        a: RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)
        for a in possible_agents
    }

    cfg = MAPPO_WITH_AUX_DEFAULT_CONFIG.copy()
    cfg.update({
        "rollouts": rollouts,
        "mini_batches": mini_batches,
        "learning_epochs": learning_epochs,
        "sequence_length": 8,
        "burn_in_steps": 0,
        "episode_start_mask_steps": 0,
        "state_preprocessor": None,
        "shared_state_preprocessor": None,
        "value_preprocessor": None,
        "learning_rate_scheduler": None,
        "mixed_precision": False,
        "aux_loss_scale": aux_loss_scale,
        "experiment": {
            "directory": "/tmp/iris_ma6_smoke",
            "experiment_name": "ticket031_slice2_smoke",
            "write_interval": 0,         # disable tensorboard writes
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    })

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
    return env, agent, memories, possible_agents, shared_policy, shared_value


def _params_finite(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if not torch.isfinite(p).all().item():
            return False
        if p.grad is not None and not torch.isfinite(p.grad).all().item():
            return False
    return True


def _grad_norm(params) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += p.grad.detach().pow(2).sum().item()
    return s ** 0.5


def run_preflight_smoke(results, device, verbose=False, num_steps=128,
                        return_env=False, aux_loss_scale=0.1):
    """Preflight smoke. By default builds and tears down its own env.

    Args:
        aux_loss_scale: 0.1 by default (matches the slice-3 YAML). Caller can
            set 0.0 to additionally verify the wired-but-off invariant on a
            real env (the bit-exact regression test is the rigorous gate).
        return_env: when True the env is NOT closed and is returned, so the
            caller can run additional tests on the same Isaac Sim instance.
    """
    env = agent = None
    try:
        env, agent, memories, possible_agents, shared_policy, shared_value = _build_pipeline(
            device, num_envs=8, rollouts=8, mini_batches=2, learning_epochs=2,
            aux_loss_scale=aux_loss_scale,
        )
        agent.init(trainer_cfg={
            "timesteps": num_steps,
            "headless": True,
            "environment_info": "log",
        })

        # Run the abridged training loop manually, mirroring AuxInjectingTrainer.
        states, infos = env.reset()
        shared_states = env.state()

        steps_run = 0
        param_hash_before = None
        for p in shared_policy.parameters():
            if p.requires_grad:
                param_hash_before = float(p.detach().sum().item())
                break

        for timestep in range(num_steps):
            agent.pre_interaction(timestep=timestep, timesteps=num_steps)

            with torch.no_grad():
                actions = agent.act(states, timestep=timestep, timesteps=num_steps)[0]

                next_states, rewards, terminated, truncated, infos = env.step(actions)
                shared_next_states = env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # ---- aux supervision injection ----
                aux = env.unwrapped.get_aux_supervision()
                infos["tri_target_position_w"] = aux["tri_target_position_w"]
                infos["tri_target_valid"] = aux["tri_target_valid"]
                # ------------------------------------

                agent.record_transition(
                    states=states, actions=actions, rewards=rewards,
                    next_states=next_states, terminated=terminated, truncated=truncated,
                    infos=infos, timestep=timestep, timesteps=num_steps,
                )

            agent.post_interaction(timestep=timestep, timesteps=num_steps)

            if not env.agents:
                with torch.no_grad():
                    states, infos = env.reset()
                    shared_states = env.state()
            else:
                states = next_states
                shared_states = shared_next_states

            steps_run += 1

        # ---- (a) finished without exception ----
        assert steps_run == num_steps, f"steps_run={steps_run}, expected {num_steps}"
        results.add_pass(f"(a) Completed {num_steps} steps without exception")

        # ---- (b) no NaN/Inf in any model parameter or gradient ----
        assert _params_finite(shared_policy), "shared_policy has NaN/Inf in params or grads"
        assert _params_finite(shared_value), "shared_value has NaN/Inf in params or grads"
        results.add_pass("(b) No NaN/Inf in policy or value parameters and gradients")

        # ---- (c) tri_target_* tensors populated (non-zero) ----
        for uid in possible_agents:
            mem = memories[uid]
            pos = mem.tensors["tri_target_position_w"]
            val = mem.tensors["tri_target_valid"]
            # Position tensors carry GT target world position which is non-zero even at
            # episode start (target spawned at non-origin location). At least some entries
            # must be non-zero across the buffer; total absolute sum > 0 is the contract.
            pos_any_nonzero = pos.abs().sum().item() > 0.0
            val_any_nonzero = val.abs().sum().item() > 0.0
            assert pos_any_nonzero, \
                f"agent {uid}: tri_target_position_w buffer is all-zero — silent registration failure?"
            assert val_any_nonzero, \
                f"agent {uid}: tri_target_valid buffer is all-zero — no agent ever had a valid sample?"
        results.add_pass("(c) tri_target_* memory tensors populated with non-zero entries")

        # ---- (d) gradient norm on policy_layer + value_layer is non-zero ----
        # We've completed several updates by now (rollouts=8 with 128 steps -> 16 updates).
        # The detection: did the policy params change vs initial? (Updates happened.)
        param_hash_after = None
        for p in shared_policy.parameters():
            if p.requires_grad:
                param_hash_after = float(p.detach().sum().item())
                break
        assert param_hash_before != param_hash_after, \
            "shared_policy parameters did not change — no update fired"

        # Verify policy_layer + value_layer gradients are non-trivial after the last update.
        # (Gradients persist on .grad after .backward; the optimizer.step() doesn't clear them.)
        policy_layer_grad = _grad_norm(shared_policy.policy_layer.parameters())
        value_layer_grad = _grad_norm(shared_value.value_layer.parameters())
        assert policy_layer_grad > 0.0, \
            f"policy_layer grad norm is zero: {policy_layer_grad}"
        assert value_layer_grad > 0.0, \
            f"value_layer grad norm is zero: {value_layer_grad}"

        # tri_head gradient invariants — depends on aux_loss_scale.
        if shared_policy.tri_head is not None:
            tri_head_grad = _grad_norm(shared_policy.tri_head.parameters())
            if aux_loss_scale == 0.0:
                assert tri_head_grad == 0.0, (
                    f"slice-2 invariant violated: tri_head grad norm should be 0 at "
                    f"aux_loss_scale=0, got {tri_head_grad}"
                )
                results.add_pass(
                    "(d) policy_layer + value_layer grads > 0; tri_head grad == 0 at scale=0"
                )
            else:
                assert tri_head_grad > 0.0, (
                    f"slice-3 invariant violated: tri_head grad norm should be > 0 at "
                    f"aux_loss_scale={aux_loss_scale}, got {tri_head_grad}. "
                    "Aux loss is wired but produces no tri_head gradient — "
                    "_update may not be computing aux_loss correctly."
                )
                results.add_pass(
                    f"(d) policy_layer + value_layer grads > 0; tri_head grad > 0 at scale={aux_loss_scale}"
                )
        else:
            results.add_pass("(d) policy_layer + value_layer grad norms > 0 (no tri_head)")

    except Exception:
        results.add_fail("Preflight smoke", traceback.format_exc())
    finally:
        if env is not None and not return_env:
            try:
                env.close()
            except Exception:
                pass

    if return_env:
        return env
