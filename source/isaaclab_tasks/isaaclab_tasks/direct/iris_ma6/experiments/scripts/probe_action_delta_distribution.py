#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ticket 044 Slice-0 — per-channel |Δaction| distribution probe.

Loads a trained MAPPO-RNN checkpoint, pins the env curriculum to a chosen
training-step via ``use_debug_initial_step``, rolls out ``episodes ×
steps_per_episode`` policy steps in deterministic-mean mode across
``num_envs`` parallel envs, captures ``_actions[t] - _actions[t-1]`` per
agent per channel, writes per-channel ``{p50, p90, p95, p99}`` to CSV.

Used to size ticket-044's per-channel ``action_slew_*`` defaults
empirically against the actual trained policy's Δaction distribution.

Usage (one probe call):
    ./isaaclab.sh -p source/isaaclab_tasks/.../experiments/scripts/probe_action_delta_distribution.py \
        --checkpoint <path>/checkpoints/agent_<step>.pt \
        --debug_step 39000 \
        --output_csv /tmp/probe_action_delta/probe_39k_ckpt_40k.csv

See ``run_action_delta_probe.bash`` for the 7-call sweep across
``debug_step ∈ {39k, 79k, 119k, 199k}`` × {nearest-greater, final} ckpts.
"""

import argparse
import os

from isaaclab.app import AppLauncher

# ---------- argparse + AppLauncher (must come before isaac imports) ----------
parser = argparse.ArgumentParser(description="Ticket 044 Slice-0 |Δa| probe")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file (e.g., .../checkpoints/agent_200000.pt)")
parser.add_argument("--debug_step", type=int, required=True,
                    help="Curriculum step to pin the env at (via use_debug_initial_step)")
parser.add_argument("--episodes", type=int, default=10,
                    help="Approximate episode count; rollout runs episodes × steps_per_episode steps")
parser.add_argument("--steps_per_episode", type=int, default=500,
                    help="Episode length in policy steps (20s × 25Hz = 500)")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Parallel envs")
parser.add_argument("--output_csv", type=str, required=True,
                    help="Destination CSV for the percentile table")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA6-Direct-Test-v0",
                    help="Gym task id")
parser.add_argument("--seed", type=int, default=42, help="Seed")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--ml_framework", type=str, default="torch",
                    choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, _cfg_overrides = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- post-launch imports ----------
import csv
import time
from typing import Dict, List

import torch
import numpy as np
import gymnasium as gym

import sys
import pickle
import yaml
from typing import Optional, Sequence

# Make the train modules importable so we can reuse MAPPORNNPolicy / MAPPORNNValue.
# We deliberately DO NOT import play_iris_mappo_rnn here because that module
# launches AppLauncher at import-time (which we already launched above) — we
# inline its helper functions below instead.
_SKRL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..", "..", "..", "scripts", "reinforcement_learning", "skrl",
))
sys.path.insert(0, _SKRL_DIR)

from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv


# ---------- inlined helpers (mirrored from play_iris_mappo_rnn.py) ----------

def get_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    """Derive run directory from checkpoint path (`.../run_dir/checkpoints/agent_*.pt`)."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    parent = os.path.dirname(checkpoint_path)
    if os.path.basename(parent) == "checkpoints":
        return os.path.dirname(parent)
    return parent


def load_agent_config(run_dir: str) -> Optional[dict]:
    """Load saved agent config (prefers pickle since YAML may have unpicklable types)."""
    pkl_path = os.path.join(run_dir, "params", "agent.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[probe] failed to load {pkl_path}: {e}")
    yaml_path = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path) as f:
                return yaml.unsafe_load(f)
        except Exception:
            try:
                with open(yaml_path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"[probe] could not parse {yaml_path}: {e}")
    return None


def get_model_hyperparams(agent_cfg: Optional[dict]) -> dict:
    """Extract model hyperparameters from agent config, with sensible defaults."""
    defaults = {
        "hidden_size": 128,
        "gru_num_layers": 1,
        "gru_hidden_size": 128,
        "initial_log_std": -1.0,
        "min_log_std": -5.0,
        "max_log_std": 2.0,
    }
    if agent_cfg is None:
        return defaults
    models = agent_cfg.get("models", {})
    policy = models.get("policy", {})
    out = defaults.copy()
    for k in out:
        if k in policy:
            out[k] = policy[k]
    return out


def get_sequence_length(agent_cfg: Optional[dict]) -> int:
    if agent_cfg is None:
        return 128
    return agent_cfg.get("agent", {}).get("sequence_length", 128)


def load_checkpoint_weights(checkpoint_path: str, possible_agents: Sequence[str], device) -> dict:
    """Load checkpoint, return per-agent state dicts (handles SKRL + custom-final formats)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "policy_state_dict" in checkpoint:
        print("[probe] detected custom final checkpoint format")
        return {a: {
            "policy": checkpoint["policy_state_dict"],
            "value": checkpoint["value_state_dict"],
            "state_preprocessor": checkpoint.get("state_preprocessor"),
            "shared_state_preprocessor": checkpoint.get("shared_state_preprocessor"),
            "value_preprocessor": checkpoint.get("value_preprocessor"),
        } for a in possible_agents}
    print(f"[probe] detected SKRL checkpoint format with agents: {list(checkpoint.keys())}")
    result = {}
    for a in possible_agents:
        if a in checkpoint:
            d = checkpoint[a]
            result[a] = {
                "policy": d["policy"],
                "value": d["value"],
                "state_preprocessor": d.get("state_preprocessor"),
                "shared_state_preprocessor": d.get("shared_state_preprocessor"),
                "value_preprocessor": d.get("value_preprocessor"),
            }
    return result


CHANNEL_LABELS = ["vx", "vy", "vz", "yaw_rate", "gimbal_yaw_rate", "gimbal_pitch_rate", "zoom_rate"]
PERCENTILES = [0.50, 0.90, 0.95, 0.99]


def build_env(args):
    """Build the iris_ma6 env with debug_initial_step pinning the curriculum."""
    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    # No cameras needed — the probe only reads _actions; bbox is computed
    # via Warp raycasting which is independent of TiledCamera.
    env_cfg.enable_tiled_cameras = False
    env_cfg.use_debug_initial_step = True
    env_cfg.debug_initial_step = int(args.debug_step)
    env_cfg.seed = args.seed
    torch.manual_seed(args.seed)
    print(f"[probe] task={args.task} num_envs={args.num_envs} debug_step={args.debug_step}")
    env = gym.make(args.task, cfg=env_cfg)
    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise ValueError(f"Task {args.task} is not a multi-agent environment")
    return env


def build_agent(env, args, raw_env):
    """Mirror play_iris_mappo_rnn.py's model + agent setup, trimmed."""
    run_dir = get_run_dir_from_checkpoint(args.checkpoint)
    print(f"[probe] checkpoint: {args.checkpoint}")
    print(f"[probe] run dir   : {run_dir}")
    agent_cfg = load_agent_config(run_dir)
    model_params = get_model_hyperparams(agent_cfg)
    sequence_length = get_sequence_length(agent_cfg)

    env_skrl = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
    device = env_skrl.device
    possible_agents = env_skrl.possible_agents

    # Shared observation spaces (same logic as play_iris_mappo_rnn.py)
    try:
        shared_observation_spaces = env_skrl.shared_observation_spaces
    except AttributeError:
        obs_shape = sum(s.shape[0] for s in env_skrl.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {a: shared_space for a in possible_agents}

    agent_weights = load_checkpoint_weights(args.checkpoint, possible_agents, device)

    obs_match = all(env_skrl.observation_spaces[a] == env_skrl.observation_spaces[possible_agents[0]]
                    for a in possible_agents)
    act_match = all(env_skrl.action_spaces[a] == env_skrl.action_spaces[possible_agents[0]]
                    for a in possible_agents)
    using_shared = obs_match and act_match

    models = {}
    if using_shared:
        shared_policy = MAPPORNNPolicy(
            observation_space=env_skrl.observation_spaces[possible_agents[0]],
            action_space=env_skrl.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=model_params["hidden_size"],
            gru_num_layers=model_params["gru_num_layers"],
            gru_hidden_size=model_params["gru_hidden_size"],
            num_envs=env_skrl.num_envs,
            sequence_length=sequence_length,
            initial_log_std=model_params["initial_log_std"],
            min_log_std=model_params["min_log_std"],
            max_log_std=model_params["max_log_std"],
        )
        shared_value = MAPPORNNValue(
            observation_space=shared_observation_spaces[possible_agents[0]],
            action_space=env_skrl.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=model_params["hidden_size"],
            gru_num_layers=model_params["gru_num_layers"],
            gru_hidden_size=model_params["gru_hidden_size"],
            num_envs=env_skrl.num_envs,
            sequence_length=sequence_length,
        )
        shared_policy.load_state_dict(agent_weights[possible_agents[0]]["policy"])
        shared_value.load_state_dict(agent_weights[possible_agents[0]]["value"])
        for a in possible_agents:
            models[a] = {"policy": shared_policy, "value": shared_value}
    else:
        for a in possible_agents:
            policy = MAPPORNNPolicy(
                observation_space=env_skrl.observation_spaces[a],
                action_space=env_skrl.action_spaces[a],
                device=device,
                hidden_size=model_params["hidden_size"],
                gru_num_layers=model_params["gru_num_layers"],
                gru_hidden_size=model_params["gru_hidden_size"],
                num_envs=env_skrl.num_envs,
                sequence_length=sequence_length,
                initial_log_std=model_params["initial_log_std"],
                min_log_std=model_params["min_log_std"],
                max_log_std=model_params["max_log_std"],
            )
            value = MAPPORNNValue(
                observation_space=shared_observation_spaces[a],
                action_space=env_skrl.action_spaces[a],
                device=device,
                hidden_size=model_params["hidden_size"],
                gru_num_layers=model_params["gru_num_layers"],
                gru_hidden_size=model_params["gru_hidden_size"],
                num_envs=env_skrl.num_envs,
                sequence_length=sequence_length,
            )
            policy.load_state_dict(agent_weights[a]["policy"])
            value.load_state_dict(agent_weights[a]["value"])
            models[a] = {"policy": policy, "value": value}

    memories = {a: RandomMemory(memory_size=1, num_envs=env_skrl.num_envs, device=device)
                for a in possible_agents}

    cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env_skrl.observation_spaces[possible_agents[0]], "device": device,
    }
    cfg["shared_state_preprocessor"] = RunningStandardScaler
    cfg["shared_state_preprocessor_kwargs"] = {
        "size": shared_observation_spaces[possible_agents[0]], "device": device,
    }
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"] = {"write_interval": 0, "checkpoint_interval": 0, "wandb": False}

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env_skrl.observation_spaces,
        action_spaces=env_skrl.action_spaces,
        device=device,
        cfg=cfg,
        shared_observation_spaces=shared_observation_spaces,
    )
    agent.init()
    agent.set_mode("eval")

    # Load preprocessor states (critical for correct evaluation)
    for a in possible_agents:
        if agent_weights[a].get("state_preprocessor") is not None:
            agent._state_preprocessor[a].load_state_dict(agent_weights[a]["state_preprocessor"])
        if agent_weights[a].get("shared_state_preprocessor") is not None:
            agent._shared_state_preprocessor[a].load_state_dict(agent_weights[a]["shared_state_preprocessor"])
        if agent_weights[a].get("value_preprocessor") is not None:
            agent._value_preprocessor[a].load_state_dict(agent_weights[a]["value_preprocessor"])

    print(f"[probe] agent ready (shared_policy={using_shared}, "
          f"hidden={model_params['hidden_size']}, gru={model_params['gru_hidden_size']}, "
          f"seq={sequence_length})")
    return env_skrl, agent, possible_agents


def main():
    set_seed(args_cli.seed)

    env = build_env(args_cli)
    raw_env = env.unwrapped
    env_skrl, agent, possible_agents = build_agent(env, args_cli, raw_env)

    n_steps = args_cli.episodes * args_cli.steps_per_episode
    print(f"[probe] rolling out {n_steps} policy steps "
          f"({args_cli.episodes} eps × {args_cli.steps_per_episode} steps), "
          f"{args_cli.num_envs} envs, deterministic mean actions")

    # Per-agent flat tensor of |Δa|: shape (n_steps × num_envs, 7).
    delta_accum: Dict[str, List[torch.Tensor]] = {a: [] for a in possible_agents}

    # Capture _actions BEFORE env.step (i.e., the post-clamp policy output stored
    # in raw_env._actions inside _pre_physics_step). The slew clip isn't merged
    # yet, so _actions = clamp(policy_output, -1, 1) is what we measure.
    prev_actions: Dict[str, torch.Tensor] = {}

    states, _ = env_skrl.reset()
    timestep = 0
    t_start = time.time()
    for t in range(n_steps):
        with torch.inference_mode():
            actions, _, outputs = agent.act(states, timestep=timestep, timesteps=n_steps)
            # Deterministic mean action (same as play_iris_mappo_rnn.py default)
            for a in possible_agents:
                actions[a] = outputs[a]["mean_actions"]
            states, _r, _term, _trunc, _i = env_skrl.step(actions)

            # Read post-clamp _actions from the env (this is what feeds into
            # cmd_vel and reward; what the slew clip would constrain).
            for a in possible_agents:
                act_now = raw_env._actions[a].detach()  # (num_envs, 7), in [-1, 1]
                if a in prev_actions:
                    delta = (act_now - prev_actions[a]).abs()  # (num_envs, 7)
                    delta_accum[a].append(delta.clone())
                prev_actions[a] = act_now.clone()

        timestep += 1
        if (t + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed
            eta = (n_steps - t - 1) / max(rate, 1e-6)
            print(f"[probe]   step {t+1}/{n_steps} | {rate:.1f} steps/s | ETA {eta:.0f}s")

    elapsed_total = time.time() - t_start
    print(f"[probe] rollout complete: {elapsed_total:.1f}s ({n_steps / elapsed_total:.1f} steps/s)")

    # --- Compute percentiles per agent × channel ---
    os.makedirs(os.path.dirname(args_cli.output_csv) or ".", exist_ok=True)
    rows = []
    ckpt_name = os.path.basename(args_cli.checkpoint)
    for a in possible_agents:
        stacked = torch.cat(delta_accum[a], dim=0)  # (n_samples, 7)
        n_samples = stacked.shape[0]
        for ch in range(7):
            vals = stacked[:, ch]
            quants = torch.quantile(vals, torch.tensor(PERCENTILES, device=vals.device))
            row = {
                "checkpoint": ckpt_name,
                "debug_step": args_cli.debug_step,
                "agent": a,
                "channel_idx": ch,
                "channel_label": CHANNEL_LABELS[ch],
                "n_samples": int(n_samples),
                "p50": float(quants[0].item()),
                "p90": float(quants[1].item()),
                "p95": float(quants[2].item()),
                "p99": float(quants[3].item()),
                "max": float(vals.max().item()),
                "mean": float(vals.mean().item()),
            }
            rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(args_cli.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[probe] wrote {len(rows)} rows to {args_cli.output_csv}")

    # Also print a compact summary table to stdout.
    print("\n[probe] per-channel percentiles (max over agents):")
    print(f"  {'channel':<20} | {'p50':>8} | {'p90':>8} | {'p95':>8} | {'p99':>8} | {'max':>8}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for ch in range(7):
        sub = [r for r in rows if r["channel_idx"] == ch]
        p50 = max(r["p50"] for r in sub)
        p90 = max(r["p90"] for r in sub)
        p95 = max(r["p95"] for r in sub)
        p99 = max(r["p99"] for r in sub)
        mx  = max(r["max"] for r in sub)
        print(f"  {CHANNEL_LABELS[ch]:<20} | {p50:>8.4f} | {p90:>8.4f} | {p95:>8.4f} | {p99:>8.4f} | {mx:>8.4f}")

    env_skrl.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
