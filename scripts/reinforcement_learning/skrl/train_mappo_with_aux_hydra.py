#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train RL agent with MAPPO-RNN + auxiliary policy triangulation head (ticket 031).

Mirrors train_mappo_rnn_hydra.py with two differences:
1. Models are MAPPOWithAuxPolicy / MAPPOWithAuxValue (extra tri head on policy).
2. Trainer wrapper calls env.unwrapped.get_aux_supervision() after env.step()
   and writes the result into the infos dict so MAPPOWithAux.record_transition
   can route it into the per-agent memories.

Usage:
    python train_mappo_with_aux_hydra.py --task Isaac-Iris-MA6-Direct-Test-v0 --num_envs 256

    # Hydra overrides (e.g., turn aux loss off for the bit-exact baseline)
    python train_mappo_with_aux_hydra.py --task Isaac-Iris-MA6-Direct-Test-v0 \\
        agent.aux_loss_scale=0.0
"""
import debugpy
import os

# Only start debugger if an environment variable is set
if os.getenv("IDE_DEBUG_MODE") == "True":
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client()

import argparse
import sys

# Parse arguments BEFORE launching Isaac Sim
parser = argparse.ArgumentParser(description="Train MAPPO-RNN+Aux agent with Hydra config system.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode (no GUI).")

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

app_args = argparse.Namespace(headless=args_cli.headless, device="cuda:0", experience="", enable_cameras=args_cli.video)
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is initialized."""

import torch
from datetime import datetime
import gymnasium as gym
import numpy as np

from mappo_with_aux import (
    MAPPOWithAux,
    MAPPO_WITH_AUX_DEFAULT_CONFIG,
    MAPPOWithAuxPolicy,
    MAPPOWithAuxValue,
)

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper


# =========================================================================
# Video recorder (verbatim copy from train_mappo_rnn_hydra; kept inline so
# this script does not depend on importing the sibling train script).
# =========================================================================

class MARLVideoRecorder:
    """Simple video recorder for MARL environments.

    Captures viewport frames at specified intervals and writes mp4 videos using moviepy.
    """

    def __init__(self, env, video_folder: str, video_interval: int = 2000, video_length: int = 200):
        self.env = env
        self.video_folder = video_folder
        self.video_interval = video_interval
        self.video_length = video_length
        self.step_count = 0
        self._frames = []
        self._recording = False
        self._video_count = 0

        os.makedirs(video_folder, exist_ok=True)
        print(f"[INFO] Video recorder: folder={video_folder}, interval={video_interval}, length={video_length}")

    def step(self):
        self.step_count += 1
        if not self._recording and self.step_count % self.video_interval == 0:
            self._recording = True
            self._frames = []
        if self._recording:
            frame = self.env.render()
            if frame is not None:
                self._frames.append(frame)
            if len(self._frames) >= self.video_length:
                self._save_video()
                self._recording = False
                self._frames = []

    def close(self):
        if self._recording and len(self._frames) > 0:
            self._save_video()
            self._recording = False
            self._frames = []

    def _save_video(self):
        if len(self._frames) == 0:
            return
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            path = os.path.join(self.video_folder, f"rl-video-step-{self.step_count - len(self._frames)}.mp4")
            fps = self.env.metadata.get("render_fps", 30) if hasattr(self.env, "metadata") else 30
            clip = ImageSequenceClip(self._frames, fps=fps)
            clip.write_videofile(path, logger="bar")
            print(f"[INFO] Video saved: {path} ({len(self._frames)} frames)")
            self._video_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to save video: {e}")


# =========================================================================
# Trainer subclass: SequentialTrainer + aux supervision injection (+ optional video).
# =========================================================================

class AuxInjectingTrainer(SequentialTrainer):
    """SequentialTrainer that calls ``env.get_aux_supervision()`` after each
    ``env.step()`` and writes the returned tensors into the ``infos`` dict.

    Mirrors the pattern by which ``shared_states`` is injected today — the env
    exposes a public READ method, the trainer wrapper transports it into
    ``infos``, the agent's ``record_transition`` reads it. See ticket 031 §3.
    """

    def __init__(self, env, agents, cfg=None, video_recorder=None, **kwargs):
        super().__init__(env=env, agents=agents, cfg=cfg, **kwargs)
        self.video_recorder = video_recorder

    def multi_agent_train(self):
        """Body mirrors VideoRecordingTrainer.multi_agent_train with one block
        added between ``env.step`` and ``record_transition``."""
        import sys as _sys
        import tqdm

        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents > 1

        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar, file=_sys.stdout,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # ----- aux supervision injection (ticket 031, slice 2) -----
                aux = self.env.unwrapped.get_aux_supervision()
                infos["tri_target_position_w"] = aux["tri_target_position_w"]
                infos["tri_target_valid"] = aux["tri_target_valid"]
                # ----------------------------------------------------------

                if not self.headless:
                    self.env.render()
                if self.video_recorder is not None:
                    self.video_recorder.step()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

        if self.video_recorder is not None:
            self.video_recorder.close()


# =========================================================================
# Model / memory factory (mirrors train_mappo_rnn_hydra.create_models).
# =========================================================================

def create_models(agent_cfg: dict, env, device, is_multi_agent: bool):
    """Create policy/value models with parameter-sharing across agents."""
    models = {}
    memories = {}

    model_cfg = agent_cfg.get("models", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 50)

    if is_multi_agent:
        possible_agents = env.possible_agents

        try:
            shared_observation_spaces = env.shared_observation_spaces
        except AttributeError:
            obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
            shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

        policy_cfg = model_cfg.get("policy", {})
        value_cfg = model_cfg.get("value", {})
        tri_head_cfg = policy_cfg.get("tri_head", {})

        shared_policy = MAPPOWithAuxPolicy(
            observation_space=env.observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=policy_cfg.get("hidden_size", 256),
            gru_num_layers=policy_cfg.get("gru_num_layers", 2),
            gru_hidden_size=policy_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            initial_log_std=policy_cfg.get("initial_log_std", -0.5),
            min_log_std=policy_cfg.get("min_log_std", -5.0),
            max_log_std=policy_cfg.get("max_log_std", 0.7),
            sequence_length=sequence_length,
            tri_head_enabled=tri_head_cfg.get("enabled", True),
            tri_head_hidden=tuple(tri_head_cfg.get("hidden", [128])),
            tri_log_var_clamp=tuple(tri_head_cfg.get("log_var_clamp", [-10.0, 4.0])),
        )

        shared_value = MAPPOWithAuxValue(
            observation_space=shared_observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=value_cfg.get("hidden_size", 256),
            gru_num_layers=value_cfg.get("gru_num_layers", 2),
            gru_hidden_size=value_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

        for agent_id in possible_agents:
            models[agent_id] = {"policy": shared_policy, "value": shared_value}
            memories[agent_id] = RandomMemory(
                memory_size=agent_cfg.get("agent", {}).get("rollouts", 800),
                num_envs=env.num_envs,
                device=device,
            )

        return models, memories, shared_observation_spaces, env.observation_spaces, env.action_spaces

    else:
        raise RuntimeError("train_mappo_with_aux_hydra requires a multi-agent env.")


# =========================================================================
# Main entry.
# =========================================================================

@hydra_task_config(args_cli.task, "skrl_mappo_rnn_aux_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train MAPPO-RNN+Aux agent with Hydra-managed configuration."""

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    seed = agent_cfg.get("seed", 42)
    set_seed(seed)

    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 50)
    burn_in_steps = agent_cfg.get("agent", {}).get("burn_in_steps", 0)
    if burn_in_steps >= sequence_length:
        raise ValueError(f"burn_in_steps ({burn_in_steps}) must be < sequence_length ({sequence_length})")

    # Resolve preprocessor / scheduler classes from string names.
    if "state_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["state_preprocessor"] == "RunningStandardScaler":
            agent_cfg["agent"]["state_preprocessor"] = RunningStandardScaler
    if "shared_state_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["shared_state_preprocessor"] == "RunningStandardScaler":
            agent_cfg["agent"]["shared_state_preprocessor"] = RunningStandardScaler
    if "value_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["value_preprocessor"] == "RunningStandardScaler":
            agent_cfg["agent"]["value_preprocessor"] = RunningStandardScaler
    if "learning_rate_scheduler" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR":
            agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    if "learning_rate" in agent_cfg.get("agent", {}):
        lr = agent_cfg["agent"]["learning_rate"]
        if isinstance(lr, str):
            agent_cfg["agent"]["learning_rate"] = float(lr)

    log_root_path = os.path.join(
        "logs", "skrl",
        agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "mappo_rnn_aux_logs"),
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mappo_rnn_aux"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    print(f"[INFO] Creating environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    video_recorder = None
    if args_cli.video:
        video_recorder = MARLVideoRecorder(
            env=env.unwrapped,
            video_folder=os.path.join(log_dir, "videos", "train"),
            video_interval=args_cli.video_interval,
            video_length=args_cli.video_length,
        )

    env = SkrlVecEnvWrapper(env)
    device = env.device

    is_multi_agent = isinstance(env.unwrapped.unwrapped, DirectMARLEnv)
    if not is_multi_agent:
        raise RuntimeError("train_mappo_with_aux_hydra requires a multi-agent env.")

    possible_agents = env.possible_agents
    print(f"[INFO] Multi-agent environment: {len(possible_agents)} agents")

    models, memories, shared_observation_spaces, observation_spaces, action_spaces = create_models(
        agent_cfg, env, device, is_multi_agent
    )

    # Finalize preprocessor kwargs.
    if "state_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"].get("state_preprocessor_kwargs") is None:
            agent_cfg["agent"]["state_preprocessor_kwargs"] = {}
        agent_cfg["agent"]["state_preprocessor_kwargs"]["size"] = observation_spaces[possible_agents[0]]
        agent_cfg["agent"]["state_preprocessor_kwargs"]["device"] = device
    if "shared_state_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"].get("shared_state_preprocessor_kwargs") is None:
            agent_cfg["agent"]["shared_state_preprocessor_kwargs"] = {}
        agent_cfg["agent"]["shared_state_preprocessor_kwargs"]["size"] = shared_observation_spaces[possible_agents[0]]
        agent_cfg["agent"]["shared_state_preprocessor_kwargs"]["device"] = device
    if "value_preprocessor" in agent_cfg.get("agent", {}):
        if agent_cfg["agent"].get("value_preprocessor_kwargs") is None:
            agent_cfg["agent"]["value_preprocessor_kwargs"] = {}
        agent_cfg["agent"]["value_preprocessor_kwargs"]["device"] = device

    # Merge defaults with YAML-loaded agent config.
    mappo_cfg = MAPPO_WITH_AUX_DEFAULT_CONFIG.copy()
    mappo_cfg.update(agent_cfg.get("agent", {}))

    agent = MAPPOWithAux(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )

    trainer_cfg = {
        "timesteps": agent_cfg.get("trainer", {}).get("timesteps", 400000),
        "headless": args_cli.headless,
        "environment_info": "log",
    }
    trainer = AuxInjectingTrainer(cfg=trainer_cfg, env=env, agents=agent, video_recorder=video_recorder)

    print("=" * 80)
    print("Starting MAPPO-RNN+AUX Training")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Device: {device}")
    print(f"Num envs: {env_cfg.scene.num_envs}")
    print(f"Num agents: {len(possible_agents)}")
    print(f"Aux loss scale: {mappo_cfg.get('aux_loss_scale')}")
    print(f"Total timesteps: {trainer_cfg['timesteps']}")
    print("=" * 80)

    trainer.train()

    for agent_id in possible_agents:
        agent_path = os.path.join(log_dir, f"agent_{agent_id}_final.pt")
        torch.save({
            "policy_state_dict": models[agent_id]["policy"].state_dict(),
            "value_state_dict": models[agent_id]["value"].state_dict(),
        }, agent_path)
        print(f"[INFO] Agent {agent_id} saved to: {agent_path}")

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
