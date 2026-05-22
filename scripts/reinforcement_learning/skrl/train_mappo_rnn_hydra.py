#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import debugpy
import os

# Only start debugger if an environment variable is set
if os.getenv("IDE_DEBUG_MODE") == "True":
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client() # The script will freeze here until you hit F5

"""
Train RL agent with MAPPO-RNN using Hydra configuration system.

This script uses Hydra for configuration management, making it fully compatible
with Ray Tune hyperparameter optimization via hydra_args overrides.

Usage:
    # Standard training
    python train_mappo_rnn_hydra.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1024

    # With Hydra overrides (for hyperparameter tuning)
    python train_mappo_rnn_hydra.py --task Isaac-Quadcopter-Direct-v0 --num_envs 1024 \
        agent.learning_rate=0.001 models.policy.hidden_size=256 agent.sequence_length=50

    # Ray Tune will use this format automatically via hydra_args
"""

import argparse
import sys

# Parse arguments BEFORE launching Isaac Sim
parser = argparse.ArgumentParser(description="Train MAPPO-RNN agent with Hydra config system.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Override agent.experiment.experiment_name in the loaded skrl cfg. "
                         "Use to avoid run-dir collisions when iterating on the same yaml.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Optional path to a skrl checkpoint .pt to warm-start training. "
                         "Restores per-agent policy, value, state/shared-state/value preprocessors, "
                         "and optimizer state. The trainer's timestep counter still starts at 0 — "
                         "this is warm-start, not full resume. When set, also auto-pins the env "
                         "curriculum to the checkpoint's training step (inferred from the filename "
                         "agent_<N>.pt) so the fine-tune sees the same env distribution the source "
                         "checkpoint last trained against. Override via --debug_initial_step or "
                         "disable via --no_curriculum_pin.")
parser.add_argument("--skip_load_optimizer", action="store_true", default=False,
                    help="When loading --checkpoint, do NOT restore optimizer state "
                         "(useful for transfer learning to a new env/reward).")
parser.add_argument("--debug_initial_step", type=int, default=None,
                    help="Override the env's debug_initial_step (the curriculum-pinned step). "
                         "When --checkpoint is passed, auto-inferred from the checkpoint's "
                         "filename (e.g. agent_400000.pt → 400000) unless this flag is set. "
                         "Forces cfg.use_debug_initial_step=True. Pass 0 (or use "
                         "--no_curriculum_pin) to skip pinning entirely.")
parser.add_argument("--no_curriculum_pin", action="store_true", default=False,
                    help="With --checkpoint, do NOT pin the curriculum to the checkpoint's "
                         "training step. Useful when you actually want the fine-tune to "
                         "re-traverse the curriculum from scratch (rare).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode (no GUI).")

# Parse known args (separates our args from Hydra args)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra (Hydra will parse hydra_args)
sys.argv = [sys.argv[0]] + hydra_args

# NOW we can import Isaac Lab and launch the simulator
from isaaclab.app import AppLauncher

# Create minimal AppLauncher args
app_args = argparse.Namespace(headless=args_cli.headless, device="cuda:0", experience="", enable_cameras=args_cli.video)
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is initialized."""

import torch
import os
from datetime import datetime
import gymnasium as gym
import numpy as np

from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Import Isaac Lab components
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper


class MARLVideoRecorder:
    """Simple video recorder for MARL environments.

    Captures viewport frames at specified intervals and writes mp4 videos using moviepy.
    Works directly with DirectMARLEnv.render() without gymnasium wrapper compatibility issues.
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
        """Call after each environment step to potentially capture a frame."""
        self.step_count += 1

        # Start recording at specified intervals
        if not self._recording and self.step_count % self.video_interval == 0:
            self._recording = True
            self._frames = []

        # Capture frame if recording
        if self._recording:
            frame = self.env.render()
            if frame is not None:
                self._frames.append(frame)

            # Stop recording after video_length frames
            if len(self._frames) >= self.video_length:
                self._save_video()
                self._recording = False
                self._frames = []

    def close(self):
        """Save any in-progress recording."""
        if self._recording and len(self._frames) > 0:
            self._save_video()
            self._recording = False
            self._frames = []

    def _save_video(self):
        """Write accumulated frames to an mp4 file."""
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


class SingleAgentWrapper:
    """Wrapper to convert single-agent env to multi-agent format with agent_0."""

    def __init__(self, env):
        self.env = env
        self.possible_agents = ["agent_0"]
        self.num_envs = env.num_envs
        self.device = env.device

        # Wrap spaces in dictionaries
        self.observation_space = {"agent_0": env.observation_space}
        self.action_space = {"agent_0": env.action_space}
        self.observation_spaces = self.observation_space
        self.action_spaces = self.action_space
        self.shared_observation_spaces = self.observation_space

    def reset(self):
        obs, info = self.env.reset()
        # Wrap per-agent data but preserve "log" key at top level for trainer logging
        info_wrapped = {"agent_0": info if info is not None else {}}
        if info is not None and "log" in info:
            info_wrapped["log"] = info["log"]
        return {"agent_0": obs}, info_wrapped

    def step(self, actions):
        action = actions["agent_0"]
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Wrap per-agent data but preserve "log" key at top level for trainer logging
        info_wrapped = {"agent_0": info if info is not None else {}}
        if info is not None and "log" in info:
            info_wrapped["log"] = info["log"]
        return (
            {"agent_0": obs},
            {"agent_0": reward},
            {"agent_0": terminated},
            {"agent_0": truncated},
            info_wrapped
        )

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class VideoRecordingTrainer(SequentialTrainer):
    """SequentialTrainer with video recording support for MARL environments.

    Overrides multi_agent_train to capture viewport frames at specified intervals
    and write mp4 videos using moviepy, bypassing gymnasium's RecordVideo wrapper
    which is incompatible with DirectMARLEnv's dict-based step returns.
    """

    def __init__(self, env, agents, cfg=None, video_recorder=None, **kwargs):
        super().__init__(env=env, agents=agents, cfg=cfg, **kwargs)
        self.video_recorder = video_recorder

    def multi_agent_train(self):
        """Override to inject video recording after each env step."""
        import sys
        import tqdm

        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents > 1

        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # capture video frame
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

        # Save any in-progress recording
        if self.video_recorder is not None:
            self.video_recorder.close()


def _infer_step_from_ckpt_filename(checkpoint_path: str) -> int | None:
    """Best-effort parse of the training step from a checkpoint filename.

    Recognizes the SKRL convention ``agent_<STEP>.pt`` (e.g. ``agent_400000.pt``).
    Returns the integer step on match, ``None`` otherwise.

    Used to auto-pin the env curriculum to the checkpoint's terminal regime
    when --checkpoint is passed without an explicit --debug_initial_step.
    """
    import os, re
    name = os.path.basename(checkpoint_path)
    m = re.match(r"agent_(\d+)\.pt$", name)
    if m:
        return int(m.group(1))
    return None


def load_checkpoint_into_agent(agent, checkpoint_path: str, possible_agents, skip_optimizer: bool = False):
    """Warm-start an already-initialized skrl MARL agent from a saved checkpoint.

    Mirrors skrl's `MultiAgent.load()` but with two adjustments:
    (1) up-front validation/warning when the checkpoint's agent UIDs don't
        match the current env's `possible_agents` (skrl's stock load only
        logs warnings deep in the loop), and
    (2) optional skipping of the per-agent optimizer state (for transfer
        learning where the optimizer's momentum is stale).

    Loads per-uid: policy, value, state_preprocessor, shared_state_preprocessor,
    value_preprocessor, optimizer. The trainer's timestep counter is unaffected.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint_path}")

    print(f"[CKPT] Loading checkpoint: {checkpoint_path}")
    modules = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)

    if not isinstance(modules, dict):
        raise ValueError(
            f"[CKPT] Unexpected checkpoint format: expected dict-of-uids, got {type(modules)}."
        )

    ckpt_agents = set(modules.keys())
    env_agents = set(possible_agents)
    if ckpt_agents != env_agents:
        missing = env_agents - ckpt_agents
        extra = ckpt_agents - env_agents
        print(f"[CKPT][WARN] Agent-id mismatch. ckpt={sorted(ckpt_agents)}, env={sorted(env_agents)}")
        if missing:
            print(f"[CKPT][WARN]   Missing from ckpt (these agents start from scratch): {sorted(missing)}")
        if extra:
            print(f"[CKPT][WARN]   Extra in ckpt (ignored): {sorted(extra)}")

    skipped_summary = []
    for uid in possible_agents:
        if uid not in modules:
            continue
        for name, data in modules[uid].items():
            if skip_optimizer and name == "optimizer":
                skipped_summary.append(f"{uid}:{name}")
                continue
            module = agent.checkpoint_modules[uid].get(name, None)
            if module is None:
                print(f"[CKPT][WARN]   No live module for {uid}:{name}, skipping")
                continue
            if not hasattr(module, "load_state_dict"):
                print(f"[CKPT][WARN]   {uid}:{name} has no load_state_dict, skipping")
                continue
            module.load_state_dict(data)
            # Note: skrl's stock load() calls module.eval() here. We don't —
            # MAPPO_RNN.post_interaction toggles train/eval around the update
            # step, so the effective mode is managed by the agent itself.

    if skipped_summary:
        print(f"[CKPT] Skipped optimizer state for: {skipped_summary}")
    print(f"[CKPT] Loaded weights for agents: {sorted(env_agents & ckpt_agents)}")


def create_models(agent_cfg: dict, env, device, is_multi_agent: bool):
    """Create policy and value models based on config."""

    models = {}
    memories = {}

    # Get config sections
    model_cfg = agent_cfg.get("models", {})
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 50)

    if is_multi_agent:
        possible_agents = env.possible_agents

        # Check for shared observation space
        try:
            shared_observation_spaces = env.shared_observation_spaces
        except AttributeError:
            obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
            shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

        # Create shared networks
        policy_cfg = model_cfg.get("policy", {})
        value_cfg = model_cfg.get("value", {})

        shared_policy = MAPPORNNPolicy(
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
        )

        shared_value = MAPPORNNValue(
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
        # Single-agent
        possible_agents = ["agent_0"]
        observation_spaces = {"agent_0": env.observation_space}
        action_spaces = {"agent_0": env.action_space}
        shared_observation_spaces = {"agent_0": env.observation_space}

        policy_cfg = model_cfg.get("policy", {})
        value_cfg = model_cfg.get("value", {})

        policy = MAPPORNNPolicy(
            observation_space=observation_spaces["agent_0"],
            action_space=action_spaces["agent_0"],
            device=device,
            hidden_size=policy_cfg.get("hidden_size", 256),
            gru_num_layers=policy_cfg.get("gru_num_layers", 2),
            gru_hidden_size=policy_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            initial_log_std=policy_cfg.get("initial_log_std", -0.5),
            min_log_std=policy_cfg.get("min_log_std", -5.0),
            max_log_std=policy_cfg.get("max_log_std", 0.7),
            sequence_length=sequence_length,
        )

        value = MAPPORNNValue(
            observation_space=shared_observation_spaces["agent_0"],
            action_space=action_spaces["agent_0"],
            device=device,
            hidden_size=value_cfg.get("hidden_size", 256),
            gru_num_layers=value_cfg.get("gru_num_layers", 2),
            gru_hidden_size=value_cfg.get("gru_hidden_size", 256),
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

        models["agent_0"] = {"policy": policy, "value": value}
        memories["agent_0"] = RandomMemory(
            memory_size=agent_cfg.get("agent", {}).get("rollouts", 800),
            num_envs=env.num_envs,
            device=device,
        )

        return models, memories, shared_observation_spaces, observation_spaces, action_spaces


@hydra_task_config(args_cli.task, "skrl_mappo_rnn_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train MAPPO-RNN agent with Hydra-managed configuration."""

    # Override with CLI args
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # Set seed
    seed = agent_cfg.get("seed", 42)
    set_seed(seed)
    # Ticket 034: env explicitly requires cfg.seed for the curriculum RNG.
    # Propagate the agent-level seed to env_cfg so existing CLI behavior is
    # unchanged (default agent_cfg seed: 42; --seed overrides).
    env_cfg.seed = seed

    # Optional CLI override for the skrl experiment name (avoids run-dir
    # collisions when iterating on the same yaml). Hydra struct-mode rejects
    # `+agent.experiment.experiment_name=...` at the command line, so the
    # override goes through here instead.
    if args_cli.experiment_name is not None:
        agent_cfg.setdefault("agent", {}).setdefault("experiment", {})[
            "experiment_name"
        ] = args_cli.experiment_name

    # Warm-start curriculum pin. When loading a checkpoint, the trainer's
    # timestep counter starts at 0 — which would restart the env curriculum
    # from scratch and undermine any fine-tuning ablation against the
    # source checkpoint's terminal regime. We pin the env's curriculum step
    # to the checkpoint's training step so the fine-tune sees the SAME env
    # distribution the checkpoint was last trained against.
    #
    # Resolution order:
    #   --no_curriculum_pin       -> skip pinning entirely (rare)
    #   --debug_initial_step N    -> explicit override
    #   --checkpoint <path>       -> auto-infer from filename (agent_<N>.pt)
    #   (none)                    -> leave env_cfg untouched
    if args_cli.checkpoint is not None and not args_cli.no_curriculum_pin:
        pin_step = args_cli.debug_initial_step
        if pin_step is None:
            pin_step = _infer_step_from_ckpt_filename(args_cli.checkpoint)
            if pin_step is None:
                raise ValueError(
                    f"--checkpoint passed without --debug_initial_step, and the "
                    f"filename {os.path.basename(args_cli.checkpoint)!r} doesn't "
                    f"match the agent_<N>.pt convention. Either pass "
                    f"--debug_initial_step <N> explicitly, or --no_curriculum_pin "
                    f"to skip pinning (rare; usually you want to pin)."
                )
        env_cfg.use_debug_initial_step = True
        env_cfg.debug_initial_step = int(pin_step)
        print(
            f"[CKPT-PIN] Curriculum pinned to step {pin_step} "
            f"(from {'--debug_initial_step' if args_cli.debug_initial_step is not None else 'checkpoint filename'}). "
            f"Env curriculum will hold at this step regardless of trainer.timestep.",
            flush=True,
        )
    elif args_cli.debug_initial_step is not None:
        # Explicit pin without --checkpoint: rare but support it (e.g., training
        # from scratch but skipping curriculum ramp-up for ablation purposes).
        env_cfg.use_debug_initial_step = True
        env_cfg.debug_initial_step = int(args_cli.debug_initial_step)
        print(
            f"[CKPT-PIN] Curriculum pinned to step {args_cli.debug_initial_step} "
            f"(no checkpoint; explicit --debug_initial_step).",
            flush=True,
        )
    elif args_cli.checkpoint is not None and args_cli.no_curriculum_pin:
        print(
            "[CKPT-PIN] --no_curriculum_pin: env curriculum starts from step 0 "
            "despite warm-start checkpoint load. Hope you know what you're doing.",
            flush=True,
        )

    # Validate burn-in
    sequence_length = agent_cfg.get("agent", {}).get("sequence_length", 50)
    burn_in_steps = agent_cfg.get("agent", {}).get("burn_in_steps", 0)

    if burn_in_steps >= sequence_length:
        raise ValueError(f"burn_in_steps ({burn_in_steps}) must be < sequence_length ({sequence_length})")

    # Process preprocessor configs
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

    # Convert learning_rate to float if it's a string (YAML may parse "1e-3" as string)
    if "learning_rate" in agent_cfg.get("agent", {}):
        lr = agent_cfg["agent"]["learning_rate"]
        if isinstance(lr, str):
            agent_cfg["agent"]["learning_rate"] = float(lr)
            print(f"[INFO] Converted learning_rate from string '{lr}' to float {agent_cfg['agent']['learning_rate']}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "mappo_rnn_logs"))
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mappo_rnn_torch"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Create video recorder (wraps unwrapped env directly, not the gym wrapper chain)
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

    # Detect multi-agent
    is_multi_agent = isinstance(env.unwrapped.unwrapped, DirectMARLEnv)

    if is_multi_agent:
        possible_agents = env.possible_agents
        print(f"[INFO] Multi-agent environment: {len(possible_agents)} agents")
    else:
        print(f"[INFO] Single-agent environment")
        env = SingleAgentWrapper(env)
        possible_agents = env.possible_agents
        is_multi_agent = True

    # Create models
    models, memories, shared_observation_spaces, observation_spaces, action_spaces = create_models(
        agent_cfg, env, device, is_multi_agent
    )

    # Finalize preprocessor kwargs (need device and observation spaces from env)
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

    # Create agent
    mappo_cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    mappo_cfg.update(agent_cfg.get("agent", {}))

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        device=device,
        cfg=mappo_cfg,
        shared_observation_spaces=shared_observation_spaces,
    )

    # Create trainer
    trainer_cfg = {
        "timesteps": agent_cfg.get("trainer", {}).get("timesteps", 400000),
        "headless": args_cli.headless,
        "environment_info": "log",
    }

    if video_recorder is not None:
        trainer = VideoRecordingTrainer(cfg=trainer_cfg, env=env, agents=agent, video_recorder=video_recorder)
    else:
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # Optional checkpoint warm-start. The SequentialTrainer's __init__ already
    # ran agent.init(), which instantiated the preprocessors and optimizers;
    # only at this point can we load their state_dicts.
    if args_cli.checkpoint is not None:
        load_checkpoint_into_agent(
            agent,
            args_cli.checkpoint,
            possible_agents,
            skip_optimizer=args_cli.skip_load_optimizer,
        )

    # Print training info
    print("=" * 80)
    print("Starting MAPPO-RNN Training")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Device: {device}")
    print(f"Num envs: {env_cfg.scene.num_envs}")
    print(f"Num agents: {len(possible_agents)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Burn-in steps: {burn_in_steps}")
    print(f"Learning rate: {agent_cfg.get('agent', {}).get('learning_rate', 'default')}")
    print(f"Total timesteps: {trainer_cfg['timesteps']}")
    print("=" * 80)

    # Train
    trainer.train()

    # Save models
    for agent_id in possible_agents:
        agent_path = os.path.join(log_dir, f"agent_{agent_id}_final.pt")
        torch.save({
            'policy_state_dict': models[agent_id]["policy"].state_dict(),
            'value_state_dict': models[agent_id]["value"].state_dict(),
        }, agent_path)
        print(f"[INFO] Agent {agent_id} saved to: {agent_path}")

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
