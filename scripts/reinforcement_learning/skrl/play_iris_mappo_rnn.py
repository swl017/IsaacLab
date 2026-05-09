# file: play_iris_mappo_rnn.py
"""
Script to play trained MAPPO RNN agents for Iris Multi-Agent task.

This script loads models trained with the train_iris_mappo_rnn.py script.
"""

import debugpy
import os

# Only start debugger if an environment variable is set
if os.getenv("IDE_DEBUG_MODE") == "True":
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client() # The script will freeze here until you hit F5

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play trained MAPPO RNN agents for Iris Multi-Agent task")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file (e.g., .../checkpoints/agent_200000.pt)")
parser.add_argument("--task", type=str, default="Isaac-Iris-MA4-Direct-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="Number of play environments")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps)")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")
parser.add_argument("--deterministic", action="store_true", default=True,
                    help="Use deterministic (mean) actions instead of sampling (default: True)")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--step",
    type=str,
    default=None,
    help="The environment step at which to evaluate the agent. If not specified, evaluates from step 0",
)
parser.add_argument(
    "--show_plots",
    action="store_true",
    default=True,
    help="Show per-agent camera + reward + gimbal plots for env 0.",
)
parser.add_argument(
    "--no_show_plots",
    dest="show_plots",
    action="store_false",
    help="Disable per-agent visualization plots.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
import time
import math
import pickle
import gymnasium as gym
import numpy as np
import yaml
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt

# Import MAPPO_RNN components
from mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG, MAPPORNNPolicy, MAPPORNNValue

# Import SKRL components
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

# Import IsaacLab components
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul


# =============================================================================
# Visualization helpers (mirrored from teleop_iris_ma6.py)
# =============================================================================

def project_point_to_image(
    point_w: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_ros: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[int, int, bool]:
    """Project a 3D world point to 2D image pixel using pinhole model."""
    point_rel = point_w - camera_pos_w
    point_cam = quat_apply(quat_inv(camera_quat_w_ros.unsqueeze(0)), point_rel.unsqueeze(0)).squeeze(0)

    x, y, z = point_cam[0].item(), point_cam[1].item(), point_cam[2].item()
    if z <= 0.01:
        return 0, 0, False

    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item()
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()

    return int(fx * x / z + cx), int(fy * y / z + cy), True


def _draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=10):
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = max(1, int((dx * dx + dy * dy) ** 0.5))
    num_dashes = max(1, length // dash_len)
    for i in range(0, num_dashes, 2):
        t0 = i / num_dashes
        t1 = min((i + 1) / num_dashes, 1.0)
        sx, sy = int(x1 + dx * t0), int(y1 + dy * t0)
        ex, ey = int(x1 + dx * t1), int(y1 + dy * t1)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def draw_bbox_overlay(
    image: np.ndarray,
    bbox_xyxy,
    bbox_empty: bool,
    agent_id: str,
    zoom_level: float = 1.0,
    target_pixel=None,
    distance_m: float | None = None,
    obs_bbox_xyxy=None,
    obs_bbox_empty: bool = True,
    bbox_aoi: float | None = None,
    replicated_bbox_xyxy=None,
    replicated_bbox_empty: bool = True,
    bg_is_ground: bool | None = None,
) -> np.ndarray:
    """Draw bbox overlay on a single agent camera image (env 0)."""
    vis = image.copy()
    h_img, w_img = vis.shape[:2]
    x_min, y_min, x_max, y_max = bbox_xyxy

    if not bbox_empty:
        color = (0, 255, 0)
        label = "Target DETECTED"
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, 2)
        w = x_max - x_min
        h = y_max - y_min
        cv2.putText(vis, f"{w}x{h}px", (x_min, max(y_min - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    else:
        color = (255, 0, 0)
        label = "Target NOT detected"

    if obs_bbox_xyxy is not None and not obs_bbox_empty:
        ox_min, oy_min, ox_max, oy_max = obs_bbox_xyxy
        obs_color = (255, 255, 0)
        for start, end in [
            ((ox_min, oy_min), (ox_max, oy_min)),
            ((ox_max, oy_min), (ox_max, oy_max)),
            ((ox_max, oy_max), (ox_min, oy_max)),
            ((ox_min, oy_max), (ox_min, oy_min)),
        ]:
            _draw_dashed_line(vis, start, end, obs_color, thickness=2, dash_len=8)

    if target_pixel is not None:
        tu, tv, tvalid = target_pixel
        if tvalid and 0 <= tu < w_img and 0 <= tv < h_img:
            cross_color = (0, 120, 255)
            size = 12
            cv2.line(vis, (tu - size, tv), (tu + size, tv), cross_color, 2)
            cv2.line(vis, (tu, tv - size), (tu, tv + size), cross_color, 2)

    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"{agent_id} | Zoom: {zoom_level:.2f}x", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if distance_m is not None:
        dist_str = f"{distance_m:.1f} m"
        if bg_is_ground is not None:
            bg_label = "GND" if bg_is_ground else "SKY"
            bg_color = (139, 90, 43) if bg_is_ground else (135, 206, 250)
            dist_str += f"  [{bg_label}]"
            cv2.putText(vis, dist_str, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_color, 1)
        else:
            cv2.putText(vis, dist_str, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if bbox_aoi is not None:
        aoi_ms = bbox_aoi * 1000.0
        aoi_steps = bbox_aoi / 0.04 if bbox_aoi > 0 else 0.0
        cv2.putText(vis, f"BBox AoI: {aoi_ms:.0f}ms ({aoi_steps:.1f} steps)",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if replicated_bbox_xyxy is not None and not replicated_bbox_empty:
        rx1, ry1, rx2, ry2 = replicated_bbox_xyxy
        rep_color = (255, 0, 255)
        for start, end in [
            ((rx1, ry1), (rx2, ry1)),
            ((rx2, ry1), (rx2, ry2)),
            ((rx2, ry2), (rx1, ry2)),
            ((rx1, ry2), (rx1, ry1)),
        ]:
            _draw_dashed_line(vis, start, end, rep_color, thickness=2, dash_len=5)

    return vis


def get_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    """Derive run directory from checkpoint path.

    .../run_dir/checkpoints/agent_*.pt -> .../run_dir/
    .../run_dir/agent_*_final.pt -> .../run_dir/
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    parent = os.path.dirname(checkpoint_path)
    if os.path.basename(parent) == "checkpoints":
        return os.path.dirname(parent)
    return parent


def load_agent_config(run_dir: str) -> Optional[dict]:
    """Load agent config from run_dir/params/agent.pkl (or agent.yaml as fallback).

    Prefers pickle format since YAML may contain Python objects that safe_load cannot handle.
    """
    # Try pickle first (handles Python objects correctly)
    pkl_path = os.path.join(run_dir, "params", "agent.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            config = pickle.load(f)
            return config if isinstance(config, dict) else None

    # Fallback to YAML (may fail if it contains Python objects)
    yaml_path = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if isinstance(config, dict) else None
        except Exception:
            print(f"[WARN] Could not load {yaml_path}. Using defaults.")
            return None

    return None


def get_model_hyperparams(agent_cfg: Optional[dict]) -> dict:
    """Extract model hyperparameters from agent config."""
    defaults = {
        "hidden_size": 256,
        "gru_num_layers": 2,
        "gru_hidden_size": 256,
        "initial_log_std": -0.5,
        "min_log_std": -5.0,
        "max_log_std": 0.7,
    }
    if agent_cfg is None:
        return defaults

    policy_cfg = agent_cfg.get("models", {}).get("policy", {})
    return {
        "hidden_size": policy_cfg.get("hidden_size", defaults["hidden_size"]),
        "gru_num_layers": policy_cfg.get("gru_num_layers", defaults["gru_num_layers"]),
        "gru_hidden_size": policy_cfg.get("gru_hidden_size", defaults["gru_hidden_size"]),
        "initial_log_std": policy_cfg.get("initial_log_std", defaults["initial_log_std"]),
        "min_log_std": policy_cfg.get("min_log_std", defaults["min_log_std"]),
        "max_log_std": policy_cfg.get("max_log_std", defaults["max_log_std"]),
    }


def get_sequence_length(agent_cfg: Optional[dict]) -> int:
    """Extract sequence length from agent config."""
    if agent_cfg is None:
        return 128  # default
    return agent_cfg.get("agent", {}).get("sequence_length", 128)


def load_checkpoint_weights(checkpoint_path: str, possible_agents: Sequence[str], device) -> dict:
    """Load checkpoint and return state dicts for each agent.

    Handles both SKRL format and custom final format.
    Returns: Dict[agent_id, {"policy": state_dict, "value": state_dict,
                             "state_preprocessor": state_dict, ...}]
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect format
    if "policy_state_dict" in checkpoint:
        # Custom final format - same weights for all agents (shared policy)
        print(f"[INFO] Detected custom final checkpoint format")
        return {agent_id: {
            "policy": checkpoint["policy_state_dict"],
            "value": checkpoint["value_state_dict"],
            "state_preprocessor": checkpoint.get("state_preprocessor"),
            "shared_state_preprocessor": checkpoint.get("shared_state_preprocessor"),
            "value_preprocessor": checkpoint.get("value_preprocessor"),
        } for agent_id in possible_agents}
    else:
        # SKRL format - per-agent weights (includes preprocessor states)
        print(f"[INFO] Detected SKRL checkpoint format with agents: {list(checkpoint.keys())}")
        result = {}
        for agent_id in possible_agents:
            if agent_id in checkpoint:
                agent_data = checkpoint[agent_id]
                result[agent_id] = {
                    "policy": agent_data["policy"],
                    "value": agent_data["value"],
                    "state_preprocessor": agent_data.get("state_preprocessor"),
                    "shared_state_preprocessor": agent_data.get("shared_state_preprocessor"),
                    "value_preprocessor": agent_data.get("value_preprocessor"),
                }
        return result


def main():
    """Play with trained MAPPO RNN agents."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create Isaac Lab environment
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    env_cfg.enable_tiled_cameras = True
    env_cfg.use_debug_initial_step = True
    env_cfg.debug_initial_step = int(args_cli.step)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Verify this is a multi-agent environment
    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise ValueError(f"Task {args_cli.task} is not a multi-agent environment")
    
    # Get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    
    # Get run directory and load config from checkpoint path
    run_dir = get_run_dir_from_checkpoint(args_cli.checkpoint)
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Run directory: {run_dir}")

    # Load agent configuration from saved params
    agent_cfg = load_agent_config(run_dir)
    if agent_cfg:
        print(f"[INFO] Loaded agent config from {run_dir}/params/agent.yaml")
    else:
        print(f"[WARN] No agent config found, using defaults")

    # Extract hyperparameters from config
    model_params = get_model_hyperparams(agent_cfg)
    sequence_length = get_sequence_length(agent_cfg)

    print(f"[INFO] Model hyperparameters:")
    print(f"  - hidden_size: {model_params['hidden_size']}")
    print(f"  - gru_num_layers: {model_params['gru_num_layers']}")
    print(f"  - gru_hidden_size: {model_params['gru_hidden_size']}")
    print(f"  - sequence_length: {sequence_length}")

    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(run_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Keep a reference to the unwrapped Isaac Lab env so we can read internal
    # state (cameras, bbox raycaster, step rewards, gimbal joints, ...) for
    # the per-agent visualization plots.
    raw_env = env.unwrapped

    # Wrap environment for SKRL
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    device = env.device

    possible_agents = env.possible_agents
    num_agents = len(possible_agents)

    print(f"[INFO] Playing {num_agents} agents using MAPPO with RNN")
    print(f"[INFO] Agents: {possible_agents}")
    print(f"[INFO] Using device: {device}")
    
    # Create shared observation spaces (same as in training)
    try:
        shared_observation_spaces = env.shared_observation_spaces
    except AttributeError:
        print("[INFO] env.shared_observation_spaces not found. Creating by concatenating all agent observations.")
        obs_shape = sum(space.shape[0] for space in env.observation_spaces.values())
        shared_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        shared_observation_spaces = {agent_id: shared_space for agent_id in possible_agents}

    # Load checkpoint weights
    agent_weights = load_checkpoint_weights(args_cli.checkpoint, possible_agents, device)

    # Check if all agents have the same observation and action spaces (for shared policy)
    obs_spaces_match = all(
        env.observation_spaces[agent_id] == env.observation_spaces[possible_agents[0]]
        for agent_id in possible_agents
    )
    action_spaces_match = all(
        env.action_spaces[agent_id] == env.action_spaces[possible_agents[0]]
        for agent_id in possible_agents
    )

    # Determine if we're using shared networks
    using_shared_policy = obs_spaces_match and action_spaces_match

    # Create models and memories for each agent
    models = {}
    memories = {}

    if using_shared_policy:
        print("[INFO] Creating shared policy and value networks.")

        # Create the shared networks once
        shared_policy = MAPPORNNPolicy(
            observation_space=env.observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=model_params["hidden_size"],
            gru_num_layers=model_params["gru_num_layers"],
            gru_hidden_size=model_params["gru_hidden_size"],
            num_envs=env.num_envs,
            sequence_length=sequence_length,
            initial_log_std=model_params["initial_log_std"],
            min_log_std=model_params["min_log_std"],
            max_log_std=model_params["max_log_std"],
        )

        shared_value = MAPPORNNValue(
            observation_space=shared_observation_spaces[possible_agents[0]],
            action_space=env.action_spaces[possible_agents[0]],
            device=device,
            hidden_size=model_params["hidden_size"],
            gru_num_layers=model_params["gru_num_layers"],
            gru_hidden_size=model_params["gru_hidden_size"],
            num_envs=env.num_envs,
            sequence_length=sequence_length,
        )

        # Load the shared model weights (use the first agent's weights)
        first_agent = possible_agents[0]
        shared_policy.load_state_dict(agent_weights[first_agent]["policy"])
        shared_value.load_state_dict(agent_weights[first_agent]["value"])
        print(f"[INFO] Loaded shared model weights from checkpoint")

        # Assign the same networks to all agents
        for agent_id in possible_agents:
            models[agent_id] = {"policy": shared_policy, "value": shared_value}
    else:
        print("[INFO] Creating separate policy networks for each agent.")
        # Load individual models for each agent
        for agent_id in possible_agents:
            # Create policy network
            policy = MAPPORNNPolicy(
                observation_space=env.observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=model_params["hidden_size"],
                gru_num_layers=model_params["gru_num_layers"],
                gru_hidden_size=model_params["gru_hidden_size"],
                num_envs=env.num_envs,
                sequence_length=sequence_length,
                initial_log_std=model_params["initial_log_std"],
                min_log_std=model_params["min_log_std"],
                max_log_std=model_params["max_log_std"],
            )

            # Create value network
            value = MAPPORNNValue(
                observation_space=shared_observation_spaces[agent_id],
                action_space=env.action_spaces[agent_id],
                device=device,
                hidden_size=model_params["hidden_size"],
                gru_num_layers=model_params["gru_num_layers"],
                gru_hidden_size=model_params["gru_hidden_size"],
                num_envs=env.num_envs,
                sequence_length=sequence_length,
            )

            # Load model weights
            policy.load_state_dict(agent_weights[agent_id]["policy"])
            value.load_state_dict(agent_weights[agent_id]["value"])
            models[agent_id] = {"policy": policy, "value": value}
            print(f"[INFO] Loaded model for agent {agent_id}")
    
    # Create memories for each agent
    memory_cfg = {
        "memory_size": 1,  # Minimal size for evaluation
        "num_envs": env.num_envs,
        "device": device,
    }
    for agent_id in possible_agents:
        memories[agent_id] = RandomMemory(**memory_cfg)
    
    # Configure MAPPO_RNN agent (minimal config for evaluation)
    cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_spaces[possible_agents[0]], "device": device}
    cfg["shared_state_preprocessor"] = RunningStandardScaler
    cfg["shared_state_preprocessor_kwargs"] = {"size": shared_observation_spaces[possible_agents[0]], "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # Disable all training-related features
    cfg["experiment"] = {
        "write_interval": 0,
        "checkpoint_interval": 0,
        "wandb": False
    }
    
    # Create MAPPO_RNN agent
    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=cfg,
        shared_observation_spaces=shared_observation_spaces
    )
    
    # Initialize agent
    agent.init()
    agent.set_mode("eval")

    # Load preprocessor states from checkpoint (CRITICAL for correct evaluation)
    print("[INFO] Loading preprocessor states from checkpoint...")
    preprocessor_loaded = False
    for agent_id in possible_agents:
        if agent_id in agent_weights:
            # Load state preprocessor
            if agent_weights[agent_id].get("state_preprocessor") is not None:
                agent._state_preprocessor[agent_id].load_state_dict(
                    agent_weights[agent_id]["state_preprocessor"]
                )
                preprocessor_loaded = True
            # Load shared state preprocessor
            if agent_weights[agent_id].get("shared_state_preprocessor") is not None:
                agent._shared_state_preprocessor[agent_id].load_state_dict(
                    agent_weights[agent_id]["shared_state_preprocessor"]
                )
            # Load value preprocessor
            if agent_weights[agent_id].get("value_preprocessor") is not None:
                agent._value_preprocessor[agent_id].load_state_dict(
                    agent_weights[agent_id]["value_preprocessor"]
                )

    if preprocessor_loaded:
        # Verify preprocessor loaded correctly
        first_agent = possible_agents[0]
        if hasattr(agent._state_preprocessor[first_agent], 'running_mean'):
            running_mean = agent._state_preprocessor[first_agent].running_mean
            print(f"[INFO] Preprocessor loaded successfully!")
            print(f"[INFO]   running_mean (first 5): {running_mean[:5].tolist()}")
        else:
            print("[INFO] Preprocessor loaded (no running_mean attribute)")
    else:
        print("[WARN] No preprocessor states found in checkpoint - using fresh scalers")

    # Reset environment
    print("Resetting the environment...")
    states, infos = env.reset()

    # ------------------------------------------------------------------
    # Visualization plots for env 0
    # Layout: (num_agents + 1) rows x 4 cols
    #   per-agent rows : camera | rewards | gimbal joints | actions
    #   global row    : distance | delay (AoI) | parallax | tri RMSE
    # X-axis is wall-sim time in seconds (env._sim_time).
    # ------------------------------------------------------------------
    show_plots = args_cli.show_plots and hasattr(raw_env, "_cameras") and len(getattr(raw_env, "_cameras", {})) > 0

    reward_plot_keys = ["action_sum", "action_delta", "bbox_center", "bbox_size",
                        "triangulation", "cbf_penalty"]
    reward_colors = {
        "action_sum": "#e74c3c",
        "action_delta": "#e67e22",
        "bbox_center": "#2ecc71",
        "bbox_size": "#27ae60",
        "triangulation": "#3498db",
        "cbf_penalty": "#9b59b6",
    }
    action_dim_labels = ["vx", "vy", "vz", "yaw_rate", "gim_yaw", "gim_pitch", "zoom"]
    action_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
                     "#3498db", "#9b59b6", "#1abc9c"]
    history_len = 200          # samples
    history_window_s = 8.0     # default visible time window (s)

    # Shared time history (one entry per env step) for all rolling plots.
    time_history: list[float] = []

    # Per-agent plot/line state
    plot_state: dict = {}
    # Global metrics (dict keyed by agent_id where per-agent; scalar list otherwise)
    global_state: dict = {
        "distance": {a: [] for a in possible_agents},
        "aoi": {a: [] for a in possible_agents},
        "parallax_deg": {a: [] for a in possible_agents},
        "tri_err_m": [],
    }
    fig = None
    if show_plots:
        plt.ion()
        fig, axes = plt.subplots(
            num_agents + 1, 4,
            figsize=(22, 4.0 * (num_agents + 1)),
            gridspec_kw={"width_ratios": [1.2, 1, 1, 1]},
            squeeze=False,
        )
        for agent_idx, agent_id in enumerate(possible_agents):
            ax_cam = axes[agent_idx, 0]
            ax_rew = axes[agent_idx, 1]
            ax_gim = axes[agent_idx, 2]
            ax_act = axes[agent_idx, 3]

            ax_cam.set_title(f"Camera - {agent_id} (env 0)")
            ax_cam.axis("off")

            ax_rew.set_title(f"Reward Components - {agent_id}")
            ax_rew.set_xlabel("Time (s)")
            ax_rew.set_ylabel("Reward")
            ax_rew.set_xlim(0, history_window_s)
            ax_rew.grid(True, alpha=0.3)
            rew_lines = {}
            for k in reward_plot_keys:
                line, = ax_rew.plot([], [], label=k, color=reward_colors[k],
                                    linewidth=1.0, alpha=0.8)
                rew_lines[k] = line
            total_line, = ax_rew.plot([], [], label="total", color="black", linewidth=1.5)
            ax_rew.legend(fontsize=6, loc="upper left")

            ax_gim.set_title(f"Gimbal Joints - {agent_id}")
            ax_gim.set_xlabel("Time (s)")
            ax_gim.set_ylabel("Angle (deg)")
            ax_gim.set_xlim(0, history_window_s)
            ax_gim.grid(True, alpha=0.3)
            gim_lines = {
                "yaw_joint_deg": ax_gim.plot([], [], color="#2ecc71", lw=1.2,
                                             alpha=0.9, label="Yaw joint")[0],
                "pitch_joint_deg": ax_gim.plot([], [], color="#e67e22", lw=1.2,
                                               alpha=0.9, label="Pitch joint")[0],
            }
            ax_gim.legend(fontsize=6, loc="upper left")

            ax_act.set_title(f"Actions - {agent_id}")
            ax_act.set_xlabel("Time (s)")
            ax_act.set_ylabel("Action (normalized)")
            ax_act.set_xlim(0, history_window_s)
            ax_act.set_ylim(-1.1, 1.1)
            ax_act.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
            ax_act.grid(True, alpha=0.3)
            act_lines = {}
            for d, label in enumerate(action_dim_labels):
                line, = ax_act.plot([], [], label=label, color=action_colors[d],
                                    linewidth=1.0, alpha=0.85)
                act_lines[label] = line
            ax_act.legend(fontsize=6, loc="upper left", ncol=2)

            plot_state[agent_id] = {
                "ax_cam": ax_cam,
                "ax_rew": ax_rew,
                "ax_gim": ax_gim,
                "ax_act": ax_act,
                "image_display": None,
                "reward_history": {k: [] for k in reward_plot_keys},
                "total_reward_history": [],
                "reward_lines": rew_lines,
                "total_reward_line": total_line,
                "gimbal_history": {"yaw_joint_deg": [], "pitch_joint_deg": []},
                "gimbal_lines": gim_lines,
                "action_history": {label: [] for label in action_dim_labels},
                "action_lines": act_lines,
            }

        # Global bottom row
        ax_dist = axes[num_agents, 0]
        ax_aoi = axes[num_agents, 1]
        ax_par = axes[num_agents, 2]
        ax_tri = axes[num_agents, 3]

        agent_palette = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#16a085"]

        global_axes = {"distance": ax_dist, "aoi": ax_aoi, "parallax_deg": ax_par, "tri_err_m": ax_tri}
        global_lines: dict = {"distance": {}, "aoi": {}, "parallax_deg": {}, "tri_err_m": None}

        ax_dist.set_title("Distance to Target (env 0)")
        ax_dist.set_xlabel("Time (s)")
        ax_dist.set_ylabel("Distance (m)")
        ax_dist.set_xlim(0, history_window_s)
        ax_dist.grid(True, alpha=0.3)
        for i, a in enumerate(possible_agents):
            line, = ax_dist.plot([], [], color=agent_palette[i % len(agent_palette)],
                                  lw=1.2, label=a)
            global_lines["distance"][a] = line
        ax_dist.legend(fontsize=6, loc="upper left")

        ax_aoi.set_title("BBox AoI (delay) — env 0")
        ax_aoi.set_xlabel("Time (s)")
        ax_aoi.set_ylabel("AoI (s)")
        ax_aoi.set_xlim(0, history_window_s)
        ax_aoi.grid(True, alpha=0.3)
        for i, a in enumerate(possible_agents):
            line, = ax_aoi.plot([], [], color=agent_palette[i % len(agent_palette)],
                                 lw=1.2, label=a)
            global_lines["aoi"][a] = line
        ax_aoi.legend(fontsize=6, loc="upper left")

        ax_par.set_title("Parallax (max LOS angle vs other agents)")
        ax_par.set_xlabel("Time (s)")
        ax_par.set_ylabel("Angle (deg)")
        ax_par.set_xlim(0, history_window_s)
        ax_par.grid(True, alpha=0.3)
        for i, a in enumerate(possible_agents):
            line, = ax_par.plot([], [], color=agent_palette[i % len(agent_palette)],
                                 lw=1.2, label=a)
            global_lines["parallax_deg"][a] = line
        ax_par.legend(fontsize=6, loc="upper left")

        ax_tri.set_title("Triangulation Error |X̂ - X*| (env 0)")
        ax_tri.set_xlabel("Time (s)")
        ax_tri.set_ylabel("Error (m)")
        ax_tri.set_xlim(0, history_window_s)
        ax_tri.grid(True, alpha=0.3)
        line_tri, = ax_tri.plot([], [], color="#34495e", lw=1.4, label="|err|")
        global_lines["tri_err_m"] = line_tri
        ax_tri.legend(fontsize=6, loc="upper left")

        fig.tight_layout()
        plt.show(block=False)
        print(f"[INFO] Plots enabled for {num_agents} agents (env 0)")
    else:
        global_axes = {}
        global_lines = {}
        if args_cli.show_plots:
            print("[INFO] Plots disabled — env has no tiled cameras available")

    # Initialize variables for the play loop
    timestep = 0
    episode_rewards = {agent_id: 0.0 for agent_id in possible_agents}
    episode_count = 0
    
    # Resolve deterministic mode (--deterministic)
    use_deterministic = args_cli.deterministic

    print("[INFO] Starting evaluation...")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Real-time mode: {args_cli.real_time}")
    print(f"[INFO] Recording video: {args_cli.video}")
    print(f"[INFO] Shared policy: {using_shared_policy}")
    print(f"[INFO] Deterministic actions: {use_deterministic}")
    print(f"[INFO] RNN sequence length: {sequence_length}")
    
    # Main play loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run everything in inference mode
        with torch.inference_mode():
            # Get actions from agent
            actions, log_probs, outputs = agent.act(states, timestep=timestep, timesteps=10000)

            # Use mean actions for deterministic evaluation
            if use_deterministic:
                for agent_id in possible_agents:
                    actions[agent_id] = outputs[agent_id]["mean_actions"]

            # Step the environment
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # ----------------------------------------------------------
            # Update visualization plots for env 0
            # ----------------------------------------------------------
            if show_plots:
                # Time axis (sim seconds)
                t_now = float(raw_env._sim_time[0].item()) if hasattr(raw_env, "_sim_time") else float(timestep) * dt

                # Detect episode reset on env 0: sim_time goes backwards (or to 0).
                # When that happens, clear every rolling history so the new
                # episode draws from scratch rather than appending non-monotonic
                # samples that would make the line jump backwards in time.
                if time_history and t_now < time_history[-1] - 1e-6:
                    time_history.clear()
                    for s in plot_state.values():
                        for k in s["reward_history"]:
                            s["reward_history"][k].clear()
                        s["total_reward_history"].clear()
                        for k in s["gimbal_history"]:
                            s["gimbal_history"][k].clear()
                        for k in s["action_history"]:
                            s["action_history"][k].clear()
                    for a in possible_agents:
                        global_state["distance"][a].clear()
                        global_state["aoi"][a].clear()
                        global_state["parallax_deg"][a].clear()
                    global_state["tri_err_m"].clear()

                time_history.append(t_now)
                if len(time_history) > history_len:
                    time_history.pop(0)
                t_min, t_max = time_history[0], time_history[-1]
                # Pre-compute LOS vectors (env 0) for parallax
                los_vecs = None
                if hasattr(raw_env, "_robots") and hasattr(raw_env, "target"):
                    target_pos_w = raw_env.target.data.root_pos_w[0]  # (3,)
                    los = torch.stack(
                        [raw_env._robots[a].data.root_pos_w[0] - target_pos_w
                         for a in possible_agents],
                        dim=0,
                    )  # (N, 3)
                    los_norm = los / (los.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                    los_vecs = los_norm

                for agent_id in possible_agents:
                    state = plot_state[agent_id]
                    camera = raw_env._cameras.get(agent_id) if hasattr(raw_env, "_cameras") else None
                    if camera is None:
                        continue

                    # --- Camera image with bbox overlay ---
                    cam_rgb = camera.data.output["rgb"][0].cpu().numpy()
                    if cam_rgb.ndim == 3 and cam_rgb.shape[2] == 4:
                        cam_rgb = cam_rgb[:, :, :3]
                    if cam_rgb.dtype != np.uint8:
                        cam_rgb = (cam_rgb * 255).clip(0, 255).astype(np.uint8)

                    a_idx = possible_agents.index(agent_id)

                    # GT bbox from raycaster
                    bbox_xyxy = (0, 0, 0, 0)
                    bbox_empty_val = True
                    if hasattr(raw_env, "bbox_raycaster_v2"):
                        bxyxy_t = raw_env.bbox_raycaster_v2.data.bboxes_xyxy[0, a_idx, 0, :]
                        bbox_empty_val = bool(raw_env.bbox_raycaster_v2.data.bbox_empty[0, a_idx, 0].item())
                        bbox_xyxy = (
                            int(bxyxy_t[0].item()), int(bxyxy_t[1].item()),
                            int(bxyxy_t[2].item()), int(bxyxy_t[3].item()),
                        )

                    zoom_val = 1.0
                    if hasattr(raw_env, "zoom_level"):
                        zoom_val = raw_env.zoom_level[0, a_idx].item()

                    # Re-project target into image (blue crosshair)
                    target_pixel = None
                    distance_m = None
                    if (
                        hasattr(raw_env, "_robots")
                        and hasattr(raw_env, "_frame_link_ids")
                        and hasattr(raw_env, "target")
                    ):
                        robot = raw_env._robots[agent_id]
                        pitch_link_idx = raw_env._frame_link_ids[agent_id]["pitch_link"]
                        cam_pos = robot.data.body_pos_w[0, pitch_link_idx]
                        pitch_link_quat = robot.data.body_quat_w[0, pitch_link_idx]

                        cam_offset_quat = torch.tensor(
                            [0.5, -0.5, 0.5, -0.5], dtype=torch.float32, device=raw_env.device
                        )
                        cam_quat_world = quat_mul(
                            pitch_link_quat.unsqueeze(0), cam_offset_quat.unsqueeze(0)
                        ).squeeze(0)
                        cam_intrinsic = camera.data.intrinsic_matrices[0]
                        target_pos = raw_env.target.data.root_pos_w[0]

                        tu, tv, tvalid = project_point_to_image(
                            target_pos, cam_pos, cam_quat_world, cam_intrinsic
                        )
                        target_pixel = (tu, tv, tvalid)
                        distance_m = (target_pos - robot.data.root_pos_w[0]).norm().item()

                    # Delayed/observed bbox + AoI
                    obs_bbox_xyxy_val = None
                    obs_bbox_empty_val = True
                    bbox_aoi_val = None
                    if getattr(raw_env, "_delay_system", None) is not None:
                        delayed_states = raw_env._delay_system.get_all_states_for_observations(
                            ego_agent_id=agent_id
                        )
                        obs_data = delayed_states[agent_id].data
                        obs_bbox_xywh = obs_data.bboxes_2d[0, 0, :]
                        obs_bbox_empty_val = obs_bbox_xywh.abs().sum().item() < 1e-6
                        bbox_aoi_val = (
                            raw_env._sim_time[0].item() - obs_data.timestamp_detection[0].item()
                        )
                        if not obs_bbox_empty_val:
                            cx = obs_bbox_xywh[0].item()
                            cy = obs_bbox_xywh[1].item()
                            bw = obs_bbox_xywh[2].item()
                            bh = obs_bbox_xywh[3].item()
                            obs_bbox_xyxy_val = (
                                int(cx - bw / 2), int(cy - bh / 2),
                                int(cx + bw / 2), int(cy + bh / 2),
                            )

                    # Replicated (calibrated noise) bbox + ground/sky flag
                    rep_bbox_xyxy_val = None
                    rep_bbox_empty_val = True
                    bg_is_ground_val = None
                    if (
                        hasattr(raw_env, "cfg")
                        and getattr(raw_env.cfg, "calibrated_bbox_noise", None) is not None
                        and raw_env.cfg.calibrated_bbox_noise.enabled
                        and getattr(raw_env.bbox_raycaster_v2.data, "bboxes_xyxy_replicated", None) is not None
                    ):
                        rep_xyxy_t = raw_env.bbox_raycaster_v2.data.bboxes_xyxy_replicated[0, a_idx, 0, :]
                        rep_empty = raw_env.bbox_raycaster_v2.data.bbox_empty_replicated[0, a_idx, 0].item()
                        rep_bbox_empty_val = bool(rep_empty)
                        if not rep_bbox_empty_val:
                            rep_bbox_xyxy_val = (
                                int(rep_xyxy_t[0].item()), int(rep_xyxy_t[1].item()),
                                int(rep_xyxy_t[2].item()), int(rep_xyxy_t[3].item()),
                            )
                        if raw_env.bbox_raycaster_v2.data.bg_is_ground is not None:
                            bg_is_ground_val = bool(
                                raw_env.bbox_raycaster_v2.data.bg_is_ground[0, a_idx, 0].item()
                            )

                    vis_image = draw_bbox_overlay(
                        cam_rgb, bbox_xyxy, bbox_empty_val, agent_id, zoom_val,
                        target_pixel=target_pixel,
                        distance_m=distance_m,
                        obs_bbox_xyxy=obs_bbox_xyxy_val,
                        obs_bbox_empty=obs_bbox_empty_val,
                        bbox_aoi=bbox_aoi_val,
                        replicated_bbox_xyxy=rep_bbox_xyxy_val,
                        replicated_bbox_empty=rep_bbox_empty_val,
                        bg_is_ground=bg_is_ground_val,
                    )

                    if state["image_display"] is None:
                        state["image_display"] = state["ax_cam"].imshow(vis_image)
                    else:
                        state["image_display"].set_data(vis_image)

                    # --- Reward components ---
                    if hasattr(raw_env, "_step_rewards") and agent_id in raw_env._step_rewards:
                        step_rewards = raw_env._step_rewards[agent_id]
                        step_total = 0.0
                        for k in reward_plot_keys:
                            val = step_rewards[k][0].item() if k in step_rewards else 0.0
                            state["reward_history"][k].append(val)
                            if len(state["reward_history"][k]) > history_len:
                                state["reward_history"][k].pop(0)
                            step_total += val
                        state["total_reward_history"].append(step_total)
                        if len(state["total_reward_history"]) > history_len:
                            state["total_reward_history"].pop(0)

                        n = len(state["total_reward_history"])
                        x_t = time_history[-n:]
                        for k in reward_plot_keys:
                            state["reward_lines"][k].set_data(x_t, state["reward_history"][k])
                        state["total_reward_line"].set_data(x_t, state["total_reward_history"])

                        state["ax_rew"].set_xlim(t_min, max(t_min + history_window_s, t_max))
                        all_vals = state["total_reward_history"][:]
                        for k in reward_plot_keys:
                            all_vals.extend(state["reward_history"][k])
                        if all_vals:
                            ymin, ymax = min(all_vals), max(all_vals)
                            margin = max(0.01, (ymax - ymin) * 0.1)
                            state["ax_rew"].set_ylim(ymin - margin, ymax + margin)

                    # --- Gimbal joint angles ---
                    if hasattr(raw_env, "gimbal_joint_idx") and agent_id in raw_env.gimbal_joint_idx:
                        robot = raw_env._robots[agent_id]
                        to_deg = 180.0 / math.pi
                        yaw_j = robot.data.joint_pos[0, raw_env.gimbal_joint_idx[agent_id]["yaw"]].item()
                        pitch_j = robot.data.joint_pos[0, raw_env.gimbal_joint_idx[agent_id]["pitch"]].item()
                        state["gimbal_history"]["yaw_joint_deg"].append(yaw_j * to_deg)
                        state["gimbal_history"]["pitch_joint_deg"].append(pitch_j * to_deg)
                        for key in state["gimbal_history"]:
                            if len(state["gimbal_history"][key]) > history_len:
                                state["gimbal_history"][key].pop(0)

                        n = len(state["gimbal_history"]["yaw_joint_deg"])
                        x_t = time_history[-n:]
                        for k, line in state["gimbal_lines"].items():
                            line.set_data(x_t, state["gimbal_history"][k])
                        state["ax_gim"].set_xlim(t_min, max(t_min + history_window_s, t_max))
                        all_angles = (state["gimbal_history"]["yaw_joint_deg"]
                                      + state["gimbal_history"]["pitch_joint_deg"])
                        if all_angles:
                            ymin, ymax = min(all_angles), max(all_angles)
                            margin = max(1.0, (ymax - ymin) * 0.1)
                            state["ax_gim"].set_ylim(ymin - margin, ymax + margin)

                    # --- Action (each dim) ---
                    if agent_id in actions:
                        act_vec = actions[agent_id][0].detach().cpu().numpy()  # (act_dim,)
                        for d, label in enumerate(action_dim_labels):
                            v = float(act_vec[d]) if d < len(act_vec) else 0.0
                            state["action_history"][label].append(v)
                            if len(state["action_history"][label]) > history_len:
                                state["action_history"][label].pop(0)
                        n = len(state["action_history"][action_dim_labels[0]])
                        x_t = time_history[-n:]
                        for label, line in state["action_lines"].items():
                            line.set_data(x_t, state["action_history"][label])
                        state["ax_act"].set_xlim(t_min, max(t_min + history_window_s, t_max))

                    # --- Global metrics: distance, AoI, parallax ---
                    a_idx_g = possible_agents.index(agent_id)
                    # distance
                    if distance_m is not None:
                        global_state["distance"][agent_id].append(distance_m)
                    else:
                        global_state["distance"][agent_id].append(float("nan"))
                    if len(global_state["distance"][agent_id]) > history_len:
                        global_state["distance"][agent_id].pop(0)
                    # AoI
                    aoi_v = bbox_aoi_val if bbox_aoi_val is not None else float("nan")
                    global_state["aoi"][agent_id].append(aoi_v)
                    if len(global_state["aoi"][agent_id]) > history_len:
                        global_state["aoi"][agent_id].pop(0)
                    # parallax (max LOS angle vs other agents, deg)
                    if los_vecs is not None and num_agents > 1:
                        cos_ij = (los_vecs[a_idx_g:a_idx_g+1] * los_vecs).sum(dim=-1)
                        cos_ij = cos_ij.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                        ang = torch.arccos(cos_ij)  # (N,)
                        ang[a_idx_g] = 0.0
                        max_par_deg = float(ang.max().item() * 180.0 / math.pi)
                    else:
                        max_par_deg = 0.0
                    global_state["parallax_deg"][agent_id].append(max_par_deg)
                    if len(global_state["parallax_deg"][agent_id]) > history_len:
                        global_state["parallax_deg"][agent_id].pop(0)

                # --- Triangulation error (env 0, scalar) ---
                tri_err = float("nan")
                tri_res = getattr(raw_env, "_triangulation_result_obs", None)
                if tri_res is None:
                    tri_res = getattr(raw_env, "_triangulation_result_gt", None)
                if (
                    tri_res is not None
                    and hasattr(raw_env, "_target_pos_w")
                    and bool(tri_res.is_valid[0, 0].item())
                ):
                    est = tri_res.position[0, 0]  # (3,)
                    truth = raw_env._target_pos_w[0]
                    tri_err = float((est - truth).norm().item())
                global_state["tri_err_m"].append(tri_err)
                if len(global_state["tri_err_m"]) > history_len:
                    global_state["tri_err_m"].pop(0)

                # --- Update global-row line plots ---
                def _autoscale(ax, all_vals, default_pad=0.5):
                    valid = [v for v in all_vals if not (isinstance(v, float) and math.isnan(v))]
                    if valid:
                        ymin, ymax = min(valid), max(valid)
                        margin = max(default_pad, (ymax - ymin) * 0.1)
                        ax.set_ylim(ymin - margin, ymax + margin)

                if global_axes:
                    n = len(time_history)
                    x_t = time_history
                    for metric in ("distance", "aoi", "parallax_deg"):
                        all_vals = []
                        for a in possible_agents:
                            line = global_lines[metric][a]
                            data = global_state[metric][a]
                            line.set_data(x_t[-len(data):], data)
                            all_vals.extend(data)
                        ax = global_axes[metric]
                        ax.set_xlim(t_min, max(t_min + history_window_s, t_max))
                        _autoscale(ax, all_vals,
                                   default_pad=0.05 if metric == "aoi" else 0.5)

                    # tri error
                    data = global_state["tri_err_m"]
                    global_lines["tri_err_m"].set_data(x_t[-len(data):], data)
                    ax_tri_err = global_axes["tri_err_m"]
                    ax_tri_err.set_xlim(t_min, max(t_min + history_window_s, t_max))
                    _autoscale(ax_tri_err, data, default_pad=0.1)

                if fig is not None:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            # Accumulate rewards
            for agent_id in possible_agents:
                episode_rewards[agent_id] += rewards[agent_id].mean().item()
            
            # Check for episode end
            any_done = False
            for agent_id in possible_agents:
                if (terminated[agent_id] | truncated[agent_id]).any():
                    any_done = True
                    break
            
            if any_done:
                episode_count += 1
                avg_rewards = {agent_id: episode_rewards[agent_id] / (timestep + 1) for agent_id in possible_agents}
                print(f"Episode {episode_count} completed at timestep {timestep}")
                print(f"  Average rewards: {avg_rewards}")
                
                # Reset episode rewards
                episode_rewards = {agent_id: 0.0 for agent_id in possible_agents}
            
            # Update states for next iteration
            states = next_states
            
            # Handle RNN state resets for finished environments
            for agent_id in possible_agents:
                finished = (terminated[agent_id] | truncated[agent_id]).nonzero(as_tuple=False)
                if finished.numel():
                    # Reset RNN states for finished environments
                    for rnn_state in agent._rnn_states[agent_id]["policy"]:
                        rnn_state[:, finished[:, 0]] = 0
                    if agent.policies[agent_id] is not agent.values[agent_id]:
                        for rnn_state in agent._rnn_states[agent_id]["value"]:
                            rnn_state[:, finished[:, 0]] = 0
        
        timestep += 1
        
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                print(f"Video recording completed ({timestep} steps)")
                break
        
        # Time delay for real-time evaluation
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # # Optional: Exit after a certain number of episodes
        # if not args_cli.video and episode_count >= 10:
        #     print(f"Completed {episode_count} episodes")
        #     break
    
    # Close the environment
    env.close()
    print("Play session completed!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()