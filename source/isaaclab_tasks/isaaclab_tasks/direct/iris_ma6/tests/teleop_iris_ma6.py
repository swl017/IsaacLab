#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard/Gamepad teleoperation script for iris_ma6 test environment.

This script allows manual control of the iris_ma6 drones to test:
- Linear velocity commands (vx, vy, vz)
- Yaw rate command
- Gimbal yaw and pitch commands
- Zoom control

Keyboard Controls:
    Movement (World Frame):
        W/S: Forward/Backward (X velocity)
        A/D: Left/Right (Y velocity)
        Q/E: Up/Down (Z velocity)
        C/V: Yaw left/right

    Gimbal Control (hold to move):
        Z/X: Gimbal yaw left/right
        T/G: Gimbal pitch up/down

    Zoom Control (toggle mode):
        I: Zoom in
        O: Zoom out
        P: Stop zoom

    Other:
        R: Reset environment
        L: Reset teleop device
        ESC: Quit

Run with:
    ./isaaclab.sh -p scripts/environments/teleoperation/teleop_iris_ma6.py --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for iris_ma6 environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Iris-MA6-Direct-Test-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "gamepad"],
    help="Device for teleoperation.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Control sensitivity factor.")
parser.add_argument("--agent_idx", type=int, default=0, help="Index of agent to control (0-2).")
parser.add_argument("--debug", action="store_true", help="Print debug values for teleop inputs.")
parser.add_argument("--show_camera", action="store_true", default=True, help="Show camera view with bbox overlay.")
parser.add_argument("--no_show_camera", dest="show_camera", action="store_false", help="Disable camera view.")
parser.add_argument("--test_gimbal_lock", action="store_true", default=False,
                    help="Lock gimbal to target (bypass policy gimbal). Adds az/el tracking plot.")
parser.add_argument("--yolo_model", type=str,
                    default="/home/usrg/mas/src/ultralytics_ros/models/yolov11m-drone.pt",
                    help="Path to YOLO model weights for live detection overlay. Set '' to disable.")
parser.add_argument("--yolo_conf", type=float, default=0.25,
                    help="YOLO confidence threshold.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import numpy as np
import time
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

import carb.input
import omni.appwindow

from isaaclab.devices import Se3Keyboard, Se3Gamepad
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

# Import the iris_ma6 environment
import isaaclab_tasks.direct.iris_ma6  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def project_point_to_image(
    point_w: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_ros: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[int, int, bool]:
    """Project a 3D world point to 2D image pixel using pinhole model.

    Args:
        point_w: (3,) world position of the point.
        camera_pos_w: (3,) camera world position.
        camera_quat_w_ros: (4,) camera orientation in ROS convention (wxyz).
            ROS: +Z forward, +X right, +Y down.
        intrinsic: (3, 3) camera intrinsic matrix.

    Returns:
        (u, v, valid): pixel coordinates and validity flag.
    """
    # Transform point to camera frame
    point_rel = point_w - camera_pos_w
    point_cam = quat_apply(quat_inv(camera_quat_w_ros.unsqueeze(0)), point_rel.unsqueeze(0)).squeeze(0)

    x, y, z = point_cam[0].item(), point_cam[1].item(), point_cam[2].item()

    if z <= 0.01:
        return 0, 0, False

    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item()
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()

    u = int(fx * x / z + cx)
    v = int(fy * y / z + cy)

    return u, v, True


def draw_bbox_overlay(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    bbox_empty: bool,
    agent_idx: int,
    zoom_level: float = 1.0,
    target_pixel: tuple[int, int, bool] | None = None,
    distance_m: float | None = None,
    sim_time: float | None = None,
    realtime_factor: float | None = None,
    obs_bbox_xyxy: tuple[int, int, int, int] | None = None,
    obs_bbox_empty: bool = True,
    bbox_aoi: float | None = None,
    dr_info: dict | None = None,
    yolo_bboxes: list | None = None,
    replicated_bbox_xyxy: tuple[int, int, int, int] | None = None,
    replicated_bbox_empty: bool = True,
    bg_is_ground: bool | None = None,
) -> np.ndarray:
    """Draw bounding box overlay on camera image.

    Args:
        image: RGB image (H, W, 3), uint8.
        bbox_xyxy: (x_min, y_min, x_max, y_max) in pixel coords from raycaster (GT).
        bbox_empty: True if no target detected by raycaster (GT).
        agent_idx: Index of the controlled agent.
        zoom_level: Current zoom level for display.
        target_pixel: (u, v, valid) from TiledCamera re-projection. Blue crosshair.
        distance_m: Distance to target in meters.
        sim_time: Simulation time in seconds.
        realtime_factor: Ratio of sim time to wall clock time.
        obs_bbox_xyxy: (x_min, y_min, x_max, y_max) delayed+noisy observation bbox.
        obs_bbox_empty: True if observed bbox is empty.
        bbox_aoi: Bbox Age-of-Information in seconds (t_current - t_capture).
        dr_info: Domain randomization state dict (target_z_scale, intrinsic_scale, progress).
        yolo_bboxes: List of (x1, y1, x2, y2, conf, cls) from YOLO detector.
        replicated_bbox_xyxy: (x_min, y_min, x_max, y_max) from detector replicator (calibrated noise).
        replicated_bbox_empty: True if replicated bbox is empty.

    Returns:
        Annotated image (H, W, 3), uint8.
    """
    vis = image.copy()
    h_img, w_img = vis.shape[:2]
    x_min, y_min, x_max, y_max = bbox_xyxy

    # GT raycaster bbox (green, solid)
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

    # Observed/delayed bbox (yellow, dashed via dotted rectangle)
    if obs_bbox_xyxy is not None and not obs_bbox_empty:
        ox_min, oy_min, ox_max, oy_max = obs_bbox_xyxy
        obs_color = (255, 255, 0)  # yellow in RGB (matplotlib renders RGB)
        # Draw dashed rectangle using short line segments
        for start, end in [
            ((ox_min, oy_min), (ox_max, oy_min)),  # top
            ((ox_max, oy_min), (ox_max, oy_max)),  # right
            ((ox_max, oy_max), (ox_min, oy_max)),  # bottom
            ((ox_min, oy_max), (ox_min, oy_min)),  # left
        ]:
            _draw_dashed_line(vis, start, end, obs_color, thickness=2, dash_len=8)

    # TiledCamera re-projection crosshair (blue)
    if target_pixel is not None:
        tu, tv, tvalid = target_pixel
        if tvalid and 0 <= tu < w_img and 0 <= tv < h_img:
            cross_color = (0, 120, 255)
            size = 12
            cv2.line(vis, (tu - size, tv), (tu + size, tv), cross_color, 2)
            cv2.line(vis, (tu, tv - size), (tu, tv + size), cross_color, 2)

    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"Agent {agent_idx} | Zoom: {zoom_level:.2f}x", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Distance to target + background type
    if distance_m is not None:
        dist_str = f"{distance_m:.1f} m"
        if bg_is_ground is not None:
            bg_label = "GND" if bg_is_ground else "SKY"
            bg_color = (139, 90, 43) if bg_is_ground else (135, 206, 250)  # brown / light blue
            dist_str += f"  [{bg_label}]"
            cv2.putText(vis, dist_str, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_color, 1)
        else:
            cv2.putText(vis, dist_str, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Simulation time and realtime factor
    if sim_time is not None and realtime_factor is not None:
        time_str = f"{sim_time:.2f} sec (x {realtime_factor:.2f})"
        cv2.putText(vis, time_str, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bbox Age-of-Information
    if bbox_aoi is not None:
        aoi_ms = bbox_aoi * 1000.0
        aoi_steps = bbox_aoi / 0.04 if bbox_aoi > 0 else 0.0
        aoi_str = f"BBox AoI: {aoi_ms:.0f}ms ({aoi_steps:.1f} steps)"
        cv2.putText(vis, aoi_str, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Domain randomization state (top-right corner)
    if dr_info is not None:
        dr_x = w_img - 250
        dr_y = 25
        dr_color = (255, 180, 0)  # orange
        cv2.putText(vis, f"DR xy-scale: {dr_info.get('target_xy_scale', 1.0):.2f}x",
                    (dr_x, dr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dr_color, 1)
        cv2.putText(vis, f"DR z-scale: {dr_info.get('target_z_scale', 1.0):.2f}x",
                    (dr_x, dr_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dr_color, 1)
        cv2.putText(vis, f"DR fov-scale: {dr_info.get('intrinsic_scale', 1.0):.2f}x",
                    (dr_x, dr_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dr_color, 1)
        cv2.putText(vis, f"DR progress: {dr_info.get('progress', 0.0):.2f}",
                    (dr_x, dr_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dr_color, 1)

    # YOLO detections (cyan, solid)
    if yolo_bboxes is not None:
        for det in yolo_bboxes:
            yx1, yy1, yx2, yy2, conf = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4]
            yolo_color = (0, 255, 255)  # cyan
            cv2.rectangle(vis, (yx1, yy1), (yx2, yy2), yolo_color, 2)
            cv2.putText(vis, f"YOLO {conf:.2f}", (yx1, max(yy1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, yolo_color, 1)

    # Replicated bbox from detector replicator (magenta, dotted)
    if replicated_bbox_xyxy is not None and not replicated_bbox_empty:
        rx1, ry1, rx2, ry2 = replicated_bbox_xyxy
        rep_color = (255, 0, 255)  # magenta
        for start, end in [
            ((rx1, ry1), (rx2, ry1)),
            ((rx2, ry1), (rx2, ry2)),
            ((rx2, ry2), (rx1, ry2)),
            ((rx1, ry2), (rx1, ry1)),
        ]:
            _draw_dashed_line(vis, start, end, rep_color, thickness=2, dash_len=5)

    # Legend
    has_obs = obs_bbox_xyxy is not None
    has_yolo = yolo_bboxes is not None and len(yolo_bboxes) > 0
    has_rep = replicated_bbox_xyxy is not None and not replicated_bbox_empty
    parts = ["green=GT"]
    if has_obs:
        parts.append("yellow=observed")
    if has_rep:
        parts.append("magenta=replicated")
    if has_yolo:
        parts.append("cyan=YOLO")
    parts.append("blue=camera")
    legend = "  ".join(parts)
    cv2.putText(vis, legend, (10, h_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return vis


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
):
    """Draw a dashed line on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = max(1, int((dx * dx + dy * dy) ** 0.5))
    num_dashes = max(1, length // dash_len)
    for i in range(0, num_dashes, 2):
        t0 = i / num_dashes
        t1 = min((i + 1) / num_dashes, 1.0)
        sx = int(x1 + dx * t0)
        sy = int(y1 + dy * t0)
        ex = int(x1 + dx * t1)
        ey = int(y1 + dy * t1)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


class IrisMA6TeleopController:
    """Teleoperation controller for iris_ma6 environment.

    Maps Se3 device inputs to iris_ma6 action space:
    - [0:3] vx, vy, vz: World frame velocity (normalized to [-1, 1])
    - [3] yaw_rate: Yaw rate (normalized to [-1, 1])
    - [4] gimbal_yaw_rate: Gimbal yaw rate (normalized to [-1, 1])
    - [5] gimbal_pitch_rate: Gimbal pitch rate (normalized to [-1, 1])
    - [6] zoom_rate: Zoom rate (normalized to [-1, 1])

    The environment scales these normalized actions to physical units.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        controlled_agent_idx: int,
        device: torch.device,
        sensitivity: float = 1.0,
    ):
        """Initialize the controller.

        Args:
            num_envs: Number of environments.
            num_agents: Number of agents per environment.
            controlled_agent_idx: Index of the agent to control.
            device: Torch device.
            sensitivity: Control sensitivity multiplier.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.controlled_agent_idx = controlled_agent_idx
        self.device = device
        self.sensitivity = sensitivity

        # Action scaling factors to normalize to [-1, 1]
        # Se3Keyboard with pos_sensitivity=0.5 gives [-0.5, 0.5] per key
        # We scale to [-1, 1] for the environment
        self.vel_scale = 2.0 * sensitivity  # 0.5 * 2.0 = 1.0
        self.yaw_rate_scale = 2.0 * sensitivity

        # Zoom state: controlled by separate keys via callbacks
        self._zoom_rate = 0.0

        # Direct keyboard polling for gimbal (bypasses Se3Keyboard's broken KEY_RELEASE)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Key codes for gimbal control
        self._key_z = carb.input.KeyboardInput.Z
        self._key_x = carb.input.KeyboardInput.X
        self._key_t = carb.input.KeyboardInput.T
        self._key_g = carb.input.KeyboardInput.G

    def set_zoom_in(self, pressed: bool):
        """Set zoom in state (called by key callback)."""
        if pressed:
            self._zoom_rate = 1.0
        elif self._zoom_rate > 0:
            self._zoom_rate = 0.0

    def set_zoom_out(self, pressed: bool):
        """Set zoom out state (called by key callback)."""
        if pressed:
            self._zoom_rate = -1.0
        elif self._zoom_rate < 0:
            self._zoom_rate = 0.0

    def reset_zoom(self):
        """Reset zoom rate to zero."""
        self._zoom_rate = 0.0

    def _is_key_pressed(self, key: carb.input.KeyboardInput) -> bool:
        """Check if a key is currently pressed using direct polling."""
        return self._input.get_keyboard_value(self._keyboard, key) > 0.5

    def process_teleop_data(
        self,
        teleop_data: tuple,
    ) -> dict[str, torch.Tensor]:
        """Process teleoperation data into environment actions.

        Args:
            teleop_data: Tuple of (pose_delta, gripper_command) from Se3 device.
                pose_delta: [x, y, z, roll, pitch, yaw] deltas
                gripper_command: bool (not used for zoom - using key callbacks instead)

        Returns:
            Dictionary mapping agent_id to action tensor (num_envs, 7).
        """
        pose_delta, _ = teleop_data  # gripper_command not used

        # Extract components from pose delta
        # Se3Keyboard: [x, y, z, roll, pitch, yaw]
        dx = pose_delta[0]  # Forward/backward -> vx
        dy = pose_delta[1]  # Left/right -> vy
        dz = pose_delta[2]  # Up/down -> vz
        # Skip rotation components - use direct polling for gimbal
        dyaw = pose_delta[5]  # Yaw (C/V keys) -> drone yaw rate

        # Map to action space (normalized to [-1, 1])
        vx = float(dx * self.vel_scale)
        vy = float(dy * self.vel_scale)
        vz = float(dz * self.vel_scale)
        yaw_rate = float(dyaw * self.yaw_rate_scale)

        # Gimbal uses direct keyboard polling (bypasses Se3Keyboard's broken KEY_RELEASE)
        # Z: gimbal yaw left (+), X: gimbal yaw right (-)
        # T: gimbal pitch up (+), G: gimbal pitch down (-)
        key_z = self._is_key_pressed(self._key_z)
        key_x = self._is_key_pressed(self._key_x)
        key_t = self._is_key_pressed(self._key_t)
        key_g = self._is_key_pressed(self._key_g)

        gimbal_yaw_rate = 0.0
        gimbal_pitch_rate = 0.0

        if key_z:
            gimbal_yaw_rate += 0.1 * self.sensitivity
        if key_x:
            gimbal_yaw_rate -= 0.1 * self.sensitivity
        if key_t:
            gimbal_pitch_rate += 0.1 * self.sensitivity
        if key_g:
            gimbal_pitch_rate -= 0.1 * self.sensitivity

        # Debug: print gimbal state if enabled (show every frame to catch issues)
        if hasattr(self, 'debug') and self.debug:
            # Increment frame counter
            if not hasattr(self, '_debug_frame'):
                self._debug_frame = 0
            self._debug_frame += 1

            # Print every 10 frames if any gimbal key was ever pressed
            if not hasattr(self, '_gimbal_was_active'):
                self._gimbal_was_active = False
            if key_z or key_x or key_t or key_g:
                self._gimbal_was_active = True

            if self._gimbal_was_active and self._debug_frame % 10 == 0:
                print(
                    f"[DEBUG] frame={self._debug_frame} "
                    f"keys(Z={key_z},X={key_x},T={key_t},G={key_g}) "
                    f"rate(yaw={gimbal_yaw_rate:.2f},pitch={gimbal_pitch_rate:.2f})"
                )

        # Zoom rate from keyboard state (set by callbacks)
        zoom_rate = self._zoom_rate * self.sensitivity

        # Clamp all actions to [-1, 1]
        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        vz = max(-1.0, min(1.0, vz))
        yaw_rate = max(-1.0, min(1.0, yaw_rate))
        gimbal_yaw_rate = max(-1.0, min(1.0, gimbal_yaw_rate))
        gimbal_pitch_rate = max(-1.0, min(1.0, gimbal_pitch_rate))
        zoom_rate = max(-1.0, min(1.0, zoom_rate))

        # Create action tensor for controlled agent
        action = torch.tensor(
            [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate],
            dtype=torch.float32,
            device=self.device,
        )

        # Expand to all environments
        action = action.unsqueeze(0).expand(self.num_envs, -1)

        # Create action dict for all agents (zero for uncontrolled agents)
        actions = {}
        for agent_idx in range(self.num_agents):
            agent_id = f"drone_{agent_idx}"
            if agent_idx == self.controlled_agent_idx:
                actions[agent_id] = action.clone()
            else:
                # Zero actions for other agents (they will hover)
                actions[agent_id] = torch.zeros(self.num_envs, 7, device=self.device)

        return actions


def main():
    """Run keyboard teleoperation with iris_ma6 environment."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_debug_initial_step = True
    env_cfg.enable_tiled_cameras = True
    if args_cli.test_gimbal_lock:
        env_cfg.debug_lock_gimbal_to_target = True
        print("[INFO] Gimbal locked to target (debug_lock_gimbal_to_target=True)")
    env_cfg.use_debug_initial_step = True
    env_cfg.debug_initial_step = 400000

    # Detector replicator uses hardcoded defaults from NoiseModelParams dataclass
    # (calibrated from experiments/calibrate_detector.py, no JSON loading needed)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Get environment info
    num_agents = len(env.cfg.possible_agents)
    controlled_agent_idx = min(args_cli.agent_idx, num_agents - 1)

    print("\n" + "=" * 60)
    print("iris_ma6 Teleoperation Controller")
    print("=" * 60)
    print(f"Number of environments: {env.num_envs}")
    print(f"Number of agents: {num_agents}")
    print(f"Controlling agent: drone_{controlled_agent_idx}")
    print(f"Device: {args_cli.teleop_device}")
    print(f"Sensitivity: {args_cli.sensitivity}")
    print("=" * 60)

    # Print controls
    print("\nControls:")
    print("-" * 40)
    print("Movement (World Frame):")
    print("  W/S: Forward/Backward (X velocity)")
    print("  A/D: Left/Right (Y velocity)")
    print("  Q/E: Up/Down (Z velocity)")
    print("  C/V: Yaw left/right")
    print("\nGimbal Control (hold to move):")
    print("  Z/X: Gimbal yaw left/right")
    print("  T/G: Gimbal pitch up/down")
    print("\nZoom Control (toggle mode):")
    print("  I: Zoom in")
    print("  O: Zoom out")
    print("  P: Stop zoom")
    print("\nOther:")
    print("  R: Reset environment")
    print("  L: Reset teleop device")
    print("  ESC: Quit")
    print("-" * 40 + "\n")

    # Reset flag
    should_reset = False

    def reset_callback():
        nonlocal should_reset
        should_reset = True
        print("[INFO] Reset requested")

    # Create teleoperation device
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.5 * args_cli.sensitivity,
            rot_sensitivity=0.5 * args_cli.sensitivity,
        )
    else:
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.5 * args_cli.sensitivity,
            rot_sensitivity=0.5 * args_cli.sensitivity,
        )

    # Add reset callback
    teleop_interface.add_callback("R", reset_callback)

    # Create controller
    controller = IrisMA6TeleopController(
        num_envs=env.num_envs,
        num_agents=num_agents,
        controlled_agent_idx=controlled_agent_idx,
        device=env.device,
        sensitivity=args_cli.sensitivity,
    )
    controller.debug = args_cli.debug

    # Zoom callbacks - toggle zoom state
    def zoom_in_callback():
        controller.set_zoom_in(True)
        print("[INFO] Zoom in")

    def zoom_out_callback():
        controller.set_zoom_out(True)
        print("[INFO] Zoom out")

    def zoom_stop_callback():
        controller.reset_zoom()
        print("[INFO] Zoom stopped")

    teleop_interface.add_callback("I", zoom_in_callback)
    teleop_interface.add_callback("O", zoom_out_callback)
    teleop_interface.add_callback("P", zoom_stop_callback)  # Stop zoom

    # Reset environment
    env.reset()
    teleop_interface.reset()
    print(f"[INFO] Initial DR state: xy={env._dr_target_scale[0, 0, 0].item():.2f}, "
          f"z={env._dr_target_scale[0, 0, 2].item():.2f}, "
          f"fov={env._dr_intrinsic_scale[0, 0].item():.2f}, "
          f"progress={env.progress_dynamics:.2f}")

    # Load YOLO model for live detection overlay (optional)
    yolo_model = None
    if args_cli.yolo_model and args_cli.yolo_model.strip():
        import os
        if os.path.isfile(args_cli.yolo_model):
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(args_cli.yolo_model)
                print(f"[INFO] YOLO model loaded: {args_cli.yolo_model}")
                print(f"[INFO] YOLO confidence threshold: {args_cli.yolo_conf}")
            except ImportError:
                print("[WARN] ultralytics not installed — YOLO overlay disabled")
                print("[WARN] Install with: pip install ultralytics")
            except Exception as e:
                print(f"[WARN] Failed to load YOLO model: {e}")
        else:
            print(f"[INFO] YOLO model not found at {args_cli.yolo_model} — YOLO overlay disabled")

    # Enable delay system at full strength for realistic observation delays
    # (env starts with mode="none" due to curriculum — override for teleop)
    if env._delay_system is not None:
        env._delay_system.set_delay_mode("fixed", progress=1.0)
        env._delay_system.set_noise_scale(1.0)
        print("[INFO] Delay system enabled (fixed mode, progress=1.0, noise_scale=1.0)")

    # Override curriculum noise_scale for teleop (curriculum starts at 0)
    if env.cfg.calibrated_bbox_noise.enabled:
        env._curriculum_noise_scale = 1.0
        env._curriculum_fp_fn_scale = 1.0

    # Camera visualization setup
    show_camera = args_cli.show_camera
    image_display = None
    fig = None
    ax_camera = None

    # Reward plot components to display (exclude diagnostic keys)
    reward_plot_keys = ["action_sum", "action_delta", "bbox_center", "bbox_size",
                        "triangulation", "cbf_penalty"]
    reward_colors = {
        "action_sum": "#e74c3c",      # red
        "action_delta": "#e67e22",     # orange
        "bbox_center": "#2ecc71",     # green
        "bbox_size": "#27ae60",       # dark green
        "triangulation": "#3498db",   # blue
        "cbf_penalty": "#9b59b6",     # purple
    }
    reward_history_len = 200  # number of steps to show
    reward_history = {k: [] for k in reward_plot_keys}
    total_reward_history = []
    reward_lines = {}
    total_reward_line = None
    ax_rewards = None

    # Gimbal tracking plot state (only when test_gimbal_lock)
    gimbal_history_len = 200
    gimbal_history = {
        "desired_az": [], "actual_az": [],
        "desired_el": [], "actual_el": [],
        "az_error_deg": [], "el_error_deg": [],
    }
    ax_gimbal = None
    gimbal_lines = {}

    if show_camera:
        plt.ion()
        if args_cli.test_gimbal_lock:
            fig, (ax_camera, ax_rewards, ax_gimbal) = plt.subplots(
                1, 3, figsize=(20, 5),
                gridspec_kw={"width_ratios": [1.2, 1, 1]},
            )
        else:
            fig, (ax_camera, ax_rewards) = plt.subplots(
                1, 2, figsize=(14, 5),
                gridspec_kw={"width_ratios": [1.2, 1]},
            )
        ax_camera.set_title(f"Camera - drone_{controlled_agent_idx}")
        ax_camera.axis("off")

        # Initialize reward plot
        ax_rewards.set_title("Reward Components (per step)")
        ax_rewards.set_xlabel("Step")
        ax_rewards.set_ylabel("Reward")
        ax_rewards.set_xlim(0, reward_history_len)
        ax_rewards.grid(True, alpha=0.3)
        for key in reward_plot_keys:
            line, = ax_rewards.plot([], [], label=key, color=reward_colors[key],
                                    linewidth=1.0, alpha=0.8)
            reward_lines[key] = line
        total_reward_line, = ax_rewards.plot([], [], label="total", color="black",
                                             linewidth=1.5)
        ax_rewards.legend(fontsize=7, loc="upper left")

        # Initialize gimbal tracking plot
        if ax_gimbal is not None:
            ax_gimbal.set_title("Gimbal Tracking (Lock-to-Target)")
            ax_gimbal.set_xlabel("Step")
            ax_gimbal.set_ylabel("Angle (deg)")
            ax_gimbal.set_xlim(0, gimbal_history_len)
            ax_gimbal.grid(True, alpha=0.3)
            gimbal_lines["desired_az"], = ax_gimbal.plot([], [], "b-", lw=1.5, label="Desired Az")
            gimbal_lines["actual_az"], = ax_gimbal.plot([], [], "b--", lw=1.0, alpha=0.7, label="Actual Az")
            gimbal_lines["desired_el"], = ax_gimbal.plot([], [], "r-", lw=1.5, label="Desired El")
            gimbal_lines["actual_el"], = ax_gimbal.plot([], [], "r--", lw=1.0, alpha=0.7, label="Actual El")
            gimbal_lines["az_error_deg"], = ax_gimbal.plot([], [], "b:", lw=1.0, alpha=0.5, label="Az err")
            gimbal_lines["el_error_deg"], = ax_gimbal.plot([], [], "r:", lw=1.0, alpha=0.5, label="El err")
            ax_gimbal.legend(fontsize=6, loc="upper left")

        fig.tight_layout()

    print("[INFO] Starting teleoperation loop...")

    # Time tracking for realtime factor
    wall_clock_start = time.time()
    sim_time_start = env.sim.current_time

    # Main loop
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get teleop data
            teleop_data = teleop_interface.advance()

            # Process into actions
            actions = controller.process_teleop_data(teleop_data)

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Update camera visualization
            if show_camera:
                agent_id = f"drone_{controlled_agent_idx}"
                camera = env._cameras[agent_id]
                camera_rgb = camera.data.output["rgb"][0].cpu().numpy()

                # Handle RGBA -> RGB
                if camera_rgb.ndim == 3 and camera_rgb.shape[2] == 4:
                    camera_rgb = camera_rgb[:, :, :3]

                # Handle float [0,1] -> uint8 [0,255]
                if camera_rgb.dtype != np.uint8:
                    camera_rgb = (camera_rgb * 255).clip(0, 255).astype(np.uint8)

                # Get bbox data from raycaster (projected from body center)
                bbox_xyxy_t = env.bbox_raycaster_v2.data.bboxes_xyxy[0, controlled_agent_idx, 0, :]
                bbox_empty_val = env.bbox_raycaster_v2.data.bbox_empty[0, controlled_agent_idx, 0].item()
                bbox_xyxy = (
                    int(bbox_xyxy_t[0].item()),
                    int(bbox_xyxy_t[1].item()),
                    int(bbox_xyxy_t[2].item()),
                    int(bbox_xyxy_t[3].item()),
                )
                zoom = env.zoom_level[0, controlled_agent_idx].item()

                # Re-project target using pitch_link body pose (fresh each frame)
                # TiledCamera.data.pos_w / quat_w_world are stale (only set at init),
                # so we read the pitch_link pose directly from the robot articulation.
                target_pixel = None
                robot = env._robots[agent_id]
                pitch_link_idx = env._frame_link_ids[agent_id]["pitch_link"]
                cam_pos = robot.data.body_pos_w[0, pitch_link_idx]
                pitch_link_quat = robot.data.body_quat_w[0, pitch_link_idx]

                # Apply camera offset rotation (same as TiledCamera config: ROS convention)
                # This maps from pitch_link frame to camera optical frame
                cam_offset_quat = torch.tensor(
                    [0.5, -0.5, 0.5, -0.5], dtype=torch.float32, device=env.device
                )
                cam_quat_world = quat_mul(
                    pitch_link_quat.unsqueeze(0), cam_offset_quat.unsqueeze(0)
                ).squeeze(0)

                # Use TiledCamera's actual intrinsic matrix (already zoom-adjusted
                # by set_intrinsic_matrices_batched each frame — no extra scaling needed)
                cam_intrinsic = camera.data.intrinsic_matrices[0]

                target_pos = env.target.data.root_pos_w[0]

                # cam_quat_world already describes the camera in ROS convention
                # (Z-forward, X-right, Y-down) because the offset [0.5,-0.5,0.5,-0.5]
                # maps body frame directly to ROS camera frame. No further conversion needed.
                tu, tv, tvalid = project_point_to_image(
                    target_pos, cam_pos, cam_quat_world, cam_intrinsic
                )
                target_pixel = (tu, tv, tvalid)

                # Compute distance from drone to target
                drone_pos = robot.data.root_pos_w[0]
                distance_m = (target_pos - drone_pos).norm().item()

                # Compute simulation time and realtime factor
                sim_time_elapsed = env.sim.current_time - sim_time_start
                wall_clock_elapsed = time.time() - wall_clock_start
                if wall_clock_elapsed > 0.01:
                    realtime_factor = sim_time_elapsed / wall_clock_elapsed
                else:
                    realtime_factor = 0.0

                # Get delayed+noisy observed bbox if delay system is active
                obs_bbox_xyxy_val = None
                obs_bbox_empty_val = True
                bbox_aoi_val = None
                if env._delay_system is not None:
                    delayed_states = env._delay_system.get_all_states_for_observations(
                        ego_agent_id=agent_id
                    )
                    obs_data = delayed_states[agent_id].data
                    # bboxes_2d is (N, T, 4) in xywh pixel format
                    obs_bbox_xywh = obs_data.bboxes_2d[0, 0, :]  # first env, first target
                    obs_bbox_empty_val = obs_bbox_xywh.abs().sum().item() < 1e-6
                    # Compute bbox Age-of-Information
                    bbox_aoi_val = (
                        env._sim_time[0].item() - obs_data.timestamp_detection[0].item()
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

                # Gather DR state for overlay
                dr_info = {
                    "target_xy_scale": env._dr_target_scale[0, 0, 0].item(),
                    "target_z_scale": env._dr_target_scale[0, 0, 2].item(),
                    "intrinsic_scale": env._dr_intrinsic_scale[0, controlled_agent_idx].item(),
                    "progress": env.progress_dynamics,
                }

                # Run YOLO inference on camera image
                yolo_bboxes = None
                if yolo_model is not None:
                    try:
                        # YOLO expects BGR; camera_rgb is RGB
                        yolo_input = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
                        results = yolo_model.predict(
                            source=yolo_input,
                            conf=args_cli.yolo_conf,
                            verbose=False,
                            device="cuda:0",
                        )
                        if results and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            yolo_bboxes = []
                            for i in range(len(boxes)):
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                conf = boxes.conf[i].item()
                                cls = int(boxes.cls[i].item())
                                yolo_bboxes.append((x1, y1, x2, y2, conf, cls))
                    except Exception as e:
                        if step_count % 100 == 0:
                            print(f"[WARN] YOLO inference error: {e}")

                # Get replicated (calibrated noise) bbox and background classification
                replicated_bbox_xyxy_val = None
                replicated_bbox_empty_val = True
                bg_is_ground_val = None
                if (
                    env.cfg.calibrated_bbox_noise.enabled
                    and env.bbox_raycaster_v2.data.bboxes_xyxy_replicated is not None
                ):
                    rep_xyxy_t = env.bbox_raycaster_v2.data.bboxes_xyxy_replicated[
                        0, controlled_agent_idx, 0, :
                    ]
                    rep_empty = env.bbox_raycaster_v2.data.bbox_empty_replicated[
                        0, controlled_agent_idx, 0
                    ].item()
                    replicated_bbox_empty_val = bool(rep_empty)
                    if not replicated_bbox_empty_val:
                        replicated_bbox_xyxy_val = (
                            int(rep_xyxy_t[0].item()),
                            int(rep_xyxy_t[1].item()),
                            int(rep_xyxy_t[2].item()),
                            int(rep_xyxy_t[3].item()),
                        )
                    if env.bbox_raycaster_v2.data.bg_is_ground is not None:
                        bg_is_ground_val = bool(
                            env.bbox_raycaster_v2.data.bg_is_ground[
                                0, controlled_agent_idx, 0
                            ].item()
                        )

                vis_image = draw_bbox_overlay(
                    camera_rgb, bbox_xyxy, bbox_empty_val, controlled_agent_idx, zoom,
                    target_pixel=target_pixel,
                    distance_m=distance_m,
                    sim_time=sim_time_elapsed,
                    realtime_factor=realtime_factor,
                    obs_bbox_xyxy=obs_bbox_xyxy_val,
                    obs_bbox_empty=obs_bbox_empty_val,
                    bbox_aoi=bbox_aoi_val,
                    dr_info=dr_info,
                    yolo_bboxes=yolo_bboxes,
                    replicated_bbox_xyxy=replicated_bbox_xyxy_val,
                    replicated_bbox_empty=replicated_bbox_empty_val,
                    bg_is_ground=bg_is_ground_val,
                )

                if image_display is None:
                    image_display = ax_camera.imshow(vis_image)
                    plt.show(block=False)
                else:
                    image_display.set_data(vis_image)

                # Update reward plots
                if hasattr(env, '_step_rewards') and agent_id in env._step_rewards:
                    step_rewards = env._step_rewards[agent_id]
                    step_total = 0.0
                    for key in reward_plot_keys:
                        val = step_rewards[key][0].item() if key in step_rewards else 0.0
                        reward_history[key].append(val)
                        if len(reward_history[key]) > reward_history_len:
                            reward_history[key].pop(0)
                        step_total += val
                    total_reward_history.append(step_total)
                    if len(total_reward_history) > reward_history_len:
                        total_reward_history.pop(0)

                    # Update line data
                    n = len(total_reward_history)
                    x_data = list(range(n))
                    for key in reward_plot_keys:
                        reward_lines[key].set_data(x_data, reward_history[key])
                    total_reward_line.set_data(x_data, total_reward_history)

                    # Rescale axes
                    ax_rewards.set_xlim(0, max(reward_history_len, n))
                    all_vals = total_reward_history[:]
                    for key in reward_plot_keys:
                        all_vals.extend(reward_history[key])
                    if all_vals:
                        ymin, ymax = min(all_vals), max(all_vals)
                        margin = max(0.01, (ymax - ymin) * 0.1)
                        ax_rewards.set_ylim(ymin - margin, ymax + margin)

                # Update gimbal tracking plot
                if ax_gimbal is not None and args_cli.test_gimbal_lock:
                    agent_id = f"drone_{controlled_agent_idx}"
                    robot = env._robots[agent_id]
                    idx = controlled_agent_idx

                    # Desired az/el: from drone to target
                    los = env._target_pos_w[0] - robot.data.root_pos_w[0]
                    desired_az = torch.atan2(los[1], los[0]).item()
                    desired_el = torch.atan2(los[2], los[:2].norm()).item()

                    # Actual az/el: from gimbal ray direction
                    # Read current gimbal joint positions
                    from isaaclab_tasks.direct.iris_ma6.iris_ma_env6_test import (
                        body_to_world_gimbal_angles, YAW_JOINT_OFFSET,
                    )
                    yaw_joint = robot.data.joint_pos[0, env.gimbal_joint_idx[agent_id]["yaw"]].item()
                    pitch_joint = robot.data.joint_pos[0, env.gimbal_joint_idx[agent_id]["pitch"]].item()
                    yaw_body = yaw_joint - YAW_JOINT_OFFSET
                    actual_az_t, actual_el_t = body_to_world_gimbal_angles(
                        torch.tensor([yaw_body], device=env.device),
                        torch.tensor([pitch_joint], device=env.device),
                        robot.data.root_quat_w[0:1],
                    )
                    actual_az = actual_az_t.item()
                    actual_el = actual_el_t.item()

                    import math
                    to_deg = 180.0 / math.pi
                    az_err = (desired_az - actual_az)
                    # Wrap to [-pi, pi]
                    az_err = (az_err + math.pi) % (2 * math.pi) - math.pi
                    el_err = desired_el - actual_el

                    gimbal_history["desired_az"].append(desired_az * to_deg)
                    gimbal_history["actual_az"].append(actual_az * to_deg)
                    gimbal_history["desired_el"].append(desired_el * to_deg)
                    gimbal_history["actual_el"].append(actual_el * to_deg)
                    gimbal_history["az_error_deg"].append(az_err * to_deg)
                    gimbal_history["el_error_deg"].append(el_err * to_deg)

                    for k in gimbal_history:
                        if len(gimbal_history[k]) > gimbal_history_len:
                            gimbal_history[k].pop(0)

                    n = len(gimbal_history["desired_az"])
                    x_data = list(range(n))
                    for k, line in gimbal_lines.items():
                        line.set_data(x_data, gimbal_history[k])

                    ax_gimbal.set_xlim(0, max(gimbal_history_len, n))
                    all_angles = (gimbal_history["desired_az"] + gimbal_history["actual_az"]
                                  + gimbal_history["desired_el"] + gimbal_history["actual_el"])
                    if all_angles:
                        ymin, ymax = min(all_angles), max(all_angles)
                        margin = max(1.0, (ymax - ymin) * 0.1)
                        ax_gimbal.set_ylim(ymin - margin, ymax + margin)

                fig.canvas.draw()
                fig.canvas.flush_events()

            # Print status periodically
            step_count += 1
            if step_count % 100 == 0:
                # Get drone state for controlled agent
                agent_id = f"drone_{controlled_agent_idx}"
                if agent_id in obs:
                    agent_obs = obs[agent_id][0]  # First env
                    pos = agent_obs[0:3]
                    vel = agent_obs[3:6]
                    gimbal_yaw = agent_obs[10]
                    gimbal_pitch = agent_obs[11]
                    zoom = agent_obs[12]
                    bbox_empty = agent_obs[17]

                    print(
                        f"[Step {step_count}] "
                        f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                        f"Vel: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) | "
                        f"Gimbal: (y={gimbal_yaw:.2f}, p={gimbal_pitch:.2f}) | "
                        f"Zoom: {zoom:.2f} | "
                        f"Target: {'Not detected' if bbox_empty > 0.5 else 'Detected'}"
                    )

            # Handle reset
            if should_reset:
                env.reset()
                should_reset = False
                xy_s = env._dr_target_scale[0, 0, 0].item()
                z_s = env._dr_target_scale[0, 0, 2].item()
                intr_s = env._dr_intrinsic_scale[0, controlled_agent_idx].item()
                print(f"[INFO] Environment reset | DR: xy={xy_s:.2f}, z={z_s:.2f}, fov={intr_s:.2f}, progress={env.progress_dynamics:.2f}")

    # Cleanup
    env.close()
    print("[INFO] Teleoperation ended")


if __name__ == "__main__":
    main()
    simulation_app.close()
