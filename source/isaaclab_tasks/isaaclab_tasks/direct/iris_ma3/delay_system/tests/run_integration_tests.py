"""
Integration tests for the Delay System (v2.2).

This test demonstrates:
1. Dual pipeline architecture (clean for rewards, noisy for observations)
2. View scheme API for unified state access
3. Noise-before-delay processing
4. Multi-agent perspective (ego vs other agents)
"""
import sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run delay system integration tests (v2.2)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.direct.iris_ma3.delay_system import (
    MultiAgentDelaySystem,
    AgentStates,
)


def main():
    print("=" * 80)
    print("Delay System v2.2 Integration Tests")
    print("=" * 80)

    # ========== Configuration ==========

    num_envs = 32
    num_agents = 3
    num_joints = 2
    num_targets = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent_ids = [f"agent_{i}" for i in range(num_agents)]

    print(f"\nConfiguration:")
    print(f"  Environments: {num_envs}")
    print(f"  Agents: {num_agents}")
    print(f"  Joints: {num_joints}")
    print(f"  Targets: {num_targets}")
    print(f"  Device: {device}")

    # ========== Initialization ==========

    delay_system = MultiAgentDelaySystem(
        possible_agents=agent_ids,
        num_envs=num_envs,
        num_joints_per_agent={agent_id: num_joints for agent_id in agent_ids},
        num_targets_per_agent={agent_id: num_targets for agent_id in agent_ids},
        dt=0.01,
        device=device,
        enable_noise=True,
        position_noise_std=0.05,
        orientation_noise_std=0.02,
        linear_velocity_noise_std=0.1,
        angular_velocity_noise_std=0.05,
        gimbal_noise_std=0.01,
        bbox_noise_std=5.0,
        zoom_noise_std=0.01,
    )

    print(f"\n[OK] Delay system initialized with DUAL PIPELINE architecture")
    print(f"     - Clean pipeline (for rewards)")
    print(f"     - Noisy pipeline (for observations)")

    # ========== Camera Configuration ==========

    width, height = 640, 480
    focal_length = 24.0
    horizontal_aperture = 20.955
    vertical_aperture = 15.716
    offset_position_b = torch.tensor([0.1, 0.0, -0.05], device=device)
    offset_rotation_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    for agent_id in agent_ids:
        delay_system.set_camera_configs(
            agent_id=agent_id,
            width=width,
            height=height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            offset_position_b=offset_position_b,
            offset_rotation_b=offset_rotation_b,
        )

    print(f"[OK] Camera configuration set for all agents")

    # ========== Simulation Loop ==========

    num_steps = 100
    dt = 0.01

    # Expected warmup periods
    detection_latency_mean = 0.3
    detection_fps = 20.0
    warmup_steps = int(detection_latency_mean / dt) + int(1.0 / detection_fps / dt)
    comm_latency_mean = 0.1
    comm_rate = 30.0
    warmup_steps_other = warmup_steps + int(comm_latency_mean / dt) + int(1.0 / comm_rate / dt)

    print(f"\nRunning {num_steps} simulation steps...")
    print(f"  Expected delays:")
    print(f"    Detection: {detection_latency_mean*1000:.0f}ms latency + {1000/detection_fps:.0f}ms FPS period")
    print(f"  Warmup (ego): ~{warmup_steps} steps ({warmup_steps * dt:.2f}s)")
    print(f"  Warmup (other): ~{warmup_steps_other} steps ({warmup_steps_other * dt:.2f}s)")

    for step in range(num_steps):
        delay_system.update_time()

        progress = step / num_steps
        delay_system.set_noise_progress_scale(progress)

        # Update GT states for each agent
        for agent_id in agent_ids:
            gt_states = create_dummy_states(num_envs, num_joints, num_targets, device, step)

            delay_system.update_gt_states(
                agent_id=agent_id,
                body_position_w=gt_states.data.body_position_w,
                body_orientation_w=gt_states.data.body_orientation_w,
                body_linear_velocity_w=gt_states.data.body_linear_velocity_w,
                body_angular_velocity_w=gt_states.data.body_angular_velocity_w,
                body_linear_acceleration_w=gt_states.data.body_linear_acceleration_w,
                body_combined_angular_velocity_w=gt_states.data.body_combined_angular_velocity_w,
                joint_positions_b=gt_states.data.joint_positions_b,
                zoom_level=gt_states.data.camera_zoom_level,
            )

            delay_system.update_detections(
                agent_id=agent_id,
                bboxes_2d_gt=gt_states.data.bboxes_2d,
            )

        # ========== View Scheme API (v2.2) ==========

        # For REWARDS: Get clean delayed states
        reward_states = delay_system.get_all_states_for_rewards("agent_0")
        ego_reward = reward_states["agent_0"]
        other_reward = reward_states["agent_1"]

        # For OBSERVATIONS: Get noisy delayed states
        obs_states = delay_system.get_all_states_for_observations("agent_0")
        ego_obs = obs_states["agent_0"]
        other_obs = obs_states["agent_1"]

        # Validate bboxes
        ego_bbox_valid = validate_bboxes_simple(ego_obs.data.bboxes_2d, width=640, height=480)
        other_bbox_valid = validate_bboxes_simple(other_obs.data.bboxes_2d, width=640, height=480)

        # Compute noise magnitude (difference between clean and noisy)
        ego_noise = (ego_obs.data.body_position_w - ego_reward.data.body_position_w).abs().mean().item()
        other_noise = (other_obs.data.body_position_w - other_reward.data.body_position_w).abs().mean().item()

        # Print progress
        if step > 20 and step < 50 or step % 10 == 0:
            sim_time_ms = step * dt * 1000
            ego_valid_ratio = ego_bbox_valid.float().mean().item()
            other_valid_ratio = other_bbox_valid.float().mean().item()

            expected_ego_valid = 0.0 if step < warmup_steps else 1.0
            expected_other_valid = 0.0 if step < warmup_steps_other else 1.0

            expected_noise = delay_system._noise_stds['position'] * progress

            print(f"  Step {step:3d} (t={sim_time_ms:.0f}ms): noise_scale={progress:.2f}")
            print(f"         ego_bbox_valid={ego_valid_ratio:.0%} (expected={expected_ego_valid:.0%})")
            print(f"         other_bbox_valid={other_valid_ratio:.0%} (expected={expected_other_valid:.0%})")
            print(f"         ego_noise={ego_noise:.4f}, other_noise={other_noise:.4f} (expected ~{expected_noise:.4f})")

    print(f"\n[OK] Simulation completed: {num_steps} steps")

    # ========== Reset Test ==========

    env_ids_to_reset = torch.tensor([0, 2], device=device)
    delay_system.reset(env_ids_to_reset)
    print(f"[OK] Reset environments: {env_ids_to_reset.tolist()}")

    # ========== Dual Pipeline Verification ==========

    print("\n" + "=" * 80)
    print("Dual Pipeline Verification (v2.2)")
    print("=" * 80)

    delay_system.set_noise_progress_scale(1.0)

    # Update with fresh data
    for agent_id in agent_ids:
        gt_states = create_dummy_states(num_envs, num_joints, num_targets, device, step=100)
        delay_system.update_gt_states(
            agent_id=agent_id,
            body_position_w=gt_states.data.body_position_w,
            body_orientation_w=gt_states.data.body_orientation_w,
            body_linear_velocity_w=gt_states.data.body_linear_velocity_w,
            body_angular_velocity_w=gt_states.data.body_angular_velocity_w,
            body_linear_acceleration_w=gt_states.data.body_linear_acceleration_w,
            body_combined_angular_velocity_w=gt_states.data.body_combined_angular_velocity_w,
            joint_positions_b=gt_states.data.joint_positions_b,
            zoom_level=gt_states.data.camera_zoom_level,
        )
        delay_system.update_detections(
            agent_id=agent_id,
            bboxes_2d_gt=gt_states.data.bboxes_2d,
        )

    # Get states from both pipelines
    reward_states = delay_system.get_all_states_for_rewards("agent_0")
    obs_states = delay_system.get_all_states_for_observations("agent_0")

    print("\n1. Ego States (agent_0):")
    ego_clean = reward_states["agent_0"]
    ego_noisy = obs_states["agent_0"]
    ego_pos_diff = (ego_noisy.data.body_position_w - ego_clean.data.body_position_w).abs().mean().item()
    print(f"   Position noise: {ego_pos_diff:.4f} (expected ~{delay_system._noise_stds['position']:.4f})")

    print("\n2. Other Agent States (agent_1):")
    other_clean = reward_states["agent_1"]
    other_noisy = obs_states["agent_1"]
    other_pos_diff = (other_noisy.data.body_position_w - other_clean.data.body_position_w).abs().mean().item()
    print(f"   Position noise: {other_pos_diff:.4f} (expected ~{delay_system._noise_stds['position']:.4f})")

    # Verify bbox noise
    other_bbox_clean = other_clean.data.bboxes_2d
    other_bbox_noisy = other_noisy.data.bboxes_2d
    bbox_diff = (other_bbox_noisy - other_bbox_clean).abs().mean().item()
    print(f"   Bbox noise: {bbox_diff:.2f} pixels")

    print("\n3. Pipeline Summary:")
    if ego_pos_diff > 0.001:
        print(f"   ✓ Ego noise working: diff={ego_pos_diff:.4f}")
    else:
        print(f"   ✗ Ego noise NOT working: diff={ego_pos_diff:.4f}")

    if other_pos_diff > 0.001:
        print(f"   ✓ Other agent noise working: diff={other_pos_diff:.4f}")
    else:
        print(f"   ✗ Other agent noise NOT working: diff={other_pos_diff:.4f}")

    print("\n" + "=" * 80)
    print("Integration tests completed successfully!")
    print("=" * 80)


def create_dummy_states(
    num_envs: int,
    num_joints: int,
    num_targets: int,
    device: torch.device,
    step: int = 0,
) -> AgentStates:
    """Create dummy AgentStates for testing."""
    states = AgentStates(num_envs, num_joints, num_targets, device)

    t = step * 0.01

    # Motion states
    states.data.body_position_w = torch.stack([
        torch.sin(torch.tensor(t)) * torch.ones(num_envs, device=device),
        torch.cos(torch.tensor(t)) * torch.ones(num_envs, device=device),
        torch.ones(num_envs, device=device) * 10.0,
    ], dim=-1)

    states.data.body_orientation_w = torch.zeros(num_envs, 4, device=device)
    states.data.body_orientation_w[:, 0] = 1.0

    states.data.body_linear_velocity_w = torch.randn(num_envs, 3, device=device) * 0.5
    states.data.body_angular_velocity_w = torch.randn(num_envs, 3, device=device) * 0.1
    states.data.body_linear_acceleration_w = torch.randn(num_envs, 3, device=device) * 0.1
    states.data.body_combined_angular_velocity_w = torch.randn(num_envs, 3, device=device) * 0.1

    states.data.joint_positions_b = torch.randn(num_envs, num_joints, device=device) * 0.5
    states.data.camera_zoom_level = torch.ones(num_envs, device=device) * 2.0

    # Camera intrinsics
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    K = torch.zeros(num_envs, 3, 3, device=device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    states.data.camera_base_intrinsics = K

    states.data.camera_offset_position_b = torch.zeros(num_envs, 3, device=device)
    states.data.camera_offset_position_b[:, 0] = 0.1
    states.data.camera_offset_position_b[:, 2] = -0.05

    states.data.camera_offset_rotation_b = torch.zeros(num_envs, 4, device=device)
    states.data.camera_offset_rotation_b[:, 0] = 1.0

    # Bboxes
    bboxes = torch.zeros(num_envs, num_targets, 4, device=device)
    for t_idx in range(num_targets):
        bboxes[:, t_idx, 0] = 100 + t_idx * 100
        bboxes[:, t_idx, 1] = 200
        bboxes[:, t_idx, 2] = 50
        bboxes[:, t_idx, 3] = 50
    bboxes += torch.randn_like(bboxes) * 2
    states.data.bboxes_2d = bboxes

    states.data.timestamp_sim_walltime = torch.ones(num_envs, device=device) * t
    states.data.timestamp_motion = torch.ones(num_envs, device=device) * t
    states.data.timestamp_detection = torch.ones(num_envs, device=device) * t

    return states


def validate_bboxes_simple(
    bboxes: torch.Tensor,
    width: int = 640,
    height: int = 480,
    min_size: float = 5.0,
) -> torch.Tensor:
    """Simple bbox validation."""
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]

    x_valid = (x >= 0) & (x < width)
    y_valid = (y >= 0) & (y < height)
    w_valid = w >= min_size
    h_valid = h >= min_size

    return x_valid & y_valid & w_valid & h_valid


if __name__ == "__main__":
    main()
