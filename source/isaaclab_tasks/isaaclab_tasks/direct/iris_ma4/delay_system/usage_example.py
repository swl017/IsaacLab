"""
Usage example for the delay system (v2.2).

This example demonstrates how to:
1. Configure the delay system with dual pipeline architecture
2. Initialize for multi-agent environment
3. Update ground truth states (fed to both clean and noisy pipelines)
4. Query states using view scheme API
5. Separate clean states (rewards) from noisy states (observations)
6. Validate bboxes AFTER delay processing
7. Use curriculum learning hooks

v2.2 KEY CHANGES:
- Dual pipeline: clean (rewards) + noisy (observations)
- Noise injected BEFORE delays (physically correct)
- New view scheme API: get_all_states_for_rewards(), get_all_states_for_observations()
- REMOVED: get_delayed_states, get_delayed_noisy_states, receive_other_agent_states
"""
import sys
# Force unbuffered output for Isaac Sim compatibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run delay system usage example")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose debug output")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.direct.iris_ma3.delay_system import (
    MultiAgentDelaySystem,
    AgentStates,
)


def main():
    print("=" * 80)
    print("Delay System v2.2 Usage Example")
    print("=" * 80)

    # ========== Configuration ==========

    # Setup
    num_envs = 4
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

    # Create delay system with noise enabled
    delay_system = MultiAgentDelaySystem(
        possible_agents=agent_ids,
        num_envs=num_envs,
        num_joints_per_agent={agent_id: num_joints for agent_id in agent_ids},
        num_targets_per_agent={agent_id: num_targets for agent_id in agent_ids},
        dt=0.01,  # 100 Hz simulation
        device=device,
        # Noise parameters
        enable_noise=True,
        position_noise_std=0.05,      # 5cm position noise
        orientation_noise_std=0.02,   # ~1 degree orientation noise
        linear_velocity_noise_std=0.1,
        angular_velocity_noise_std=0.05,
        gimbal_noise_std=0.01,
        bbox_noise_std=5.0,           # 5 pixel bbox noise
        zoom_noise_std=0.01,
    )

    print(f"\n[OK] Delay system initialized for {num_agents} agents, {num_envs} environments")

    # ========== Camera Configuration (REQUIRED) ==========
    # Set camera intrinsics and offsets for each agent.
    # These are static fields that don't change during simulation.

    # Camera parameters (realistic 640x480 camera)
    width, height = 640, 480
    focal_length = 24.0  # mm
    horizontal_aperture = 20.955  # mm (roughly matches 45 deg FOV)
    vertical_aperture = 15.716   # mm

    # Camera offset in body frame (gimbal mounted slightly below and forward)
    offset_position_b = torch.tensor([0.1, 0.0, -0.05], device=device)  # 10cm forward, 5cm below
    offset_rotation_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # Identity quaternion

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
    dt = 0.01  # 100 Hz simulation

    # Expected delay parameters (defaults from delay_system_cfg.py)
    detection_latency_mean = 0.3  # 300ms detection latency
    detection_fps = 20.0  # 20 Hz detector
    motion_time_constant = 0.1  # 100ms first-order lag
    warmup_steps = int(detection_latency_mean / dt) + int(1.0 / detection_fps / dt)  # ~35 steps

    print(f"\nRunning {num_steps} simulation steps...")
    print(f"  Expected delays (configured):")
    print(f"    Motion: τ={motion_time_constant*1000:.0f}ms (first-order lag time constant)")
    print(f"    Detection: {detection_latency_mean*1000:.0f}ms latency + {1000/detection_fps:.0f}ms FPS period")
    print(f"  Warmup: ~{warmup_steps} steps ({warmup_steps * dt:.2f}s) for bboxes to propagate")

    for step in range(num_steps):
        # Advance simulation time
        delay_system.update_time()

        # Curriculum learning: gradually increase noise
        progress = step / num_steps
        delay_system.set_noise_progress_scale(progress)

        # Update ground truth states for each agent
        for agent_id in agent_ids:
            # In real env, these would come from physics simulation
            gt_states = create_dummy_states(num_envs, num_joints, num_targets, device, step)

            # Update motion states
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

            # Update detection data (v2.1: NO valid_mask parameter)
            delay_system.update_detections(
                agent_id=agent_id,
                bboxes_2d_gt=gt_states.data.bboxes_2d,  # [N, T, 4]
            )

        # ========== View Scheme API (v2.2) ==========

        # For REWARDS: Get clean delayed states (no noise)
        reward_states = delay_system.get_all_states_for_rewards("agent_0")
        ego_clean = reward_states["agent_0"]
        other_clean = reward_states["agent_1"]

        # For OBSERVATIONS: Get noisy delayed states
        obs_states = delay_system.get_all_states_for_observations("agent_0")
        ego_noisy = obs_states["agent_0"]
        other_noisy = obs_states["agent_1"]

        # Access delayed data directly
        ego_position = ego_noisy.data.body_position_w  # [num_envs, 3]
        ego_bboxes = ego_noisy.data.bboxes_2d  # [num_envs, num_targets, 4]

        other_position = other_noisy.data.body_position_w
        other_bboxes = other_noisy.data.bboxes_2d

        # ========== Bbox Validation (v2.1+ pattern) ==========
        # Users should validate bboxes AFTER delay processing
        # In real code, use: bbox_valid = bbox_raycaster.validate_bbox(bboxes)
        # Here we simulate with simple bounds check
        bbox_valid = validate_bboxes_simple(ego_bboxes, width=640, height=480)
        obs_bbox_valid = bbox_valid  # Same thing for noisy observations

        # ========== Clean vs Noisy Comparison ==========
        # Clean states (for rewards - no noise)
        ego_pos_clean = ego_clean.data.body_position_w
        other_pos_clean = other_clean.data.body_position_w

        # Noisy states (for observations - with noise)
        ego_pos_noisy = ego_noisy.data.body_position_w
        other_pos_noisy = other_noisy.data.body_position_w

        # Compute noise magnitude (useful for debugging)
        ego_noise_mag = (ego_pos_noisy - ego_pos_clean).abs().mean()
        other_noise_mag = (other_pos_noisy - other_pos_clean).abs().mean()

        # Use timestamp information (if available)
        if ego_noisy.timestamps is not None:
            motion_staleness = ego_noisy.timestamps.timestamps['motion'].staleness
            detection_staleness = ego_noisy.timestamps.timestamps['detection'].staleness
        else:
            motion_staleness = torch.zeros(num_envs, device=device)
            detection_staleness = torch.zeros(num_envs, device=device)

        # Print progress every 20 steps
        if step % 20 == 0:
            valid_ratio = bbox_valid.float().mean().item()

            # Expected bbox_valid: 0% during warmup, 100% after
            expected_valid = 0.0 if step < warmup_steps else 1.0

            # Show simulation time and expected vs actual bbox_valid
            sim_time_ms = step * dt * 1000
            print(f"  Step {step:3d} (t={sim_time_ms:.0f}ms): noise={progress:.2f}, "
                  f"bbox_valid={valid_ratio:.0%} (expected={expected_valid:.0%})")

    print(f"\n[OK] Simulation completed: {num_steps} steps")

    # ========== Reset Example ==========

    env_ids_to_reset = torch.tensor([0, 2], device=device)
    delay_system.reset(env_ids_to_reset)

    print(f"[OK] Reset environments: {env_ids_to_reset.tolist()}")

    # ========== Curriculum Learning Hooks Example ==========

    print("\n" + "=" * 80)
    print("Curriculum Learning Hooks Example")
    print("=" * 80)

    # Adjust delay parameters during training (easier -> harder)
    # v2.2: Need to apply to BOTH pipelines (clean and noisy)
    print("\nAdjusting delay parameters for curriculum learning:")

    def set_delay_params(motion_tc, orientation_tc, joint_tc, fps_mean, latency_mean, dropout_rate):
        """Helper to apply settings to both pipelines."""
        for ds in [delay_system._delay_system_clean, delay_system._delay_system_noisy]:
            ds.set_time_constants(
                motion_tc=motion_tc,
                orientation_tc=orientation_tc,
                joint_tc=joint_tc,
            )
            ds.set_detection_latency_params(
                fps_mean=fps_mean,
                latency_mean=latency_mean,
                dropout_rate=dropout_rate,
            )

    # Easy settings (fast, low latency)
    set_delay_params(
        motion_tc=0.02,       # 20ms time constant
        orientation_tc=0.02,
        joint_tc=0.01,
        fps_mean=30.0,        # 30 Hz detector
        latency_mean=0.1,     # 100ms latency
        dropout_rate=0.02,    # 2% dropout
    )
    print("  [Easy] motion_tc=0.02, detection_fps=30, latency=0.1s, dropout=2%")

    # Medium settings
    set_delay_params(
        motion_tc=0.05,
        orientation_tc=0.05,
        joint_tc=0.02,
        fps_mean=20.0,
        latency_mean=0.2,
        dropout_rate=0.05,
    )
    print("  [Medium] motion_tc=0.05, detection_fps=20, latency=0.2s, dropout=5%")

    # Hard settings (realistic)
    set_delay_params(
        motion_tc=0.1,
        orientation_tc=0.1,
        joint_tc=0.05,
        fps_mean=15.0,
        latency_mean=0.3,
        dropout_rate=0.10,
    )
    print("  [Hard] motion_tc=0.1, detection_fps=15, latency=0.3s, dropout=10%")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


def create_dummy_states(
    num_envs: int,
    num_joints: int,
    num_targets: int,
    device: torch.device,
    step: int = 0,
) -> AgentStates:
    """
    Create dummy AgentStates for demonstration.

    In a real environment, these would come from physics simulation.
    """
    states = AgentStates(num_envs, num_joints, num_targets, device)

    # Simulate moving drone
    t = step * 0.01  # time in seconds

    # Motion states (circular motion)
    states.data.body_position_w = torch.stack([
        torch.sin(torch.tensor(t)) * torch.ones(num_envs, device=device),
        torch.cos(torch.tensor(t)) * torch.ones(num_envs, device=device),
        torch.ones(num_envs, device=device) * 10.0,  # 10m altitude
    ], dim=-1)

    # Body orientation (identity quaternion with small random perturbation)
    # Quaternion format: (w, x, y, z)
    states.data.body_orientation_w = torch.zeros(num_envs, 4, device=device)
    states.data.body_orientation_w[:, 0] = 1.0  # w=1, identity quaternion

    states.data.body_linear_velocity_w = torch.randn(num_envs, 3, device=device) * 0.5
    states.data.body_angular_velocity_w = torch.randn(num_envs, 3, device=device) * 0.1
    states.data.body_linear_acceleration_w = torch.randn(num_envs, 3, device=device) * 0.1
    states.data.body_combined_angular_velocity_w = torch.randn(num_envs, 3, device=device) * 0.1

    # Joint states (gimbal pitch/yaw)
    states.data.joint_positions_b = torch.randn(num_envs, num_joints, device=device) * 0.5

    # Camera zoom
    states.data.camera_zoom_level = torch.ones(num_envs, device=device) * 2.0  # 2x zoom

    # Detection bboxes (simulate detections in frame)
    # Format: [x, y, w, h] in pixels
    bboxes = torch.zeros(num_envs, num_targets, 4, device=device)
    for t_idx in range(num_targets):
        bboxes[:, t_idx, 0] = 100 + t_idx * 100  # x center
        bboxes[:, t_idx, 1] = 200                 # y center
        bboxes[:, t_idx, 2] = 50                  # width
        bboxes[:, t_idx, 3] = 50                  # height
    # Add some noise to simulate detection jitter
    bboxes += torch.randn_like(bboxes) * 2
    states.data.bboxes_2d = bboxes

    # Timestamps
    states.data.timestamp_sim_walltime = torch.ones(num_envs, device=device) * t
    states.data.timestamp_motion = torch.ones(num_envs, device=device) * t
    states.data.timestamp_detection = torch.ones(num_envs, device=device) * t

    return states


def validate_bboxes_simple(
    bboxes: torch.Tensor,  # [N, T, 4]
    width: int = 640,
    height: int = 480,
    min_size: float = 5.0,
) -> torch.Tensor:
    """
    Simple bbox validation (simulates bbox_raycaster.validate_bbox).

    In real code, use your bbox_raycaster implementation.

    Args:
        bboxes: Bounding boxes [N, T, 4] in (x, y, w, h) format
        width: Image width
        height: Image height
        min_size: Minimum bbox dimension

    Returns:
        valid: Boolean mask [N, T]
    """
    x = bboxes[..., 0]  # [N, T]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]

    # Check bounds
    x_valid = (x >= 0) & (x < width)
    y_valid = (y >= 0) & (y < height)

    # Check minimum size
    w_valid = w >= min_size
    h_valid = h >= min_size

    return x_valid & y_valid & w_valid & h_valid


if __name__ == "__main__":
    main()
