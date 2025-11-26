"""
Usage example for the delay system.

This example demonstrates how to:
1. Configure the delay system
2. Initialize for multi-agent environment
3. Update ground truth states
4. Query delayed states from different perspectives
5. Use timestamp information
"""
import sys
# Force unbuffered output for Isaac Sim compatibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run delay system test suite")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--test-verbose", action="store_true", help="Enable verbose test debug output")
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.direct.iris_ma3.delay_system import DelaySystem, DelaySystemCfg, AgentStates


def main():
    # ========== Configuration ==========

    # Create delay system configuration
    cfg = DelaySystemCfg()

    # Customize parameters (optional)
    cfg.dt_sim = 0.01  # 100 Hz simulation
    cfg.detection_fps_mean = 20.0  # 20 Hz detector
    cfg.detection_latency_mean = 0.3  # 300ms processing latency
    cfg.inter_agent_comm_latency_mean = 0.1  # 100ms inter-agent comm

    # ========== Initialization ==========

    # Setup
    num_envs = 4
    num_agents = 3
    num_joints = 2
    num_targets = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent_ids = [f"agent_{i}" for i in range(num_agents)]

    # Create delay system
    delay_system = DelaySystem(
        cfg=cfg,
        agent_ids=agent_ids,
        num_envs=num_envs,
        num_joints=num_joints,
        num_targets=num_targets,
        device=device,
    )

    print(f"✓ Delay system initialized for {num_agents} agents, {num_envs} environments")

    # ========== Simulation Loop ==========

    num_steps = 100

    for step in range(num_steps):
        # Advance simulation time
        delay_system.step()

        # Update ground truth states for each agent
        for agent_id in agent_ids:
            # In real env, these would come from physics simulation
            gt_states = create_dummy_states(num_envs, num_joints, num_targets, device)
            delay_system.update_agent_gt_states(agent_id, gt_states)

        # Query delayed states from agent_0's perspective
        agent_0_view = delay_system.get_all_agent_states_for_ego("agent_0")

        # agent_0_view contains:
        # - agent_0: Fast ego processing
        # - agent_1: Slow inter-agent communication
        # - agent_2: Slow inter-agent communication

        ego_states = agent_0_view["agent_0"]
        other_states = agent_0_view["agent_1"]

        # Access delayed data
        ego_position = ego_states.data.body_position_w  # [num_envs, 3]
        ego_bboxes = ego_states.data.bboxes_2d  # [num_envs, num_targets, 4]

        other_position = other_states.data.body_position_w
        other_bboxes = other_states.data.bboxes_2d

        # Use timestamp information (if timestamps are set)
        if ego_states.timestamps is not None:
            motion_staleness = ego_states.timestamps.timestamps['motion'].staleness
            detection_staleness = ego_states.timestamps.timestamps['detection'].staleness

            # Use staleness in observations or rewards
            avg_detection_staleness = detection_staleness.mean()

            if step % 10 == 0:
                print(f"Step {step}: Avg detection staleness = {avg_detection_staleness:.3f}s")

        # Query from different agent's perspective
        agent_1_view = delay_system.get_all_agent_states_for_ego("agent_1")
        # Now agent_1 is ego, agent_0 and agent_2 are "others"

    print(f"✓ Simulation completed: {num_steps} steps")

    # ========== Reset Example ==========

    # Reset specific environments
    env_ids_to_reset = torch.tensor([0, 2], device=device)
    delay_system.reset(env_ids_to_reset)

    print(f"✓ Reset environments: {env_ids_to_reset.tolist()}")


def create_dummy_states(num_envs: int, num_joints: int, num_targets: int, device: torch.device) -> AgentStates:
    """Create dummy AgentStates for demonstration.

    Note: AgentStates initialization already creates all fields with zeros.
    This minimal example only populates motion and joint fields to demonstrate
    the delay system's core functionality. Detection fields (bboxes_2d, etc.)
    are left as zeros to avoid dimension mismatch issues in the current
    delay system architecture.
    """
    states = AgentStates(num_envs, num_joints, num_targets, device)

    # Populate only motion and joint fields for this demonstration
    # Motion states
    states.data.body_position_w = torch.randn(num_envs, 3, device=device)
    states.data.body_linear_velocity_w = torch.randn(num_envs, 3, device=device)

    # Joint states
    states.data.joint_positions_b = torch.randn(num_envs, num_joints, device=device)

    # Camera zoom (demonstrates first-order lag for mechanical actuators)
    states.data.camera_zoom_level = torch.ones(num_envs, device=device) * 1.0  # 1x zoom

    # Note: Detection fields (bboxes_2d, camera_ray_directions_w, camera_ray_origins_w)
    # remain as zeros. The current delay system architecture processes all detection
    # fields through a single sampler, which doesn't support mixed dimensions.

    # Set timestamps
    states.data.timestamp_sim_walltime = torch.ones(num_envs, device=device) * 0.1
    states.data.timestamp_motion = torch.ones(num_envs, device=device) * 0.1
    states.data.timestamp_detection = torch.ones(num_envs, device=device) * 0.1

    return states


if __name__ == "__main__":
    main()
