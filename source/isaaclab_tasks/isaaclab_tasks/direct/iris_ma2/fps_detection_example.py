# fps_detection_example.py
"""
Example showing how to use FPS-limited detection system.
Demonstrates the separation of FPS throttling and detection validity.
"""

import torch
from delay_comm_system_optimized import DelayedObservationManager

def main():
    # Configuration
    num_envs = 1024
    num_agents = 2
    dt = 0.01  # 100 Hz simulation
    detection_fps = 30.0  # 20 Hz detection rate (camera FPS)
    bbox_validity_rate = 0.7  # 70% of detections have bbox within image bounds

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create delay manager with FPS throttling
    delay_manager = DelayedObservationManager(
        num_envs=num_envs,
        num_agents=num_agents,
        dt=dt,
        device=device,
        motion_time_constant=0.1,
        gimbal_time_constant=0.03,
        detection_mean_latency=0.00,  # 50ms detection latency
        detection_std_latency=0.00,   # 10ms std
        detection_fps=detection_fps,   # <-- FPS throttling parameter
        comm_mean_delay=0.1,
        comm_std_delay=0.03,
        comm_dropout_rate=0.05
    )

    # Print configuration
    print("="*70)
    print("FPS THROTTLING AND DETECTION VALIDITY TEST")
    print("="*70)
    print(f"Simulation Rate:     {1/dt:.0f} Hz (every step)")
    print(f"Detection FPS:       {detection_fps:.0f} Hz (throttled)")
    print(f"Expected Period:     {1000/detection_fps:.1f} ms between captures")
    print(f"BBox Validity Rate:  {bbox_validity_rate*100:.0f}% (simulated)")
    print(f"Number of Envs:      {num_envs}")
    print(f"Number of Agents:    {num_agents}")
    print("="*70)
    print()

    # Track statistics for each agent
    frames_captured = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
    frames_with_valid_bbox = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
    frames_received_delayed = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)

    for step in range(500):  # 5 seconds of simulation at 100 Hz
        delay_manager.update_time()

        for agent_id in range(num_agents):
            # Generate simulated bbox data [N, 4] = [x, y, w, h]
            bboxes = torch.rand(num_envs, 4, device=device) * 100

            # Simulate detection validity: bbox center is within image bounds
            # This is independent of FPS throttling
            bbox_is_valid = torch.rand(num_envs, device=device) < bbox_validity_rate

            # Add detection (FPS-throttled internally)
            # The system will:
            # 1. Check if enough time passed (FPS throttling)
            # 2. Store frame data for envs that pass FPS check
            # 3. Store bbox_is_valid for those same frames
            delay_manager.add_detection(agent_id, bboxes, bbox_is_valid)

            # Retrieve delayed detection
            delayed_detection, detection_valid_mask = delay_manager.get_delayed_detection(agent_id)

            # delayed_detection.valid: whether we have received data (frame available)
            # detection_valid_mask: whether bbox is valid (within bounds) for received frames

            has_frame = delayed_detection.valid  # [N] - do we have a delayed frame?
            bbox_valid_in_frame = detection_valid_mask.squeeze(-1)  # [N, 1] -> [N]

            # Count statistics
            frames_received_delayed[:, agent_id] += has_frame.long()
            frames_with_valid_bbox[:, agent_id] += torch.logical_and(has_frame, bbox_valid_in_frame).long()

        # Print progress every second
        if (step + 1) % 100 == 0:
            elapsed_time = (step + 1) * dt
            print(f"Time: {elapsed_time:.2f}s ({step+1} steps)")

            for agent_id in range(num_agents):
                # Calculate average across environments
                avg_frames = frames_received_delayed[:, agent_id].float().mean().item()
                avg_valid = frames_with_valid_bbox[:, agent_id].float().mean().item()

                frame_rate = (avg_frames / (step + 1)) * 100
                valid_rate = (avg_valid / (step + 1)) * 100

                print(f"  Agent {agent_id}:")
                print(f"    Frames received:  {avg_frames:.1f} ({frame_rate:.1f}% of steps)")
                print(f"    Valid bboxes:     {avg_valid:.1f} ({valid_rate:.1f}% of steps)")
            print()

    # Final statistics
    total_steps = 500
    expected_frame_rate = detection_fps * dt  # Should match FPS throttling
    expected_frames = expected_frame_rate * total_steps
    expected_valid_bbox_rate = expected_frame_rate * bbox_validity_rate

    print("="*70)
    print("FINAL STATISTICS (averaged across all environments)")
    print("="*70)
    print(f"Total simulation steps: {total_steps}")
    print(f"Expected frame rate:    {expected_frame_rate*100:.1f}% ({expected_frames:.0f} frames)")
    print(f"Expected valid bbox rate: {expected_valid_bbox_rate*100:.1f}% ({expected_valid_bbox_rate*total_steps:.0f} valid)")
    print()

    for agent_id in range(num_agents):
        # Average across all environments
        avg_frames = frames_received_delayed[:, agent_id].float().mean().item()
        avg_valid = frames_with_valid_bbox[:, agent_id].float().mean().item()

        frame_rate = avg_frames / total_steps
        valid_rate = avg_valid / total_steps

        print(f"Agent {agent_id}:")
        print(f"  Frames received (delayed):    {avg_frames:.1f} ({frame_rate*100:.1f}%)")
        print(f"  Frames with valid bbox:       {avg_valid:.1f} ({valid_rate*100:.1f}%)")
        print(f"  Frame rate ratio to expected: {frame_rate/expected_frame_rate:.3f}x")
        print(f"  Valid bbox ratio to expected: {valid_rate/expected_valid_bbox_rate:.3f}x")

        # Conditional validity rate (of received frames, how many have valid bbox?)
        if avg_frames > 0:
            conditional_validity = (avg_valid / avg_frames) * 100
            print(f"  Valid bbox % (of received):   {conditional_validity:.1f}% (expected ~{bbox_validity_rate*100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
