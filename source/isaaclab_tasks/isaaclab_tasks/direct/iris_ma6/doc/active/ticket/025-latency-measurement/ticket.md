## Ticket: Latency measurement suite (C1, C2, C4, C5)

**What**: Implement ROS2-based latency and dropout measurement tools for the sensor/communication pipeline. Four measurement targets:
- C1: IMU-to-policy latency (proprioceptive)
- C2: Detection inference latency under realistic load
- C4: Raw image transport latency
- C5: Datalink latency and dropout rate (agent-to-agent, agent-to-GCS)

**Why**: The iris_ma6 delay system uses estimated latencies (5ms ego motion, 100ms detection, 500ms inter-agent) that have never been validated against the real pipeline. If real comms latency is 200ms (not 500ms) or detection is 50ms (not 100ms), the policy trains against wrong assumptions. These measurements directly set delay system parameters.

**Depends on**: Nothing (independent of flight test harness). Can be developed in parallel with ticket-023.

**Scripts**:

### C1: IMU-to-Policy Latency
- **Approach**: Use `ros2 topic delay` on the IMU topic chain, or a custom node that timestamps message receipt vs. header stamp
- **Minimal script**: Subscribe to IMU topic at policy node location, compute `receive_time - header.stamp`
- **Output**: CSV with per-message latency, summary statistics

### C2: detector_latency_bench.py
- **What**: Feed representative images to the detector at a configurable rate, measure inference time under realistic load
- **How**:
  1. Load images from a directory (collected from outdoor flights)
  2. Publish images to detector input topic at configurable FPS
  3. Run triangulation node concurrently (to simulate realistic GPU/CPU load)
  4. Measure per-frame: publish_time → detection_result_time
- **Output**: CSV with per-frame inference latency, summary statistics (mean, std, percentiles)

### C4: image_transport_latency.py
- **What**: Measure camera-to-ROS2 pipeline latency
- **How**:
  1. Point camera at a display showing a high-resolution millisecond timer
  2. Subscribe to camera image topic
  3. OCR or manual comparison of displayed time vs. ROS2 header timestamp
  4. Automated approach: display a known pattern at known times, detect in image
- **Caveat**: Measures full pipeline (sensor → driver → ROS2 topic). Actual detector-input path may differ. Document the measurement path.
- **Output**: CSV with per-frame transport latency

### C5: datalink_latency_ping.py
- **What**: Measure one-way communication latency and packet loss between agents
- **How**:
  1. Node A publishes timestamped ping messages at configurable rate (e.g., 50 Hz)
  2. Node B receives, records (send_timestamp, receive_timestamp, sequence_number)
  3. Compute one-way latency (requires clock sync — use PTP/NTP or estimate from round-trip)
  4. Detect packet loss from sequence number gaps
  5. Compute burst statistics (consecutive drops)
  6. Collect both agent-to-agent and agent-to-GCS paths
- **Configuration**: Test at multiple locations, with different link hardware (WiFi, radio, LTE)
- **Note**: Link technology selection is a separate ticket. This script is link-agnostic.
- **Output**: CSV with per-message latency, dropout events, burst statistics

**Scope boundary**:
- DO: Implement measurement scripts (not flight control — these are passive observers or bench tools)
- DO: Output standardized CSV format for each measurement type
- DO: Include summary statistics computation (mean, std, percentiles, packet loss %)
- DO NOT: Send flight commands (these are observation/measurement only)
- DO NOT: Select link technology (separate ticket per user comment)
- DO NOT: Implement the full detector pipeline (assume it exists, just measure it)

**Affected files**:
- NEW: Scripts in `offboard_py/offboard_py/` or a new `latency_tools/` subdirectory
- MODIFY: `offboard_py/setup.py` — add entry points (if placed in offboard_py)

**Acceptance criteria**:
- C1: Reports IMU latency statistics with >1000 samples
- C2: Reports detection inference latency under concurrent triangulation load
- C4: Reports image transport latency with documented measurement path
- C5: Reports one-way latency, packet loss rate, burst dropout statistics per link condition
- All: Output CSV with consistent column format; print summary statistics to console
- All: Configurable duration, sample count, output directory

**Reference**: [checklist.md](../021-sim2real-measurement-checklist/checklist.md) items C1, C2, C4, C5

**Flow**: Light (measurement tools, no complex control logic)
