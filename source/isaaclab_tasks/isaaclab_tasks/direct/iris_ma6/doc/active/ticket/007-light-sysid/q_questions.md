## Stage Q — Open Questions

### Assumptions requiring confirmation

1. **Which drone instance to test?** Single drone (instance 0) vs all agents.
   → **Answer**: Single is enough (instance 0, namespace `px4_1`).

2. **Offboard command interface**: New node in offboard_py, standalone script, or new package?
   → **Answer**: New ROS2 node in `offboard_py`.

3. **iris_ma6 reference data format**: Are raw timeseries saved by auto_tune.py?
   → **Answer**: Match auto_tune.py output format. (Raw timeseries NOT saved — must regenerate.)

4. **Simulation clock source**: Which topic carries authoritative sim clock?
   → **Answer**: PegasusSimulator publishes `/clock`, but duplicate topics arrive. Refer to `los_rate_controller.py` for dedup pattern.

5. **Gimbal command interface**: Exact topic names and message types?
   → **Answer**: `/px4_1/gimbal_cmd_los_rate` (Vector3: x=az_rate, y=el_rate rad/s).

6. **Gimbal reference curves**: Do they exist?
   → **Answer**: No. Both iris_ma6 and gimbal stabilizer use LOS rate as input. Compare response to same commanded step — no iris_ma6 gimbal simulation needed.

7. **Recording duration and settling criteria**: Match auto_tune.py windows?
   → **Answer**: Use 10 seconds per test.

8. **Output location**: Where should files live?
   → **Answer**: `ros2_ws/src/offboard_py/sysid/` — tightly coupled to ROS2 implementation.

### Architectural decisions requiring human input

9. **ROS2 node vs standalone**: Which script architecture?
   → **Answer**: ROS2 node (depends on MAVROS).

10. **Sequential vs automated test execution**: Land between tests?
    → **Answer**: No landing. `arm → hover → test1 → hover(settle 10s) → test2 → ...`

11. **Online vs post-processing metrics**: Compute during recording or post-hoc?
    → **Answer**: Post-hoc.

12. **Plot format**: PDF or HTML?
    → **Answer**: PDF.

13. **Multi-agent gimbal namespacing**: Single node or per-drone?
    → **Answer**: Each drone gets namespaced topics.

### Additional decisions (from design review)

14. **PX4 interface**: Drop uxDDS (`fmu/*`) entirely. Use MAVROS-only, all ENU.

15. **Offboard mode**: Velocity setpoints only. No position mode.

16. **Attitude perturbation**: Velocity impulse recovery ≠ direct attitude perturbation. auto_tune.py adaptation is separate work.

17. **MAVROS IMU**: Gimbal stabilizer subscribes to `mavros/imu/data`. offboard_control.py republishes this.

18. **Clock**: Subscribe `/clock` in sim (`use_sim_time: true`). Wallclock in real tests.

19. **Velocity streaming rate**: 100 Hz sim-time rate (matching iris_ma6 physics dt).

20. **Gimbal LOS rate test**: Short pulses — init → 0.5s pitch → hold 0.5s → 0.5s yaw → hold 0.5s (avoid joint limits).

21. **Gimbal LOS stabilization test**: Yaw drone → pulse x-vel 1s (pitch disturbance) → pulse y-vel 1s (roll disturbance). Measure LOS drift.

22. **Launch**: New `tmux/isaac_sysid.tmuxp.yaml` integrating MAVROS + sysid node. Don't reuse `isaac_mavros.tmuxp.yaml`.
