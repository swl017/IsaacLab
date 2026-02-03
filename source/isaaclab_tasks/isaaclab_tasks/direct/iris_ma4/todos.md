# TODOs for iris_ma4

1. (Raised at 2026-01-30 21:00, resolved at 2026-02-01 00:00) Does `detection_age` actually relect delayed timestamps?
    > Summary of All Changes
        File	Change
        multi_agent_delay_system_v2.py	1. Added RAW_TIMESTAMP_FIELDS constant
        2. Added timestamp dimension in _build_field_dims()
        3. Added timestamp delay configs in _build_field_configs()
        4. Store timestamp in update_detections()
        5. Retrieve delayed timestamp in _build_agent_states()
        6. Initialize timestamp in _initialize_pipelines_from_gt_states()
        run_tests.py	Added run_timestamp_tests() verification tests

2. (Raised at 2026-01-30 23:00, resolved at 2026-01-31 12:00) `Warning: Triangulation covariance computation failed: linalg.inv: (Batch element 0): The diagonal element 1 is zero, the inversion could not be completed because the input matrix is singular.` all the time.
    - Triangulation Singularity Fix (triang_cov_reward_torch.py:280-293)
       Added minimum diagonal floor (1e-6) to prevent near-zero covariance
       Added try/except with pseudo-inverse fallback for edge cases

3. (Raised at 2026-01-30 23:00) 30,000 step training session ETA 12 hours, at decimation=4 dt=1/100.

4. (Raised at 2026-01-30 23:00, resolved 2026-02-01 23:00) Check initial state (especially gimbal pointing to the target and formation) and randomization(including robot, gimbal orientation, zoom level) is correct and respects curriculum and physical limits(gimbal tilt limits, max speed, distance, height, etc.) at reset.

5. (Raised at 2026-02-01 00:50, resolved 2026-02-01 02:30) Gimbal joint_id inconsistant between set_joint_position and reading actual joint angles. We need to review the structure inside Iris USDA model.
    - Now Convention Established
        joint_positions_b tensor uses [pitch, yaw, roll] convention:
        Index 0: pitch
        Index 1: yaw
        Index 2: roll
        This convention is camera-centric and matches what compute_camera_orientation_from_gimbal() expects.

6. (Raised at 2026-02-01 03:00, resolved 2026-02-01 23:00) Robot and gimbal not facing the target correctly at reset.

7. (Raised at 2026-02-01 23:10, resolved 2026-02-02 00:24) Add randomized target movement.
```
# Definition
self.target = RigidObject(self.cfg.target_cfg)

# Update target acceleration with capped velocity
# while being above certain altitude (10m) and within a geofenced area (from 50m x 50m to 1,000m x 1,000m curriculum)

# The target can fly in straight or in circular path.
# Target's target velocity and flight path can be updated at random period

```

8. (Raised at 2026-02-02 00:25, resolved at 2026-02-02 00:40) Can we better implement `direction_change_prob` at
@source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma4/target_movement/target_movement.py#L283 in terms of distribution, learnability and curriculum. What interpretation can we give around this whole 'target movement' issue?

9. Running `iris_ma4_mappo_rnn_tuning6_20260203_014758` experiment...
    - triangulation_quality = exp(-trace_cov) (iris_ma_env4.py (lines 802-806)) will saturate fast and is unit-sensitive (trace is in ~m²). If trace is commonly >1, this term becomes near-zero almost always; if trace is <<1, it’s almost always ~1. Consider exp(-trace_cov / s) with a tuned s, or a gentler map like 1 / (1 + trace_cov / s) or -log(trace_cov + eps).

10. (Raised at 2026-02-03 13:11) For 3 or more agents
    - Triangulation validity currently requires all agents’ bboxes valid (bbox_valid_mask.all(dim=1) in _compute_triangulation_covariance). With detection dropout, this can zero out triangulation rewards often and create “dead” learning periods; consider “at least 2 valid views” (for 2 agents: both; for >2: any pair) instead of all.

11. Running `iris_ma4_mappo_rnn_tuning6_20260203_151152` experiment...
    - Training collapsed at 20k step.