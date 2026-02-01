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