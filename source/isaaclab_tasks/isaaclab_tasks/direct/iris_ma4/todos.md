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

4. (Raised at 2026-01-30 23:00) Check initial state (especially gimbal pointing to the target and formation) and randomization is correct and respects curriculum and physical limits, like gimbal tilt limits, at reset. Gimbal joint_id inconsistancy must be resolved prior to this issue.

5. (Raised at 2026-02-01 00:50) Gimbal joint_id inconsistant between set_joint_position and reading actual joint angles. We need to review the structure inside Iris USDA model.
    ```
    # Implementation
    """ NOTE: Abuse of variables/terms: 
        1.  cmd_vel[:, idx, 4] and cmd_vel[:, idx, 5] are used as target angles (not rates) instead.
        2.  targets are applied to in roll, pitch, yaw order, regardless of joint_ids yaw, pitch, roll order.
    """
    robot.set_joint_position_target(
        target=torch.stack([
            torch.zeros_like(gimbal_roll_stabilizing), 
            torch.clamp(self.cmd_vel[:, idx, 4], min=self.cfg.gimbal.pitch_limits[0], max=self.cfg.gimbal.pitch_limits[1]),
            torch.clamp(self.cmd_vel[:, idx, 5], min=self.cfg.gimbal.yaw_limits[0], max=self.cfg.gimbal.yaw_limits[1])], dim=-1),
        joint_ids=[
            self.gimbal_joint_idx[agent_id]["yaw"],
            self.gimbal_joint_idx[agent_id]["roll"],
            self.gimbal_joint_idx[agent_id]["pitch"],
        ]
    )
    """ Debug logs """
    gimbal_targets = torch.stack([
            gimbal_roll_stabilizing, 
            torch.clamp(self.cmd_vel[:, idx, 4], min=self.cfg.gimbal.pitch_limits[0], max=self.cfg.gimbal.pitch_limits[1]), 
            torch.clamp(self.cmd_vel[:, idx, 5], min=self.cfg.gimbal.yaw_limits[0], max=self.cfg.gimbal.yaw_limits[1])], dim=-1)
    carb.log_warn(f"gimbal targets (roll, pitch, yaw): {gimbal_targets.cpu().numpy().round(2)}")
    
    gimbal_actual = torch.stack([
            robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["roll"]], 
            robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["pitch"]], 
            robot.data.joint_pos[:, self.gimbal_joint_idx[agent_id]["yaw"]]], dim=-1)
    carb.log_warn(f"gimbal actual (roll, pitch, yaw): {gimbal_actual.cpu().numpy().round(2)}")
    ```
    ```
    # Terminal output
    # Gimbal actually turns in pitch direction in the simulation with pitch targets applied
    2026-01-31 15:46:39 [30,537ms] [Warning] [isaaclab_tasks.direct.iris_ma4.iris_ma_env4] gimbal targets (roll, pitch, yaw): [[ 0.   -0.79  0.  ]
    [ 0.   -0.79  0.  ]
    [ 0.   -0.79  0.  ]
    [ 0.   -0.79  0.  ]]
    2026-01-31 15:46:39 [30,538ms] [Warning] [isaaclab_tasks.direct.iris_ma4.iris_ma_env4] gimbal actual (roll, pitch, yaw): [[-0.79 -0.    0.  ]
    [-0.79 -0.    0.  ]
    [-0.79 -0.    0.  ]
    [-0.79 -0.    0.  ]]

    # Transient (roll command shows up when actual yaw value is above certain value)
    # Gimbal actually turns in yaw direction in the simulation with yaw targets applied
    2026-01-31 16:41:39 [150,708ms] [Warning] [isaaclab_tasks.direct.iris_ma4.iris_ma_env4] gimbal targets (roll, pitch, yaw): [[ 0.79 -0.    3.49]
    [-0.79 -0.    3.49]
    [ 0.79 -0.    3.49]
    [ 0.79 -0.    3.49]]
    2026-01-31 16:41:39 [150,708ms] [Warning] [isaaclab_tasks.direct.iris_ma4.iris_ma_env4] gimbal actual (roll, pitch, yaw): [[ 0.    3.49  0.  ]
    [ 0.    3.49  0.  ]
    [ 0.    3.49  0.  ]
    [-0.    3.49 -0.  ]]

    2026-01-31 16:43:10 [241,762ms] [Warning] [isaaclab_tasks.direct.iris_ma4.iris_ma_env4] gimbal targets (roll, pitch, yaw): [[ 0.   -0.    3.49]
    [ 0.   -0.    3.49]
    [ 0.   -0.    3.49]
    [ 0.   -0.    3.49]]
    2026-01-31 16:43:10 [241,763ms] [Warning] [isaaclab_tasks.direct.iris_ma4.iris_ma_env4] gimbal actual (roll, pitch, yaw): [[-0.    1.41 -0.  ]
    [-0.    1.41 -0.  ]
    [-0.    1.41 -0.  ]
    [ 0.    1.41  0.  ]]
    ```