## Ticket: Reward weight tuning for action smoothness

**What**: The trained policy (928b9585f2, 400k steps) produces high-frequency oscillations across all action dimensions during deployment. The parity report (`doc/experiments/parity_report.png`) shows rapid switching in velocity commands (vx, vy, vz) and gimbal outputs (gimbal_az, gimbal_el, zoom) at both untrained and trained checkpoints. The trained checkpoint (step 800k, right column) still exhibits dense oscillatory control despite stable value estimates.

**Why**: Oscillatory actions are not deployable to real drones — actuator wear, energy waste, and unstable gimbal footage. The current action smoothness penalties (`action_delta_penalty_scale = -1.0`, `action_sum_penalty_scale = -2.0`) are dwarfed by task rewards (`bbox_center_reward_scale = 60.0`, `bbox_size_reward_scale = 60.0`, `triangulation_reward_scale = 5.0`), so the policy has little incentive to produce smooth commands. This is the primary blocker for sim-to-real transfer.

**Scope boundary**:
- Do NOT change the reward function structure (reward terms stay the same, only weights change)
- Do NOT change the curriculum schedule or level transitions
- Do NOT change the network architecture or learning rate schedule (928b9585f2 fix is validated)
- Do NOT add new reward terms

**Affected modules**:
- `iris_ma_env6_test_cfg.py` — reward weight fields (lines 367–411)
- `iris_ma_env6_test.py` — `_get_rewards()` (lines 1079–1445) for understanding reward assembly
- `doc/reward_spec.md` — for authoritative reward term definitions

**Acceptance criteria**:
- Action delta (frame-to-frame change) RMS decreases by at least 50% compared to 928b9585f2 baseline
- Task performance (triangulation d0, bbox metrics) does not degrade by more than 15%
- Parity report visually shows smoother action trajectories without dense high-frequency switching

**Flow**: Full QRISPY
