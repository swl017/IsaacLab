# Experiment: 2026-05-26_01-38-37_mappo_rnn_torch_02c34fe802_ticket040_042_pegasus_matched_params

**Commit**: `02c34fe8021a67299639a6caf71d09c15c30e5b5` — "Now velocity step response matches PegasusSimulator/PX4 SITL almost exactly"
**Date**: 2026-05-26 (run started 01:38:37, **in-flight** at time of writing)
**Experiment ID**: ad-hoc (not in `experiment_registry.py`); ticket-040 + ticket-042 manual run
**Base**: `5e4d3f4db4_ticket040_042_pegasus_matched_params` (sibling run from 00:12-25 same day) — same physics_mode/EKF lag config, but trained against the prior `PX4_MATCHED_CONTROLLER_CFG` (default-plant tuned gains, mis-paired with pegasus plant).

---

## 1. Hypothesis

Pairing the pegasus plant (ticket 040) with the **gain set tuned against that plant** ([px4_matched_pegasus.py](../../controller/tuning/tuning_results/px4_matched_pegasus.py), produced this session by `sysid_replicator.py --plant pegasus --ekf-lag`) closes the sim2sim closed-loop gap to PegasusSimulator + PX4 SITL. The prior `5e4d3f4db4` run had the right plant but the wrong gains (default-plant tuned values); sim-to-sim transfer fails and it motivates this run.

Concretely, the sysid_replicator run that produced these gains hit:
- `vel_5_settling_time` gap: **0.4%** (PASS, ticket-040 threshold < 15%)
- `yaw_settling_time` gap: **40%** (PASS, threshold < 50%)
- `vel_5_ss_error` gap: 31.6% (iris-ma6 tighter than PX4, benign direction)
- Score 0.0168 (best of 4 ablation runs)

Hypothesis: with the controller's closed-loop response now matching PX4 SITL, the policy should train cleanly and outperform pre-040 baselines on every tracked metric when transfered to PegasusSimulator with `mas` stack — specifically no transient instability from the over-aggressive racing-class plant's response, and no controller-induced action chatter.

## 2. Configuration Delta

Key deltas from the prior `5e4d3f4db4` run (which shared most of the env cfg):

```python
# Controller gains — flipped from default-plant tuned to pegasus-plant tuned
# via the new __post_init__ auto-pairing in iris_ma_env6_test_cfg.py.
# Effective values (from logs/.../params/env.yaml:512-547):
drone_controller.velocity.Kp_vel   = (1.62,    1.62,    3.8909)   # was default-plant gains
drone_controller.velocity.Ki_vel   = (0.3059,  0.3059,  1.0336)
drone_controller.attitude.Kp_att   = (6.8903,  6.8903,  2.7571)
drone_controller.rate.Kp_rate      = (0.7495,  0.7495,  0.9877)
drone_controller.rate.Ki_rate      = (0.3938,  0.3938,  0.2305)
drone_controller.rate.Kd_rate      = (0.00373, 0.00373, 0.00000)

# Inherited from 5e4d3f4db4 (ticket 040 + 042 active):
physics_mode                       = "pegasus"      # ticket 040 plant (k_f=8.55e-6, omega_max=1100, linear-diag drag)
expected_body_mass                 = 1.5            # USD mass sanity check
delay_system_params.use_bulk_ego_motion_latency = False  # ticket 042 — per-channel EKF lag
delay_system_params.ego_orientation_latency_mean_s     = 0.018
delay_system_params.ego_angular_velocity_latency_mean_s = 0.015
# (pos/vel latency = 0 ms; lin_acc = 35 ms — ticket 041 measurements)
```

Trainer/agent:
```yaml
trainer.timesteps        : 400000           # logs/.../params/agent.yaml:62
agent.experiment_name    : 02c34fe802_ticket040_042_pegasus_matched_params
scene.num_envs           : 1024
env.seed                 : 42
```

## 3. Results

### 3.1 Training Curves

**Status**: in-flight — TB events file still being written (last update 2026-05-26 17:30; run started 01:38). Best checkpoint snapshot timestamp 04:04 (well before mid-training), suggesting reward improved early then plateaued or was overtaken by a later checkpoint. Checkpoints present: `agent_{40k,80k,120k,160k,200k}.pt`.

Metric: `pair_valid_rate` — **TODO** (extract from TB after completion)

Metric: `all_invalid_rate` — **TODO**

Metric: `Total timesteps (mean)` — **TODO**

Metric: `drone_0_triangulation` — **TODO**

Metric: `drone_0_bbox_center` — **TODO**

Metric: `collision_per_env` — **TODO**

Metric: `drone_0_cbf_penalty` — **TODO**

Metric: `drone_0_action_sum` — **TODO**  *(expect lower vs 5e4d3f4db4 — pegasus plant + tuned gains should reduce control activity)*

Metric: `drone_0_action_delta` — **TODO**  *(expect lower — closed-loop bandwidth now matched, less chatter)*

### 3.2 Behavior Observations

**TODO** after completion. Provisional notes:
- The sibling commit's message ("Now velocity step response matches PegasusSimulator/PX4 SITL almost exactly") sets the expectation: this is the first iris_ma6 training run with both **plant** (ticket 040) AND **gain set** (this session's sysid_replicator output) aligned to PegasusSimulator + PX4 SITL.
- USD body mass logged as 1.5 kg at env construction (matches `expected_body_mass`, no divergence warning).

## 4. Analysis / Open question

**TODO** after completion. Hooks to evaluate:

1. **vs `a3f94fdc8b_ticket037_critic_priv_obs_no_dr`** — pre-040 baseline (racing-class plant + default-plant gains). The expected sim2real-faithful regression is some loss of episode return (lower T/W → harder recovery from disturbances); the gain is that the deployed policy will not be surprised by PX4's slower / less authoritative response.

2. **Decision rule for the ticket-040 ±10% acceptance gate** — Compare mean episode return at 200k vs the pre-040 baseline. Within ±10% = ship; >10% drop = investigate whether the policy is fighting the lower T/W in ways that domain randomization could close.

## 5. Decision / Todo

**TODO** after run completes:
- [ ] Extract TB metrics into the §3.1 placeholders above (see [evaluate/](../../evaluate/) or `_isaac_sim/python.sh` + tensorboard for the extraction pattern used in prior experiment docs).
- [ ] If results pass the ±10% gate against pre-040 baseline, mark ticket 040's acceptance criterion 5 (do-no-harm training run) DONE.
- [ ] If results are clean, this becomes the new training baseline for future ticket work — propagate by updating `_template.md`-implied references that point at the old baseline.
- [ ] If results regress, the candidate root causes (ordered by likelihood):
  - Lower T/W ratio (2.6 vs 82) means policies can no longer "muscle through" disturbances → may need domain randomization on Kp_vel_z (low end) to teach robust behavior at the realistic authority level.
  - EKF lag on attitude (18 ms) adds phase lag the prior baselines did not see → may need delay-aware policy (LSTM / recurrence already present — should help).
  - Linear-diag drag (Pegasus) vs quadratic drag (default) has lower drag at low speed and higher at high speed — could affect target-following at the speeds the policy operates at.

## References

- Commit [02c34fe802](https://github.com/anthropics/IsaacPX4) — the gain-set flip plus its supporting infrastructure (sysid_replicator.py extensions, tuning_results/__init__.py cleanup, env cfg auto-pairing).
- Session log: [progress.txt entry for 2026-05-26](../active/progress.txt) — sysid replicator pegasus-plant + EKF-lag integration; full ablation results (v1–v4) and final gain selection.
- Sysid output: [replicator_metrics_pegasus.json](../../controller/sysid_output/analysis/replicator_metrics_pegasus.json), [replicator_*_pegasus.pdf](../../controller/sysid_output/analysis/).
- Exported gains: [px4_matched_pegasus.py](../../controller/tuning/tuning_results/px4_matched_pegasus.py).
- Ticket 040: [040-match-pegasus-physics-parameters/ticket.md](../active/ticket/040-match-pegasus-physics-parameters/ticket.md).
- Ticket 041: [041-px4-ekf-state-lag-measurement/ticket.md](../active/ticket/041-px4-ekf-state-lag-measurement/ticket.md) — `ekf_state_lag.json` (delay values used here).
- Ticket 042: [042-per-channel-ekf-latency-iris-ma6/ticket.md](../active/ticket/042-per-channel-ekf-latency-iris-ma6/ticket.md).
- Run logs: [logs/skrl/iris_ma6/2026-05-26_01-38-37_mappo_rnn_torch_02c34fe802_ticket040_042_pegasus_matched_params/](../../../../../../../../../logs/skrl/iris_ma6/2026-05-26_01-38-37_mappo_rnn_torch_02c34fe802_ticket040_042_pegasus_matched_params/).
