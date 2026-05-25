## Ticket 042 — Apply ticket 041's per-channel EKF latency to iris_ma6 training-time obs

**Status**: Proposed
**Created**: 2026-05-25
**Type**: Implementation + validation (consumes ticket 041's measured deliverable).
**Target setup**: iris_ma6 training (any task in the registry); Pegasus SITL parity as the load-bearing case.
**Deliverable**:
1. The config patch (drafted, validated, committed to `delay_system_v3/delay_cfg_v3.py` + `iris_ma_env6_test_cfg.py`).
2. A bit-exact regression test for the legacy bulk path.
3. A short A/B training run (200k steps × 1 seed × 2 configs) that confirms the new defaults don't degrade convergence vs. the pre-patch baseline.
4. One-page experiment writeup in `doc/experiments/`.

**What**: Replace iris_ma6's bundled `5±2 ms` ego-motion latency with the per-channel values measured by [ticket 041](../041-px4-ekf-state-lag-measurement/ticket.md) — two regimes:

| Channel | Pre-patch (bulk) | Post-patch (per-channel) | Source |
|---|---|---|---|
| `body_position_w` | 5 ± 2 ms | **0 ± 5 ms** | GPS-fused, lag-compensated (`OutputPredictor`) |
| `body_velocity_w` | 5 ± 2 ms | **0 ± 5 ms** | GPS-fused, lag-compensated |
| `body_orientation_w` (quat) | 5 ± 2 ms | **18 ± 5 ms** | conservative upper bound (yaw 0, roll/pitch 17) |
| `body_angular_velocity_w` | 5 ± 2 ms | **15 ± 5 ms** | IMU-driven (raw IMU + MAVLink) |
| `body_linear_acceleration_w` | 5 ± 2 ms | **35 ± 5 ms** | IMU spec-force + truth-side dv/dt lag |
| First-order TC | 5 ms (uniform) | **0 ms in SITL**; raise to 0.25 s under DR for real-hw | quiescent in lockstep (no IMU bias → no fusion correction) |

A draft patch was authored in the same session that closed ticket 041; it lives at `delay_system_v3/delay_cfg_v3.py` (new per-channel knobs on `DelaySystemKeyParams` + per-field overrides in `create_delay_cfg_from_params`) and `iris_ma_env6_test_cfg.py` (env defaults flipped to the per-channel path). The legacy bulk pipeline remains accessible via `use_bulk_ego_motion_latency=True` for opt-in regression.

**Why**: Ticket 041 measured PX4 EKF2 output-vs-truth lag in Pegasus SITL and found **two regimes** that the current single-bulk model does not represent — half the channels are too lagged (GPS-fused, should be 0) and half are too fresh (IMU-driven, should be 15–35 ms). The measurement is the load-bearing reference for sim2sim parity; ticket 040 (Pegasus physics parity) closes the plant gap and this ticket closes the corresponding EKF-channel gap. Without 042, iris_ma6 trains against a uniformly-aged obs vector that differs measurably from what the deployed policy sees through MAVROS.

The change is also the entry point for real-hardware sim2real prep: when ticket 043 lands (real-hardware EKF re-measurement), the same per-channel surface admits the larger TCs (0.25 s on IMU-driven channels) that real flight controllers exhibit — no schema change needed.

**Blocked on**: nothing. Ticket 041's JSON deliverable is on disk. The config patch is drafted.

**Depends on**:
- [ticket 041](../041-px4-ekf-state-lag-measurement/ticket.md) (DONE) — measurement source. Values quoted above are from `controller/sysid_output/ekf_state_lag/ekf_state_lag.json`.
- [delay-system-v3](../029-delay-system-redesign-DONE/ticket.md) (DONE) — the ringbuffer + field-override architecture that the patch consumes. No changes to internals.

**Distinct from**:
- [ticket 043 (proposed)](../043-real-hw-ekf-lag-remeasurement/) — real-hardware re-measurement. The patch's per-channel surface is designed to admit those values when they land; 042 sticks with the SITL numbers.
- [ticket 044 (proposed)](../044-sysid-bode-fit-imu-channels/) — proper Bode-plot system identification for the IMU-driven channels. 042 uses constant τ; 044 fits a transfer function. Only execute 044 if 042 shows the constant approximation is insufficient.
- [ticket 045 (proposed)](../045-split-orientation-yaw-rp/) — refactor `body_orientation_w` into yaw + roll/pitch fields. 042 uses the conservative 18 ms upper bound on the bundled quat; 045 is only required if that bound visibly hurts policy quality.

### Patch summary (already drafted)

1. **`delay_system_v3/delay_cfg_v3.py`**:
   - `DelaySystemKeyParams` gains 11 new fields: `use_bulk_ego_motion_latency` + 5 `ego_<channel>_latency_mean_s` + 5 `ego_<channel>_fol_tau_s` + `ego_per_channel_latency_std_s`.
   - `create_delay_cfg_from_params` builds per-channel `DelayPipelineCfgV3` overrides keyed by field name (`body_position_w`, `body_velocity_w`, `body_orientation_w`, `body_angular_velocity_w`, `body_linear_acceleration_w`) and injects them into `cfg.delay_cfg.ego.field_overrides` alongside the existing `bboxes_2d` override.
   - `use_bulk_ego_motion_latency=True` skips the per-channel block entirely; the legacy bulk pipeline remains the ego default.
2. **`iris_ma_env6_test_cfg.py`**:
   - `delay_system_params` defaults flipped to `use_bulk_ego_motion_latency=False` and explicit per-channel values from the measurement.
   - Legacy bulk knobs (`ego_motion_latency_*`) retained so explicit-opt-in experiments keep working.

Smoke-tested in the same session — the per-channel mode produces correct field overrides, and the bulk-mode regression produces a config tree identical to pre-patch (only `bboxes_2d` overrides the ego default).

### Slices

**Slice 1 — bit-exact regression test**.
- Write a unit test under `delay_system_v3/tests/` that:
  - Instantiates `DelaySystemKeyParams(use_bulk_ego_motion_latency=True, burst_dropout_enabled=False, ...)` with the pre-patch values
  - Calls `create_delay_cfg_from_params`
  - Computes a stable hash (e.g., `repr(asdict(cfg))`) and compares against a checked-in golden hash
- Same test in per-channel mode with default values produces a *different* known hash (locks the post-patch defaults).
- AppLauncher template per [iris_ma6/CLAUDE.md](../../../CLAUDE.md) "Generating Tests for Functional Modules".

**Slice 2 — short A/B training run**.
- New experiment in `experiments/registry.py`: `validation_obs_v2_short` — 200k steps, 1 seed, 2 configs:
  - `baseline_bulk`: `use_bulk_ego_motion_latency=True` (post-patch code, legacy behavior)
  - `treatment_per_channel`: per-channel values (post-patch default)
- Same task (`Isaac-Iris-MA6-Direct-Test-v0`), same RNG, same curriculum.
- Metrics tracked (via existing metric_tracker):
  - Mean reward (last-10 avg); KL band stability; LR adaptation behavior
  - Task-quality: `bbox_center`, `pair_valid_rate`, `all_invalid_rate`, `track_lost_fraction`, `triangulation_rmse`
  - Safety: `collision_per_env`, `cbf_penalty`
  - Action-smoothness: `action_sum`, `action_delta`
- Acceptance bar: treatment within ±5% of baseline on each metric, OR a clear interpretable improvement. Regressions >5% on any metric require diagnosis.

**Slice 3 — DR sanity for `*_fol_tau_s`**.
- 100k-step run with `ego_angular_velocity_fol_tau_s` and `ego_linear_acceleration_fol_tau_s` set to 0.25 s under domain randomization (mimics real-hardware EKF2_TAU defaults).
- Acceptance bar: KL stays in adaptive band, mean reward within ±10% of treatment_per_channel (Slice 2). Failure here is the early signal that ticket 044 (transfer-function fit) is load-bearing for sim2real.
- Optional — defer if Slice 2 passes cleanly and real-hardware data (ticket 043) hasn't landed yet.

### Method (training-time validation)

1. **Pre-flight**: bit-exact regression test (Slice 1) passes before any training launches.
2. **Slice 2 launch**:
   - Branch from current `main`. Apply the patch (already drafted). Land.
   - Register `validation_obs_v2_short` experiment. Run on the dev machine (~3 wall-hours per 200k-step run at current sim rate).
   - Tensorboard analysis: pull last-10 reward, KL trace, task-quality metrics via the existing experiments framework.
   - Decision rule: if all metrics within ±5%, treatment_per_channel becomes the new default. If any regress >5%, root-cause before merging the patch.
3. **Slice 3** (optional, sequential): same setup with the larger FOL TCs. Same decision rule.

### Acceptance criteria

| Criterion | Result |
|---|---|
| `delay_system_v3/delay_cfg_v3.py` and `iris_ma_env6_test_cfg.py` carry the drafted patch | landed |
| `use_bulk_ego_motion_latency=True` produces bit-exact pre-patch config tree | unit test green |
| Default per-channel config matches the values from `ticket 041`'s `ekf_state_lag.json` | confirmed by smoke test |
| Slice 2 A/B run completes; treatment within ±5% on all tracked metrics | tensorboard + writeup |
| `doc/experiments/<date>_ticket042_per_channel_ekf_latency.md` documents the run | written |
| No regression in any existing experiment in the registry (spot-check on a representative one) | confirmed |
| Slice 3 (optional): IMU-channel FOL TC = 0.25 s training tolerates the rolloff | KL stable, reward within ±10% |

### Scope boundary

- **DO**: land the per-channel config patch; validate via short A/B; document the result.
- **DO**: keep the legacy bulk path opt-in for any future experiment that wants to reproduce pre-patch behavior.
- **DO**: identify, but do not implement, the conditions under which tickets 044 / 045 become load-bearing.
- **DO NOT**: re-tune controller gains. The ego obs lag change can interact with controller bandwidth, but PID re-tuning is a separate ticket (out of scope).
- **DO NOT**: change bbox latency (310 ms), detection FPS staleness, dropout, or inter-agent comms (500 ms) — those are sibling channels with their own sources and were not touched by ticket 041.
- **DO NOT**: implement the orientation-split refactor (ticket 045). The conservative 18 ms upper bound on the quaternion is sufficient until evidence shows otherwise.
- **DO NOT**: measure real-hardware lag (ticket 043). 042 stays in sim.

### Risk

Low to medium.

1. **Larger latency on IMU-driven channels** (5 → 15/18/35 ms) is the main concern. The body_rate channel feeds straight into the policy's rate-controller observation; a 10 ms increase in apparent obs lag could theoretically affect closed-loop stability. Mitigation: Slice 2's KL-band + reward checks catch this. If it manifests, the recourse is to delay-compensate the rate channel (e.g., a Smith predictor) — that's a follow-up ticket, not a blocker for 042.
2. **Zero latency on position / velocity** (5 → 0 ms) is a *reduction* in obs lag. Almost certainly benign; if anything, gives the policy fresher position info. Watch for `bbox_center` improving but `action_delta` increasing (sharper closed-loop).
3. **Field-name mismatch** is the silent failure mode: if the field names in the patch (`body_position_w` etc.) don't match what the env actually registers in `AgentStates`, the per-channel overrides silently fall through to the bulk default pipeline. Mitigation: add an assertion in `create_delay_cfg_from_params` that every entry in `motion_field_overrides` matches a known field name. Easy to add.
4. **Quaternion DC bias** (~9 mrad on yaw, ~4.5 mrad on roll observed in [ticket 041](../041-px4-ekf-state-lag-measurement/ticket.md)) is a separate calibration issue, not a latency issue. Not in scope for 042; documented in 041's notes.

### Coupling

- **Ticket 040** (Pegasus physics parity) — sister sim2sim ticket. 040 closes the plant gap; 042 closes the EKF-channel gap. After both land, the residual sim2sim error has a documented decomposition.
- **Ticket 029 (DONE)** — delay system v3 internals. 042 lives entirely inside the existing architecture; no changes to the pipeline / ringbuffer / dual-cache logic.
- **Tickets 043 / 044 / 045** — successors gated on 042's training results.
- **Experiments framework** — adds one experiment to the registry; existing pipeline runs unchanged.

### Affected files

**Edits** (drafted; need to be confirmed and committed):
- [delay_system_v3/delay_cfg_v3.py](../../../delay_system_v3/delay_cfg_v3.py) — 11 new fields on `DelaySystemKeyParams`; per-channel `field_overrides` block in `create_delay_cfg_from_params`.
- [iris_ma_env6_test_cfg.py](../../../iris_ma_env6_test_cfg.py) — `delay_system_params` defaults flipped to per-channel mode with measured values.

**New**:
- `delay_system_v3/tests/test_per_channel_latency.py` — bit-exact regression + per-channel-mode smoke test (Slice 1).
- `experiments/registry.py` — add `validation_obs_v2_short` (Slice 2).
- `doc/experiments/<date>_ticket042_per_channel_ekf_latency.md` — writeup of the A/B result.

### References

- [ticket 041](../041-px4-ekf-state-lag-measurement/ticket.md) — measurement source.
- `controller/sysid_output/ekf_state_lag/ekf_state_lag.json` — numeric ground truth for the per-channel defaults.
- [delay_system_v3/CONTEXT.md](../../../delay_system_v3/CONTEXT.md) — pipeline architecture.
- PX4 EKF2 source: `OutputPredictor` at [output_predictor.cpp:168](/home/usrg/IsaacPX4/PX4-Autopilot/src/modules/ekf2/EKF/output_predictor.cpp#L168); `EKF2_TAU_POS/VEL = 0.25 s` defaults at [ekf2_params.c:1125,1136](/home/usrg/IsaacPX4/PX4-Autopilot/src/modules/ekf2/ekf2_params.c#L1125).
- [iris_ma6/CLAUDE.md](../../../CLAUDE.md) — test-writing conventions for Slice 1.

**Flow**: Low to medium. The code change is straightforward (already drafted). The risk is in the training validation — a 200k-step run is cheap enough to be the gate, and the legacy bulk path is the rollback. Estimated 1–2 days: (a) commit the patch + test; (b) launch Slice 2 (~3 wall-h); (c) analyze; (d) writeup; (e) optionally Slice 3.
