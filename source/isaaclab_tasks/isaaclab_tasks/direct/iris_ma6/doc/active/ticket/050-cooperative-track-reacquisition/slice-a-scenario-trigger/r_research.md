# Stage R — Codebase Research Report (Slice A)

Facts only. File refs without a module prefix are `iris_ma_env6_test.py`.

## Detection state (where "track" lives)
- Reward path: `_per_agent_bbox_nonempty[N,A]` (bool) via `bbox.abs().sum(-1) > 1e-6` (`:1799–1813`);
  `num_valid_detections[N]` = sum over agents.
- Accumulators `_detection_stats = {total_steps, invalid_all_agents, pair_valid_count}[N]`
  (`:594–598`); incremented `:1811–1813` (`invalid_all += num_valid==0`, `pair_valid += num_valid>=2`);
  reset in `_reset_idx` (`:3349`).
- Obs path: `_vis_bbox_empty[agent]→[N]` from EMA confidence < 0.4 (`:1428–1431`, `:557–560`) — vis only.
- `pair_valid` is the per-step predicate `num_valid_detections >= 2` (`:1813`), not a stored tensor.

## Triangulation
- `TriangulationResult`: `position[N,T,3]`, `covariance[N,T,3,3]`, `quality_metric[N,T]`, `is_valid[N,T]`.
- Reward cov @ GT target (`use_gt_target=True`, `:1820/:1664`); obs cov @ triangulated midpoint (`:2372`).
- Quality modes: trace/sqrt_trace/det/max_eig (`triangulation.py:896–907`).
- Obs tail = 6D `tri_pos(3) ++ tri_std(3)`. Invalid fallback: `tri_pos→zeros` (`:2380–2384`),
  `tri_std→−1.0` sentinel (`:2399–2403`). **No NaN propagates** (corrects parent ticket's "NaN" claim).

## Observation topology (ACTUAL — corrects parent ticket C3)
- Ego 30D; **inter-agent = 16D per other** (`:2186`, `:2250–2283`):
  `pos(3), vel(3), ray_w(3), combined_ang_vel_w(3), zoom(1), bbox_empty(1), data_age(1), bbox_age(1)`.
- `other_ray_w = camera_ray_directions_w[:,0,:]` (`:2268`) = peer's **bbox-based world-frame ray to
  target** (NOT gimbal pointing). Degenerate when peer bbox empty (center=0→points to image corner);
  validity signalled by co-located `other_bbox_empty`.
- Docstring at `:164/:178` ("14D … gimbal_azimuth_world") is **STALE** — this is what the parent
  ticket quoted.

## Loss mechanisms available
- Detector miss (`bbox_raycaster_v2/detector_replicator.py:_apply_miss :282–328`): dual sigmoid in
  `sqrt(w*h)`, sky/ground-conditioned; sets `bbox_empty`. Small/far targets miss more → the "far"
  (zoom) loss path occurs naturally.
- Dropout (`delay_system_v3/`): i.i.d. Bernoulli (`DropoutCfg.probability=0.05`) + burst
  Gilbert-Elliott (`BurstDropoutCfg`). Burst state `(N,A,A)` per directional channel j→i (NOT shared);
  i.i.d. supports per-(env,agent) `Tensor[N,A]` rate (`multi_agent_wrapper.py:1006–1054`).
  → independent per-agent loss confirmed; single-agent loss arises naturally (Q#7).
- On drop: hold-last-value + frozen timestamp → AoI grows (`delay_pipeline_v3.py:631–640`). Dropout
  ages data (grows `bbox_age`); it does NOT set `bbox_empty`. (Two distinct loss signatures.)

## Curriculum
- ~13 sequential phases. i.i.d. dropout = 160k–180k; burst = 200k–220k; FP/FN miss ramp ≈100k–120k;
  target-motion = 40k–80k.
- Single global 0→1 ramp per phase. Floor preserved by expand-range sampling
  `uniform(min, min+p·(max−min))` + per-(env,agent) `Uniform(0, global_p)` (`progress_helper.py`).
- Raise ceiling without floor → increase a phase's `*_max` (e.g. `target_controller_cfg.max_speed_end`,
  `initial_states_cfg.target_distance_max`, `cylinder_diameter_max`), leave `*_min`/`*_start` fixed.
- In-bounds knobs: target `max_speed_start=1→max_speed_end=5`, `update_interval_*`, evasion @30% of
  phase, `enable_z_motion`; init `target_distance 10→40`, `cylinder_diameter 20→100`, spread 2→5,
  zoom max 3→4. Target behavior profiles (kamikaze/evasive/stealth) defined but hardcoded to
  `["standard"]` (`target_controller.py:581`). `max_lin_vel`/slew OUT (sysid lock).

## Metrics & TensorBoard
- `experiments/metrics/metric_tracker.py` (eval): visibility_ratio, tri_valid_ratio,
  `track_loss_count` (valid→invalid transitions), `max_track_gap`, trace_sigma, rmse, success.
  All TEAM-AGGREGATE; none condition on "a peer still saw it." No per-agent peer-assisted-deficit
  / re-acquisition metric exists anywhere.
- `timeseries_tracker.py`: per-step visibility, tri_valid, distance_to_target, target_speed,
  viewing_angle, cbf_penalty (+ valid-only rmse, sqrt_trace_sigma).
- Training TB hook: env → `self.extras["log"][key]` (`:3355–3357`); existing
  `Detection/all_invalid_rate`, `Detection/pair_valid_rate` (`:3328–3334`); skrl trainer forwards
  via `track_data("Info / <key>", v)` (`train_mappo_rnn_hydra.py:299–302`).

## Key tensors for Slice A
| Attr | Shape | Meaning | Loc |
|---|---|---|---|
| `_per_agent_bbox_nonempty` | [N,A] bool | per-agent valid detection (reward path) | :1799–1813 |
| `_detection_stats[*]` | [N] | episode accumulators | :594/:1811 |
| `other_ray_w` | [N,3] | peer target bearing (valid iff other_bbox_empty=0) | :2268 |
| `TriangulationResult.covariance` | [N,T,3,3] | fused-estimate covariance | triangulation.py |

## Gaps / inconsistencies (factual)
1. Stale obs docstring (:164/:178) — peer channel is 16D w/ bbox-ray bearing, not 14D gimbal.
2. Two loss signatures: detector miss sets bbox_empty; dropout grows bbox_age (stale-but-present).
3. No NaN at single-agent loss (zeroed + −1.0 sentinel).
4. No per-agent peer-assisted-deficit / re-acquisition metric exists.
5. Target behavior profiles defined but disabled (`["standard"]`).
6. Dropout independence holds (per-channel burst state; per-(env,agent) i.i.d. rate).

## Design-shaping conclusions (carry into Stage I)
- Peer bearing already in obs (#1) → narrows Slice C premise.
- Single-agent loss does not NaN (#3) → de-risks Slice A instrumentation.
