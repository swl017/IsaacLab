# Stage I — Design Document (Slice A): Cooperation Trigger + Re-acquisition Instrumentation

## Problem statement
Training never creates single-agent track-loss events (cause C1), so there is no gradient for
peer-assisted recovery. Slice A (a) raises the scenario *ceiling* so the target leaves individual
FOVs and enables independent per-agent detection loss (manufacturing the hand-off), and (b) adds
first-class track-loss / re-acquisition instrumentation. It changes NO reward and NO obs — only
makes the event occur and measurable. The gate is event existence at a non-trivial rate, NOT
recovery success (recovery is Slice B).

## Proposed approach
Two independently-flagged, default-off components, composing with the t034 per-(env,agent)
anti-forgetting sampler so the easy floor is preserved automatically.

**Component 1 — `enable_track_loss_scenario` (ceiling raise).** When on, raises only `*_max`/end
values of in-bounds knobs (target behaviour + initial conditions) and activates INDEPENDENT
per-agent detection loss within the cooperative training window:
- far-target sub-mode (recovery needs zoom-in): raise `initial_states_cfg.target_distance_max`
  and initial-zoom ceiling so the target pixel size drops below the detector-miss sigmoid for one
  agent at a time.
- edge-of-frustum sub-mode (recovery needs gimbal slew): raise `target_controller_cfg.max_speed_end`
  and shorten `update_interval_*_end` so the target crosses a single agent's FOV boundary; optionally
  enable the already-defined-but-disabled non-standard behaviour profiles.
- dropout sub-mode: bring the existing INDEPENDENT per-(env,agent) i.i.d. detector-dropout/miss into
  the cooperative window (currently late 160k-180k), channels independent (no shared mask) →
  P(exactly one of 2 lost)=2p(1-p) naturally.
Floor untouched (`*_min`/`*_start` fixed); flag off = bit-exact baseline.

**Component 2 — `cooperation_metrics/` (new module, `ReacquisitionTracker`).** Per-(env,agent)
stateful tracker consuming the reward-path detection signal; emits gate metrics to `extras["log"]`
(training TB) and to the eval `MetricTracker`.

## Effective-track signal — reachable-set-within-FOV gate (no time constant)
Track-loss is defined by a physical validation gate, NOT an arbitrary staleness time. A detection
is actionable only while a max-speed target's reachable set still fits the camera FOV:

    d_i(t) = (bbox_i nonempty) AND ( v_max * bbox_age_i  <  k * R_i * tan(FOV_eff_i / 2) )

- v_max: curriculum max target speed (known per env).
- R_i: range to target — privileged GT range (legitimate; this is a metric), avoids monocular ambiguity.
- FOV_eff_i: effective half-FOV from zoom (already computed for ray unprojection / ego `effective_hfov`).
- k: dimensionless FOV-fraction, default 1.0 (literal "could be outside frame"); the only residual
  choice, a sensitivity knob, not seconds.
This unifies both loss modes: at AoI=0 the reachable set is a point and the gate reduces to "is the
detection in frame" (FOV-exit); as AoI grows it is the probabilistic version (dropout staleness).
Back-solved equivalent τ = (R/v_max)·tan(FOV_eff/2) is an OUTPUT, adaptive per-event (zoom-in → lose
faster). It is a standard tracking/association gate — defensible on the ticket's estimation spine.

## Re-acquisition metric
Per agent i, team detection d∈{0,1}^A:
- Peer-assisted deficit interval = maximal run where d_i=0 AND ∃ j≠i: d_j=1.
- Onset tag: cold (starts at t=0) | mid-loss (starts on 1→0). mid-loss sub-tagged far/edge/dropout
  (dropout ⇐ gate failed by staleness while bbox nonempty; far/edge ⇐ bbox empty, small vs out-of-bounds).
- Qualifying only if interval length ≥ τ_min (event flicker filter).
- Success = interval ends with d_i→1 held ≥ τ_hold; time_to_reacq = t_end − t_onset.
- τ_min, τ_hold: event-counting hygiene only (~couple hundred ms defaults; metric insensitive away
  from flicker regime; report sensitivity sweep).

Emitted (per cause + aggregate): `Coop/track_loss_event_rate` (per-episode qualifying mid-loss count
= GATE METRIC), `Coop/reacq_success_rate`, `Coop/time_to_reacq_mean`, `Coop/team_track_maintenance`
(frac steps Σd≥2). Plus diagnostics to disambiguate emergent behaviour: inter-agent distance change
and ego-bearing-to-peer alignment within recovery windows.

Proposed gate threshold (Q#5, confirm in S/P): mean qualifying mid-loss events ≥ 0.5 / episode at
the ceiling, with cold deficits also present.

## Key interfaces and data flow
```
_get_rewards()  [per step]
  ├─ existing: _per_agent_bbox_nonempty[N,A]
  ├─ gather bbox_age[N,A] (delay AoI), R[N,A] (GT range), FOV_eff[N,A] (zoom), v_max[N]
  └─ ReacquisitionTracker.update(detected, bbox_age, R, fov_eff, v_max, t)   [WRITE once/step]
_reset_idx(env_ids)
  ├─ ReacquisitionTracker.episode_summary(env_ids) → extras["log"]["Coop/..."]   [READ]
  └─ ReacquisitionTracker.reset(env_ids)
evaluate.py / MetricTracker ← same per-step signal, post-hoc aggregation
```
- `cooperation_metrics/` standalone (no iris_ma6 deps); module scaffold per qrispy Appendix B.1.
- Scenario knobs: new ceiling fields on existing cfgs gated by `enable_track_loss_scenario`; dropout
  window shift via `curriculum_cfg`. No new cross-module dependency.

## What this does NOT include
Reward change / bbox-90 rebalance (B); new obs / explicit peer-bearing / EKF (C — peer bearing
`other_ray_w` already in obs); architecture (D); `max_lin_vel`/slew (sysid lock); privileged-critic
GT-target actor wiring (optional, deferred); >2-agent scaling.

## Open risks
1. far/edge tagging needs target pixel projection at onset; if not cheap, collapse to single `fov`
   cause for Slice A.
2. Ceiling raise may dent floor performance despite anti-forgetting; short A/B (flag off vs on) at
   easy settings to confirm no regression.
3. Low recovery rate is EXPECTED & acceptable (no Slice-B reward → no incentive); gate is event
   existence, not reacq_success_rate.
4. 2-agent rarity: "peer retains while one loses" may be rarer than with 3; gate-verification run
   confirms rate, else fall back to 3 agents.
5. GT range in the metric is privileged — fine for measurement, but must NOT leak into obs/reward.
