# Ticket 043 — Prev-action obs + first-order LP on cmd_vel

**Status**: Slice 1 landed (patch + unit tests). Slice 2 launches pending (3 × 200k runs).
**Date**: 2026-05-26
**Base commit**: 02c34fe802 (ticket 040/042 pegasus-matched params)
**Branch**: iris_ma6-training

## Summary

Two architectural smoothness levers landed on the iris_ma6 env, both behind feature flags so the pre-patch trajectory remains opt-in:

1. **Prev-action obs (`enable_prev_action_obs`)** — append the previous applied filtered command (7D: vx, vy, vz, yaw_rate in physical units; gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate normalized) to each agent's ego observation. Bumps per-agent obs dim by +7 (47 → 54 at A=2, no triangulation). Source is `_cmd_vel_filt` so the obs channel reflects what the controller actually consumed, not the raw normalized action — this keeps the channel semantically uniform across the batch under per-(env, agent) curriculum + DR jitter on `_max_lin_vel`, and matches the m/s units the deployment policy emits through MAVROS.

2. **First-order LP on cmd_vel (`enable_action_lowpass`)** — per-channel EMA with `α[ch] = 1 − exp(−dt/τ[ch])`, `dt = sim.dt * decimation = 0.04 s`. τ defaults: 0.08 s on vel xy/z & yaw_rate (cutoff ≈ 2 Hz), 0.04 s on gimbal channels (stiff loop), 0.10 s on zoom (visual stability). Filtered command overwrites `cmd_vel` so `_apply_action` consumes it. `_actions` (raw normalized policy output) is left untouched so the existing `action_delta` reward continues to penalize the policy's raw decision, not the filter output.

Default `IrisMA6TestEnvCfg` flips both flags `True`. With both flags `False`, `cmd_vel` equals `action * scale` byte-for-byte (verified in Slice 1).

## Patch

- [iris_ma_env6_test_cfg.py](../../iris_ma_env6_test_cfg.py) — new cfg block (`enable_prev_action_obs`, `enable_action_lowpass`, six per-channel τ defaults). Conditional obs-dim bump in `__post_init__`: `obs_dim += 7` when `enable_prev_action_obs`.
- [iris_ma_env6_test.py](../../iris_ma_env6_test.py) — env-side state:
  - `__init__`: `_cmd_vel_filt: (N, A, 7)` zeroed; `_lp_alpha: (7,)` precomputed from cfg τ and `sim.dt * decimation`.
  - `_pre_physics_step`: after the raw scaled `cmd_vel[:, idx, :]` write, branch on `enable_action_lowpass` (EMA + writeback) / `enable_prev_action_obs` (passthrough mirror) / neither (no-op, bit-exact).
  - `_get_observations` (both delay-system and GT paths): append `_cmd_vel_filt[:, idx, :]` to `ego_obs` *before* inter-agent concat, gated on `enable_prev_action_obs`. The ego tail layout becomes `[…existing 31D…, prev-action (7D)]` followed by inter-agent dims.
  - `_reset_idx`: `_cmd_vel_filt[env_ids] = 0.0` next to the existing `cmd_vel[env_ids] = 0.0`.
- [tests/test_action_smoothness_v1.py](../../tests/test_action_smoothness_v1.py) — Slice 1 unit test (12 checks).
- [experiments/experiment_registry.py](../../experiments/experiment_registry.py) — three Slice-2 experiments registered:
  - `validation_action_smoothness_short_baseline` — both flags off (pre-patch).
  - `validation_action_smoothness_short_prev_action_only` — prev-action on, LP off (diagnostic).
  - `validation_action_smoothness_short_full` — both on (shipping candidate).

## Slice 1 — Unit test results

```
TICKET 043 — ACTION SMOOTHNESS (prev-action obs + cmd_vel LP) TEST SUITE
--- Category 1 — cfg defaults / obs-dim accounting ---
  ✓ cfg defaults: flags=True, τ values match ticket 043
  ✓ cfg obs_dim: prev-action-off → 47 (regression); on → 54 (+ tri = 60)
--- Category 2 — pure-math LP tests ---
  ✓ LP convergence: x→0.9817 after 4τ (8 steps) ≈ 1−e⁻⁴
  ✓ large-τ attenuation: α=0.0392 → first step 0.0392, ~4τ converges
  ✓ sampling-rate invariance: dt=0.04 → 0.993262, dt=0.02 → 0.993262, both = 1−exp(−t/τ)
--- Category 3a — env buffers / α derivation ---
  ✓ buffers + α: _cmd_vel_filt=(4, 2, 7), α=[0.3935, 0.3935, 0.3935, 0.3935, 0.6321, 0.6321, 0.3297]
--- Category 3b — flags OFF regression ---
  ✓ flags-off regression: cmd_vel matches action*scale; _cmd_vel_filt untouched
--- Category 3c — LP runtime semantics ---
  ✓ LP first step: cmd_vel == α × cmd_vel_raw, _cmd_vel_filt mirrors
  ✓ LP convergence (12 steps): cmd_vx ≥ 0.95 × target across envs/agents
--- Category 3d — prev-action concat into ego obs ---
  ✓ prev-action in obs: ego[31:38] == _cmd_vel_filt for all agents
--- Category 3e — LP-off passthrough into _cmd_vel_filt ---
  ✓ LP off + prev-action on: _cmd_vel_filt mirrors cmd_vel (passthrough)
--- Category 3f — reset clears filter state ---
  ✓ reset zeros _cmd_vel_filt for reset envs only

Total: 12 | Passed: 12 | Failed: 0
```

Sampling-rate invariance is exact (1e−12 across dt=0.04 vs dt=0.02 at the same physical time) — the `1 − exp(−dt/τ)` discretization is the closed-form continuous-time pole, not an Euler approximation.

## Slice 2 — A/B plan (not yet launched)

| Run | enable_prev_action_obs | enable_action_lowpass | Purpose |
|---|---|---|---|
| baseline | False | False | Pre-patch behavior; bit-exact to t040/t042 baseline |
| prev_action_only | True | False | Isolates the obs-channel contribution |
| full | True | True | Shipping candidate |

All three: 200k steps, seed=42, default curriculum, branched from `02c34fe802`. Acceptance:
- `full` reduces `cmd_vel_delta` ≥ 30% vs `baseline` at 200k.
- `full` task-quality metrics (bbox_center, bbox_size, pair_valid_rate, triangulation_rmse, collision_per_env) within ±5% of `baseline`.
- `prev_action_only` is the diagnostic — if it alone closes most of the smoothness gap, τ can be tuned more aggressively in a follow-up.

## Slice 3 — τ sensitivity sweep (conditional)

Run only if `full` shows directionally correct but insufficient smoothness gain. 100k-step runs at τ_vel ∈ {0.04, 0.08, 0.16} s with gimbal/zoom fixed at default.

## Risks

1. **Obs-dim change invalidates pre-043 checkpoints.** Forward-only; pre-patch checkpoints continue to work with both flags `False`. New runs start from scratch.
2. **τ defaults too aggressive (over-smoothing) or too lax (no effect).** Slice 2's task-quality bar and `prev_action_only` diagnostic catch both failure modes.
3. **RunningStandardScaler refits to the new 7 dims.** Brief settling expected at training start; no special handling needed.
4. **First-policy-step transient at reset.** `_cmd_vel_filt` is zeroed at reset → first step attenuated by `α ≈ 0.39` at τ=0.08. Documented; not a bug — consistent with "no prior command exists."

## Coupling

- **Ticket 040** (Pegasus physics parity, landed) — τ defaults are tuned for 040's matched-controller gain set.
- **Ticket 042** (per-channel EKF lag, landed) — orthogonal. Prev-action obs operates on the env's outgoing command, not the EKF-lagged ingoing state.
- **Ticket 002** (reward weight retuning) — follow-up. After 043, `action_delta_penalty_scale` and bbox weights should be revisited against the new action geometry.

## Next steps

1. Launch the three Slice-2 runs sequentially (~24 wall-h each).
2. Tensorboard pull at 40k / 80k / 120k / 200k for early-stop on obvious regression.
3. Update this writeup with Slice-2 results; decide on Slice-3 sweep.
