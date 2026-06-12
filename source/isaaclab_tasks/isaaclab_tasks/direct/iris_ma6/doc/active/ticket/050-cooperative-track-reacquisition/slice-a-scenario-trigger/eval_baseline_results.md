# Slice A — gate verification + C2 baseline (eval on t048 wide policy)

**Date:** 2026-06-12
**Command:**
```
./isaaclab.sh -p experiments/evaluate.py \
  --checkpoint logs/skrl/iris_ma6/2026-06-11_05-17-30_mappo_rnn_torch_ticket048_net_width_wide/agent_drone_0_final.pt \
  --task Isaac-Iris-MA6-Direct-Test-v0 --headless --num_envs 256 \
  --track_loss_scenario --step 400000 --output /tmp/t050_sliceA_baseline.json
```
Policy loaded for INFERENCE (competent tracker) so peer-assisted deficits actually occur.

## results["reacquisition"] (601 completed episodes)
| metric | value | meaning |
|---|---|---|
| track_loss_event_rate | **2.05** | GATE (target ≥0.5) — **PASS, 4× over** |
| reacq_success_rate | **0.385** | **C2 baseline** — Slice B must beat this |
| time_to_reacq_s | 0.46 | mean recovery time |
| team_track_maintenance | 0.53 | fraction of steps with ≥2 agents tracking |
| cold_deficit_rate | 0.19 | per-episode cold (episode-start) deficits |
| event_rate_fov | 2.03 | ≈ all losses are FOV-exit |
| event_rate_dropout | 0.018 | dropout sub-mode contributes little at these settings |
| event_rate_far / edge | 0 / 0 | tag_far_edge=False + target_pixel_inbounds not fed (deferred) |
| reacq_dist_delta_mean | -0.086 | agent does NOT close in on peer |
| reacq_bearing_align_mean | 0.798 | on recovery, ego bearing ≈ aligns with a peer's |

## Interpretation
- **C1 solved**: scenario manufactures single-agent peer-assisted loss (2.05/ep). Cooperation gradient exists.
- **C2 baseline = 0.385**: the selfish policy recovers ~39% of deficits. Slice B target: raise this.
- **Q#13 prediction confirmed**: naive "re-point toward peer bearing" emerges (align≈0.80); sophisticated
  "close in to peer" does NOT (dist_delta≈−0.09). The latter is what the Slice-B covariance reward should pay for.
- Losses dominated by FOV-exit; the dropout sub-mode is minor at current ceilings (could raise dropout_prob
  or feed target_pixel_inbounds for far/edge split — both follow-ups).
