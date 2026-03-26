# Experiment: <2026-03-26_14-56-04_mappo_rnn_torch_3da98de790_no_collision_termination>

**Commit**: `3da98de79003823eddb75b8d74d081726bbfeb03`
**Date**: 2026-03-26
**Experiment ID**: `a1_with_aoi`
**Base**: `37371694949b81ac5996d087a04dda5b2aa4fb03`<2026-03-26_04-39-35_mappo_rnn_torch_3737169494_tracking_curriculum>

Training in process 72k.

---

## 1. Hypothesis
The pair_valid_rate cliff at ~20k is caused by collision termination creating a death spiral:
shorter episodes → less bbox reward → conservative policy → can't track → worse performance.
Removing collision termination (penalty only, like MA5) should eliminate or soften the cliff.

Also reverted lambda_cbf to 1.0 (from 10.0) and tracking_end_step to 60k (from 150k) to
return to the cleanest baseline config with only the termination change.

## 2. Configuration Delta
- Removed collision termination (collision now penalty-only, episode continues)
- `lambda_cbf`: 10.0 → 1.0 (revert to original)
- `tracking_end_step`: 150k → 60k (revert to original)
- `collision_penalty_scale`: -100 (kept, matching MA5)
- Vectorized initial_states_generator (per-agent randomization)

## 3. Results (72k/200k)

### 3.1 Training Curves
**Metric: `pair_valid_rate`**
- Peaks at 0.85 at 16k, dips to 0.31 at 42k, then **recovers** — 0.81 at 72k.
- This is the first MA6 experiment where pair_valid_rate recovers from the cliff.
- Previous experiments (e36e8ca123, 5d2b8f8eca, 4dc2f44b42) showed permanent decline or weak recovery.
- The dip at 36-42k correlates with a collision spike (see below).

**Metric: `all_invalid_rate`**
- Low (1-4%) during 0-34k, spikes to 24.6% at 36k, then oscillates 1-19%.
- Volatile but trending down. At 72k: 1.5%.

**Metric: `Total timesteps (mean)`**
- Full 499 steps during 0-18k (no collision termination = full episodes!).
- Sudden drop to ~51 at 22k, gradual decline to ~16 at 72k.
- **This is concerning** — episodes are still getting cut short by some other termination
  (likely OOB). Needs investigation.

**Metric: `drone_0_triangulation`**
- Zero until 60k (coordination curriculum), then rapid ramp to 3.18 at 72k.
- Already outperforming several previous experiments at the same step count.

**Metric: `drone_0_bbox_center`**
- Rises to ~29 by 20k, dips to 21 at 36k, then **stays elevated** (18-32 range).
- At 72k: 26.7. In previous experiments, bbox_center decayed to 13-15 by 70k.
- The reward signal is NOT collapsing — the policy maintains tracking quality.

**Metric: `collision_fraction`**
- Elevated vs previous experiments (0.2-0.3% vs 0.02-0.04%) — expected without termination.
- Large spike at 36k (5.7%), coinciding with the pair_valid_rate dip.
- Trending down after 36k: the policy is learning to avoid collisions from penalty alone.

**Metric: `drone_0_cbf_penalty`**
- Much smaller than previous experiments (-0.07 at 72k vs -0.6 in others).
- `lambda_cbf=1` keeps CBF penalty proportional rather than dominating.

**Metric: `drone_0_collision`**
- Steady ~-0.25 during 0-34k, spike to -3.4 at 36k, then oscillates lower.
- Collision penalty is providing signal without overwhelming other rewards.

**Metric: `drone_0_action_sum`**
- Monotonic increase in magnitude: -5.1 → -8.1 at 72k.
- Higher than previous experiments (-6 to -7 range), suggesting more active flight.
- The policy is NOT learning to be conservative — it's learning to move and track.

### 3.2 Behavior Observations
Not tested yet. (Training in process)

## 4. Analysis / Open question
- **Collision termination was the root cause of the permanent cliff.**
  Removing it allows pair_valid_rate to recover (0.31 → 0.81 at 72k), bbox_center to stay
  elevated (26.7 vs 13-15 in previous experiments), and triangulation to ramp quickly.
- The 36k dip is a transient learning phase where the policy experimented with aggressive
  strategies (collision_fraction spiked to 5.7%), then recovered.
- **Total timesteps declining to 16 is the open question.** Without collision termination,
  what is ending episodes? Likely OOB (out-of-bounds) termination. If agents fly too far
  from the target, they may hit world bounds. This needs investigation — if OOB is too
  aggressive, it could become the new bottleneck replacing collision termination.
- The volatile oscillations in pair_valid_rate and all_invalid_rate (compared to smooth
  curves in previous experiments) may be because the policy now has freedom to explore
  aggressive strategies without the death penalty of episode termination.

## 5. Decision / Todo
- **Investigate OOB termination**: What causes Total timesteps to drop to 16? Check
  `_get_dones()` for non-collision termination conditions.
- ~~Continue training to 200k to see if pair_valid_rate stabilizes above 0.8.~~ No point continuing with early termination.
- If OOB is too aggressive, consider relaxing bounds or adding a soft boundary penalty
  (similar to how collision was changed from termination to penalty).
- This result validates the termination hypothesis — proceed with this config as the new baseline.
- Try with lower z action penalty