# Experiment: <2026-03-21_21-20-19_mappo_rnn_torch_0004dcd9df_max_lin_vel_curriculum>

**Commit**: `0004dcd9dff1f0b160dc717f1b29062a45ccd239`
**Date**: 2026-03-21_21-20-19
**Experiment ID**: `a1_with_aoi`
**Base**: `40e862f68d3d3bdc6a39530cb86ca1329ef5a696`<2026-03-21_20-03-47_mappo_rnn_torch_40e862f68d_test_wo_triangulation>

---

## 1. Hypothesis
Start from lower speed to achieve higher `pair_valid_rate` at 20k step

## 2. Configuration Delta
max lin vel 10m/s flat -> 5m/s to 10m/s curriculum

## 3. Results

### 3.1 Training Curves
Achieve higher `pair_valid_rate` at 20k step

### 3.2 Behavior Observations
Didn't check

## 4. Analysis / Open question
Easier task, faster learning.
Does this translate to faster learning overall, including the increasing difficulty with curriculum?

## 5. Decision / Todo
Keep max lin vel curriculum.
