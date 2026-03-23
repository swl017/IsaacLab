# Experiment: <2026-03-23_17-33-02_mappo_rnn_torch_67dfcdf4ce_obs_redesign>

**Commit**: `67dfcdf4ced25e14a09ea56a6ffeba06aa20ac12`
**Date**: 2026-03-23_17-33-02
**Experiment ID**: `a1_with_aoi`
**Base**: `0004dcd9dff1f0b160dc717f1b29062a45ccd239`<2026-03-21_21-20-19_mappo_rnn_torch_0004dcd9df_max_lin_vel_curriculum>

---

## 1. Hypothesis
Observation redesign: ray directions, raw gimbal joint angles for both ego and other observation were missing, so
adding them must help finding/tracking the target.

## 2. Configuration Delta


## 3. Results

### 3.1 Training Curves
Metric: `pair_valid_rate`
- Similar until 80k
- Much better at 80k-100k(noise curriculum)
- Worse from 100k(delay curriculum)
- **But still the performance degrade in the 20k-80k region** 

Metric: `drone_0_triangulation`
- Dominate until 100k
- Worse from 100k
- Half the score at 200k

Metric: `Total timesteps (mean)`
- Survives almost the entire episodes(~500 stpes), whereas the base experiments are only around half(200~300 steps)

### 3.2 Behavior Observations
Was not tested

## 4. Analysis / Open question
- `drone_0_triangulation` tells the hypothesis is only half correct. The change makes this experiment dominate in 
the triangulation until 100k(noise), but performs worse under delay. However, inter-agent observation IS the heart of
the multi-agent algorithm, so we will keep this change.
- `pair_valid_rate` still degrades in the 20k-80k region. Contrast to iris_ma5(`logs/skrl/iris_ma5_ablations_2026-02-28/2026-02-24_00-45-58_a1_with_aoi_seed42`),
iris_ma6's tilting body dynamics seems to be significantly harder to learn. At the tile limits, even the gimbal 
stabilization controller can't keep the LOS stable, etc.

## 5. Decision / Todo
Seperate the curriculums for ego max lin vel to target max lin vel.