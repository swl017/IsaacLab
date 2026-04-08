## Design Document: Burst Dropout (Gilbert-Elliott Model)

### Problem statement

The delay system's dropout is i.i.d. Bernoulli per step. Real wireless links exhibit bursty loss (5-20 consecutive frames). A correlated loss model is needed for robust sim-to-real transfer.

### Proposed approach

Gilbert-Elliott two-state Markov chain per directional communication channel. Channel j→i has independent state: Good (low dropout ~1%) or Bad (high dropout ~90%). New `BurstDropoutSampler` conforms to existing sampler interface. Flag-based switching (`burst_enabled`). Shared sampler at MultiAgentDelaySystemV3 level passes masks to individual pipelines. Curriculum-gated: p_onset ramped from 0 after i.i.d. dropout phase. Observations only, not rewards.

### Key interfaces and data flow

- `BurstDropoutSampler`: owns Markov state (num_envs, num_agents, num_agents). advance() → transitions + caches. sample_mask(i,j) → cached slice.
- Integration: MultiAgentDelaySystemV3 owns sampler, advances in set_time(), passes masks to pipelines via get_delayed().
- Config: BurstDropoutCfg nested in MultiAgentDelayCfgV3.
- Curriculum: burst_dropout_start/end_step in CurriculumCfg, p_onset ramped.

### What this does NOT include

- No AoI/staleness changes, no latency spikes, no per-field burst granularity, no reward-path burst.

### Open risks (resolved)

1. O(agents^2) state: acceptable at current scale.
2. Burst mode replaces per-pipeline i.i.d. dropout (engineer decision).
3. External mask passed as argument to advance() (option A, engineer decision).
