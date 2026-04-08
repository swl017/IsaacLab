## Codebase Research Report

### Module inventory

| Module | File | Purpose |
|---|---|---|
| `DropoutCfg` | `delay_cfg_v3.py:118-142` | Config: enabled, sampling, probability, rate_distribution |
| `DropoutSampler` | `sampling_strategies.py:338-437` | i.i.d. Bernoulli mask generator. Shape (num_envs,) |
| `PerAgentDropoutSampler` | `sampling_strategies.py:764-872` | Multi-agent variant. Exported but NOT integrated in pipeline V3 |
| `DelayPipelineV3` | `delay_pipeline_v3.py:37-572` | 4-stage pipeline. Owns one DropoutSampler. `_apply_dropout()` at lines 440-476 |
| `UnifiedDelaySystem` | `delay_system_v3.py:30-365` | Per-field pipeline registry. `set_dropout_rate()` broadcasts to all pipelines |
| `MultiAgentDelaySystemV3` | `multi_agent_wrapper.py:61-909` | Per-agent heterogeneity. `set_dropout_rate()` applies base+offset per agent |
| `CurriculumCfg` | `curriculum_cfg.py` | Dropout phase: 160k-180k. `get_dropout_progress()` returns [0,1] |

### Data models

- Dropout mask flow: CurriculumCfg → effective_rate → DropoutSampler._rates → sample_mask() → _apply_dropout()
- Two-path: query(allow_dropout=True) for obs, query(allow_dropout=False) for rewards
- Pipeline instantiation per field-type: ego motion (no dropout), ego detection (dropout enabled), other agents (dropout enabled)
- advance/query split: advance() calls _apply_dropout() with sampler, query() returns cached data

### Conventions observed

- Sampler interface: sample_mask() → Tensor[bool], set_rate(float), reset(env_ids), initialize()
- Config composition: DropoutCfg → DelayPipelineCfgV3 → PerspectiveCfg → UnifiedDelayCfgV3 → MultiAgentDelayCfgV3
- Curriculum control: single set_dropout_rate(rate) called every step
- Per-agent heterogeneity: additive offsets from dropout_offset_range
- Reset: env calls delay_system.reset(env_ids) + randomize_per_agent_params(env_ids) at lines 2035-2036

### Gaps and inconsistencies

1. PerAgentDropoutSampler exported but unused — candidate for deletion
2. No shared dropout state between pipelines on same channel — factual observation about architecture
