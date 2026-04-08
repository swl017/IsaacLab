## Open Questions

### Assumptions requiring confirmation

1. **Per-agent vs per-channel vs per-field granularity**: Per-channel (all fields from one comm link drop together) is most realistic. Per-field is unrealistic (transport-layer loss affects all data).
2. **Markov state shape**: Shared `BurstDropoutSampler` object holds state `(num_envs, num_agents, num_agents)` for asymmetric per-directional-channel dropout. Channel j→i independent of i→j.
3. **Relationship to existing DropoutCfg**: Flag-based — `burst_enabled=True` → Gilbert-Elliott, `False` → original Bernoulli. Same interface, easy A/B comparison.
4. **Curriculum phase placement**: Separate phase after i.i.d. dropout (200k-220k). i.i.d. dropout phase (160k-180k) completes first.
5. **Burst dropout and rewards path**: Only for observations, not rewards.
6. **Reset behavior**: Reset Markov state to Good on env reset (for delay system and RNN hidden state warmup).
7. **p_onset and p_recovery ramping**: Both vary — different burst lengths via randomized p_recovery per episode.

### Architectural decisions requiring human input

8. **Sampler replacement vs wrapper**: New `BurstDropoutSampler` class with same `sample_mask()` interface. Pipeline selects sampler based on config.
9. **State persistence across advance/query**: Markov transition in `advance()`, cached mask reused in `query()`.
10. **File placement**: New file `burst_dropout.py` inside `delay_system_v3/`.

### Engineer decisions

- Per-channel with asymmetric directional links (j→i independent of i→j)
- Burst mode replaces per-pipeline i.i.d. dropout on affected channels
- External mask passed as argument to `advance()` (option A)
