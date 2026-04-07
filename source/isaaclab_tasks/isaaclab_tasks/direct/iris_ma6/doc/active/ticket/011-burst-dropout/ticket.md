## Ticket: Burst dropout — correlated packet loss model in delay system

**What**: Replace the i.i.d. Bernoulli dropout in the delay system with a correlated burst dropout model. Real wireless links (WiFi, telemetry radios) drop packets in bursts of 5-20 consecutive frames, not independently per frame.

**Why**: The current delay system applies dropout as i.i.d. Bernoulli per step (`dropout_probability=5%`). This means the policy sees isolated single-frame gaps that are easily interpolated. Real wireless links exhibit bursty loss: when interference or congestion occurs, multiple consecutive packets are lost (5-20 frames at 30 FPS = 170-670ms blackout). The policy needs to handle sustained information blackouts, not just isolated drops. Without burst training, the policy may panic or diverge during real-world link degradation.

**Evidence**:
- Ticket-005 §1.2: "Burst dropout — real wireless drops packets in bursts (5-20 consecutive), not i.i.d." listed as gap
- Ticket-005 §1.2: "Packet loss pattern: i.i.d. dropout, probability=5% — Partial (no burst/correlated loss)"
- Additional gap §3: "Burst dropout — real wireless drops packets in bursts (5-20 consecutive), not i.i.d."
- Current implementation: `DelayPipelineV3` applies dropout via `torch.bernoulli` per step per field, independently

**Scope**:
1. **Burst dropout model**: Implement a Gilbert-Elliott (two-state Markov) model for each communication channel:
   - **Good state**: low dropout probability (e.g., 1%)
   - **Bad state**: high dropout probability (e.g., 80-100%)
   - Transition probabilities: `p_good→bad` (burst onset), `p_bad→good` (burst recovery)
   - Mean burst length = 1 / `p_bad→good` (target: 5-20 steps at observation rate)
   - Mean good length = 1 / `p_good→bad` (target: 50-200 steps)
2. **Per-agent, per-field state**: Each agent's communication channel has its own Markov state (some agents may be in burst while others are fine). Optionally per-field (ego vs inter-agent) or per-channel (all fields from one agent drop together — more realistic)
3. **Configuration**: `BurstDropoutCfg` with `p_onset`, `p_recovery`, `burst_dropout_prob`, `good_dropout_prob`, `enabled` flag
4. **Curriculum gating**: Burst dropout activates after i.i.d. dropout phase (e.g., 200k+ steps). Ramp `p_onset` from 0 to target value
5. **Integration**: Replace or augment the existing Bernoulli dropout in `DelayPipelineV3._apply_dropout()`

**Scope boundary**:
- Do NOT change the delay system architecture (ringbuffer, advance/query split)
- Do NOT modify the AoI or staleness computation — burst dropout is applied after delay, before observation
- Do NOT model heavy-tail latency in this ticket (that's a separate mechanism — latency spikes vs packet loss)
- Keep i.i.d. dropout as a fallback mode (`burst_enabled=False` → original Bernoulli)

**Affected modules**:
- `delay_system_v3/delay_pipeline_v3.py` — `_apply_dropout()` method
- `delay_system_v3/delay_cfg.py` — add `BurstDropoutCfg`
- `iris_ma_env6_test_cfg.py` — wire BurstDropoutCfg into delay config
- `curriculum/curriculum_cfg.py` — burst dropout curriculum phase

**Key references**:
- Gilbert-Elliott model: two-state Markov chain for bursty channel
- Current dropout: `delay_pipeline_v3.py`, `_apply_dropout()` method (Bernoulli per step)
- Delay system calling contract: `delay_system_v3/CONTEXT.md`

**Acceptance criteria**:
1. Gilbert-Elliott model implemented with configurable transition probabilities
2. Per-agent Markov state (agents experience independent burst patterns)
3. Mean burst length configurable (default: 10 steps at observation rate)
4. Curriculum-gated: disabled at early training, ramped after i.i.d. dropout phase
5. Backward compatible: `burst_enabled=False` → identical to current Bernoulli dropout
6. Test: verify burst statistics match configured mean burst length over 1000+ steps

**Flow**: Full QRISPY
