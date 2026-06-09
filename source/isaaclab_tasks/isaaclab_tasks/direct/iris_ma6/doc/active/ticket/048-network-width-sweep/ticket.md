## Ticket 048 — Network width sweep (MLP encoder + GRU hidden, MAPPO-RNN)

**Status**: Proposed (queued, no scheduling commitment — see §Method for sequencing options).
**Created**: 2026-06-02
**Type**: Agent-cfg sweep + A/B validation. No env-side code changes.
**Target setup**: iris_ma6 training, post-046/047 cfg state if scheduled after those; otherwise post-045. Pegasus SITL + PX4 SITL transfer remains the load-bearing case.
**Deliverable**:
1. Three agent-cfg sweep entries varying `hidden_size` and `gru_hidden_size` proportionally: baseline (current 64/64), mid (128/128), wide (256/256). `gru_num_layers` held at 1.
2. Sequential A/B at 200k × seed=42 per config (~24 wall-h each, ~72h total). Single-seed v1; multi-seed re-run is a follow-up if the headline result is marginal.
3. One-page experiment writeup in `doc/experiments/` comparing sample efficiency, final-policy quality, smoothness, and wall-time cost across the three widths.
4. Cfg-default flip of `agents/skrl_mappo_rnn_cfg.yaml` to the width that lands the best Pareto point (quality vs cost), gated on sweep acceptance.

**What**: The current MAPPO-RNN cfg uses `hidden_size = gru_hidden_size = 64`, with the comment in [skrl_mappo_rnn_cfg.yaml:11](../../../agents/skrl_mappo_rnn_cfg.yaml#L11): *"Tune it for robust final policy later. May take longer to train."* That tuning hasn't happened. The model's default constructor signature in [mappo_rnn.py:1096](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1096) uses 256/256/2 — the cfg was deliberately downsized to 64/64/1 for training speed early in the project, and never revisited. The t044 + t045 results provide a fresh motivation to revisit it:

- t045's 9-call probe showed 100% slew saturation across every channel × every checkpoint × every curriculum stage. The policy uses 100% of the available bandwidth budget on bbox-tracking demands and has nothing left for cooperative bearing maneuvers.
- One *possible* explanation (not the only one) is **policy capacity**: a 64-wide MLP + 64-wide GRU may not have enough representational capacity to learn a *strategy* that allocates slew bandwidth across single-agent tracking AND multi-agent geometry simultaneously. Saturation could partly be a "single-objective optimization" symptom from a network that can only encode one priority at a time.
- Tickets 046 (task-difficulty) and 047 (reward retune) attack the *demand* side. Ticket 048 attacks the *capacity* side. The two paths are complementary, not mutually exclusive.

**The three sweep configurations**:

| Config | hidden_size | gru_hidden_size | Approx. params (per agent, obs=54) | Ratio to baseline |
|---|---|---|---|---|
| `baseline` (current) | 64 | 64 | ~33k | 1.0× |
| `mid` | 128 | 128 | ~123k | ~3.7× |
| `wide` | 256 | 256 | ~480k | ~14.5× |

`gru_num_layers = 1` held across all three to avoid entangling depth with width. Depth (2-layer GRU) is a separate axis the cfg comment also flags; deferred to a follow-up ticket if width sweep alone doesn't move the needle.

**Why hidden_size + gru_hidden_size *together***: the MLP encoder produces a `hidden_size`-dim input to the GRU, which then expands to `gru_hidden_size`. If encoder is narrow and GRU is wide, the encoder becomes the bottleneck (and vice versa). Holding the ratio fixed (1:1) at each scale isolates "more capacity" as a single variable without introducing bottleneck artifacts. Decoupled sweeps (e.g., 64/128 or 128/64) are a possible follow-up if 1:1 width turns out to plateau.

**Why `gru_num_layers` held at 1**: 2-layer GRUs have known training instabilities under PPO (vanishing-gradient through the recurrent depth interacting with the clipped surrogate). The cfg's own note ("May take longer to train") refers to this. Keeping depth at 1 isolates the width effect cleanly. Depth-sweep is its own thing.

**What we expect to see (hypothesis, not assumption)**:
1. **Sample efficiency**: wider networks fit faster *per env step* but may need more steps to converge because of larger optimizer-state mass. Expect mid and wide to *trail* baseline early (≤ 40k) and *match-or-exceed* by 120k+.
2. **Triangulation recovery**: wider networks have more capacity to encode "use bandwidth for cooperative bearing." If capacity was the bottleneck, mid or wide should show triangulation > 21 (t045 plateau). If saturation is purely demand-side, all three widths will plateau identically and the ticket's value is in the negative result.
3. **Smoothness**: the slew clip is the bandwidth ceiling regardless of capacity. `slew_sat_*` should stay near 1.0 on at least the velocity channels. Wider networks may *redistribute* saturation (less on some channels, more on others) but not eliminate it.
4. **Wall-time per env step**: roughly 1.5-2× slower for wide (256/256/1) vs baseline (64/64/1) on a 3090 — the GRU forward+backward dominates, and the GRU's quadratic cost in hidden_size is the binding factor. Mid (128/128) should be ~1.2-1.5×.
5. **Training stability**: KL band should stay healthy with no LR retune (the KL-adaptive scheduler self-corrects). If KL drifts > 0.04 sustained, the wider network's effective learning rate is too high relative to its parameter count → LR retune becomes a follow-up requirement.

### Patch summary

1. **`agents/skrl_mappo_rnn_cfg.yaml`** — no default change in Slice 1. The cfg-default flip is gated on Slice-2 acceptance.

2. **`experiments/experiment_registry.py`** — three Slice-1 entries:
   - `validation_net_width_baseline`: cfg unchanged (hidden_size=64, gru_hidden_size=64, gru_num_layers=1). The current shipping default. May be omitted if the latest baseline run (t045 or post-046/047) is already seed-42-aligned.
   - `validation_net_width_mid`: `models.policy.hidden_size: 128`, `models.policy.gru_hidden_size: 128`, mirrored to `models.value.*`.
   - `validation_net_width_wide`: `models.policy.hidden_size: 256`, `models.policy.gru_hidden_size: 256`, mirrored to `models.value.*`.

   Hydra override paths confirmed against the existing `sweep_agents_n*` pattern in the registry which uses `models.policy.hidden_size` / `models.policy.gru_hidden_size` overrides successfully.

3. **No env-side code changes.** No new flags. No env_cfg changes (this ticket changes the *agent* cfg, not the env cfg). Slew clip, prev-action obs, spawn geometry, reward weights, EKF lag — all unchanged.

### Slices

**Slice 0 — None.** No empirical pre-flight. The motivation is qualitative (t045 saturation evidence + the long-standing cfg note).

**Slice 1 — Register three experiments.**
- Add the three registry entries. AST + Hydra-path validation.
- No new unit test needed — the model construction path is exercised by every existing training/eval test. A 30-second smoke run of each config to confirm the env builds + the model loads is enough.

**Slice 2 — Sequential A/B (3 runs × 200k).**
- All three runs use seed=42 to keep the seed-noise floor constant. Multi-seed re-run is a follow-up if results are within seed noise.
- Wall-time budget: ~24h × 3 = ~72h. Schedule as sequential (single GPU), not parallel.
- Metrics tracked (in addition to the standard Slice-2 metric panel):
  - **Sample efficiency**: reward curve at {40k, 80k, 120k, 160k, 200k}. Higher area under curve = better sample efficiency. The 40k value isolates "warmup speed" (where wider networks may trail).
  - **Final-policy quality (load-bearing)**: triangulation, bbox_center, pair_valid_rate, tracking_lost — measured at 200k. The acceptance test for "did capacity unlock anything."
  - **Smoothness redistribution**: per-channel `slew_sat_*` — do wider networks saturate differently or just identically?
  - **Wall-time / step**: derived from event-file timestamps (TB writes wall time per scalar). Expected ratio: wide ≈ 1.5-2.0× baseline, mid ≈ 1.2-1.5× baseline.
  - **Training stability**: KL band, value_loss tail, σ trace. Any sustained drift outside the healthy band is a stop-the-run trigger.
- Acceptance bar (a "successful" sweep delivers a Pareto frontier; the bars are guidance, not pass/fail):
  - At least one widened config (mid or wide) shows **triangulation ≥ 32** at 200k (a 50% recovery vs t045's 21 plateau, mid-bar between t045 and t043 baseline).
  - Smoothness preserved: `total_rms` within ±10% of the corresponding baseline run.
  - No new safety regression (`collision_fraction ≤ 2× t043 baseline`).
  - Wall-time / step on the chosen Pareto point ≤ 2.5× baseline (operational ceiling — bigger is fine if it ships, but anything more is a sim2real pain).

**Slice 3 — cfg-default flip + writeup (conditional on Slice-2 Pareto winner).**
- If mid (128/128) is the Pareto winner: flip cfg default. Update session docs.
- If wide (256/256) is materially better than mid (e.g., +5 triangulation points): still consider mid as the shipping default because of the 1.5× wall-time cost. The writeup should explicitly state the Pareto trade.
- If baseline is the Pareto winner (i.e., width didn't help): no cfg flip. The ticket's value is in the documented negative result — and the t045 saturation story becomes "demand-side bottleneck, not capacity-side." This *strengthens* the case for Ticket 049 (loosen slew, deferred).

### Method (training-time validation)

This is a queued ticket — when to schedule depends on the user's path through 046/047:

- **Option A — schedule before 046+047**: gives a clean capacity-vs-demand attribution. If width alone recovers triangulation, 046+047 may be unneeded. But the prior chain shows demand-side issues are real, so this ordering risks running an expensive sweep on the wrong axis.
- **Option B — schedule after 046+047**: each widened config benefits from the lowered task difficulty AND retuned reward, so a successful width result is also a clean "best of all worlds" runner. Adds clarity to the Pareto frontier.
- **Option C — schedule in parallel with 047**: doable if there's a second GPU. Otherwise 047 is faster (1 run × 24h vs 3 runs × 24h).

My recommendation: **B**. The user's stated priority is reducing task difficulty first; network width is exploration on a different axis and benefits from being layered on top of a cleaner baseline. But this is a sequencing call, not a load-bearing decision — the ticket itself is independent.

Run sequence (whichever option is chosen):
1. Apply registry patch. Smoke-test the three configurations build without errors.
2. Launch `validation_net_width_baseline` (if not already covered by a prior seed-42 run).
3. Launch `validation_net_width_mid`. Monitor at 40k / 80k / 120k for early stop on stability issues.
4. Launch `validation_net_width_wide`. Same monitoring.
5. Synthesize the Pareto comparison. Decide cfg-default flip.

### Acceptance criteria

| Criterion | Result |
|---|---|
| Three registry entries land + AST clean | pending |
| Smoke run: all three configs build the env + load the model without error | pending |
| Slice-2 sweep completed (3 × 200k @ seed=42) | tensorboard |
| At least one widened config shows `triangulation ≥ 32` at 200k | tensorboard |
| Widened config(s) preserve smoothness (`total_rms` within ±10% of baseline width) | tensorboard |
| `collision_fraction ≤ 2× t043 baseline` on all three runs | tensorboard |
| Wall-time / step on the chosen Pareto winner ≤ 2.5× baseline | event-file analysis |
| `doc/experiments/<date>_ticket048_network_width_sweep.md` writeup landed with the Pareto comparison | written |
| Cfg-default flip applied (if Pareto winner is mid or wide) | pending |

### Scope boundary

- **DO**: sweep `hidden_size` and `gru_hidden_size` jointly at 1:1 ratio across three scales (64/128/256). Hold `gru_num_layers = 1`. Hold all other agent-cfg knobs constant (LR, sequence_length, rollouts, mini_batches, KL thresholds, etc.). A/B + writeup. Flip default if Pareto winner is not baseline.
- **DO NOT**: change `gru_num_layers` (depth). Deferred to a follow-up if width sweep plateaus.
- **DO NOT**: change `sequence_length`. Affects RNN truncated-BPTT span and is its own dimension.
- **DO NOT**: retune LR, KL bands, or mini-batches for the wider configs. KL-adaptive scheduler should self-correct; if it doesn't, LR retune is a follow-up (separate ticket).
- **DO NOT**: change env-side cfg (slew clip, spawn, rewards, EKF lag). 048 is an agent-cfg-only sweep.
- **DO NOT**: mix in architectural changes (Transformer, residuals, layer-norm in MLP). Width sweep is the experimental unit.
- **DO NOT**: revert any prior ticket (043, 044, 045, 046, 047).

### Risk

Low to medium.

1. **Wider networks fit slower under the same LR** — the optimizer step on a 14×-bigger parameter set takes longer to settle. Slice-2 may show wide trailing baseline at 200k even if it would eventually outperform. Mitigation: monitor reward curve shape. If wide is still climbing strongly at 200k (positive slope), extend to 400k for that config only. If wide is flat by 160k, the run has converged at its capacity-limit. Add to the writeup either way.

2. **Bigger network exacerbates compute cost** — at 256/256/1, wall-time per env step roughly doubles, GPU memory grows. Mitigation: if the wide run can't fit at `num_envs = 1024`, drop to `num_envs = 512` for that config alone. Document the trade in the writeup. Slice-2 num_envs alignment must be tracked or the sample-efficiency comparison is muddled.

3. **KL bands drift** — the KL-adaptive scheduler may not pick the right LR for a 256/256 network because the per-step KL distribution shifts with network width. Watch KL trace: if it sustains > 0.04 for > 20k steps, kill the run; LR retune becomes a precursor. Mitigation: documented in the run log; not a v1 fix.

4. **Wider doesn't help** (the headline negative result). Mitigation: the writeup positions this as *evidence* that the t045 saturation bottleneck is demand-side, not capacity-side. Strengthens the case for Ticket 049 (loosen slew). Not wasted compute — the negative result has documentation value.

5. **Wider helps too much without rewards retune** — wide config might learn a policy that ignores bbox tracking and overweights triangulation under the *current* reward weights (where triangulation is severely under-weighted at 5:60). If 048 is run *after* 047, this is moot. If run *before* 047, watch bbox metrics carefully — sudden bbox_center collapse with triangulation overshoot means the capacity unlocked a degenerate strategy under the imbalanced reward, not a robust one.

6. **Existing checkpoints incompatible** — pre-048 checkpoints are 64-width. Loading into a 128 or 256 config requires fresh training. Forward-only training; not blocking.

### Coupling

- **Ticket 044** (slew clip, landed) — width sweep operates *under* the slew clip. The clip is unchanged; what changes is the policy's capacity to allocate the clipped bandwidth.
- **Ticket 045** (envelope downshift, landed) — sweep operates under the t045 envelope or whichever spawn/reward regime is chosen at scheduling time.
- **Ticket 046** (closer spawn + 2D target, scheduled before 048 in Method §Option B) — wider networks may *better exploit* the freed slew bandwidth that 046 unlocks. Composes additively.
- **Ticket 047** (reward retune, scheduled before 048 in Method §Option B) — wider networks need a balanced reward to allocate capacity sensibly. 047 first makes 048 cleaner.
- **Ticket 049** (loosen slew, deferred) — 048's negative result, if it lands, strengthens the case for 049 (capacity wasn't the bottleneck → demand-side ceiling is real → the slew clip itself is the structural constraint).
- **Trainer infrastructure** — the existing MAPPO-RNN trainer in `mappo_rnn.py` is unchanged. Models are constructed from cfg dictionaries already (see `MAPPORNNPolicy.__init__` signature accepting `hidden_size` / `gru_hidden_size`); no plumbing additions needed.

### Affected files

**Edits (Slice 1)**:
- [experiments/experiment_registry.py](../../../experiments/experiment_registry.py) — register `validation_net_width_{baseline, mid, wide}`.

**Edits (Slice 3, conditional cfg-default flip)**:
- [agents/skrl_mappo_rnn_cfg.yaml](../../../agents/skrl_mappo_rnn_cfg.yaml) — flip `models.policy.hidden_size` and `models.policy.gru_hidden_size` (and matching `models.value.*`) to the Pareto winner. Remove or update the "Tune it for robust final policy later" comment that motivated this ticket.

**New**:
- `doc/experiments/<date>_ticket048_network_width_sweep.md` — Slice-2 writeup with Pareto comparison.

### References

- [agents/skrl_mappo_rnn_cfg.yaml:10-21](../../../agents/skrl_mappo_rnn_cfg.yaml#L10) — current width settings (64/64/1) + the "Tune later" comment that motivates this ticket.
- [mappo_rnn.py:1093-1117](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1093) — `MAPPORNNBaseModel.__init__` showing default 256/256/2 — the original design target that the cfg downsized away from.
- [mappo_rnn.py:1186-1212](../../../../../../../scripts/reinforcement_learning/skrl/mappo_rnn.py#L1186) — `MAPPORNNPolicy` construction; accepts hidden_size / gru_hidden_size / gru_num_layers as constructor args.
- [t045 probe summary](../../../experiments/scripts/probe_action_delta_t045_2026-06-01_21-57-15/summary_pivot.txt) — 100% slew saturation evidence; one of the two hypotheses for the saturation is capacity-bound, this ticket tests it.
- [Ticket 044](../044-action-slew-rate-clip/ticket.md) — slew clip context; preserved under the sweep.
- [Ticket 046](../046-closer-spawn-and-2d-target-motion/ticket.md), [Ticket 047](../047-reward-retune-bbox-vs-triangulation/ticket.md) — task-difficulty + reward-shape predecessors. The Pareto comparison in 048's writeup will reference whichever of these are landed at scheduling time.
- `sweep_agents_n*` entries in `experiments/experiment_registry.py` — existing pattern for `models.policy.hidden_size` / `models.policy.gru_hidden_size` overrides; this ticket follows the same convention.

**Flow**: Low to medium. Three small registry entries + three sequential 200k runs + one synthesis writeup. Estimated 4 days: (a) Slice-1 registry patch + smoke (1 hour); (b) Slice-2 sequential launch (~72 wall-h); (c) Pareto analysis + writeup; (d) conditional Slice-3 cfg-default flip.
