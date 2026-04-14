## Ticket: Gate detection-pipeline split behind a config flag (post-mortem of 120k cliff)

**What**: Introduce `detection_pipeline_split: bool = True` on `UnifiedDelaySystemCfg` (or the nearest equivalent on `delay_system_v3`). When `False`, `get_delayed()` routes both `use_noise=False` and `use_noise=True` calls through a single shared `DelayPipelineV3` per perspective for `bboxes_2d` fields — i.e. the pre-ticket-020 behavior where the noisy observation path is silently overridden by the clean-GT reward path. When `True` (default), retain the current split (`_ego_detection_pipelines` / `_other_detection_pipelines`).

**Why**: The `2695ffe1e3_revert_to_2be3` post-mortem identified a pre-existing shared-pipeline idempotency-guard bug at `2be3b9a3d9`: because `_get_rewards` (L365) runs before `_get_observations` (L384), the clean reward call hits `advance()` first and caches the clean result into both `_cached_data_no_dropout` and `_cached_data_with_dropout`; the second call from `_get_observations` with `use_noise=True` hits the `_last_advance_time == t_current` guard and returns early, so `query()` serves the cached clean data. The Gaussian bbox noise ramped by `_noise_scale` (100k–120k) was silently discarded during 2be3 training. Ticket 020 fixes this by giving `bboxes_2d` their own detection pipelines with independent buffers and idempotency guards — which means observation-path corruption is now real for the first time, including for 2be3-era curriculum phases (Gaussian bbox noise, fixed/random delay, IID dropout). This is the most likely cause of the universal 120k cliff in every post-`3461991973` run.

**Hypothesis under test**: With `detection_pipeline_split=False` and everything else at the current commit, the observation path should again return cached clean data from the reward call (reproducing 2be3-era behavior for `bboxes_2d`). The 120k fixed-delay cliff should disappear and training should track the `2be3` trajectory through 220k. If it does, ticket 020's bug fix is confirmed as the effective-difficulty inflection — the fidelity stack additions (detector replicator, FP/FN, burst dropout) are not the root cause of the cliff on their own, they're amplifiers of a curriculum that was previously silently defanged.

**Scope boundary**:
- Add the flag to the delay-system cfg; thread it through `UnifiedDelaySystem.__init__` and `get_delayed()`.
- When `False`: do NOT instantiate `_ego_detection_pipelines` / `_other_detection_pipelines`; route detection queries through the regular `_ego_pipelines` / `_other_pipelines` just like pre-ticket-020.
- `set_dropout_rate`, `set_delay_mode`, `reset`, `step`, and `_all_pipelines()` must skip the detection dicts when the flag is off (no-op branch, not conditional iteration).
- Leave `set_dropout_rate_by_perspective()` behavior intact — that fix is orthogonal to the pipeline split.
- Do NOT change any detector-replicator, burst-dropout, FP/FN, or curriculum code.
- Do NOT remove the split path — this is a diagnostic flag, not a rollback.

**Affected files**:
- `delay_system_v3/delay_cfg_v3.py` — add flag
- `delay_system_v3/delay_system_v3.py` — conditional pipeline instantiation and routing
- `delay_system_v3/multi_agent_wrapper.py` — forward the flag
- `iris_ma_env6_test_cfg.py` — surface the flag at env-cfg level so runs can toggle it without code changes

**Key constraint**: The detection pipelines currently have independent sampler instances. When the flag is off, the observation path must share the reward path's pipeline *exactly* — same buffer, same idempotency guard. Anything less than that doesn't reproduce pre-ticket-020 behavior.

**Verification**:
1. **Teleop smoke (direct bug reproduction)**: Run `teleop_iris_ma6.py` with `set_noise_scale(1.0)` manually forced (teleop's default leaves `_noise_scale=0`, which masks the bug even when flag=off). With `detection_pipeline_split=False`, the yellow overlay should match green GT regardless of noise scale — that confirms the pre-fix override is reproduced. With flag on, the overlay should diverge from GT. Also verify with FP/FN enabled: flag off should hide FN-blink/FP-offset, flag on should show them.
2. **Parity run**: 20k-step headless run with flag off, seed-matched to 2be3. Check `pair_valid_rate` at 8k/16k/20k matches 2be3 (≥0.90 at 16k). Since `_noise_scale` is 0 pre-100k, flag state shouldn't matter for this phase — this is a sanity check that the flag doesn't break anything unrelated.
3. **Full training**: 220k run with flag off. The decision criterion is pair_valid at 140k. 120k is where fixed-delay onset starts (first curriculum event that ticket 020 actually reveals to observations). If pair_valid stays above ~0.75 through 140k with flag off, the cliff is confirmed ticket-020-induced. If it collapses anyway, the cause is elsewhere — proceed to ticket 017–019 audit.

**Estimated impact**: Diagnostic only. No training/performance change expected when flag is on (default).

**Flow**: Direct implementation. Decoupled from tickets 017–019 audit.

**Parent**: post-mortem in `doc/experiments/2026-04-14_01-23-40_mappo_rnn_torch_2695ffe1e3_revert_to_2be3.md` §6 step 1.
