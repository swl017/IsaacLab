# Bisect: post-fix collapse attribution (min_steps vs ticket 029)

**Date**: 2026-04-15
**Purpose**: Attribute the `087be7697c` post-fix collapse to either `min_steps: 2→0` or ticket 029 (dual-cache pipeline + bbox single-field).
**Anchors**:
- `2695ffe1e3_revert_to_2be3` (pre-fix): reward 2587 @ 40k, pair_valid 0.88.
- `087be7697c_revert_to_2be3` (post-fix): reward 913 @ 40k, pair_valid 0.46.

## Design

Two bisect runs, each 50k steps, seed 42, same launch recipe.

| Branch (worktree) | Base | Delta from base | Purpose |
|---|---|---|---|
| `bisect/min_steps` | `2695ffe1e3` | cherry-pick `087be7697c` only → `min_steps: 2 → 0` + FP/FN horizon pushout | Isolate `min_steps=0` |
| `bisect/ticket029` | `fb6fc29778` (ticket 029 commit) | FP/FN horizon hunk cherry-picked; `min_steps` stays at **2** | Isolate ticket 029 |

FP/FN horizon is carried forward on both branches (it's at 2.2M steps — outside the 50k window). So the only live variable between the two branches is `min_steps`.

## Results

| step | run | reward | pair_v | trk_l | bb_c | bb_s | sigma |
|----:|---|------:|------:|------:|----:|----:|------:|
| 40k | pre-fix 2695 (min=2, no-t029) | 2587 | 0.880 | 0.010 | 40.8 | 46.0 | 0.39 |
| 40k | post-fix 087 (min=0, +t029) | 913 | 0.463 | 0.714 | 16.0 | 27.1 | 0.39 |
| 40k | **B1** (min=0, no-t029) | **−6265** | **0.029** | **0.999** | 0.1 | 0.6 | **2.01** (ceiling) |
| 40k | **B2** (min=2, +t029) | **3171** | **0.873** | **0.005** | 43.8 | 45.3 | 0.14 |

Full trajectory by checkpoint in `/tmp/analyze_all.py`.

## Conclusion

**`min_steps: 2 → 0` is the cause of the post-fix collapse.** Ticket 029 is not only innocent, it's a **net improvement** (B2 beats pre-fix by +584 reward at 40k).

- **B1** (min_steps=0 alone, without ticket 029): reward **−6265** @ 40k, pair_valid 0.029, tracking_lost 0.999, sigma maxed out at 2.01. The policy never learned tracking at all. Complete training failure.
- **B2** (ticket 029 alone, min_steps kept at 2): reward **3171** @ 40k, pair_valid 0.873, tracking_lost 0.005, sigma 0.14. Better than pre-fix on every metric except sigma (which compressed more aggressively; the policy exploited the fix more efficiently).
- **Post-fix** (ticket 029 + min_steps=0): reward 913 @ 40k — ticket 029's improvements partially compensate for `min_steps=0`'s damage (B1 −6265 → post-fix 913 = +7178 reward buffer), but can't fully salvage it.

### Why `min_steps=0` breaks training

`LatencyCfg.min_steps` is the floor for step-quantized latency lookups from the circular buffer. With `min_steps=2`, the pipeline always reads a slot at least 2 steps behind the current write, giving the buffer time to warm up and preventing read-after-write races in any mode that enables latency. With `min_steps=0`:

- The latency sampler can clamp `time_lags` to 0, meaning the buffer read returns the just-appended slot.
- In "fixed"/"random" modes, this breaks delay semantics (observation and reward both read the current write, identical no-delay output).
- Even in "none" mode (which short-circuits the latency stage), the LatencySampler is still initialized from the distribution with `min_steps=0`, consuming different random numbers from the global RNG stream and distorting downstream randomization.
- The test `test_min_steps_zero.py` already flagged that `buf[0]` semantics returns the write slot, and that "timestamp regression" at latency jumps is a known contract. Training is not robust to those semantics.

See `delay_system_v3/tests/test_result_min_steps.txt` for the pre-existing validation suite.

## Recommendation

Revert `delay_cfg_v3.py:86` → `min_steps: int = 2` (restore the pre-087 default). Keep ticket 029 (dual-cache pipeline + bbox single-field — both the structural fix and the measured improvement).

The resulting configuration = `2695ffe1e3` + ticket 029 + FP/FN horizon pushout (if desired). Based on B2, this should give reward ≈ 3000–3200 @ 40k with pair_valid ≈ 0.85–0.88 — better than pre-fix.

## Bisect run artifacts (logs)

- B2: [`logs/skrl/iris_ma6/2026-04-15_03-33-46_mappo_rnn_torch_fb6fc29778_bisect_ticket029_only/`](../../../../../../../logs/skrl/iris_ma6/2026-04-15_03-33-46_mappo_rnn_torch_fb6fc29778_bisect_ticket029_only/)
- B1: [`logs/skrl/iris_ma6/2026-04-15_11-55-15_mappo_rnn_torch_560236e08b_bisect_min_steps/`](../../../../../../../logs/skrl/iris_ma6/2026-04-15_11-55-15_mappo_rnn_torch_560236e08b_bisect_min_steps/)

Branches preserved locally (`git branch`): `bisect/min_steps`, `bisect/ticket029`. Delete with `git branch -D bisect/min_steps bisect/ticket029` if no longer needed.
