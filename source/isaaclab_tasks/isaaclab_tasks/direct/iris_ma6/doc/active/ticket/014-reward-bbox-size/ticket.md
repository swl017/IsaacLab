## Ticket: Re-evaluate bbox size reward term — drop or replace with learned objective

**What**: The `bbox_size_reward` (scale=60.0) rewards the policy for maintaining target bbox area at ~20% of the image. This heuristic is problematic: (1) the optimal bbox size depends on the detector replicator's miss rate curve — smaller targets get missed more, so the policy should zoom in more than 20% suggests, (2) the optimal size is not fixed but depends on detector parameters that may change with calibration, (3) the size reward competes with the triangulation reward which benefits from geometric diversity (agents at different distances/angles). Consider dropping bbox_size_reward and letting the policy learn optimal zoom from the miss rate signal and triangulation quality.

**Why**: With the detector replicator active (ticket-010), there's a natural learning signal for zoom: if the target is too small, detections are missed (FN), triangulation degrades, and the policy gets lower triangulation reward. The policy should discover its own optimal zoom level from this signal rather than being told "20% is good." The heuristic bbox_size_reward may conflict with what the detector model actually requires.

**Evidence**:
- Ticket-010 teleop testing: replicator miss rate depends on target pixel size. The "optimal" size is wherever the miss rate becomes acceptably low — this is a function of the sigmoid parameters, not a fixed 20%
- `bbox_center_reward` (centering) can be justified — centering implies correct anticipation of target motion. But `bbox_size_reward` has no such justification — it's a proxy for "be at the right distance" which triangulation already rewards
- Current reward scale: bbox_center=60, bbox_size=60, triangulation=5. The size reward dominates the triangulation signal

**Scope**:
1. **Ablation**: Train with `bbox_size_reward_scale=0.0` (disabled) and compare against baseline. Metrics: pair_valid_rate, triangulation error, zoom behavior
2. **Analysis**: Does the policy discover a natural zoom level from miss rate + triangulation signals alone?
3. **Decision**: Based on ablation, either drop bbox_size_reward permanently or replace with a detector-aware reward (e.g., reward proportional to detection confidence or inversely proportional to miss probability)

**Scope boundary**:
- Do NOT change bbox_center_reward (centering is justified for anticipation)
- Do NOT change triangulation reward structure
- Do NOT change the detector replicator (ticket-010/013 scope)
- This is an ablation + decision ticket, not a guaranteed code change

**Affected modules**:
- `iris_ma_env6_test.py` — `_get_rewards()` bbox_size_reward term
- `iris_ma_env6_test_cfg.py` — `bbox_size_reward_scale`
- `experiments/experiment_registry.py` — new ablation experiment

**Dependencies**:
- Ticket-013 (calibration refit) should complete first so the replicator matches real YOLO — ablation results against wrong replicator are not meaningful

**Acceptance criteria**:
1. Ablation experiment defined and run (with/without bbox_size_reward)
2. Metrics compared: pair_valid_rate, triangulation RMSE, average zoom level, detection rate
3. Decision documented with data

**Flow**: Light (I → S → Y → PR) — single ablation experiment
