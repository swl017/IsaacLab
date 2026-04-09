## Ticket: Separate detection buffer for bbox observation path

**What**: Add separate detection pipelines in `UnifiedDelaySystem` for `bboxes_2d` fields so the noisy observation path has its own latency buffer, independent of the raw reward path.

**Why**: The delay pipeline's `advance()` method has an idempotency guard that runs only once per sim step. Because rewards (`use_noise=False`) are computed before observations (`use_noise=True`), the single shared pipeline always populates its latency buffer with clean GT bboxes. The noisy observation path then returns this clean data instead of the detector-replicator output — silently bypassing all FN misses, calibrated localization noise, and FP injection. The yellow "delayed bbox" overlay in teleop was identical to green GT bbox, confirming the bug.

**Root cause**: In `UnifiedDelaySystem.get_delayed()`, both `use_noise=False` and `use_noise=True` calls routed to the same `DelayPipelineV3` instance. The pipeline's circular buffer was filled by whichever call arrived first (rewards), and the idempotency guard prevented the second call (observations) from appending its own data.

**Fix** (all in `delay_system_v3.py`):
- Added `_ego_detection_pipelines` and `_other_detection_pipelines` dicts, populated only for fields with suffix `bboxes_2d`
- `get_delayed()` routes to detection pipeline when `use_noise=True` and one exists
- All bulk operations (`set_delay_mode`, `set_dropout_rate`, `reset`, `step`, etc.) include detection pipelines via `_all_pipelines()` helper
- Detection pipelines use the same config as regular pipelines; in `fixed` mode (teleop, late curriculum) delay values are deterministic and identical

**Related fixes in this session**:
1. **Delay mode not active in teleop** (`teleop_iris_ma6.py`): Uncommented `set_delay_mode("fixed", progress=1.0)` — env starts with `mode="none"` for curriculum, teleop never overrode it
2. **Detector replicator was a no-op** (`detector_replicator.py`): When `params_path=""` (default), `_params` stayed `None` and `apply()` returned immediately. Fixed to fall back to hardcoded `NoiseModelParams()` defaults
3. **Burst dropout zeroed ego i.i.d. dropout** (`multi_agent_wrapper.py` + `delay_system_v3.py`): `set_dropout_rate()` zeroed ALL pipelines when burst sampler was active, but burst only applies to "other" perspective. Added `set_dropout_rate_by_perspective()` to preserve ego dropout

**Files modified**:
- `delay_system_v3/delay_system_v3.py` — detection pipeline dicts, routing, bulk ops
- `delay_system_v3/multi_agent_wrapper.py` — `set_dropout_rate()` ego/other split
- `bbox_raycaster_v2/detector_replicator.py` — `NoiseModelParams()` fallback
- `tests/teleop_iris_ma6.py` — delay mode override uncommented

**Design notes**:
- Detection pipelines have independent sampler instances. In `fixed` mode, latency is deterministic from config so both paths produce identical delay. In `random` mode, they draw from the same distribution independently — statistically equivalent, not identical per-step
- Only `bboxes_2d` fields get detection pipelines (controlled by `_DETECTION_SUFFIXES`). All other fields continue sharing a single pipeline per perspective
- These detection pipelines can later be replaced by real YOLO inference buffers, where GT is still the geometric projection from bbox raycaster

**Verification**: Run `python teleop_iris_ma6.py --num_envs 2 --enable_camera --test_gimbal_lock`. Yellow bbox should blink (FN misses), show noise offset from green GT, and display nonzero AoI.
