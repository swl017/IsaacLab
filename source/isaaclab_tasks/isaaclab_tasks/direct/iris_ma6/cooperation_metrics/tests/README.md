# cooperation_metrics Test Suite

Tests for `ReacquisitionTracker` (ticket 050, Slice A) — the peer-assisted track-loss /
re-acquisition instrument.

## Test Files
- **run_tests.py**: standalone runner (no pytest; Isaac Sim compatible).
  - Effective-track gate: AoI=0 reduces to in-frame detection; staleness trips the gate.
  - Event logic: mid-loss deficit + successful re-acquisition; sub-`tau_min` flicker rejected;
    failed hold (re-loss before `tau_hold`) not counted as success.
  - Cause tagging: cold (episode-start) vs mid-loss; dropout (nonempty-but-stale).
  - Peer-assisted precondition: simultaneous loss by all agents is not a deficit.
  - Aggregates: team-track maintenance fraction; multi-env/agent batch; idempotency; reset.

## Running
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/cooperation_metrics/tests/run_tests.py
./isaaclab.sh -p .../run_tests.py --test-verbose
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p .../run_tests.py   # CPU only
```

## Expected Results
All tests pass on CPU and GPU. The tracker is measurement-only — it does not touch obs/reward.

## Conventions
`test_result.txt` (overwritten each run) and `error_log.txt` (appended) track the iterative
debug loop per the project test protocol.
