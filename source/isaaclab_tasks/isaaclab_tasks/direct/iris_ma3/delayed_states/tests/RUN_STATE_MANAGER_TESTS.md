# Quick Guide: Running MultiAgentStateManager Tests

## Quick Start

```bash
# Navigate to test directory
cd source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delayed_states/tests

# Run all MultiAgentStateManager tests
python run_all_tests.py -k "test_initialization or test_noise or test_update_gt or test_curriculum or test_detection_processing or test_state_consistency or test_multi_agent or test_communication_integration or test_reset or test_camera or test_zoom or test_partial or test_get_all or test_disabled"
```

## Run by Priority

### P0: Critical Tests (5 tests)
```bash
python run_all_tests.py -k "test_initialization or test_noise_reproducibility or test_update_gt_states or test_curriculum_learning or test_detection_processing"
```

**Tests**:
1. ✅ `test_initialization` - Basic setup
2. ✅ `test_noise_reproducibility` - Seeded noise works
3. ✅ `test_update_gt_states` - State updates work
4. ✅ `test_curriculum_learning` - Progress scaling works
5. ✅ `test_detection_processing` - Detection pipeline works

**Expected runtime**: ~5-10 seconds

### P1: Integration Tests (5 tests)
```bash
python run_all_tests.py -k "test_state_consistency or test_multi_agent_integration or test_communication_integration or test_noise_formula_consistency or test_reset_functionality"
```

**Tests**:
1. ✅ `test_state_consistency` - GT→delayed→noisy chain
2. ✅ `test_multi_agent_integration` - Multiple agents work
3. ✅ `test_communication_integration` - Broadcast/receive works
4. ✅ `test_noise_formula_consistency` - Formulas match iris_ma_env3.py
5. ✅ `test_reset_functionality` - Reset works correctly

**Expected runtime**: ~10-15 seconds

### P2: Validation Tests (6 tests)
```bash
python run_all_tests.py -k "test_camera_pose_updates or test_zoom_noise_clamping or test_partial_env_update or test_detection_with_invalid_mask or test_get_all_states or test_noise_disabled"
```

**Tests**:
1. ✅ `test_camera_pose_updates` - Camera poses updated
2. ✅ `test_zoom_noise_clamping` - Zoom clamped to [1, 10]
3. ✅ `test_partial_env_update` - Env-specific updates
4. ✅ `test_detection_with_invalid_mask` - Invalid masks handled
5. ✅ `test_get_all_states` - Batch access works
6. ✅ `test_noise_disabled` - No noise when disabled

**Expected runtime**: ~10-15 seconds

## Run Individual Tests

### Test Noise Reproducibility
```bash
python run_all_tests.py -k test_noise_reproducibility
```

### Test Curriculum Learning
```bash
python run_all_tests.py -k test_curriculum_learning
```

### Test State Consistency
```bash
python run_all_tests.py -k test_state_consistency
```

## Run with Options

### Verbose Output
```bash
python run_all_tests.py -k test_initialization -v
```

### Stop on First Failure
```bash
python run_all_tests.py -k test_noise_reproducibility -x
```

### With Coverage Report
```bash
python run_all_tests.py -k test_update_gt_states --coverage
```

## Expected Output

### Successful Test
```
test_multi_agent_state_manager.py::test_initialization PASSED
✓ Initialization test passed

test_multi_agent_state_manager.py::test_noise_reproducibility PASSED
✓ Noise reproducibility test passed

================================================================================
✓ All tests passed!
================================================================================
```

### Failed Test
```
test_multi_agent_state_manager.py::test_noise_reproducibility FAILED

E       AssertionError: Noise should be identical with same seed
E       assert tensor([[0.0123, ...]], device='cuda:0')
E        != tensor([[0.0456, ...]], device='cuda:0')

================================================================================
✗ Tests failed with exit code: 1
================================================================================
```

## Test File Structure

```
tests/
├── run_all_tests.py                         # Main test runner
├── test_multi_agent_state_manager.py        # NEW: StateManager tests
├── test_channel_buffer.py                   # Existing: Channel tests
├── test_multi_agent_comm.py                 # Existing: Comm tests
└── ...
```

## Common Issues

### Issue: Import Error
```
ImportError: No module named 'isaaclab_tasks.direct.iris_ma3.delayed_states'
```

**Solution**: Run from correct directory or set PYTHONPATH:
```bash
cd /path/to/IsaacLab
export PYTHONPATH=$PYTHONPATH:$(pwd)/source
```

### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `num_envs` in test fixtures or run on CPU:
```python
@pytest.fixture
def basic_config(device):
    return {
        "num_envs": 2,  # Reduced from 4
        ...
    }
```

### Issue: Test Hangs
**Solution**: Check if GPU is available and working:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Debugging Tests

### Run Single Test with Print Statements
```bash
python run_all_tests.py -k test_initialization -s
```
The `-s` flag shows print statements from tests.

### Run with Python Debugger
```bash
python -m pdb run_all_tests.py -k test_noise_reproducibility
```

### Check Test Collection
```bash
python run_all_tests.py --collect-only
```

## Integration with CI/CD

### Run in Headless Mode (Default)
```bash
python run_all_tests.py --headless
```

### Run with Specific GPU
```bash
CUDA_VISIBLE_DEVICES=0 python run_all_tests.py
```

### Generate JUnit XML Report
```bash
pytest test_multi_agent_state_manager.py --junitxml=test_results.xml
```

## Performance Benchmarks

| Test | Expected Time | GPU Memory |
|------|---------------|------------|
| test_initialization | <1s | ~100 MB |
| test_noise_reproducibility | 1-2s | ~200 MB |
| test_update_gt_states | 1-2s | ~200 MB |
| test_curriculum_learning | 2-3s | ~300 MB |
| test_detection_processing | 1-2s | ~200 MB |
| **Total (All 16 tests)** | **15-25s** | **~500 MB** |

## Continuous Testing

### Watch for Changes (requires pytest-watch)
```bash
pip install pytest-watch
ptw test_multi_agent_state_manager.py
```

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
cd source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delayed_states/tests
python run_all_tests.py --priority P0
if [ $? -ne 0 ]; then
    echo "Tests failed! Fix before committing."
    exit 1
fi
```

## Next Steps

After tests pass:
1. ✅ Tests passing → Integration ready
2. 📝 Update documentation if needed
3. 🚀 Test in full environment (iris_ma3_env.py)
4. 🔄 Run regression tests
5. 📊 Profile performance

## References

- Test suite documentation: [MULTI_AGENT_STATE_MANAGER_TESTS.md](MULTI_AGENT_STATE_MANAGER_TESTS.md)
- API documentation: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- Implementation: [delayed_states.py](../delayed_states.py)

---

**Quick Reference**: Just run `python run_all_tests.py --priority P0` to verify critical functionality!
