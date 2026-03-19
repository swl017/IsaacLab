# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab is a GPU-accelerated, open-source robotics simulation framework built on NVIDIA Isaac Sim. It supports reinforcement learning, imitation learning, and motion planning for robotics research with sim-to-real transfer capabilities.

## Development Commands

### Main Entry Point
- `./isaaclab.sh` - Main utility script for all Isaac Lab operations
- `./isaaclab.sh -h` - Show help with all available options

### Environment Setup
- `./isaaclab.sh -c [NAME]` - Create conda environment (default: env_isaaclab)
- `./isaaclab.sh -i [LIB]` - Install extensions and RL frameworks (default: all)
- `./isaaclab.sh -v` - Setup VSCode settings

### Development Tasks
- `./isaaclab.sh -f` - Run pre-commit formatting and linting
- `./isaaclab.sh -t` - Run all unit tests
- `./isaaclab.sh -d` - Build documentation

### Running Scripts
- `./isaaclab.sh -p [SCRIPT]` - Run Python scripts with Isaac Lab environment
- `./isaaclab.sh -s [ARGS]` - Launch Isaac Sim with extensions

### Training and Simulation
```bash
# Train with different RL frameworks
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Cartpole-v0
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0

# Play trained models
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Cartpole-v0 --checkpoint /path/to/model

# Run environment demos
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0
./isaaclab.sh -p scripts/demos/quadcopter.py
```

## Code Architecture

### Core Framework Structure
- **source/isaaclab/**: Core framework with simulation, environments, sensors, assets
- **source/isaaclab_tasks/**: Pre-built RL environments (direct and manager-based)
- **source/isaaclab_assets/**: Robot and sensor asset configurations
- **source/isaaclab_rl/**: RL framework integration (skrl, RSL-RL, RL-Games, SB3)
- **source/isaaclab_mimic/**: Imitation learning extensions

### Environment Types
1. **Direct Environments** (`source/isaaclab_tasks/direct/`): Direct physics control with custom reward/observation functions
2. **Manager-Based Environments** (`source/isaaclab_tasks/manager_based/`): Modular environments using manager classes for observations, rewards, commands

### Key Concepts

#### Environment Workflows
- **Direct Workflow**: Inherit from `DirectRLEnv`, implement `_get_observations()`, `_get_rewards()`, `_get_dones()`, `_reset_idx()`
- **Manager-Based Workflow**: Use `ManagerBasedRLEnv` with manager classes for modularity

#### Manager Classes (Manager-Based)
- `ObservationManager`: Handles observation collection and processing
- `RewardManager`: Manages reward computation with multiple terms
- `TerminationManager`: Handles episode termination conditions
- `ActionManager`: Processes and applies actions to assets
- `EventManager`: Manages randomization and curriculum events
- `CommandManager`: Handles high-level commands (velocity, pose targets)

#### Asset System
- `Articulation`: Multi-body robotic systems (manipulators, quadrupeds, humanoids)
- `RigidObject`: Single rigid bodies with collision
- `DeformableObject`: Soft body simulations
- Assets configured via dataclass configs with `_cfg.py` suffix

#### Sensors
- Camera (RGB, depth, segmentation) with RTX acceleration
- Contact sensors for force/torque measurement
- IMU sensors for orientation and acceleration
- Ray casting for LIDAR-like distance measurements
- Frame transformers for coordinate system conversion

### Project Conventions

#### File Organization
- Environment configs: `*_env_cfg.py`
- Agent configs: `agents/` subdirectories with framework-specific files
- Task implementations: Environment name matches directory (e.g., `cartpole/cartpole_env.py`)

#### Configuration System
- Uses dataclasses with `@configclass` decorator
- Hierarchical configs with inheritance
- MISSING values for required parameters
- Runtime config validation and type checking

#### Multi-Framework Support
Each environment supports multiple RL frameworks:
- **RSL-RL**: Configurations in `agents/rsl_rl_ppo_cfg.py`
- **SKRL**: Configurations in `agents/skrl_*_cfg.yaml`
- **RL-Games**: Configurations in `agents/rl_games_*_cfg.yaml`
- **Stable-Baselines3**: Configurations in `agents/sb3_*_cfg.yaml`

### Development Workflow

1. **Environment Creation**: Use `./isaaclab.sh -n` to create new environments from templates
2. **Asset Integration**: Add robot URDF/USD files and create asset configs in `isaaclab_assets`
3. **Agent Training**: Create agent configs for your preferred RL framework
4. **Testing**: Always run `./isaaclab.sh -t` before committing changes
5. **Code Quality**: Run `./isaaclab.sh -f` for formatting and linting

### Common File Locations
- Training logs: `logs/[framework]/[env_name]/`
- Hydra outputs: `outputs/[date]/[time]/`
- Checkpoint saves: `logs/[framework]/[env_name]/[timestamp]/`
- Documentation: `docs/` (Sphinx-based)

### Physics and Simulation
- Built on NVIDIA Isaac Sim (Omniverse/USD)
- GPU-accelerated physics simulation
- Support for rigid bodies, articulated systems, deformable objects
- Ray-tracing based sensors and cameras
- Parallelized environments for faster training

### Testing and Quality Assurance
- Comprehensive test suite in `test/` directories
- Pre-commit hooks for code formatting (black, isort)
- Type checking with pyright
- Documentation building with Sphinx
- Continuous integration workflows

## Important Conventions
- Quaternion: `wxyz` convention

## Generating Tests for Functional Modules

When creating tests for functional modules (e.g., controllers, sensors, safety systems, utilities), follow these guidelines to ensure Isaac Sim compatibility and consistency.

### Test Directory Structure

For any functional module, create a `tests/` subdirectory:

```
my_module/
├── __init__.py
├── my_component.py
├── other_component.py
├── doc/
└── tests/
    ├── __init__.py
    ├── run_tests.py           # Standalone test runner
    ├── test_my_component.py   # Individual test files
    ├── test_other_component.py
    └── README.md              # Single README describing all tests
```

**Reference Example**: See [source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/](source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/safety/) for a complete implementation.

### Required AppLauncher Template

**CRITICAL**: Every test script that imports Isaac Sim or Isaac Lab modules **MUST** include the AppLauncher initialization template at the very beginning, **before any other module imports**.

```python
#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Brief description of what this test file does.
"""
import argparse
from isaaclab.app import AppLauncher

# Add AppLauncher args
parser = argparse.ArgumentParser(description="Run [module name] test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW you can import other modules (torch, isaaclab, etc.)
import torch
import sys
from typing import Dict, List

# Import your module components
from isaaclab_tasks.direct.my_module import MyComponent, MyComponentCfg
```

**Why This Matters**: Isaac Sim requires specific initialization before other libraries are imported. Failing to use this template will cause import errors and crashes.

### Standalone Test Runner (`run_tests.py`)

Create a standalone test runner that:
1. Uses the AppLauncher template
2. Implements a custom test results tracker
3. Runs tests without pytest (to avoid Isaac Sim import conflicts)
4. Provides clear pass/fail reporting

**Template Structure**:
```python
#!/usr/bin/env python3
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run [module] test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import traceback

# Test results tracking
class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
        # Print error details

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print(f"\nTotal: {total}, Passed: {len(self.passed)}, Failed: {len(self.failed)}")
        return len(self.failed) == 0 and len(self.errors) == 0

def run_component_tests(results: TestResults, device: torch.device):
    """Run tests for specific component."""
    print("\n" + "="*80)
    print("Testing MyComponent")
    print("="*80)

    # Test 1: Initialization
    try:
        cfg = MyComponentCfg()
        component = MyComponent(cfg, num_envs=16, device=device)
        assert component.num_envs == 16
        results.add_pass("Initialization")
    except Exception as e:
        results.add_fail("Initialization", str(e))

    # Add more tests...

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = TestResults()

    try:
        run_component_tests(results, device)
    except Exception as e:
        results.add_error("Component suite", traceback.format_exc())

    success = results.print_summary()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

### Test Documentation (README.md)

Generate **exactly one README.md** per test directory that includes:

1. **Overview**: Brief description of what is being tested
2. **Test Files**: List each test file with bullet points of test categories
3. **Running Tests**: Clear instructions with command examples
4. **Test Organization**: Describe test structure (classes, functions, scenarios)
5. **Expected Results**: What passing tests should demonstrate
6. **Common Issues**: Troubleshooting guide
7. **Contributing**: Guidelines for adding new tests

**Example Structure**:
```markdown
# [Module Name] Test Suite

Brief description of the module being tested.

## Test Files

- **test_component_a.py**: Tests for ComponentA
  - Initialization and configuration
  - Core functionality (computation, detection, etc.)
  - Edge cases and boundary conditions
  - Reset and state management

- **test_component_b.py**: Tests for ComponentB
  - [Categories specific to this component]

## Running Tests

### Run All Tests
\`\`\`bash
# Using standalone runner (recommended for Isaac Sim modules)
./isaaclab.sh -p source/my_module/tests/run_tests.py

# Using pytest (if compatible)
./isaaclab.sh -p -m pytest source/my_module/tests/ -v
\`\`\`

### Run Specific Tests
\`\`\`bash
# Single component
./isaaclab.sh -p source/my_module/tests/run_tests.py --component component_a

# With CPU only
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/my_module/tests/run_tests.py
\`\`\`

## Test Organization

[Describe how tests are organized - by class, by functionality, etc.]

## Expected Results

- GPU/CPU compatibility: Tests run on both devices
- Numerical accuracy: Appropriate tolerances for floating-point operations
- Edge case handling: Proper behavior with invalid/boundary inputs
- State management: Reset clears internal state correctly

## Common Issues

### CUDA Out of Memory
Reduce `num_envs` in test fixtures to lower memory usage.

### Import Errors
Ensure module is installed: `./isaaclab.sh -i`

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Ensure all existing tests pass
3. Update this README for new test files
4. Verify tests on both CPU and GPU
```

### Test Best Practices

1. **Device Agnostic**: Tests should work on both CUDA and CPU
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Parameterize Environments**: Use fixtures or variables for `num_envs`
   ```python
   num_envs = 16  # Easy to adjust for memory constraints
   ```

3. **Numerical Tolerances**: Use appropriate tolerances for floating-point comparisons
   ```python
   assert torch.allclose(result, expected, atol=1e-5)
   ```

4. **Test Independence**: Each test should be self-contained and not depend on others
   ```python
   # Create fresh instances for each test
   cfg = MyComponentCfg()
   component = MyComponent(cfg, num_envs, device)
   ```

5. **Clear Test Names**: Use descriptive names that explain what is being tested
   ```python
   results.add_pass("Distance computation with symmetric agents")
   ```

6. **Test Categories**: Organize tests into logical groups
   - Initialization and configuration
   - Core functionality
   - Edge cases and boundary conditions
   - Integration with other components
   - Reset and state management

7. **2-step Scheme**: First test with simple values, then with realistic values when testing the core funcionality and integration.

8. **Error Handling**: Catch exceptions and provide useful error messages
   ```python
   try:
       # Test code
       results.add_pass("Test name")
   except Exception as e:
       results.add_fail("Test name", traceback.format_exc())
   ```

### Debug Printouts and Result Statistics

Tests should include comprehensive printouts for debugging and tracking results. Follow these guidelines:

#### 1. Test Suite Headers

Print clear section headers to organize test output:

```python
def run_component_tests(results: TestResults, device: torch.device):
    """Run tests for specific component."""
    print("\n" + "=" * 80)
    print("Testing MyComponent")
    print("=" * 80)

    num_envs = 16
    num_agents = 3
    print(f"Configuration: {num_envs} environments, {num_agents} agents")
```

#### 2. Individual Test Status

Show pass/fail status immediately after each test:

```python
# In TestResults class
def add_pass(self, test_name: str):
    self.passed.append(test_name)
    print(f"  ✓ {test_name}")

def add_fail(self, test_name: str, error: str):
    self.failed.append((test_name, error))
    print(f"  ✗ {test_name}")
    # Print first few lines of error for immediate feedback
    error_lines = error.split('\n')[:5]
    for line in error_lines:
        print(f"    {line}")

def add_error(self, test_name: str, error: str):
    self.errors.append((test_name, error))
    print(f"  ERROR {test_name}")
    print(f"    {error}")
```

#### 3. Detailed Error Information

For failed tests, print full traceback and relevant values:

```python
try:
    result = component.compute_value(input_tensor)
    expected = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert torch.allclose(result, expected, atol=1e-5)
    results.add_pass("Value computation")
except AssertionError as e:
    # Print diagnostic information
    error_msg = f"Expected: {expected}\nGot: {result}\nDifference: {(result - expected).abs().max().item()}"
    results.add_fail("Value computation", error_msg)
except Exception as e:
    results.add_fail("Value computation", traceback.format_exc())
```

#### 4. Comprehensive Test Summary

Print detailed summary at the end with statistics:

```python
def print_summary(self):
    total = len(self.passed) + len(self.failed) + len(self.errors)
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total}")
    print(f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)" if total > 0 else "Passed: 0")
    print(f"Failed:      {len(self.failed)} ({100*len(self.failed)/total:.1f}%)" if total > 0 else "Failed: 0")
    print(f"Errors:      {len(self.errors)} ({100*len(self.errors)/total:.1f}%)" if total > 0 else "Errors: 0")

    if self.failed:
        print("\n" + "-" * 80)
        print("FAILED TESTS:")
        print("-" * 80)
        for test_name, error in self.failed:
            print(f"\n{test_name}:")
            print(f"  {error}")

    if self.errors:
        print("\n" + "-" * 80)
        print("TEST ERRORS:")
        print("-" * 80)
        for test_name, error in self.errors:
            print(f"\n{test_name}:")
            print(f"  {error}")

    print("=" * 80)
    return len(self.failed) == 0 and len(self.errors) == 0
```

#### 5. Progress Indicators for Long Tests

For tests with multiple iterations or steps:

```python
# Test with progress tracking
try:
    phi_values = []
    print("  Running 20-step simulation...", end="", flush=True)

    for i in range(20):
        # Update progress every 5 steps
        if i % 5 == 0 and i > 0:
            print(f".{i}", end="", flush=True)

        # Run test step
        phi, tau = component.compute(inputs, dt=0.1)
        phi_values.append(phi.mean().item())

    print(" done")

    # Check results
    assert max(phi_values) <= 1.0
    results.add_pass("Multi-step simulation")
except Exception as e:
    print(" FAILED")
    results.add_fail("Multi-step simulation", traceback.format_exc())
```

#### 6. Intermediate Value Debugging

Print key intermediate values for complex tests:

```python
# Debug mode flag (can be set via command line argument)
DEBUG = False  # Set to True for detailed debug output

try:
    distances = detector.compute_distances(positions)

    if DEBUG:
        print(f"    Distance matrix shape: {distances.shape}")
        print(f"    Distance range: [{distances.min().item():.3f}, {distances.max().item():.3f}]")
        print(f"    Mean distance: {distances.mean().item():.3f}")

    assert distances.shape == (num_envs, num_agents, num_agents)
    results.add_pass("Distance computation")
except Exception as e:
    results.add_fail("Distance computation", str(e))
```

#### 7. Test Execution Summary Header

Print configuration at the start of test suite:

```python
def main():
    """Main test runner."""
    print("=" * 80)
    print("SAFETY MODULE TEST SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU name:      {torch.cuda.get_device_name(0)}")
    print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = TestResults()

    # Run tests...
```

#### 8. Expected vs Actual Comparisons

For assertion failures, show clear expected vs actual values:

```python
try:
    result = component.process(input_data)
    expected_shape = (num_envs, num_agents)

    assert result.shape == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, got {result.shape}"

    expected_range = (0.0, 1.0)
    actual_min = result.min().item()
    actual_max = result.max().item()

    assert actual_min >= expected_range[0] and actual_max <= expected_range[1], \
        f"Value out of range: expected [{expected_range[0]}, {expected_range[1]}], " \
        f"got [{actual_min:.4f}, {actual_max:.4f}]"

    results.add_pass("Output validation")
except AssertionError as e:
    results.add_fail("Output validation", str(e))
```

#### 9. Performance Statistics (Optional)

Track and report execution time for performance-critical tests:

```python
import time

try:
    start_time = time.time()

    # Run test
    for _ in range(100):
        result = component.compute(large_input)

    elapsed = time.time() - start_time
    avg_time = elapsed / 100 * 1000  # ms per call

    print(f"    Average execution time: {avg_time:.3f} ms")

    results.add_pass(f"Performance test (avg: {avg_time:.2f}ms)")
except Exception as e:
    results.add_fail("Performance test", traceback.format_exc())
```

#### 10. Verbose Mode Support

Add optional verbose flag for detailed output:

```python
# In argument parser
# Add `--test-verbose` to avoid conflict with AppLauncher's existing `--verbose` flag
parser.add_argument("--test-verbose", action="store_true",
                   help="Enable verbose debug output")
args_cli = parser.parse_args()

# Pass verbose flag to test functions
VERBOSE = args_cli.verbose

# Use in tests
def run_component_tests(results: TestResults, device: torch.device, verbose: bool = False):
    if verbose:
        print(f"  Input tensor: {input_tensor}")
        print(f"  Configuration: {cfg}")
```

**Example Output Format**:

```
================================================================================
SAFETY MODULE TEST SUITE
================================================================================
Device:        cuda
Torch version: 2.0.0
CUDA version:  11.8
GPU name:      NVIDIA RTX 4090
Started:       2025-01-15 10:30:45

================================================================================
Testing CollisionDetector
================================================================================
Configuration: 16 environments, 3 agents
  ✓ Initialization
  ✓ Distance computation
  ✓ Collision detection (no collision)
  ✓ Collision detection (with collision)
  ✓ Collision penalties
  ✓ Reset functionality

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 6
Passed:      6 (100.0%)
Failed:      0 (0.0%)
Errors:      0 (0.0%)
================================================================================
```

### Running Tests

```bash
# Run standalone test runner
./isaaclab.sh -p source/my_module/tests/run_tests.py

# Run with verbose output
./isaaclab.sh -p source/my_module/tests/run_tests.py --test-verbose

# Run with specific device
CUDA_VISIBLE_DEVICES="" ./isaaclab.sh -p source/my_module/tests/run_tests.py  # CPU only
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/my_module/tests/run_tests.py   # GPU 0

# Run with pytest (if using pytest-compatible tests)
./isaaclab.sh -p -m pytest source/my_module/tests/ -v
./isaaclab.sh -p -m pytest source/my_module/tests/test_specific.py::TestClass::test_method -v

# Run with coverage
./isaaclab.sh -p -m pytest source/my_module/tests/ --cov=my_module --cov-report=html
```

### Automating Test Execution and Debugging

When developing or debugging tests, use an iterative approach to systematically eliminate errors until all tests pass.

#### Iterative Test-Fix Workflow

Follow this workflow to automate test execution and error resolution:

1. **Execute test script** and capture full output
2. **Log output** to `test_result.txt` (overwrite each iteration)
3. **Analyze errors** from the logged output
4. **Track unique errors** in `error_log.txt` (append, don't overwrite)
5. **Fix identified issues** in test code or implementation
6. **Repeat** until all tests pass

#### Test Output Logging

**CRITICAL**: Always log test output to files for analysis and debugging.

```bash
# Run test and log output to test_result.txt (overwrites each time)
./isaaclab.sh -p source/my_module/tests/run_tests.py > source/my_module/tests/test_result.txt 2>&1

# Check the results
cat source/my_module/tests/test_result.txt

# Or tail for large outputs
tail -100 source/my_module/tests/test_result.txt
```

#### Error Tracking System

Maintain two separate log files in the `tests/` directory:

1. **`test_result.txt`**: Current test run output (OVERWRITTEN each iteration)
   - Contains the complete output from the most recent test execution
   - Used to identify what needs fixing in the current iteration
   - Always overwrite with new results

2. **`error_log.txt`**: Cumulative error history (APPENDED each iteration)
   - Tracks all unique error patterns encountered
   - Prevents repeating the same mistakes
   - Documents the debugging journey
   - Include iteration number, error type, and fix applied

#### Claude Code Agent Workflow for Automated Testing

**CRITICAL INSTRUCTIONS FOR CLAUDE CODE AGENT**: When asked to develop or debug tests, follow this fully automated iterative workflow:

1. **Run test and log output** (iteration 1)
2. **Read `test_result.txt`** to analyze errors
3. **Read `error_log.txt`** to avoid repeating past mistakes
4. **Fix identified errors** using Edit/Write tools
5. **Document fix** by appending to `error_log.txt`
6. **Repeat steps 1-5** until tests pass or max iterations reached

**Maximum Iterations**: Default to 10 iterations. Stop early if tests pass.

**Example Automated Workflow**:

```bash
# Iteration 1: Run test
./isaaclab.sh -p source/my_module/tests/run_tests.py > source/my_module/tests/test_result.txt 2>&1

# Read test_result.txt - analyze errors
# Read error_log.txt - check previous attempts
# Fix errors in code using Edit/Write tools
# Append fix to error_log.txt

# Iteration 2: Run test again
./isaaclab.sh -p source/my_module/tests/run_tests.py > source/my_module/tests/test_result.txt 2>&1

# Read test_result.txt - analyze NEW errors
# Read error_log.txt - avoid repeating fixes
# Fix NEW errors
# Append NEW fix to error_log.txt

# Continue until tests pass or max_iterations reached
```

**Key Points for Claude Code Agent**:
- Always use `> test_result.txt 2>&1` to overwrite with latest results
- Always use `>> error_log.txt` to append error history (never overwrite)
- Read BOTH files before each fix iteration
- Fix one error at a time, starting with the first error in `test_result.txt`
- Document each fix with iteration number, error type, root cause, and solution

#### Automated Test Iteration Tracking

Append to `error_log.txt` after each iteration:

```bash
# After fixing errors in iteration N
cat >> source/my_module/tests/error_log.txt << 'EOF'
================================================================================
ITERATION N - $(date)
================================================================================
Error Type: [ImportError/AssertionError/TypeError/etc]
Location: [file.py:line]
Message: [error message from test_result.txt]

Root Cause: [why the error occurred]
Fix Applied: [what was changed to fix it]

Files Modified:
- [file1.py]: [description of changes]
- [file2.py]: [description of changes]
--------------------------------------------------------------------------------
EOF
```

#### Fully Automated Helper Script (Optional)

For users who want a standalone automation script:

```python
#!/usr/bin/env python3
"""
Fully automated test iteration script.
Usage: python iterate_tests.py [--max-iterations N]
"""
import subprocess
import os
import argparse
from datetime import datetime

def run_test_iteration(test_script: str, test_dir: str, iteration: int):
    """Run a single test iteration and log results."""

    # File paths
    result_file = os.path.join(test_dir, "test_result.txt")
    error_log = os.path.join(test_dir, "error_log.txt")

    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}")
    print(f"{'='*80}")
    print(f"Running: {test_script}")
    print(f"Logging to: {result_file}")

    # Run test and capture output
    result = subprocess.run(
        ["./isaaclab.sh", "-p", test_script],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout
    )

    # Write current result (overwrite)
    with open(result_file, 'w') as f:
        f.write(f"Iteration {iteration}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

    # Check for errors
    has_errors = result.returncode != 0

    if has_errors:
        # Append to error log (don't overwrite)
        with open(error_log, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ITERATION {iteration} - {datetime.now()}\n")
            f.write(f"{'='*80}\n")

            # Extract and log unique error patterns
            if "Traceback" in result.stdout:
                f.write("Python Exception Found:\n")
                # Extract traceback
                lines = result.stdout.split('\n')
                in_traceback = False
                for line in lines:
                    if "Traceback" in line:
                        in_traceback = True
                    if in_traceback:
                        f.write(f"  {line}\n")
                        if line.strip() and not line.startswith(' '):
                            in_traceback = False

            if "FAILED" in result.stdout or "ERROR" in result.stdout:
                f.write("\nFailed/Error tests:\n")
                lines = result.stdout.split('\n')
                for line in lines:
                    if "✗" in line or "ERROR" in line:
                        f.write(f"  {line}\n")

            f.write("\n[ACTION REQUIRED] Analyze errors and apply fixes before next iteration\n")

    print(f"\nTest {'PASSED' if not has_errors else 'FAILED'}")
    print(f"Check {result_file} for details")
    if has_errors:
        print(f"Error patterns logged to {error_log}")

    return not has_errors

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated test iteration")
    parser.add_argument("--test-script", default="source/my_module/tests/run_tests.py",
                       help="Path to test script")
    parser.add_argument("--test-dir", default="source/my_module/tests",
                       help="Directory for test logs")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum number of iterations (default: 10)")
    args = parser.parse_args()

    iteration = 1

    print("="*80)
    print("AUTOMATED TEST ITERATION")
    print("="*80)
    print(f"Test script: {args.test_script}")
    print(f"Test directory: {args.test_dir}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Results logged to: {os.path.join(args.test_dir, 'test_result.txt')}")
    print(f"Error history: {os.path.join(args.test_dir, 'error_log.txt')}")

    while iteration <= args.max_iterations:
        success = run_test_iteration(args.test_script, args.test_dir, iteration)

        if success:
            print(f"\n{'='*80}")
            print(f"✓ ALL TESTS PASSED in iteration {iteration}!")
            print(f"{'='*80}")
            break

        print(f"\n✗ Iteration {iteration} failed.")
        print(f"→ Fix errors based on {os.path.join(args.test_dir, 'test_result.txt')}")
        print(f"→ Check {os.path.join(args.test_dir, 'error_log.txt')} for previous attempts")

        iteration += 1

    if iteration > args.max_iterations:
        print(f"\n{'='*80}")
        print(f"✗ REACHED MAXIMUM ITERATIONS ({args.max_iterations})")
        print(f"{'='*80}")
        print(f"Tests still failing. Review logs:")
        print(f"  - {os.path.join(args.test_dir, 'test_result.txt')}")
        print(f"  - {os.path.join(args.test_dir, 'error_log.txt')}")
        exit(1)

    exit(0)
```

**Usage**:

```bash
# Run with defaults (10 iterations)
python source/my_module/tests/iterate_tests.py

# Run with custom max iterations
python source/my_module/tests/iterate_tests.py --max-iterations 20

# Run with custom paths
python source/my_module/tests/iterate_tests.py \
    --test-script source/other_module/tests/run_tests.py \
    --test-dir source/other_module/tests \
    --max-iterations 15
```

**Note for Claude Code Users**: This script only runs tests and logs results. **You (Claude Code agent) must analyze logs and fix errors between iterations**. The script does NOT automatically fix code - that's your job!

#### Manual Iterative Workflow

If not using automation, follow this manual process:

**Iteration 1:**
```bash
# Run test
./isaaclab.sh -p source/my_module/tests/run_tests.py > source/my_module/tests/test_result.txt 2>&1

# Read results
cat source/my_module/tests/test_result.txt

# Document errors found
echo "=== ITERATION 1 ===" >> source/my_module/tests/error_log.txt
echo "$(date)" >> source/my_module/tests/error_log.txt
grep -A 5 "ERROR\|FAILED\|Traceback" source/my_module/tests/test_result.txt >> source/my_module/tests/error_log.txt
echo "" >> source/my_module/tests/error_log.txt

# Fix identified issues in code...
```

**Iteration 2:**
```bash
# Run test again (overwrites test_result.txt)
./isaaclab.sh -p source/my_module/tests/run_tests.py > source/my_module/tests/test_result.txt 2>&1

# Read NEW results
cat source/my_module/tests/test_result.txt

# Document NEW errors (append to error_log.txt)
echo "=== ITERATION 2 ===" >> source/my_module/tests/error_log.txt
echo "$(date)" >> source/my_module/tests/error_log.txt
grep -A 5 "ERROR\|FAILED\|Traceback" source/my_module/tests/test_result.txt >> source/my_module/tests/error_log.txt
echo "Fixed: [describe what was fixed from iteration 1]" >> source/my_module/tests/error_log.txt
echo "" >> source/my_module/tests/error_log.txt

# Continue fixing...
```

**Repeat until:** `test_result.txt` shows all tests passing

#### Error Log Format

Structure `error_log.txt` to track debugging progress:

```
================================================================================
ITERATION 1 - 2025-01-15 10:30:00
================================================================================
Error Type: ImportError
Location: run_tests.py, line 35
Message: cannot import name 'CollisionDetector' from 'isaaclab_tasks.direct.iris_ma3.safety'

Root Cause: Missing __init__.py export
Fix Applied: Added CollisionDetector to __init__.py exports

--------------------------------------------------------------------------------

================================================================================
ITERATION 2 - 2025-01-15 10:35:00
================================================================================
Error Type: AssertionError
Location: test_collision_detector.py, line 125
Message: Shape mismatch: expected (16, 3, 3), got (16, 3)

Root Cause: compute_distances returning 2D instead of 3D tensor
Fix Applied: Fixed distance computation to return pairwise matrix

Previous iteration fixed: ImportError

--------------------------------------------------------------------------------

================================================================================
ITERATION 3 - 2025-01-15 10:40:00
================================================================================
ALL TESTS PASSED ✓

Total iterations: 3
Total unique errors fixed: 2
- ImportError (missing __init__ export)
- Shape mismatch (incorrect tensor dimensions)
```

#### Best Practices for Iterative Testing

1. **One Error Type at a Time**: Fix the first error encountered, then rerun
   - Don't try to fix all errors simultaneously
   - Early errors may mask later ones

2. **Read Previous Iterations**: Before fixing, check `error_log.txt`
   - Avoid repeating fixes that didn't work
   - Learn from previous iteration attempts

3. **Document Fixes**: Always note what you fixed in `error_log.txt`
   - Helps track progress
   - Provides debugging history

4. **Use Timeouts**: For tests that might hang
   ```bash
   timeout 300 ./isaaclab.sh -p source/my_module/tests/run_tests.py > test_result.txt 2>&1
   ```

5. **Check Exit Codes**: Verify test success programmatically
   ```bash
   ./isaaclab.sh -p source/my_module/tests/run_tests.py > test_result.txt 2>&1
   if [ $? -eq 0 ]; then
       echo "Tests passed!"
   else
       echo "Tests failed - check test_result.txt"
   fi
   ```

6. **Incremental Testing**: Test components individually first
   - Run single test functions before full suite
   - Reduces debugging complexity

7. **Clean State Between Iterations**: Sometimes stale state causes issues
   ```bash
   # Clear Python cache
   find source/my_module -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

   # Re-run test
   ./isaaclab.sh -p source/my_module/tests/run_tests.py > test_result.txt 2>&1
   ```

#### Example Error Analysis Workflow

When analyzing `test_result.txt`:

1. **Identify Error Type**
   - ImportError → Check imports and __init__.py
   - AttributeError → Check class/function definitions
   - AssertionError → Check test expectations vs actual behavior
   - TypeError → Check function signatures and types

2. **Locate Error Source**
   - Read traceback from bottom to top
   - Identify the file and line number
   - Check if error is in test code or implementation

3. **Reproduce Minimal Case**
   - Isolate the failing test
   - Create minimal reproduction if needed

4. **Apply Fix**
   - Fix the root cause, not symptoms
   - Verify fix makes sense

5. **Update Error Log**
   - Document error type, cause, and fix
   - Note iteration number

6. **Rerun Tests**
   - Overwrite `test_result.txt` with new run
   - Check if error is resolved
   - Move to next error if any

#### File Structure for Test Logging

```
my_module/
└── tests/
    ├── __init__.py
    ├── run_tests.py
    ├── test_component_a.py
    ├── test_component_b.py
    ├── README.md
    ├── test_result.txt         # Current run output (overwritten)
    ├── error_log.txt           # Cumulative error history (appended)
    └── iterate_tests.py        # Optional automation script
```

**Note**: Add `test_result.txt` and `error_log.txt` to `.gitignore` if they contain sensitive or transient information. However, keeping `error_log.txt` in version control can be valuable for documentation.

### File Naming Conventions

- Test files: `test_[component_name].py`
- Test runner: `run_tests.py`
- Documentation: `README.md` (one per tests/ directory)
- Init file: `__init__.py` (can be empty or export test utilities)
- Current test output: `test_result.txt` (overwritten each run, optional `.gitignore`)
- Error history: `error_log.txt` (appended each run, keep in repo for docs)

## Guide on Generating Technical Documents
### File structure
```
my_module/
├── doc/
├    └── SOME_PLAN_OR_RECORD_OF_WORK.md     # Organize execution plan / provide the summary to the work done for each task.
└── tests/
```
