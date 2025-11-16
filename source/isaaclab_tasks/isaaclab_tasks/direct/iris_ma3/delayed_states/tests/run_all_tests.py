#!/usr/bin/env python3
"""
Master test runner for delayed_states.py test suite.

Runs all tests and provides summary report.

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --verbose          # Verbose output
    python run_all_tests.py --priority P0      # Run only P0 tests
    python run_all_tests.py --priority all     # Run all tests
    python run_all_tests.py --coverage         # Run with coverage report
"""
import argparse
from isaaclab.app import AppLauncher

# Create parser for both AppLauncher and test runner args
parser = argparse.ArgumentParser(description="Run delayed_states.py test suite")
parser.add_argument(
    "--priority",
    choices=["P0", "P1", "P2", "all"],
    default="all",
    help="Test priority level to run (default: all)"
)
parser.add_argument(
    "--coverage",
    action="store_true",
    help="Generate coverage report"
)
parser.add_argument(
    "--markers", "-m",
    type=str,
    help="Run tests matching given mark expression (pytest -m)"
)
parser.add_argument(
    "--keyword", "-k",
    type=str,
    help="Run tests matching keyword expression (pytest -k)"
)
parser.add_argument(
    "--failfast", "-x",
    action="store_true",
    help="Stop on first failure"
)

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import pytest


def main():
    args = args_cli  # Use the parsed args

    # Build pytest arguments
    pytest_args = []

    # Add test directory
    import os
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytest_args.append(test_dir)

    # Always use verbose for clarity and show prints
    pytest_args.append("-v")
    pytest_args.append("-s")  # Show print statements from tests

    # Priority-based test selection
    if args.priority == "P0":
        # Core user-requested tests
        pytest_args.extend([
            "-k",
            "test_throttle_only or test_dropout_only or test_latency_only or "
            "test_combined_impairments or test_broadcast or test_receive or "
            "test_initialization or test_noise_reproducibility or test_update_gt_states or "
            "test_curriculum_learning or test_detection_processing"
        ])
    elif args.priority == "P1":
        # Critical tests
        pytest_args.extend([
            "-k",
            "test_buffer_corruption or test_sample_and_hold or test_variable_delay or "
            "test_impairment_order or test_valid_mask_semantics or "
            "test_statistics_accuracy or test_reset_specific_envs or "
            "test_end_to_end_detection or test_motion_and_gimbal or "
            "test_detection_communication_integration or "
            "test_state_consistency or test_multi_agent_integration or "
            "test_communication_integration or test_noise_formula_consistency or "
            "test_reset_functionality"
        ])
    elif args.priority == "P2":
        # Important tests
        pytest_args.extend([
            "-k",
            "test_throttle_zero or test_throttle_infinite or test_dropout_zero or "
            "test_dropout_full or test_latency_zero or test_latency_clamping or "
            "test_dynamic_parameter or test_lazy_buffer or test_multi_key or "
            "test_full_mesh or test_asymmetric or test_receive_before_send or "
            "test_communication_statistics or test_first_order_lag_convergence or "
            "test_update_time_constant or test_quaternion_slerp or "
            "test_quaternion_shortest or test_first_order_lag_reset or "
            "test_time_management or test_detection_parameter or test_comm_parameter or "
            "test_complete_reset or test_input_validity or test_validity_persistence or "
            "test_per_environment_validity or "
            "test_camera_pose_updates or test_zoom_noise_clamping or "
            "test_partial_env_update or test_detection_with_invalid_mask or "
            "test_get_all_states or test_noise_disabled"
        ])
    # If priority == "all", don't add any filter

    # Coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma3/delayed_states",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Markers
    if args.markers:
        pytest_args.extend(["-m", args.markers])

    # Keyword
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Fail fast
    if args.failfast:
        pytest_args.append("-x")

    # Add color
    pytest_args.append("--color=yes")

    # Add summary
    pytest_args.append("-ra")

    print("=" * 80)
    print("Running delayed_states.py Test Suite")
    print("=" * 80)
    print(f"Priority: {args.priority}")
    print(f"Coverage: {'Yes' if args.coverage else 'No'}")
    print(f"Test directory: {test_dir}")
    print("=" * 80)
    print()

    # Run tests
    exit_code = pytest.main(pytest_args)

    # Summary
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests failed with exit code: {exit_code}")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
