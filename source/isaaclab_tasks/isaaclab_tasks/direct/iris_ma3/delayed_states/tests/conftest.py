"""
Pytest configuration and test statistics collection for delayed_states tests.
"""

import pytest
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class TestStats:
    """Statistics for a single test."""
    name: str
    status: str  # passed, failed, skipped
    duration: float
    test_class: str
    details: Dict[str, Any] = field(default_factory=dict)


class TestStatisticsPlugin:
    """Pytest plugin to collect and display test statistics."""

    def __init__(self):
        self.test_stats: List[TestStats] = []
        self.session_start_time = None

    def pytest_sessionstart(self, session):
        """Called before test run begins."""
        self.session_start_time = time.time()

    def pytest_runtest_logreport(self, report):
        """Called for each test phase (setup, call, teardown)."""
        if report.when == 'call':
            # Extract test class and name
            test_parts = report.nodeid.split("::")
            test_class = test_parts[-2] if len(test_parts) >= 2 else "Unknown"
            test_name = test_parts[-1] if test_parts else report.nodeid

            # Create test stats
            stats = TestStats(
                name=test_name,
                status=report.outcome,
                duration=report.duration,
                test_class=test_class
            )

            # Add failure info if test failed
            if report.failed:
                stats.details['error'] = str(report.longrepr)[:100]  # First 100 chars

            self.test_stats.append(stats)

    def pytest_sessionfinish(self, session, exitstatus):
        """Called after all tests complete."""
        if not self.test_stats:
            return

        total_duration = time.time() - self.session_start_time

        # Group tests by class
        tests_by_class: Dict[str, List[TestStats]] = {}
        for stat in self.test_stats:
            if stat.test_class not in tests_by_class:
                tests_by_class[stat.test_class] = []
            tests_by_class[stat.test_class].append(stat)

        # Calculate summary statistics
        total_tests = len(self.test_stats)
        passed = sum(1 for s in self.test_stats if s.status == 'passed')
        failed = sum(1 for s in self.test_stats if s.status == 'failed')
        skipped = sum(1 for s in self.test_stats if s.status == 'skipped')

        # Print summary
        print("\n" + "=" * 100)
        print("TEST STATISTICS SUMMARY")
        print("=" * 100)

        # Overall summary
        print(f"\n{'Overall Results':<40} {'Count':<10} {'Duration':<15} {'Status':<10}")
        print("-" * 100)
        print(f"{'Total Tests':<40} {total_tests:<10} {total_duration:.3f}s")
        print(f"{'Passed':<40} {passed:<10} {sum(s.duration for s in self.test_stats if s.status == 'passed'):.3f}s"
              f"       {'✓ PASS' if failed == 0 else ''}")
        if failed > 0:
            print(f"{'Failed':<40} {failed:<10} {sum(s.duration for s in self.test_stats if s.status == 'failed'):.3f}s"
                  f"       {'✗ FAIL'}")
        if skipped > 0:
            print(f"{'Skipped':<40} {skipped:<10}")

        # Per-class breakdown
        print(f"\n{'Test Class Breakdown':<40} {'Tests':<10} {'Passed':<10} {'Failed':<10} {'Avg Time':<15}")
        print("-" * 100)

        for test_class in sorted(tests_by_class.keys()):
            class_tests = tests_by_class[test_class]
            class_passed = sum(1 for s in class_tests if s.status == 'passed')
            class_failed = sum(1 for s in class_tests if s.status == 'failed')
            avg_duration = sum(s.duration for s in class_tests) / len(class_tests)

            status_icon = "✓" if class_failed == 0 else "✗"
            print(f"{status_icon} {test_class:<37} {len(class_tests):<10} {class_passed:<10} {class_failed:<10} {avg_duration:.4f}s")

        # Slowest tests
        print(f"\n{'Top 10 Slowest Tests':<60} {'Duration':<15} {'Status':<10}")
        print("-" * 100)

        slowest = sorted(self.test_stats, key=lambda x: x.duration, reverse=True)[:10]
        for i, stat in enumerate(slowest, 1):
            status_icon = "✓" if stat.status == 'passed' else "✗"
            test_name = stat.name[:55] + "..." if len(stat.name) > 55 else stat.name
            print(f"{i:2}. {status_icon} {test_name:<55} {stat.duration:.4f}s       {stat.status.upper()}")

        # Failed tests details
        failed_tests = [s for s in self.test_stats if s.status == 'failed']
        if failed_tests:
            print(f"\n{'Failed Tests Details':<60}")
            print("-" * 100)
            for stat in failed_tests:
                print(f"✗ {stat.test_class}::{stat.name}")
                if 'error' in stat.details:
                    print(f"  Error: {stat.details['error']}")

        # Performance metrics
        avg_duration = sum(s.duration for s in self.test_stats) / len(self.test_stats)
        min_duration = min(s.duration for s in self.test_stats)
        max_duration = max(s.duration for s in self.test_stats)

        print(f"\n{'Performance Metrics':<40}")
        print("-" * 100)
        print(f"{'Average test duration':<40} {avg_duration:.4f}s")
        print(f"{'Fastest test':<40} {min_duration:.4f}s")
        print(f"{'Slowest test':<40} {max_duration:.4f}s")
        print(f"{'Total execution time':<40} {total_duration:.3f}s")
        print(f"{'Tests per second':<40} {total_tests / total_duration:.2f}")

        print("\n" + "=" * 100)

        # Exit status summary
        if exitstatus == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"✗ TEST RUN FAILED (exit code: {exitstatus})")
        print("=" * 100 + "\n")


def pytest_configure(config):
    """Register the plugin."""
    plugin = TestStatisticsPlugin()
    config.pluginmanager.register(plugin, 'test_statistics')
