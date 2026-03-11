# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auto-tuning utilities for DroneController.

This module provides auto-tuning capabilities for the DroneController.
Run the tuning script directly:
    ./isaaclab.sh -p .../tuning/auto_tune.py --headless

Note: The tuning classes are not imported here to avoid AppLauncher
initialization during package import.
"""

# Don't import from auto_tune.py - it starts AppLauncher
# Use lazy import or run auto_tune.py directly as a script
