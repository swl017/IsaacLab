# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""cooperation_metrics — track-loss / re-acquisition instrumentation (ticket 050, Slice A).

Measures single-agent, peer-assisted track-loss and re-acquisition events: the cooperative
gradient the later slices (team/difference reward, peer-bearing channel) must exploit. The
effective-track signal is a reachable-set-within-FOV validation gate, not a staleness time.

Measurement-only: does not modify observations, rewards, or dynamics.
"""

# Configurations
from .cooperation_metrics_cfg import ReacquisitionTrackerCfg

# Core classes
from .reacquisition_tracker import ReacquisitionTracker

__all__ = [
    # Configurations
    "ReacquisitionTrackerCfg",
    # Core classes
    "ReacquisitionTracker",
]
