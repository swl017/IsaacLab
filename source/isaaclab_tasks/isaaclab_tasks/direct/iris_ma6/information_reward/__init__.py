# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""information_reward — team / difference (counterfactual) information reward (ticket 050, Slice B).

Bearing-only Fisher-information-with-prior over the target position. Produces a team quality
(A-optimality trace readout, matching the existing reward) and a per-agent difference reward equal
to each agent's marginal information contribution — dense, defined at 0/1/N bearings, and large
exactly when an agent re-acquires a target a peer already holds. Reward-side only (no obs change).
"""

# Configurations
from .information_reward_cfg import InformationRewardCfg

# Core classes
from .information_reward import InformationReward

__all__ = [
    # Configurations
    "InformationRewardCfg",
    # Core classes
    "InformationReward",
]
