# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Iris MA6 v1 environment (heading-frame observation redesign — ticket 022).

Inherits from IrisMA6TestEnvCfg and overrides the observation space to match the
redesigned layout. The environment code lives in `iris_ma_env6_v1.py`; this config
keeps everything else identical to the test env.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .delay_system_v3 import DistributionCfg, create_delay_cfg_from_params
from .iris_ma_env6_test_cfg import IrisMA6TestEnvCfg


@configclass
class IrisMA6V1EnvCfg(IrisMA6TestEnvCfg):
    """Iris MA6 v1 env config with heading-frame observation layout.

    Observation space per agent (actor):
    - Ego (31D): vel_v1(3), roll(1), pitch(1), [cos psi, sin psi](2),
      ang_vel_b(3), lin_acc_b(3), ray_v1(3),
      gimbal_yaw_j(1), gimbal_pitch_j(1), gimbal_roll_j(1),
      combined_ang_vel_w(3), motion_aoi(1), bbox_aoi(1),
      zoom(1), effective_hfov(1), bbox(4), bbox_empty(1)
    - Inter-agent (19D per other): delta_pos_v1(3), delta_vel_v1(3),
      ray_other_v1(3), combined_ang_vel_w(3), alpha_conv(1),
      baseline_mag(1), alpha_base(1), zoom(1), bbox_empty(1),
      data_age(1), bbox_age(1)
    - Optional triangulation tail (4D): tri_pos_v1(3), scalar_unc(1)
    """

    def __post_init__(self):
        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {self.num_agents}")

        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {a: 7 for a in self.possible_agents}
        self.bbox_raycaster_v2.num_cameras_per_env = self.num_agents

        self.delay_system = create_delay_cfg_from_params(self.delay_system_params)
        if self.calibrated_bbox_noise.enabled:
            self.delay_system.noise.bbox_std = DistributionCfg(
                type="constant", value=0.0, min_value=0.0
            )

        # v1 observation layout (ticket 022): heading-frame ego + geometry features + 4D tri tail.
        obs_dim = 31 + 19 * (self.num_agents - 1)
        if self.enable_triangulation:
            obs_dim += 4  # tri_pos_v1 (3) + scalar_unc (1)
        self.observation_spaces = {a: obs_dim for a in self.possible_agents}
