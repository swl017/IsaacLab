import torch

import isaaclab.utils.math as math_utils

class ParameterRandomizerCfg:
    pass

class ParameterRandomizer:
    def __init__(self, num_envs: int, device: torch.device):
        self.device = device
        self.num_envs = num_envs
        self._ALL_INDICES = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

