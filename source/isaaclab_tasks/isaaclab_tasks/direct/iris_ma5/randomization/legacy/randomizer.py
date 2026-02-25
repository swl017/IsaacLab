from .initial_states import InitialStatesRandomizer
from .parameters import ParameterRandomizer
from .target_sampling import TargetSampler, TargetSamplerCfg

class Randomizer:
    def __init__(self, num_envs, device, gimbal_pitch_limits=None):
        """
        Initialize randomizer with all randomization modules.

        Args:
            num_envs: Number of parallel environments
            device: PyTorch device (cpu or cuda)
            gimbal_pitch_limits: Tuple of (min, max) pitch angles in radians.
                               If None, uses default from TargetSamplerCfg.
        """
        self.initial_states = InitialStatesRandomizer(num_envs, device)
        self.parameter = ParameterRandomizer(num_envs, device)

        # Create target sampler with gimbal constraints
        target_cfg = TargetSamplerCfg()
        if gimbal_pitch_limits is not None:
            target_cfg.pitch_limit_min = gimbal_pitch_limits[0]
            target_cfg.pitch_limit_max = gimbal_pitch_limits[1]

        self.targets = TargetSampler(target_cfg, device)