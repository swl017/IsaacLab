# domain_randomization_config.py
"""
Domain randomization configuration for measurement noise in triangulation.
"""

import torch
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg


@configclass
class MeasurementNoiseCfg:
    """Configuration for measurement noise parameters."""
    
    # Position noise (meters)
    position_std_min: float = 0.0
    position_std_max: float = 0.1
    
    # Orientation noise (radians)
    orientation_std_min: float = 0.0
    orientation_std_max: float = 0.01
    
    # Gimbal angle noise (radians)
    gimbal_yaw_std_min: float = 0.0
    gimbal_yaw_std_max: float = 0.005
    gimbal_pitch_std_min: float = 0.0
    gimbal_pitch_std_max: float = 0.005
    
    # Pixel detection noise (pixels)
    pixel_std_min: float = 0.3
    pixel_std_max: float = 1.0
    
    # Camera intrinsic noise
    focal_length_std_ratio_min: float = 0.0  # fraction of focal length
    focal_length_std_ratio_max: float = 0.01
    principal_point_std_min: float = 0.0  # pixels
    principal_point_std_max: float = 2.0
    
    # Zoom level noise
    zoom_std_min: float = 0.0
    zoom_std_max: float = 0.05
    
    # Curriculum: gradually increase noise over training
    enable_curriculum: bool = True
    curriculum_start_step: int = 0
    curriculum_end_step: int = 100000


def randomize_measurement_noise(env, env_ids: torch.Tensor, noise_cfg: MeasurementNoiseCfg):
    """
    Randomize measurement noise covariances for specified environments.
    Called at episode reset to set noise levels for the episode.
    
    Args:
        env: The environment instance
        env_ids: Environment indices to randomize
        noise_cfg: Noise configuration
    """
    device = env.device
    B = len(env_ids)
    
    # Curriculum factor: 0 at start, 1 at end
    if noise_cfg.enable_curriculum:
        progress = (env.common_step_counter - noise_cfg.curriculum_start_step) / \
                   (noise_cfg.curriculum_end_step - noise_cfg.curriculum_start_step)
        curriculum_alpha = torch.clamp(torch.tensor(progress, device=device), 0.0, 1.0)
    else:
        curriculum_alpha = 1.0
    
    # Sample noise standard deviations for this episode
    # Position noise
    pos_std = torch.rand(B, 3, device=device) * \
              (noise_cfg.position_std_max - noise_cfg.position_std_min) + \
              noise_cfg.position_std_min
    pos_std *= curriculum_alpha
    env.measurement_noise['Sigma_twb'][env_ids] = torch.diag_embed(pos_std ** 2)
    
    # Orientation noise (Euler angles)
    ori_std = torch.rand(B, 3, device=device) * \
              (noise_cfg.orientation_std_max - noise_cfg.orientation_std_min) + \
              noise_cfg.orientation_std_min
    ori_std *= curriculum_alpha
    env.measurement_noise['Sigma_phiwb'][env_ids] = torch.diag_embed(ori_std ** 2)
    
    # Gimbal yaw noise
    yaw_std = torch.rand(B, 1, device=device) * \
              (noise_cfg.gimbal_yaw_std_max - noise_cfg.gimbal_yaw_std_min) + \
              noise_cfg.gimbal_yaw_std_min
    yaw_std *= curriculum_alpha
    env.measurement_noise['Sigma_alpha'][env_ids] = (yaw_std ** 2).unsqueeze(-1)
    
    # Gimbal pitch noise
    pitch_std = torch.rand(B, 1, device=device) * \
                (noise_cfg.gimbal_pitch_std_max - noise_cfg.gimbal_pitch_std_min) + \
                noise_cfg.gimbal_pitch_std_min
    pitch_std *= curriculum_alpha
    env.measurement_noise['Sigma_beta'][env_ids] = (pitch_std ** 2).unsqueeze(-1)
    
    # Pixel noise (isotropic)
    pix_std = torch.rand(B, device=device) * \
              (noise_cfg.pixel_std_max - noise_cfg.pixel_std_min) + \
              noise_cfg.pixel_std_min
    pix_std *= curriculum_alpha
    env.measurement_noise['Sigma_pix'][env_ids] = torch.diag_embed(
        torch.stack([pix_std, pix_std], dim=1) ** 2
    )
    
    # Camera intrinsic noise (fx, fy, cx, cy)
    if noise_cfg.focal_length_std_ratio_max > 0:
        focal_std_ratio = torch.rand(B, 2, device=device) * \
                         (noise_cfg.focal_length_std_ratio_max - noise_cfg.focal_length_std_ratio_min) + \
                         noise_cfg.focal_length_std_ratio_min
        focal_std_ratio *= curriculum_alpha
        
        pp_std = torch.rand(B, 2, device=device) * \
                (noise_cfg.principal_point_std_max - noise_cfg.principal_point_std_min) + \
                noise_cfg.principal_point_std_min
        pp_std *= curriculum_alpha
        
        # Diagonal covariance [var(fx), var(fy), var(cx), var(cy)]
        K_var = torch.zeros(B, 4, device=device)
        # Assuming fx and fy are around 1000 pixels (typical)
        K_var[:, 0] = (focal_std_ratio[:, 0] * 1000) ** 2
        K_var[:, 1] = (focal_std_ratio[:, 1] * 1000) ** 2
        K_var[:, 2] = pp_std[:, 0] ** 2
        K_var[:, 3] = pp_std[:, 1] ** 2
        
        env.measurement_noise['Sigma_K'][env_ids] = torch.diag_embed(K_var)
    
    # Store noise std for applying during observations
    env.measurement_noise['position_std'][env_ids] = pos_std
    env.measurement_noise['orientation_std'][env_ids] = ori_std
    env.measurement_noise['gimbal_yaw_std'][env_ids] = yaw_std.squeeze(-1)
    env.measurement_noise['gimbal_pitch_std'][env_ids] = pitch_std.squeeze(-1)


def apply_observation_noise(
    true_value: torch.Tensor,
    noise_std: torch.Tensor,
    device: torch.device | str
) -> torch.Tensor:
    """
    Apply zero-mean Gaussian noise to observations.
    
    Args:
        true_value: [..., D] true value
        noise_std: [..., D] or [...] standard deviation
        device: torch device
    
    Returns:
        noisy_value: [..., D] = true_value + N(0, noise_std^2)
    """
    noise = torch.randn_like(true_value, device=device)
    if noise_std.shape != true_value.shape:
        noise_std = noise_std.unsqueeze(-1).expand_as(true_value)
    return true_value + noise * noise_std


# Event term functions for Isaac Lab's event manager
def randomize_noise_on_reset(env, env_ids: torch.Tensor):
    """Event term: randomize noise at episode reset."""
    randomize_measurement_noise(env, env_ids, env.cfg.measurement_noise)


def randomize_noise_periodic(env, env_ids: torch.Tensor):
    """Event term: periodically randomize noise during episode."""
    # This can be used to change noise characteristics mid-episode
    # For now, we only randomize at reset
    pass