# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-accelerated camera image processing for domain randomization.

This module provides computational simulation of different camera configurations
through crop and resize operations on rendered images.

Pipeline:
    Render (1920x1080) -> Crop (FOV simulation) -> Resize (resolution simulation)

The processing maintains correct intrinsic matrix transformations through
the pipeline, enabling accurate 3D reconstruction from processed images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .domain_randomization_cfg import CameraRandomizationCfg


class CameraProcessor:
    """GPU-accelerated camera image processor for resolution/FOV randomization.

    This class implements computational camera randomization by:
    1. Sampling FOV scale and focal length parameters
    2. Computing symmetric crop regions (centered principal points)
    3. Applying crop+resize via differentiable grid_sample
    4. Tracking intrinsic matrix transformations

    Constraints:
        - Centered principal points only (symmetric crops)
        - Square pixels only (f_x = f_y)

    Example:
        ```python
        cfg = CameraRandomizationCfg()
        processor = CameraProcessor(cfg, num_envs=256, num_agents=3, device="cuda")

        # Randomize parameters for some environments
        processor.randomize(env_ids=torch.arange(128))

        # Process rendered images
        images = torch.randn(256, 3, 1080, 1920, device="cuda")  # NCHW
        processed, intrinsics = processor.process_images_batched(
            images, output_size=(360, 640)
        )
        ```
    """

    def __init__(
        self,
        cfg: CameraRandomizationCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
    ):
        """Initialize the camera processor.

        Args:
            cfg: Camera randomization configuration.
            num_envs: Number of parallel environments.
            num_agents: Number of agents per environment.
            device: PyTorch device for tensor allocation.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = torch.device(device) if isinstance(device, str) else device

        # Total number of cameras
        self.num_cameras = num_envs * num_agents

        # Pre-allocate buffers
        # Crop parameters: (crop_x, crop_y, crop_w, crop_h)
        self.crop_params = torch.zeros(
            num_envs, num_agents, 4, device=self.device, dtype=torch.float32
        )

        # Intrinsic matrices: 3x3 per camera
        self.intrinsic_matrices = torch.zeros(
            num_envs, num_agents, 3, 3, device=self.device, dtype=torch.float32
        )

        # Target sizes per camera
        self.target_sizes = torch.zeros(
            num_envs, num_agents, 2, device=self.device, dtype=torch.int32
        )

        # FOV scales for logging/debugging
        self.fov_scales = torch.ones(
            num_envs, num_agents, device=self.device, dtype=torch.float32
        )

        # Focal lengths at render resolution
        self.focal_lengths = torch.ones(
            num_envs, num_agents, device=self.device, dtype=torch.float32
        ) * ((cfg.focal_length_range[0] + cfg.focal_length_range[1]) / 2.0)

        # Initialize with default parameters (full FOV, middle focal length)
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize all cameras with default (non-randomized) parameters."""
        # Default: full FOV, middle resolution, middle focal length
        default_fov_scale = 1.0
        default_focal_length = (
            self.cfg.focal_length_range[0] + self.cfg.focal_length_range[1]
        ) / 2.0

        # Get default target size
        if self.cfg.discrete_resolutions:
            default_target = self.cfg.discrete_resolutions[0]  # Highest resolution
        else:
            default_target = (self.cfg.target_width_range[1], self.cfg.target_height_range[1])

        # Set for all cameras
        self.fov_scales.fill_(default_fov_scale)
        self.focal_lengths.fill_(default_focal_length)
        self.target_sizes[:, :, 0] = default_target[0]
        self.target_sizes[:, :, 1] = default_target[1]

        # Compute crop params and intrinsics
        self._compute_all_transforms()

    def _compute_all_transforms(self):
        """Compute crop parameters and intrinsic matrices for all cameras."""
        W_render = self.cfg.render_width
        H_render = self.cfg.render_height

        # Compute crop dimensions from FOV scales
        crop_w = (W_render * self.fov_scales).int()
        crop_h = (H_render * self.fov_scales).int()

        # Symmetric crop for centered principal point
        crop_x = ((W_render - crop_w.float()) / 2.0).int()
        crop_y = ((H_render - crop_h.float()) / 2.0).int()

        # Store crop parameters
        self.crop_params[:, :, 0] = crop_x.float()
        self.crop_params[:, :, 1] = crop_y.float()
        self.crop_params[:, :, 2] = crop_w.float()
        self.crop_params[:, :, 3] = crop_h.float()

        # Compute intrinsic matrices
        # After crop+resize: f_final = f_render * (target_w / crop_w)
        W_target = self.target_sizes[:, :, 0].float()
        H_target = self.target_sizes[:, :, 1].float()

        scale = W_target / crop_w.float().clamp(min=1.0)
        f_final = self.focal_lengths * scale

        # Centered principal points
        cx_final = W_target / 2.0
        cy_final = H_target / 2.0

        # Build intrinsic matrices
        self.intrinsic_matrices.zero_()
        self.intrinsic_matrices[:, :, 0, 0] = f_final  # f_x
        self.intrinsic_matrices[:, :, 1, 1] = f_final  # f_y (square pixels)
        self.intrinsic_matrices[:, :, 0, 2] = cx_final  # c_x
        self.intrinsic_matrices[:, :, 1, 2] = cy_final  # c_y
        self.intrinsic_matrices[:, :, 2, 2] = 1.0

    def randomize(self, env_ids: torch.Tensor | None = None):
        """Randomize camera parameters for specified environments.

        Args:
            env_ids: Environment indices to randomize. If None, randomizes all.
        """
        if not self.cfg.enabled:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        num_envs_to_randomize = len(env_ids)

        # Determine number of samples based on per-agent randomization
        if self.cfg.randomize_per_agent:
            num_samples = num_envs_to_randomize * self.num_agents
            sample_shape = (num_envs_to_randomize, self.num_agents)
        else:
            num_samples = num_envs_to_randomize
            sample_shape = (num_envs_to_randomize,)

        # Sample FOV scales
        fov_scales = torch.empty(sample_shape, device=self.device).uniform_(
            self.cfg.fov_scale_range[0], self.cfg.fov_scale_range[1]
        )

        # Sample focal lengths
        focal_lengths = torch.empty(sample_shape, device=self.device).uniform_(
            self.cfg.focal_length_range[0], self.cfg.focal_length_range[1]
        )

        # Sample target resolutions
        if self.cfg.discrete_resolutions:
            num_options = len(self.cfg.discrete_resolutions)
            indices = torch.randint(num_options, sample_shape, device=self.device)
            resolutions = torch.tensor(
                self.cfg.discrete_resolutions, device=self.device, dtype=torch.int32
            )
            target_sizes = resolutions[indices]  # (num_samples, 2)
        else:
            widths = torch.randint(
                self.cfg.target_width_range[0],
                self.cfg.target_width_range[1] + 1,
                sample_shape,
                device=self.device,
            )
            heights = torch.randint(
                self.cfg.target_height_range[0],
                self.cfg.target_height_range[1] + 1,
                sample_shape,
                device=self.device,
            )
            target_sizes = torch.stack([widths, heights], dim=-1)

        # Expand to per-agent if not randomizing per agent
        if not self.cfg.randomize_per_agent:
            fov_scales = fov_scales.unsqueeze(-1).expand(-1, self.num_agents)
            focal_lengths = focal_lengths.unsqueeze(-1).expand(-1, self.num_agents)
            target_sizes = target_sizes.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Update buffers
        self.fov_scales[env_ids] = fov_scales
        self.focal_lengths[env_ids] = focal_lengths
        self.target_sizes[env_ids] = target_sizes

        # Recompute transforms for affected environments
        self._compute_transforms_for_envs(env_ids)

    def _compute_transforms_for_envs(self, env_ids: torch.Tensor):
        """Compute crop parameters and intrinsics for specific environments."""
        W_render = self.cfg.render_width
        H_render = self.cfg.render_height

        fov_scales = self.fov_scales[env_ids]
        focal_lengths = self.focal_lengths[env_ids]
        target_sizes = self.target_sizes[env_ids]

        # Compute crop dimensions
        crop_w = (W_render * fov_scales).int()
        crop_h = (H_render * fov_scales).int()
        crop_x = ((W_render - crop_w.float()) / 2.0).int()
        crop_y = ((H_render - crop_h.float()) / 2.0).int()

        # Update crop parameters
        self.crop_params[env_ids, :, 0] = crop_x.float()
        self.crop_params[env_ids, :, 1] = crop_y.float()
        self.crop_params[env_ids, :, 2] = crop_w.float()
        self.crop_params[env_ids, :, 3] = crop_h.float()

        # Compute intrinsics
        W_target = target_sizes[:, :, 0].float()
        H_target = target_sizes[:, :, 1].float()
        scale = W_target / crop_w.float().clamp(min=1.0)
        f_final = focal_lengths * scale

        cx_final = W_target / 2.0
        cy_final = H_target / 2.0

        self.intrinsic_matrices[env_ids, :, 0, 0] = f_final
        self.intrinsic_matrices[env_ids, :, 1, 1] = f_final
        self.intrinsic_matrices[env_ids, :, 0, 2] = cx_final
        self.intrinsic_matrices[env_ids, :, 1, 2] = cy_final
        self.intrinsic_matrices[env_ids, :, 2, 2] = 1.0

    def process_images_batched(
        self,
        images: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply crop and resize to rendered images using grid_sample.

        All images are processed to the same output size for batched training.
        The intrinsic matrices are adjusted to match the output size.

        Args:
            images: Rendered images with shape (N, C, H, W) or (N, H, W, C).
                    N should be num_envs * num_agents.
            output_size: Output (H, W) size. If None, uses (360, 640) default.

        Returns:
            processed: Processed images with shape (N, C, H_out, W_out).
            intrinsics: Adjusted intrinsic matrices with shape (N, 3, 3).
        """
        if output_size is None:
            output_size = (360, 640)  # Default nHD

        H_out, W_out = output_size
        N = images.shape[0]
        H_render = self.cfg.render_height
        W_render = self.cfg.render_width

        # Ensure NCHW format
        if images.shape[-1] in [1, 3, 4]:  # Likely NHWC
            images = images.permute(0, 3, 1, 2)

        # Flatten crop params to (N, 4)
        crop_params = self.crop_params.view(N, 4)

        # Build sampling grids
        grids = self._build_crop_grids(
            crop_params, (H_render, W_render), output_size
        )

        # Apply grid sample for differentiable crop+resize
        processed = F.grid_sample(
            images.float(),
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Compute intrinsics for fixed output size
        intrinsics = self._compute_intrinsics_for_output(output_size)

        return processed, intrinsics

    def _build_crop_grids(
        self,
        crop_params: torch.Tensor,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Build sampling grids for grid_sample.

        Args:
            crop_params: Crop parameters (N, 4) with (crop_x, crop_y, crop_w, crop_h).
            input_size: Input image size (H, W).
            output_size: Output image size (H, W).

        Returns:
            grids: Sampling grids with shape (N, H_out, W_out, 2).
        """
        N = crop_params.shape[0]
        H_in, W_in = input_size
        H_out, W_out = output_size

        crop_x = crop_params[:, 0]
        crop_y = crop_params[:, 1]
        crop_w = crop_params[:, 2]
        crop_h = crop_params[:, 3]

        # Create normalized output grid coordinates [-1, 1]
        y_out = torch.linspace(-1, 1, H_out, device=self.device)
        x_out = torch.linspace(-1, 1, W_out, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_out, x_out, indexing="ij")

        # Expand grid for batch
        grid_x = grid_x.unsqueeze(0).expand(N, -1, -1)  # (N, H_out, W_out)
        grid_y = grid_y.unsqueeze(0).expand(N, -1, -1)

        # Compute input coordinates for crop region
        # grid_sample uses [-1, 1] where -1 is left/top edge, 1 is right/bottom edge
        # Map output [-1, 1] to crop region in input [-1, 1]

        # Crop region bounds in normalized coordinates
        x_min = 2.0 * crop_x / W_in - 1.0  # (N,)
        x_max = 2.0 * (crop_x + crop_w) / W_in - 1.0
        y_min = 2.0 * crop_y / H_in - 1.0
        y_max = 2.0 * (crop_y + crop_h) / H_in - 1.0

        # Reshape for broadcasting
        x_min = x_min.view(N, 1, 1)
        x_max = x_max.view(N, 1, 1)
        y_min = y_min.view(N, 1, 1)
        y_max = y_max.view(N, 1, 1)

        # Map output grid [-1, 1] to input crop region
        # output -1 -> x_min, output 1 -> x_max
        grid_x_mapped = (grid_x + 1) / 2 * (x_max - x_min) + x_min
        grid_y_mapped = (grid_y + 1) / 2 * (y_max - y_min) + y_min

        # Stack to (N, H_out, W_out, 2) - grid_sample expects (x, y) order
        grids = torch.stack([grid_x_mapped, grid_y_mapped], dim=-1)

        return grids

    def _compute_intrinsics_for_output(
        self, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """Compute intrinsic matrices for a fixed output size.

        Args:
            output_size: Output image size (H, W).

        Returns:
            intrinsics: Intrinsic matrices with shape (N, 3, 3).
        """
        H_out, W_out = output_size
        N = self.num_envs * self.num_agents

        # Flatten parameters
        focal_lengths = self.focal_lengths.view(N)
        crop_w = self.crop_params[:, :, 2].view(N)

        # Compute scale from crop to output
        scale = W_out / crop_w.clamp(min=1.0)

        # Final focal length
        f_final = focal_lengths * scale

        # Centered principal points
        cx = W_out / 2.0
        cy = H_out / 2.0

        # Build intrinsic matrices
        intrinsics = torch.zeros(N, 3, 3, device=self.device)
        intrinsics[:, 0, 0] = f_final
        intrinsics[:, 1, 1] = f_final
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        intrinsics[:, 2, 2] = 1.0

        return intrinsics

    def get_intrinsic_matrices(self) -> torch.Tensor:
        """Get current intrinsic matrices for all cameras.

        Returns:
            Intrinsic matrices with shape (num_envs, num_agents, 3, 3).
        """
        return self.intrinsic_matrices.clone()

    def get_crop_params(self) -> torch.Tensor:
        """Get current crop parameters for all cameras.

        Returns:
            Crop parameters with shape (num_envs, num_agents, 4).
            Each entry is (crop_x, crop_y, crop_w, crop_h).
        """
        return self.crop_params.clone()

    def get_fov_scales(self) -> torch.Tensor:
        """Get current FOV scales for all cameras.

        Returns:
            FOV scales with shape (num_envs, num_agents).
        """
        return self.fov_scales.clone()
