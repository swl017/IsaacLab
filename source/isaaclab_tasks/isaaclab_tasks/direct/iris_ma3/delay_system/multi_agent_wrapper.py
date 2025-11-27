"""
Multi-Agent Delay System Wrapper (v2.2)

Dual-pipeline architecture for proper noise-before-delay simulation.

Key Architecture:
- TWO separate DelaySystem instances:
  - Clean pipeline (for rewards): GT → delays → clean delayed states
  - Noisy pipeline (for observations): GT → noise injection → delays → noisy delayed states
- Noise is injected BEFORE delays (physically correct)
- View scheme API for unified state access

API:
- get_all_states_for_rewards(ego_id) → Dict[agent_id → AgentStates] (clean)
- get_all_states_for_observations(ego_id) → Dict[agent_id → AgentStates] (noisy)
"""

from __future__ import annotations
from typing import Dict, List, Any
import torch
import copy

from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .delay_system import DelaySystem
from .delay_system_cfg import DelaySystemCfg
from .agent_states import AgentStates


class MultiAgentDelaySystem:
    """
    Multi-agent delay system with dual pipeline architecture (v2.2).

    This wrapper maintains TWO separate DelaySystem instances:
    1. Clean pipeline (for rewards) - processes clean GT through delays
    2. Noisy pipeline (for observations) - processes noisy GT through delays

    The noise is injected BEFORE delays, which is physically correct because:
    - Noise represents sensor measurement errors at the point of sensing
    - Delays represent processing/communication latency AFTER sensing
    """

    def __init__(
        self,
        possible_agents: List[str],
        num_envs: int,
        num_joints_per_agent: Dict[str, int],
        num_targets_per_agent: Dict[str, int],
        dt: float,
        device: torch.device,
        # Noise parameters (curriculum-compatible)
        enable_noise: bool = False,
        position_noise_std: float = 0.0,
        orientation_noise_std: float = 0.0,
        linear_velocity_noise_std: float = 0.0,
        angular_velocity_noise_std: float = 0.0,
        linear_acceleration_noise_std: float = 0.0,
        gimbal_noise_std: float = 0.0,
        bbox_noise_std: float = 0.0,
        zoom_noise_std: float = 0.0,
        noise_seed: int = 0,
    ):
        """
        Initialize multi-agent delay system with dual pipeline.

        Args:
            possible_agents: List of agent IDs (e.g., ["agent_0", "agent_1", ...])
            num_envs: Number of parallel environments
            num_joints_per_agent: Dict mapping agent_id -> number of joints
            num_targets_per_agent: Dict mapping agent_id -> number of targets
            dt: Physics timestep (seconds)
            device: torch.device for all tensors
            enable_noise: Enable observation noise
            *_noise_std: Noise standard deviations for curriculum
            noise_seed: Random seed for reproducibility
        """
        # Store configuration
        self._possible_agents = possible_agents
        self._num_envs = num_envs
        self._num_joints_per_agent = num_joints_per_agent
        self._num_targets_per_agent = num_targets_per_agent
        self._dt = dt
        self._device = device

        # Noise configuration
        self._enable_noise = enable_noise
        self._noise_progress = 0.0  # Curriculum scale [0, 1]
        self._noise_stds = {
            'position': position_noise_std,
            'orientation': orientation_noise_std,
            'linear_velocity': linear_velocity_noise_std,
            'angular_velocity': angular_velocity_noise_std,
            'linear_acceleration': linear_acceleration_noise_std,
            'gimbal': gimbal_noise_std,
            'bbox': bbox_noise_std,
            'zoom': zoom_noise_std,
        }

        # Create noise generators (one per agent for reproducibility)
        self._noise_generators: Dict[str, torch.Generator] = {}
        for i, agent_id in enumerate(possible_agents):
            gen = torch.Generator(device=device)
            gen.manual_seed(noise_seed + i)
            self._noise_generators[agent_id] = gen

        # Camera configuration storage
        self._camera_configs: Dict[str, Dict[str, Any]] = {}

        # Create AgentStates buffers for each agent (ground truth storage)
        self._agent_states: Dict[str, AgentStates] = {}
        for agent_id in possible_agents:
            self._agent_states[agent_id] = AgentStates(
                num_envs=num_envs,
                num_joints=num_joints_per_agent[agent_id],
                num_targets=num_targets_per_agent[agent_id],
                device=device,
            )

        # Get common parameters (assumes homogeneous agents)
        first_agent = possible_agents[0]
        num_joints = num_joints_per_agent[first_agent]
        num_targets = num_targets_per_agent[first_agent]

        # ====================================================================
        # DUAL PIPELINE: Two separate DelaySystem instances
        # ====================================================================

        # Clean pipeline (for rewards) - processes clean GT
        self._delay_system_clean = DelaySystem(
            cfg=DelaySystemCfg(),
            agent_ids=possible_agents,
            num_envs=num_envs,
            num_joints=num_joints,
            num_targets=num_targets,
            device=device,
        )

        # Noisy pipeline (for observations) - processes GT with noise
        self._delay_system_noisy = DelaySystem(
            cfg=DelaySystemCfg(),
            agent_ids=possible_agents,
            num_envs=num_envs,
            num_joints=num_joints,
            num_targets=num_targets,
            device=device,
        )

    def set_camera_configs(
        self,
        agent_id: str,
        width: int,
        height: int,
        focal_length: float,
        horizontal_aperture: float,
        vertical_aperture: float,
        offset_position_b: torch.Tensor,  # [3]
        offset_rotation_b: torch.Tensor,  # [4] quaternion
    ) -> None:
        """
        Set camera configuration for an agent.

        Constructs the camera intrinsics matrix K and stores camera offset.
        """
        # Construct intrinsics matrix
        K = self._construct_intrinsics_matrix(
            width=width,
            height=height,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        )

        # Store camera config
        self._camera_configs[agent_id] = {
            'intrinsics': K,
            'offset_position_b': offset_position_b,
            'offset_rotation_b': offset_rotation_b,
        }

        # Update AgentStates buffer with camera config
        states = self._agent_states[agent_id]
        states.data.camera_offset_position_b[:] = offset_position_b
        states.data.camera_offset_rotation_b[:] = offset_rotation_b
        states.data.camera_base_intrinsics[:] = K

    def update_time(self) -> None:
        """
        Advance simulation time by dt for BOTH pipelines.

        CRITICAL: Must be called at start of each environment step,
        before any state updates or queries.
        """
        self._delay_system_clean.step(self._dt)
        self._delay_system_noisy.step(self._dt)

    def set_noise_progress_scale(self, progress: float) -> None:
        """
        Set curriculum progress for noise scaling.

        Noise is scaled as: actual_noise = base_noise * progress

        Args:
            progress: Curriculum progress in [0.0, 1.0]
                - 0.0: No noise (perfect observations)
                - 1.0: Full noise (configured std values)
        """
        self._noise_progress = max(0.0, min(1.0, progress))

    def update_gt_states(
        self,
        agent_id: str,
        body_position_w: torch.Tensor,           # [N, 3]
        body_orientation_w: torch.Tensor,        # [N, 4] quaternion
        body_linear_velocity_w: torch.Tensor,    # [N, 3]
        body_angular_velocity_w: torch.Tensor,   # [N, 3]
        body_linear_acceleration_w: torch.Tensor,  # [N, 3]
        body_combined_angular_velocity_w: torch.Tensor,  # [N, 3]
        joint_positions_b: torch.Tensor,         # [N, J]
        zoom_level: torch.Tensor,                # [N] or [N, 1]
    ) -> None:
        """
        Update ground-truth states for an agent.

        This feeds states to BOTH pipelines:
        1. Clean pipeline: GT → delay system
        2. Noisy pipeline: GT → noise injection → delay system
        """
        # Get AgentStates buffer
        states = self._agent_states[agent_id]

        # Ensure zoom_level is [N] not [N, 1]
        if zoom_level.dim() == 2 and zoom_level.shape[1] == 1:
            zoom_level = zoom_level.squeeze(1)

        # Update state fields in GT buffer
        states.data.body_position_w = body_position_w
        states.data.body_orientation_w = body_orientation_w
        states.data.body_linear_velocity_w = body_linear_velocity_w
        states.data.body_angular_velocity_w = body_angular_velocity_w
        states.data.body_linear_acceleration_w = body_linear_acceleration_w
        states.data.body_combined_angular_velocity_w = body_combined_angular_velocity_w
        states.data.joint_positions_b = joint_positions_b
        states.data.camera_zoom_level = zoom_level

        # ====================================================================
        # Feed to BOTH pipelines
        # ====================================================================

        # 1. Clean pipeline: direct GT
        self._delay_system_clean.update_agent_gt_states(agent_id, states)

        # 2. Noisy pipeline: inject noise BEFORE delays
        noisy_states = copy.deepcopy(states)
        self._inject_gaussian_noise_to_raw_sensors(noisy_states, agent_id)
        self._delay_system_noisy.update_agent_gt_states(agent_id, noisy_states)

    def update_detections(
        self,
        agent_id: str,
        bboxes_2d_gt: torch.Tensor,  # [N, T, 4] (x, y, w, h)
    ) -> None:
        """
        Update detection data (bounding boxes) for an agent.

        Feeds to BOTH pipelines with noise applied to noisy pipeline.

        Note: bboxes_2d_valid_mask has been REMOVED. Users should validate
        bboxes AFTER delay processing using:
            bbox_valid = bbox_raycaster.validate_bbox(delayed_bboxes)
        """
        # Update in GT buffer
        states = self._agent_states[agent_id]
        states.data.bboxes_2d = bboxes_2d_gt

        # Note: DelaySystem updates are handled in update_agent_gt_states
        # The bboxes are stored in AgentStates and processed through the pipeline

    # ========================================================================
    # View Scheme API (v2.2)
    # ========================================================================

    def get_all_states_for_rewards(self, ego_agent_id: str) -> Dict[str, AgentStates]:
        """
        Get ALL agent states for REWARD computation (clean, no noise).

        Uses the clean pipeline to provide accurate delayed states for:
        - Triangulation covariance computation
        - Formation quality rewards
        - Any reward that needs accurate ground truth

        Args:
            ego_agent_id: The ego agent requesting states

        Returns:
            Dict[agent_id -> AgentStates] where:
            - ego_agent_id: States with ego perspective delays (fast local)
            - other agents: States with inter-agent comm delays (slow network)

        Example:
            all_states = delay_system.get_all_states_for_rewards("agent_0")
            ego = all_states["agent_0"]
            other = all_states["agent_1"]
        """
        return self._delay_system_clean.get_all_agent_states_for_ego(ego_agent_id)

    def get_all_states_for_observations(self, ego_agent_id: str) -> Dict[str, AgentStates]:
        """
        Get ALL agent states for OBSERVATION computation (noisy).

        Uses the noisy pipeline to provide realistic observations:
        - Sensor noise applied BEFORE delays (physically correct)
        - Derived fields computed from noisy delayed sensors

        Args:
            ego_agent_id: The ego agent requesting states

        Returns:
            Dict[agent_id -> AgentStates] where:
            - ego_agent_id: Noisy states with ego perspective delays
            - other agents: Noisy states with inter-agent comm delays

        Example:
            all_states = delay_system.get_all_states_for_observations("agent_0")
            ego = all_states["agent_0"]
            other = all_states["agent_1"]

            # Validate bboxes AFTER getting delayed states
            bbox_valid = bbox_raycaster.validate_bbox(ego.data.bboxes_2d)
        """
        return self._delay_system_noisy.get_all_agent_states_for_ego(ego_agent_id)

    def broadcast_state(
        self,
        sender_id: str,
        state_keys: List[str],
    ) -> None:
        """
        Broadcast selected state fields from sender to all other agents.

        Note: Communication is handled automatically by DelaySystem during
        update_agent_gt_states(). This method exists for API compatibility.
        """
        # Communication handled automatically by DelaySystem
        pass

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Reset delay system for specified environments.

        Resets BOTH pipelines.
        """
        self._delay_system_clean.reset(env_ids)
        self._delay_system_noisy.reset(env_ids)

        # Reset noise RNG
        for agent_id in self._possible_agents:
            gen = self._noise_generators[agent_id]
            seed = gen.initial_seed()
            gen.manual_seed(seed)

    # ========================================================================
    # Curriculum Learning Hooks
    # ========================================================================

    def set_time_constants(
        self,
        motion_tc: float = None,
        orientation_tc: float = None,
        joint_tc: float = None,
        zoom_tc: float = None,
    ) -> None:
        """
        Set time constants for first-order lag filters (BOTH pipelines).
        """
        self._delay_system_clean.set_time_constants(
            motion_tc=motion_tc,
            orientation_tc=orientation_tc,
            joint_tc=joint_tc,
            zoom_tc=zoom_tc,
        )
        self._delay_system_noisy.set_time_constants(
            motion_tc=motion_tc,
            orientation_tc=orientation_tc,
            joint_tc=joint_tc,
            zoom_tc=zoom_tc,
        )

    def set_detection_latency_params(
        self,
        fps_mean: float = None,
        latency_mean: float = None,
        dropout_rate: float = None,
    ) -> None:
        """
        Set detection latency parameters (BOTH pipelines).
        """
        self._delay_system_clean.set_detection_latency_params(
            fps_mean=fps_mean,
            latency_mean=latency_mean,
            dropout_rate=dropout_rate,
        )
        self._delay_system_noisy.set_detection_latency_params(
            fps_mean=fps_mean,
            latency_mean=latency_mean,
            dropout_rate=dropout_rate,
        )

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _construct_intrinsics_matrix(
        self,
        width: int,
        height: int,
        focal_length: float,
        horizontal_aperture: float,
        vertical_aperture: float,
    ) -> torch.Tensor:
        """Construct camera intrinsics matrix K [3, 3]."""
        fx = focal_length * width / horizontal_aperture
        fy = focal_length * height / vertical_aperture
        cx = width / 2.0
        cy = height / 2.0

        K = torch.tensor([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=torch.float32, device=self._device)

        return K

    def _inject_gaussian_noise_to_raw_sensors(
        self,
        states: AgentStates,
        agent_id: str,
    ) -> AgentStates:
        """
        Inject Gaussian noise to RAW sensor fields.

        This is called BEFORE passing states to the noisy delay pipeline,
        ensuring noise is applied before delays (physically correct).

        Applied to RAW SENSORS:
        - body_position_w, body_orientation_w
        - body_linear/angular_velocity_w
        - body_linear_acceleration_w
        - joint_positions_b
        - bboxes_2d, camera_zoom_level

        NOT applied to DERIVED fields (computed from noisy inputs later).
        """
        if not self._enable_noise or self._noise_progress == 0.0:
            return states

        scale = self._noise_progress
        rng = self._noise_generators[agent_id]

        # Position
        if self._noise_stds['position'] > 0:
            states.data.body_position_w = states.data.body_position_w + torch.randn(
                states.data.body_position_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['position'] * scale

        # Orientation (quaternion perturbation)
        if self._noise_stds['orientation'] > 0:
            states.data.body_orientation_w = self._perturb_quaternion(
                states.data.body_orientation_w,
                self._noise_stds['orientation'] * scale,
                rng
            )

        # Linear velocity
        if self._noise_stds['linear_velocity'] > 0:
            states.data.body_linear_velocity_w = states.data.body_linear_velocity_w + torch.randn(
                states.data.body_linear_velocity_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['linear_velocity'] * scale

        # Angular velocity
        if self._noise_stds['angular_velocity'] > 0:
            states.data.body_angular_velocity_w = states.data.body_angular_velocity_w + torch.randn(
                states.data.body_angular_velocity_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['angular_velocity'] * scale

        # Linear acceleration
        if self._noise_stds['linear_acceleration'] > 0:
            states.data.body_linear_acceleration_w = states.data.body_linear_acceleration_w + torch.randn(
                states.data.body_linear_acceleration_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['linear_acceleration'] * scale

        # Gimbal joint positions
        if self._noise_stds['gimbal'] > 0:
            states.data.joint_positions_b = states.data.joint_positions_b + torch.randn(
                states.data.joint_positions_b.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['gimbal'] * scale

        # Bounding boxes
        if self._noise_stds['bbox'] > 0:
            states.data.bboxes_2d = self._inject_realistic_bbox_noise(
                bboxes=states.data.bboxes_2d,
                position_noise_std=self._noise_stds['bbox'] * scale,
                scale_noise_std=self._noise_stds['bbox'] * scale * 0.1,
                scale_dependency_factor=0.5,
                aspect_ratio_noise_range=0.1,
                rng=rng,
            )

        # Zoom level
        if self._noise_stds['zoom'] > 0:
            states.data.camera_zoom_level = states.data.camera_zoom_level + torch.randn(
                states.data.camera_zoom_level.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['zoom'] * scale
            states.data.camera_zoom_level = torch.clamp(states.data.camera_zoom_level, min=1.0)

        return states

    def _perturb_quaternion(
        self,
        quat: torch.Tensor,  # [N, 4]
        angle_std: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Perturb quaternion with small rotation noise."""
        N = quat.shape[0]

        # Random axes (uniform on sphere)
        axis = torch.randn(N, 3, generator=rng, device=self._device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

        # Random angles
        angle = torch.randn(N, generator=rng, device=self._device) * angle_std

        # Convert to quaternion
        noise_quat = quat_from_angle_axis(angle, axis)

        # Apply perturbation
        perturbed = quat_mul(quat, noise_quat)

        # Normalize
        return perturbed / (perturbed.norm(dim=-1, keepdim=True) + 1e-8)

    def _inject_realistic_bbox_noise(
        self,
        bboxes: torch.Tensor,  # [N, T, 4] (x, y, w, h)
        position_noise_std: float,
        scale_noise_std: float,
        scale_dependency_factor: float = 0.5,
        aspect_ratio_noise_range: float = 0.1,
        rng: torch.Generator = None,
    ) -> torch.Tensor:
        """Inject realistic bounding box noise with scale dependence."""
        x = bboxes[..., 0]
        y = bboxes[..., 1]
        w = bboxes[..., 2]
        h = bboxes[..., 3]

        # Scale-dependent noise factor
        area = w * h + 1e-6
        sqrt_area = torch.sqrt(area)
        scale_factor = 1.0 + scale_dependency_factor / sqrt_area

        # Position noise
        x_noise = torch.randn(x.shape, device=x.device, generator=rng) * position_noise_std * scale_factor
        y_noise = torch.randn(y.shape, device=y.device, generator=rng) * position_noise_std * scale_factor

        # Scale noise (shared for aspect preservation)
        common_scale_noise = torch.randn(w.shape, device=w.device, generator=rng) * scale_noise_std
        aspect_ratio_noise = (torch.rand(w.shape, device=w.device, generator=rng) - 0.5) * 2 * aspect_ratio_noise_range

        w_noisy = w * (1.0 + common_scale_noise + aspect_ratio_noise)
        h_noisy = h * (1.0 + common_scale_noise - aspect_ratio_noise)

        # Clamp to valid values
        w_noisy = torch.clamp(w_noisy, min=1.0)
        h_noisy = torch.clamp(h_noisy, min=1.0)

        return torch.stack([x + x_noise, y + y_noise, w_noisy, h_noisy], dim=-1)
