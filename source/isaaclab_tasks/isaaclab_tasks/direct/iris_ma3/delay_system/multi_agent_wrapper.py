"""
Multi-Agent Delay System Wrapper

API-compatible wrapper around DelaySystem to enable drop-in replacement
of MultiAgentStateManager in existing environments.

Key features:
- Exact API match to MultiAgentStateManager
- Uses DelaySystem internally for principled delay/noise modeling
- Separates delayed (clean) states for rewards vs delayed+noisy states for observations
- Supports noise curriculum scaling
- Handles camera config → intrinsics matrix conversion
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import torch
import copy

from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .delay_system import DelaySystem
from .delay_system_cfg import DelaySystemCfg
from .agent_states import AgentStates


class MultiAgentDelaySystem:
    """
    Multi-agent delay system with API compatibility for MultiAgentStateManager.

    This wrapper provides the same API as MultiAgentStateManager but uses the
    new DelaySystem internally for more principled delay and noise modeling.
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
        Initialize multi-agent delay system.

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

        # Create AgentStates buffers for each agent
        self._agent_states: Dict[str, AgentStates] = {}
        for agent_id in possible_agents:
            self._agent_states[agent_id] = AgentStates(
                num_envs=num_envs,
                num_joints=num_joints_per_agent[agent_id],
                num_targets=num_targets_per_agent[agent_id],
                device=device,
            )

        # Create DelaySystem with default config
        # Note: Using default DelaySystemCfg for now. In the future, this could be
        # parameterized to allow custom delay/noise characteristics.
        #
        # DelaySystem assumes all agents have the same number of joints/targets.
        # For heterogeneous agents, we'd need to modify DelaySystem or create separate instances.
        # For now, we assume homogeneous agents (valid for iris_ma_env3).
        first_agent = possible_agents[0]
        num_joints = num_joints_per_agent[first_agent]
        num_targets = num_targets_per_agent[first_agent]

        self._delay_system = DelaySystem(
            cfg=DelaySystemCfg(),  # Use default config
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

        Constructs the camera intrinsics matrix K:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]

        Where:
            fx = focal_length * width / horizontal_aperture
            fy = focal_length * height / vertical_aperture
            cx = width / 2.0
            cy = height / 2.0

        Args:
            agent_id: Agent identifier
            width: Image width (pixels)
            height: Image height (pixels)
            focal_length: Physical focal length (mm)
            horizontal_aperture: Sensor width (mm)
            vertical_aperture: Sensor height (mm)
            offset_position_b: Camera offset position in body frame [3]
            offset_rotation_b: Camera offset rotation quaternion [4] (w, x, y, z)
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
        # Broadcast to all environments
        states = self._agent_states[agent_id]
        states.data.camera_offset_position_b[:] = offset_position_b
        states.data.camera_offset_rotation_b[:] = offset_rotation_b
        states.data.camera_base_intrinsics[:] = K

    def update_time(self) -> None:
        """
        Advance simulation time by dt.

        This updates the internal clock used for:
        - FPS throttling (detector runs at 20 Hz, not every timestep)
        - Latency simulation (data becomes available after delay)
        - Staleness tracking (age of received data)

        CRITICAL: Must be called at start of each environment step,
        before any state updates or queries.
        """
        self._delay_system.step(self._dt)

    def set_noise_progress_scale(self, progress: float) -> None:
        """
        Set curriculum progress for noise scaling.

        Noise is scaled as: actual_noise = base_noise * progress

        Args:
            progress: Curriculum progress in [0.0, 1.0]
                - 0.0: No noise (perfect observations)
                - 1.0: Full noise (configured std values)

        Typical usage in environment:
            progress = (step - start) / (end - start)
            state_manager.set_noise_progress_scale(progress)
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

        This method:
        1. Populates AgentStates object with provided fields
        2. Computes derived fields (camera pose, combined velocities)
        3. Passes to DelaySystem for delay application
        4. Internally creates delayed (clean) states

        Args:
            agent_id: Agent identifier
            body_position_w: Body position in world frame
            body_orientation_w: Body orientation quaternion (w, x, y, z)
            body_linear_velocity_w: Linear velocity in world frame
            body_angular_velocity_w: Angular velocity in world frame
            body_linear_acceleration_w: Linear acceleration in world frame
            body_combined_angular_velocity_w: Body + gimbal angular velocity
            joint_positions_b: Gimbal joint angles [pitch, yaw, ...]
            zoom_level: Optical zoom level (1.0 = no zoom)
        """
        # Get AgentStates buffer for this agent
        states = self._agent_states[agent_id]

        # Ensure zoom_level is [N] not [N, 1]
        if zoom_level.dim() == 2 and zoom_level.shape[1] == 1:
            zoom_level = zoom_level.squeeze(1)

        # Update state fields
        states.data.body_position_w = body_position_w
        states.data.body_orientation_w = body_orientation_w
        states.data.body_linear_velocity_w = body_linear_velocity_w
        states.data.body_angular_velocity_w = body_angular_velocity_w
        states.data.body_linear_acceleration_w = body_linear_acceleration_w
        states.data.body_combined_angular_velocity_w = body_combined_angular_velocity_w
        states.data.joint_positions_b = joint_positions_b
        states.data.camera_zoom_level = zoom_level

        # Pass to DelaySystem for processing
        # DelaySystem will:
        # - Apply delays to raw sensor fields
        # - Recompute derived fields from delayed inputs
        # - Store delayed states internally
        self._delay_system.update_agent_gt_states(agent_id, states)

    def update_detections(
        self,
        agent_id: str,
        bboxes_2d_gt: torch.Tensor,  # [N, T, 4] (x, y, w, h)
    ) -> None:
        """
        Update detection data (bounding boxes) for an agent.

        Applies detector-specific characteristics:
        - FPS throttling (20 Hz update rate)
        - Detection latency (300ms avg)
        - Dropout (occasional missed detections)
        - Pixel noise (bbox jitter)

        Args:
            agent_id: Agent identifier
            bboxes_2d_gt: Ground-truth bounding boxes in pixels [N, T, 4]

        Note:
            Must be called AFTER update_gt_states() for camera pose.

        IMPORTANT: bboxes_2d_valid_mask has been REMOVED from the delay pipeline.
        Users should validate bboxes AFTER delay processing using:
            bbox_valid = bbox_raycaster.validate_bbox(delayed_bboxes)  # [N, T]
        This avoids confusion between frame-level validity and detection validity.
        """
        # Update bboxes in AgentStates buffer
        states = self._agent_states[agent_id]
        states.data.bboxes_2d = bboxes_2d_gt

        # DelaySystem will handle:
        # - FPS throttle (sample-and-hold at 20 Hz)
        # - Latency (stochastic ~300ms)
        # - Dropout (random detection failures)
        # Note: Pixel noise applied later in get_delayed_noisy_states()

    def broadcast_state(
        self,
        sender_id: str,
        state_keys: List[str],
    ) -> None:
        """
        Broadcast selected state fields from sender to all other agents.

        Simulates wireless communication with:
        - Communication latency (~100-200ms)
        - Packet loss (dropout)
        - Age tracking (total data age = sensor delay + comm delay)

        Args:
            sender_id: Agent broadcasting their state
            state_keys: List of field names to broadcast, e.g.:
                ['position', 'orientation', 'linear_velocity',
                 'combined_angular_velocity', 'bboxes_2d',
                 'bboxes_2d_valid_mask', 'camera_ray_directions_w']

        Typical usage:
            for agent_id in agents:
                state_manager.broadcast_state(
                    sender_id=agent_id,
                    state_keys=['position', 'orientation', ...]
                )
        """
        # Map state_keys to AgentStates field names
        # NOTE: bboxes_2d_valid_mask has been REMOVED from the delay pipeline.
        FIELD_NAME_MAP = {
            'position': 'body_position_w',
            'orientation': 'body_orientation_w',
            'linear_velocity': 'body_linear_velocity_w',
            'combined_angular_velocity': 'body_combined_angular_velocity_w',
            'bboxes_2d': 'bboxes_2d',
            'camera_ray_directions_w': 'camera_ray_directions_w',
            'camera_intrinsics': 'camera_base_intrinsics',
            'joint_positions': 'joint_positions_b',
        }

        # Convert state_keys to AgentStates field names
        field_names = [FIELD_NAME_MAP.get(key, key) for key in state_keys]

        # DelaySystem handles communication internally via _apply_communication()
        # which is called during update_agent_gt_states()
        # We just need to trigger the communication by updating the broadcast fields
        # Note: In DelaySystem, communication is handled in _apply_communication()
        # which is called automatically during the step() method.
        pass  # Communication handled automatically by DelaySystem

    def get_delayed_states(self, agent_id: str) -> AgentStates:
        """
        Get delayed (clean) states for rewards.

        Returns delayed states WITHOUT noise injection.
        Used for:
        - Reward computation (need accurate delayed ground truth)
        - Triangulation covariance (need clean delayed geometry)

        Characteristics:
        - Sensor delays applied (motion ~10ms, detection ~300ms)
        - FPS throttling applied (detection at 20 Hz)
        - NO observation noise
        - Derived fields computed from delayed raw sensors

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStates with delayed but noise-free data
        """
        return self._delay_system.get_ego_states(agent_id)

    def get_delayed_noisy_states(self, agent_id: str) -> AgentStates:
        """
        Get delayed AND noisy states for observations.

        Returns delayed states WITH noise injection.
        Used for:
        - Agent observations (realistic sensor noise)
        - Policy input (what the agent actually sees)

        v2.0 Architecture (BREAKING CHANGE):
        - Noise injected to RAW SENSORS before delays
        - Derived fields computed from noisy delayed sensors
        - Preserves geometric consistency

        Characteristics:
        - Gaussian noise added to RAW sensors:
            * Position (pos_std * progress)
            * Orientation (ori_std * progress)
            * Linear velocity (lin_vel_std * progress)
            * Angular velocity (ang_vel_std * progress)
            * Gimbal angles (gimbal_std * progress)
            * Bounding boxes (bbox_std * progress)
            * Zoom level (zoom_std * progress)
        - Sensor delays applied to noisy sensors
        - FPS throttling applied
        - Derived fields computed from delayed noisy inputs
        - Noise scaled by curriculum progress

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStates with delayed and noisy data
        """
        # Get ground-truth states (clean)
        gt_states = self._agent_states[agent_id]

        # Clone to avoid modifying GT
        noisy_gt = copy.deepcopy(gt_states)

        # Inject noise to RAW sensors BEFORE delays (v2.0)
        self._inject_gaussian_noise_to_raw_sensors(noisy_gt, agent_id)

        # Process noisy GT through delay system
        # This will apply delays and recompute derived fields from noisy delayed inputs
        delayed_noisy = self._delay_system.process_noisy_gt_states(agent_id, noisy_gt)

        return delayed_noisy

    def receive_other_agent_states(
        self,
        receiver_id: str,
    ) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Get states received from other agents via communication.

        Returns states broadcasted by other agents, including:
        - Communication delay (additional latency on top of sensor delay)
        - Packet loss (some states may be missing)
        - Age tracking (total age = sensor delay + comm delay)

        Args:
            receiver_id: Agent receiving the data

        Returns:
            Dict[sender_id -> Dict[state_key -> (data, valid_mask, data_age)]]

            Where:
            - sender_id: ID of agent who sent this data
            - state_key: Field name (e.g., 'position', 'bboxes_2d')
            - data: Tensor with actual data [N, ...]
            - valid_mask: Boolean mask [N] (False if dropped/stale)
            - data_age: Age of data in seconds [N] (sensor delay + comm delay)

        Example:
            received = state_manager.receive_other_agent_states("agent_0")

            if "agent_1" in received and received["agent_1"] is not None:
                other_state = received["agent_1"]
                pos = other_state['position'][0]      # [N, 3]
                pos_valid = other_state['position'][1]  # [N] bool
                pos_age = other_state['position'][2]    # [N] seconds
        """
        # Get all agent states from DelaySystem
        all_states = self._delay_system.get_all_agent_states_for_ego(receiver_id)

        # Filter out receiver's own state and convert to expected dict format
        received_states = {}
        for sender_id, agent_states in all_states.items():
            if sender_id != receiver_id and agent_states is not None:
                # Convert AgentStates to Dict[state_key -> (data, valid_mask, data_age)]
                received_states[sender_id] = self._agent_states_to_dict(agent_states)

        return received_states

    def _agent_states_to_dict(
        self,
        agent_states: AgentStates,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Convert AgentStates to dictionary format for legacy compatibility.

        Maps full field names to short names and wraps data in tuples:
        (data, valid_mask, data_age)

        Field name mapping:
        - body_position_w -> position
        - body_orientation_w -> orientation
        - joint_positions_b -> joint_positions
        - camera_base_intrinsics -> camera_intrinsics
        - camera_ray_directions_w -> camera_ray_directions_w

        SIMPLIFIED ARCHITECTURE (v2.1):
        - bboxes_2d_valid_mask has been REMOVED from the delay pipeline
        - Users should validate bboxes AFTER delay processing using:
            bbox_valid = bbox_raycaster.validate_bbox(delayed_bboxes)  # [N, T]
        - All valid_mask fields are set to True (always valid)
        - All data_age fields are set to 0 (use timestamps if needed)

        Args:
            agent_states: AgentStates object from DelaySystem

        Returns:
            Dict[state_key -> (data, valid_mask, data_age)] with short field names
        """
        N = self._num_envs

        # Simplified: All fields are always valid, age tracking via timestamps
        default_valid_mask = torch.ones(N, dtype=torch.bool, device=self._device)
        default_data_age = torch.zeros(N, dtype=torch.float32, device=self._device)

        state_dict = {
            # Motion states
            'position': (
                agent_states.data.body_position_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
            'orientation': (
                agent_states.data.body_orientation_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
            'linear_velocity': (
                agent_states.data.body_linear_velocity_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
            'angular_velocity': (
                agent_states.data.body_angular_velocity_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),

            # Joint states
            'joint_positions': (
                agent_states.data.joint_positions_b,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),

            # Camera states
            'camera_position': (
                agent_states.data.camera_position_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
            'camera_orientation': (
                agent_states.data.camera_orientation_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
            'camera_intrinsics': (
                agent_states.data.camera_base_intrinsics,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),

            # Detection states (bboxes)
            # NOTE: Users should validate bboxes AFTER using bbox_raycaster.validate_bbox()
            'bboxes_2d': (
                agent_states.data.bboxes_2d,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
            'camera_ray_directions_w': (
                agent_states.data.camera_ray_directions_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            ),
        }

        # Add combined angular velocity if available
        if hasattr(agent_states.data, 'body_combined_angular_velocity_w'):
            state_dict['combined_angular_velocity'] = (
                agent_states.data.body_combined_angular_velocity_w,
                default_valid_mask.clone(),
                default_data_age.clone()
            )

        return state_dict

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Reset delay system for specified environments.

        Clears:
        - Delay buffers (sample-and-hold histories)
        - Communication buffers (pending broadcasts)
        - Timestamp trackers (age counters)
        - Noise generators (RNG state reset to seed)

        Args:
            env_ids: Environment indices to reset [E]

        Note:
            Does NOT reset camera configs (persistent across episodes)
        """
        # Reset DelaySystem
        self._delay_system.reset(env_ids)

        # Reset noise RNG for specified envs (reproducibility)
        # Note: torch.Generator doesn't support per-environment reset,
        # so we reset all generators. This is acceptable for episode resets.
        for agent_id in self._possible_agents:
            gen = self._noise_generators[agent_id]
            seed = gen.initial_seed()
            gen.manual_seed(seed)

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
        """
        Construct camera intrinsics matrix K [3, 3].

        Uses pinhole camera model:
            fx = focal_length * width / horizontal_aperture
            fy = focal_length * height / vertical_aperture
            cx = width / 2.0
            cy = height / 2.0

        Returns:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
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
        Inject Gaussian noise to RAW sensor fields only (v2.0).

        Physically correct approach: noise added to raw sensors BEFORE delays,
        then derived fields are computed from noisy delayed inputs.

        Applied to RAW SENSORS:
        - body_position_w (state estimator)
        - body_orientation_w (IMU)
        - body_linear_velocity_w (state estimator)
        - body_angular_velocity_w (IMU)
        - body_linear_acceleration_w (IMU)
        - joint_positions_b (gimbal encoders)
        - bboxes_2d (object detector)
        - camera_zoom_level (zoom motor encoder)

        NOT applied to DERIVED fields (computed from noisy inputs):
        - camera_position_w (from noisy body_position_w + offset)
        - camera_orientation_w (from noisy body_orientation_w + joint_positions_b)
        - camera_ray_directions_w (from noisy camera_orientation_w + bboxes_2d)

        Modifies states in-place, then returns for chaining.

        Args:
            states: AgentStates to modify (should be GT states, not delayed)
            agent_id: Agent identifier (for noise generator)

        Returns:
            Modified states (same object, for chaining)
        """
        if not self._enable_noise or self._noise_progress == 0.0:
            return states

        scale = self._noise_progress
        rng = self._noise_generators[agent_id]

        # Position
        if self._noise_stds['position'] > 0:
            states.data.body_position_w += torch.randn(
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
            states.data.body_linear_velocity_w += torch.randn(
                states.data.body_linear_velocity_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['linear_velocity'] * scale

        # Angular velocity
        if self._noise_stds['angular_velocity'] > 0:
            states.data.body_angular_velocity_w += torch.randn(
                states.data.body_angular_velocity_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['angular_velocity'] * scale

        # Linear acceleration
        if self._noise_stds['linear_acceleration'] > 0:
            states.data.body_linear_acceleration_w += torch.randn(
                states.data.body_linear_acceleration_w.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['linear_acceleration'] * scale

        # Gimbal joint positions
        if self._noise_stds['gimbal'] > 0:
            states.data.joint_positions_b += torch.randn(
                states.data.joint_positions_b.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['gimbal'] * scale

        # Bounding boxes (realistic noise with scale dependence and aspect ratio preservation)
        if self._noise_stds['bbox'] > 0:
            states.data.bboxes_2d = self._inject_realistic_bbox_noise(
                bboxes=states.data.bboxes_2d,
                position_noise_std=self._noise_stds['bbox'] * scale,
                scale_noise_std=self._noise_stds['bbox'] * scale * 0.1,  # 10% of position std for size
                scale_dependency_factor=0.5,
                aspect_ratio_noise_range=0.1,  # ±10% aspect ratio variation
                rng=rng,
            )

        # Zoom level
        if self._noise_stds['zoom'] > 0:
            states.data.camera_zoom_level += torch.randn(
                states.data.camera_zoom_level.shape,
                generator=rng,
                device=self._device
            ) * self._noise_stds['zoom'] * scale
            # Clamp to valid range (zoom >= 1.0)
            states.data.camera_zoom_level = torch.clamp(states.data.camera_zoom_level, min=1.0)

        return states

    def _inject_uniform_noise_to_raw_sensors(
        self,
        states: AgentStates,
        agent_id: str,
    ) -> AgentStates:
        """
        Inject Uniform noise to RAW sensor fields only (v2.0).

        Use cases:
        - Quantization errors (IMU/encoder resolution limits)
        - Bounded perturbations (known sensor error bounds)
        - Domain randomization (wider uniform exploration)

        Noise range: [-noise_std, +noise_std] uniform distribution

        Applied to same RAW SENSORS as Gaussian version:
        - body_position_w, body_orientation_w
        - body_linear/angular_velocity_w
        - joint_positions_b
        - bboxes_2d, camera_zoom_level

        NOT applied to DERIVED fields (computed from noisy inputs).

        Modifies states in-place, then returns for chaining.

        Args:
            states: AgentStates to modify (should be GT states, not delayed)
            agent_id: Agent identifier (for noise generator)

        Returns:
            Modified states (same object, for chaining)
        """
        if not self._enable_noise or self._noise_progress == 0.0:
            return states

        scale = self._noise_progress
        rng = self._noise_generators[agent_id]

        # Helper: uniform noise in [-bound, +bound]
        def uniform_noise(shape, bound):
            return (torch.rand(shape, generator=rng, device=self._device) - 0.5) * 2 * bound

        # Position
        if self._noise_stds['position'] > 0:
            bound = self._noise_stds['position'] * scale
            states.data.body_position_w += uniform_noise(
                states.data.body_position_w.shape, bound
            )

        # Orientation (quaternion perturbation with uniform angles)
        if self._noise_stds['orientation'] > 0:
            angle_bound = self._noise_stds['orientation'] * scale
            states.data.body_orientation_w = self._perturb_quaternion_uniform(
                states.data.body_orientation_w,
                angle_bound,
                rng
            )

        # Linear velocity
        if self._noise_stds['linear_velocity'] > 0:
            bound = self._noise_stds['linear_velocity'] * scale
            states.data.body_linear_velocity_w += uniform_noise(
                states.data.body_linear_velocity_w.shape, bound
            )

        # Angular velocity
        if self._noise_stds['angular_velocity'] > 0:
            bound = self._noise_stds['angular_velocity'] * scale
            states.data.body_angular_velocity_w += uniform_noise(
                states.data.body_angular_velocity_w.shape, bound
            )

        # Linear acceleration
        if self._noise_stds['linear_acceleration'] > 0:
            bound = self._noise_stds['linear_acceleration'] * scale
            states.data.body_linear_acceleration_w += uniform_noise(
                states.data.body_linear_acceleration_w.shape, bound
            )

        # Gimbal joint positions
        if self._noise_stds['gimbal'] > 0:
            bound = self._noise_stds['gimbal'] * scale
            states.data.joint_positions_b += uniform_noise(
                states.data.joint_positions_b.shape, bound
            )

        # Bounding boxes (realistic noise with scale dependence and aspect ratio preservation)
        # Note: Uses Gaussian for scale noise and uniform for aspect, regardless of overall noise type
        if self._noise_stds['bbox'] > 0:
            states.data.bboxes_2d = self._inject_realistic_bbox_noise(
                bboxes=states.data.bboxes_2d,
                position_noise_std=self._noise_stds['bbox'] * scale,
                scale_noise_std=self._noise_stds['bbox'] * scale * 0.1,  # 10% of position std for size
                scale_dependency_factor=0.5,
                aspect_ratio_noise_range=0.1,  # ±10% aspect ratio variation
                rng=rng,
            )

        # Zoom level
        if self._noise_stds['zoom'] > 0:
            bound = self._noise_stds['zoom'] * scale
            states.data.camera_zoom_level += uniform_noise(
                states.data.camera_zoom_level.shape, bound
            )
            # Clamp to valid range (zoom >= 1.0)
            states.data.camera_zoom_level = torch.clamp(states.data.camera_zoom_level, min=1.0)

        return states

    def _perturb_quaternion(
        self,
        quat: torch.Tensor,  # [N, 4]
        angle_std: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """
        Perturb quaternion with small rotation noise.

        Uses axis-angle → quaternion conversion:
        1. Sample random axis (uniform on unit sphere)
        2. Sample angle (Gaussian with std=angle_std)
        3. Convert to quaternion
        4. Multiply with original quaternion
        5. Normalize

        Args:
            quat: Input quaternions [N, 4] (w, x, y, z)
            angle_std: Standard deviation of rotation angle (radians)
            rng: Random number generator

        Returns:
            Perturbed quaternions [N, 4]
        """
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

    def _perturb_quaternion_uniform(
        self,
        quat: torch.Tensor,  # [N, 4]
        angle_bound: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """
        Perturb quaternion with uniform rotation noise.

        Uses axis-angle → quaternion conversion:
        1. Sample random axis (uniform on unit sphere)
        2. Sample angle (Uniform in [-angle_bound, +angle_bound])
        3. Convert to quaternion
        4. Multiply with original quaternion
        5. Normalize

        Args:
            quat: Input quaternions [N, 4] (w, x, y, z)
            angle_bound: Maximum rotation angle (radians)
            rng: Random number generator

        Returns:
            Perturbed quaternions [N, 4]
        """
        N = quat.shape[0]

        # Random axes (uniform on sphere)
        axis = torch.randn(N, 3, generator=rng, device=self._device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

        # Random angles (uniform in [-bound, +bound])
        angle = (torch.rand(N, generator=rng, device=self._device) - 0.5) * 2 * angle_bound

        # Convert to quaternion
        noise_quat = quat_from_angle_axis(angle, axis)

        # Apply perturbation
        perturbed = quat_mul(quat, noise_quat)

        # Normalize
        return perturbed / (perturbed.norm(dim=-1, keepdim=True) + 1e-8)

    def _perturb_quaternion_exp_map(
        self,
        quat: torch.Tensor,  # [N, 4]
        tangent_std: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """
        Perturb quaternion using exponential map (proper tangent space perturbation).

        This method is more numerically stable than axis-angle for very small rotations
        (< 0.5 degrees) because it works directly in the tangent space of SO(3).

        Algorithm:
        1. Sample 3D tangent vector (Gaussian with std=tangent_std)
        2. Compute rotation angle: theta = ||tangent||
        3. For small theta: use first-order approximation q_noise ≈ [1, tangent/2]
        4. For large theta: use full exponential map q_noise = [cos(θ/2), sin(θ/2)*axis]
        5. Apply perturbation: q_perturbed = q * q_noise
        6. Normalize

        Args:
            quat: Input quaternions [N, 4] (w, x, y, z)
            tangent_std: Standard deviation for tangent space noise (radians)
            rng: Random number generator

        Returns:
            Perturbed quaternions [N, 4]

        Note:
            - tangent_std corresponds to the std of the rotation angle in radians
            - For tangent_std < 0.5 degrees (~0.00873 rad), this is more accurate
            - For larger angles, axis-angle method is equally accurate
        """
        N = quat.shape[0]

        # Sample tangent vector in so(3) (Lie algebra of SO(3))
        # Each component has std = tangent_std / sqrt(3) so total magnitude std ≈ tangent_std
        component_std = tangent_std / (3 ** 0.5)
        tangent = torch.randn(N, 3, generator=rng, device=self._device) * component_std

        # Compute rotation angle (magnitude of tangent vector)
        theta = torch.norm(tangent, dim=-1)  # [N]

        # Small angle threshold (below which first-order approximation is accurate)
        small_angle_threshold = 1e-6

        # Create mask for small vs large angles
        small_angle_mask = theta < small_angle_threshold

        # For small angles: first-order approximation
        # q ≈ [1, tangent/2] (unnormalized, close to identity)
        q_small_w = torch.ones(N, device=self._device)
        q_small_xyz = tangent / 2.0  # [N, 3]

        # For large angles: full exponential map
        # q = [cos(θ/2), sin(θ/2) * tangent/θ]
        half_theta = theta / 2.0
        sin_half_theta = torch.sin(half_theta)
        cos_half_theta = torch.cos(half_theta)

        # Avoid division by zero for small theta
        theta_safe = torch.where(theta > small_angle_threshold, theta, torch.ones_like(theta))
        axis = tangent / theta_safe.unsqueeze(-1)  # [N, 3]

        q_large_w = cos_half_theta
        q_large_xyz = sin_half_theta.unsqueeze(-1) * axis  # [N, 3]

        # Select based on angle magnitude
        noise_w = torch.where(small_angle_mask, q_small_w, q_large_w)
        noise_xyz = torch.where(small_angle_mask.unsqueeze(-1), q_small_xyz, q_large_xyz)

        # Construct noise quaternion [w, x, y, z]
        noise_quat = torch.cat([noise_w.unsqueeze(-1), noise_xyz], dim=-1)  # [N, 4]

        # Normalize noise quaternion
        noise_quat = noise_quat / (torch.norm(noise_quat, dim=-1, keepdim=True) + 1e-8)

        # Apply perturbation: q_perturbed = q * q_noise
        perturbed = quat_mul(quat, noise_quat)

        # Normalize result
        return perturbed / (torch.norm(perturbed, dim=-1, keepdim=True) + 1e-8)

    def _perturb_quaternion_auto(
        self,
        quat: torch.Tensor,  # [N, 4]
        angle_std: float,
        rng: torch.Generator,
        exp_map_threshold: float = 0.00873,  # 0.5 degrees in radians
    ) -> torch.Tensor:
        """
        Auto-select quaternion perturbation method based on noise magnitude.

        Automatically switches between:
        - Exponential map: For very small angles (< threshold), more numerically stable
        - Axis-angle: For larger angles, computationally simpler

        Args:
            quat: Input quaternions [N, 4] (w, x, y, z)
            angle_std: Standard deviation of rotation angle (radians)
            rng: Random number generator
            exp_map_threshold: Threshold below which to use exponential map (radians).
                              Default: 0.00873 rad (~0.5 degrees)

        Returns:
            Perturbed quaternions [N, 4]

        Note:
            - Threshold of 0.5 degrees is recommended based on numerical analysis
            - For typical RL training noise levels (1-5 degrees), axis-angle is fine
            - For very precise applications (sub-degree), exponential map is preferred
        """
        if angle_std < exp_map_threshold:
            # Use exponential map for very small angles (more numerically stable)
            return self._perturb_quaternion_exp_map(quat, angle_std, rng)
        else:
            # Use standard axis-angle method for larger angles
            return self._perturb_quaternion(quat, angle_std, rng)

    def _inject_realistic_bbox_noise(
        self,
        bboxes: torch.Tensor,  # [N, T, 4] (x, y, w, h)
        position_noise_std: float,
        scale_noise_std: float,
        scale_dependency_factor: float = 0.5,
        aspect_ratio_noise_range: float = 0.1,
        rng: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Inject realistic bounding box noise with scale dependence and aspect ratio preservation.

        This implements three realistic noise characteristics:
        1. **Scale-dependent position error**: Small bboxes (distant objects) have larger
           relative position error than large bboxes (close objects).
        2. **Scale-dependent size error**: Similar scale dependence for width/height.
        3. **Aspect ratio preservation with variation**: Width and height have correlated
           scale noise (preserves approximate aspect ratio) plus uniform aspect noise.

        Args:
            bboxes: Input bounding boxes [N, T, 4] in (x, y, w, h) format
            position_noise_std: Base std deviation for position noise (pixels)
            scale_noise_std: Base std deviation for scale (w/h) noise (relative)
            scale_dependency_factor: Controls scale dependence strength.
                                    Higher = more noise for small bboxes. Default: 0.5
            aspect_ratio_noise_range: Uniform noise range for aspect ratio perturbation.
                                     Value of 0.1 means ±10% aspect variation. Default: 0.1
            rng: Random number generator. If None, uses default RNG.

        Returns:
            Noisy bounding boxes [N, T, 4] with realistic error characteristics

        Note:
            - Small bboxes → larger relative error (farther objects less precise)
            - Aspect ratio approximately preserved but with realistic variation
            - All noise clipped to ensure bbox validity (w, h >= 1.0)
        """
        # Extract components
        x = bboxes[..., 0]  # [N, T]
        y = bboxes[..., 1]  # [N, T]
        w = bboxes[..., 2]  # [N, T]
        h = bboxes[..., 3]  # [N, T]

        # Compute scale-dependent noise factor
        # Small bboxes (small area) → large scale_factor → more noise
        area = w * h + 1e-6  # Add epsilon to avoid division by zero
        sqrt_area = torch.sqrt(area)
        scale_factor = 1.0 + scale_dependency_factor / sqrt_area  # [N, T]

        # Position noise (scale-dependent)
        x_noise = torch.randn(x.shape, device=x.device, generator=rng) * position_noise_std * scale_factor
        y_noise = torch.randn(y.shape, device=y.device, generator=rng) * position_noise_std * scale_factor

        # Common scale noise (Gaussian) - shared between w and h for aspect preservation
        common_scale_noise = torch.randn(w.shape, device=w.device, generator=rng) * scale_noise_std

        # Aspect ratio perturbation (UNIFORM random)
        # This adds ±aspect_ratio_noise_range variation to aspect ratio
        aspect_ratio_noise = (torch.rand(w.shape, device=w.device, generator=rng) - 0.5) * 2 * aspect_ratio_noise_range

        # Apply noise to width and height
        # Width gets: common scale + positive aspect noise
        # Height gets: common scale - negative aspect noise
        # This preserves approximate aspect ratio while allowing realistic variation
        w_noisy = w * (1.0 + common_scale_noise + aspect_ratio_noise)
        h_noisy = h * (1.0 + common_scale_noise - aspect_ratio_noise)

        # Clamp to ensure valid bboxes
        w_noisy = torch.clamp(w_noisy, min=1.0)
        h_noisy = torch.clamp(h_noisy, min=1.0)

        # Stack and return
        return torch.stack([x + x_noise, y + y_noise, w_noisy, h_noisy], dim=-1)
