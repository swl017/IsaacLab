"""
Main delay system orchestration for multi-agent environments.

This module coordinates all delay components to provide realistic
time-shifting phenomena for sim-to-real transfer.
"""

from __future__ import annotations
import torch
from typing import Dict, List, Optional
from isaaclab.envs.common import AgentID

from .agent_states import AgentStates
from .delay_system_cfg import DelaySystemCfg
from .timestamp_manager import TimestampManager
from .sampler_chain import SamplerChain
from .stochastic_sampler import StochasticSampler
from .field_configs import (
    FIELD_GROUPS,
    RAW_SENSOR_FIELDS,
    DERIVED_FIELDS,
    create_motion_sampler,
    create_orientation_sampler,
    create_joint_sampler,
    create_detection_sampler,
    create_communication_sampler,
    create_passthrough_sampler,
    get_all_groups,
    get_raw_field_group,
    get_derived_fields_by_dependency_level,
    get_derived_field_info,
)
from . import derived_field_computers


class DelaySystem:
    """
    Main delay system for multi-agent environment.

    Coordinates samplers, timestamp tracking, and perspective-specific delays
    to simulate realistic sensor latency, communication delays, and data staleness.
    """

    def __init__(
        self,
        cfg: DelaySystemCfg,
        agent_ids: List[AgentID],
        num_envs: int,
        num_joints: int,
        num_targets: int,
        device: torch.device,
    ):
        """
        Args:
            cfg: Delay system configuration
            agent_ids: List of agent identifiers
            num_envs: Number of parallel environments
            num_joints: Number of joints per agent
            num_targets: Number of targets per agent
            device: Device to allocate tensors on
        """
        self.cfg = cfg
        self.agent_ids = agent_ids
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.num_targets = num_targets
        self.device = device

        # Current simulation time
        self.t_sim = torch.zeros(num_envs, device=device)

        # Ground truth states storage (most recent)
        self.gt_states: Dict[AgentID, AgentStates] = {}
        for agent_id in agent_ids:
            self.gt_states[agent_id] = AgentStates(num_envs, num_joints, num_targets, device)

        # Delayed states storage (for each agent's perspective)
        # Structure: delayed_states[ego_agent][other_agent] = delayed AgentStates
        self.delayed_states: Dict[AgentID, Dict[AgentID, AgentStates]] = {}
        for ego_agent in agent_ids:
            self.delayed_states[ego_agent] = {}
            for other_agent in agent_ids:
                self.delayed_states[ego_agent][other_agent] = AgentStates(
                    num_envs, num_joints, num_targets, device
                )

        # Initialize timestamp manager
        field_groups = get_all_groups()
        self.timestamp_manager = TimestampManager(agent_ids, field_groups, num_envs, device)

        # Create base samplers (shared by all agents)
        self._create_base_samplers()

        # Create perspective-specific communication samplers
        self._create_communication_samplers()

    def _create_base_samplers(self):
        """Create base samplers PER AGENT.

        CRITICAL: Each agent must have its own sampler instances because samplers
        like FirstOrderLagSampler maintain internal state (filtered_state) that would
        get corrupted if different agents' data passes through the same sampler.

        Structure: self.motion_samplers[agent_id][field] = sampler
        """
        # Motion samplers: {agent_id: {field: sampler}}
        self.motion_samplers: Dict[AgentID, Dict[str, SamplerChain]] = {}
        for agent_id in self.agent_ids:
            self.motion_samplers[agent_id] = {}
            for field in FIELD_GROUPS['motion']:
                self.motion_samplers[agent_id][field] = create_motion_sampler(
                    field_name=field,
                    num_envs=self.num_envs,
                    time_constant=self.cfg.motion_time_constant,
                    dt=self.cfg.dt_sim,
                    device=self.device,
                )

        # Orientation samplers: {agent_id: {field: sampler}}
        self.orientation_samplers: Dict[AgentID, Dict[str, SamplerChain]] = {}
        for agent_id in self.agent_ids:
            self.orientation_samplers[agent_id] = {}
            for field in FIELD_GROUPS['orientation']:
                self.orientation_samplers[agent_id][field] = create_orientation_sampler(
                    num_envs=self.num_envs,
                    time_constant=self.cfg.orientation_time_constant,
                    dt=self.cfg.dt_sim,
                    device=self.device,
                )

        # Joint samplers: {agent_id: {field: sampler}}
        self.joint_samplers: Dict[AgentID, Dict[str, SamplerChain]] = {}
        for agent_id in self.agent_ids:
            self.joint_samplers[agent_id] = {}
            for field in FIELD_GROUPS['joints']:
                self.joint_samplers[agent_id][field] = create_joint_sampler(
                    num_joints=self.num_joints,
                    num_envs=self.num_envs,
                    time_constant=self.cfg.joint_time_constant,
                    dt=self.cfg.dt_sim,
                    device=self.device,
                )

        # Zoom sampler: {agent_id: sampler}
        self.zoom_samplers: Dict[AgentID, SamplerChain] = {}
        for agent_id in self.agent_ids:
            self.zoom_samplers[agent_id] = create_motion_sampler(
                field_name='camera_zoom_level',
                num_envs=self.num_envs,
                time_constant=self.cfg.joint_time_constant,  # Similar to joints (mechanical actuator)
                dt=self.cfg.dt_sim,
                device=self.device,
            )

        # Detection bbox samplers: {agent_id: sampler}
        self.detection_bbox_samplers: Dict[AgentID, StochasticSampler] = {}
        for agent_id in self.agent_ids:
            self.detection_bbox_samplers[agent_id] = create_detection_sampler(
                fps_mean=self.cfg.detection_fps_mean,
                fps_std=self.cfg.detection_fps_std,
                latency_mean=self.cfg.detection_latency_mean,
                latency_std=self.cfg.detection_latency_std,
                dropout_rate=self.cfg.detection_dropout_rate,
                num_envs=self.num_envs,
                dt=self.cfg.dt_sim,
                device=self.device,
            )

        # Detection rays samplers: {agent_id: sampler}
        self.detection_rays_samplers: Dict[AgentID, StochasticSampler] = {}
        for agent_id in self.agent_ids:
            self.detection_rays_samplers[agent_id] = create_detection_sampler(
                fps_mean=self.cfg.detection_fps_mean,
                fps_std=self.cfg.detection_fps_std,
                latency_mean=self.cfg.detection_latency_mean,
                latency_std=self.cfg.detection_latency_std,
                dropout_rate=self.cfg.detection_dropout_rate,
                num_envs=self.num_envs,
                dt=self.cfg.dt_sim,
                device=self.device,
            )

        # Passthrough sampler (stateless, can be shared)
        self.passthrough_sampler = create_passthrough_sampler(
            num_envs=self.num_envs,
            device=self.device,
        )

    def _create_communication_samplers(self):
        """Create communication samplers PER AGENT for ego vs. other perspectives.

        CRITICAL: Each agent must have its own sampler instances because StochasticSampler
        maintains internal state (held_data) that would get corrupted if different agents'
        data passes through the same sampler.

        Structure: self.ego_comm_samplers[agent_id][field] = sampler
        """
        # Collect all field names that need communication samplers
        comm_field_groups = ['motion', 'orientation', 'joints', 'zoom', 'detection_bbox', 'detection_rays']
        all_comm_fields = []
        for group in comm_field_groups:
            all_comm_fields.extend(FIELD_GROUPS.get(group, []))

        # Ego communication samplers (fast, local) - per AGENT, per FIELD
        self.ego_comm_samplers: Dict[AgentID, Dict[str, StochasticSampler]] = {}
        for agent_id in self.agent_ids:
            self.ego_comm_samplers[agent_id] = {}
            for field_name in all_comm_fields:
                self.ego_comm_samplers[agent_id][field_name] = create_communication_sampler(
                    rate_min=1.0 / self.cfg.dt_sim,  # Run at sim rate
                    rate_max=1.0 / self.cfg.dt_sim,
                    latency_mean=self.cfg.ego_comm_latency,
                    latency_std=0.0,  # Constant
                    dropout_rate=0.0,  # No dropout for ego
                    num_envs=self.num_envs,
                    dt=self.cfg.dt_sim,
                    device=self.device,
                )

        # Inter-agent communication samplers (slow, network) - per AGENT, per FIELD
        self.inter_agent_comm_samplers: Dict[AgentID, Dict[str, StochasticSampler]] = {}
        for agent_id in self.agent_ids:
            self.inter_agent_comm_samplers[agent_id] = {}
            for field_name in all_comm_fields:
                self.inter_agent_comm_samplers[agent_id][field_name] = create_communication_sampler(
                    rate_min=self.cfg.inter_agent_comm_rate_min,
                    rate_max=self.cfg.inter_agent_comm_rate_max,
                    latency_mean=self.cfg.inter_agent_comm_latency_mean,
                    latency_std=self.cfg.inter_agent_comm_latency_std,
                    dropout_rate=self.cfg.inter_agent_comm_dropout_rate,
                    num_envs=self.num_envs,
                    dt=self.cfg.dt_sim,
                    device=self.device,
                )

    def update_agent_gt_states(self, agent_id: AgentID, states: AgentStates):
        """
        Record ground truth states for an agent.

        Args:
            agent_id: Agent identifier
            states: Ground truth AgentStates
        """
        # Store ground truth
        self.gt_states[agent_id] = states

        # Compute derived fields for GT states (camera_position_w, camera_orientation_w, etc.)
        # This ensures GT states have all fields populated for debugging/visualization
        self._compute_derived_fields_inplace(states)

        # Update delayed states for all perspectives
        self._update_delayed_states_for_agent(agent_id)

    def process_noisy_gt_states(
        self,
        agent_id: AgentID,
        noisy_gt_states: AgentStates,
        is_ego: bool = True,
    ) -> AgentStates:
        """
        Process noisy ground-truth states through delay pipeline WITHOUT storing.

        Used for v2.0 noise architecture where noise is injected to raw sensors
        before delays, enabling derived fields to be computed from noisy delayed inputs.

        This method applies the same processing as update_agent_gt_states but:
        1. Does NOT store the GT states
        2. Does NOT update delayed_states for all perspectives
        3. Returns delayed states with specified perspective (ego or inter-agent)

        Args:
            agent_id: Agent identifier
            noisy_gt_states: Ground truth AgentStates with noise already injected to raw sensors
            is_ego: If True, apply ego perspective (fast local comm).
                   If False, apply inter-agent perspective (slow network comm).

        Returns:
            AgentStates with delays applied and derived fields recomputed

        Example usage (in wrapper):
            # For ego noisy states:
            gt_states = self._agent_states[agent_id]
            noisy_gt = copy.deepcopy(gt_states)
            self._inject_gaussian_noise_to_raw_sensors(noisy_gt, agent_id)
            delayed_noisy = delay_system.process_noisy_gt_states(agent_id, noisy_gt, is_ego=True)

            # For other agent noisy states (with inter-agent comm delay):
            delayed_noisy_other = delay_system.process_noisy_gt_states(agent_id, noisy_gt, is_ego=False)
        """
        # Apply base processing (delays + derived field computation) using per-agent samplers
        base_processed = self._apply_base_processing(agent_id, noisy_gt_states)

        # Apply communication based on perspective using per-agent comm samplers
        delayed_noisy = self._apply_communication(
            agent_id,
            base_processed,
            is_ego=is_ego,
        )

        return delayed_noisy

    def _update_delayed_states_for_agent(self, agent_id: AgentID):
        """
        Update delayed states for an agent across all perspectives.

        Args:
            agent_id: Agent whose states are being updated
        """
        gt_states = self.gt_states[agent_id]

        # Apply base processing (motion filter, detection) using per-agent samplers
        base_processed_states = self._apply_base_processing(agent_id, gt_states)

        # Update for each ego agent's perspective
        for ego_agent in self.agent_ids:
            if ego_agent == agent_id:
                # Ego perspective: fast local communication
                final_states = self._apply_communication(
                    agent_id,
                    base_processed_states,
                    is_ego=True,
                )
            else:
                # Other agent perspective: slow inter-agent communication
                final_states = self._apply_communication(
                    agent_id,
                    base_processed_states,
                    is_ego=False,
                )

            # Store delayed states
            self.delayed_states[ego_agent][agent_id] = final_states

    def _apply_base_processing(self, agent_id: AgentID, states: AgentStates) -> AgentStates:
        """
        Apply base processing with Phase 2 derived field pipeline.

        Phase 2 Architecture:
        1. Apply delays to RAW sensor fields only (using per-agent samplers)
        2. Recompute DERIVED fields from delayed raw sensor inputs
        3. Process in dependency order (level 1, then level 2, etc.)

        Args:
            agent_id: Agent identifier (used to select per-agent samplers)
            states: Ground truth AgentStates

        Returns:
            AgentStates with delays applied to raw sensors and derived fields recomputed
        """
        processed_states = AgentStates(
            self.num_envs, self.num_joints, self.num_targets, self.device
        )

        # ========== Step 1: Apply delays to RAW sensor fields ==========

        # Get sampler for each raw field based on its group (using per-agent samplers)
        for field_name, field_info in RAW_SENSOR_FIELDS.items():
            group = field_info['group']
            raw_data = getattr(states.data, field_name)

            # Select appropriate per-agent sampler based on group
            if group == 'motion':
                sampler = self.motion_samplers[agent_id].get(field_name)
            elif group == 'orientation':
                sampler = self.orientation_samplers[agent_id].get(field_name)
            elif group == 'joints':
                sampler = self.joint_samplers[agent_id].get(field_name)
            elif group == 'zoom':
                sampler = self.zoom_samplers[agent_id]
            elif group == 'detection_bbox':
                sampler = self.detection_bbox_samplers[agent_id]
            elif group == 'camera_intrinsics':
                sampler = self.passthrough_sampler  # Stateless, can be shared
            else:
                # Unknown group, skip
                setattr(processed_states.data, field_name, raw_data.clone())
                continue

            # Apply delay
            if sampler is not None:
                delayed_data, info = sampler.update(raw_data, self.t_sim)
                setattr(processed_states.data, field_name, delayed_data)
            else:
                # No sampler for this field, just copy
                setattr(processed_states.data, field_name, raw_data.clone())

        # ========== Step 2: Recompute DERIVED fields from delayed raw inputs ==========

        # Process in dependency order (level 1, then level 2, etc.)
        derived_by_level = get_derived_fields_by_dependency_level()
        for level in sorted(derived_by_level.keys()):
            for field_name in derived_by_level[level]:
                field_info = get_derived_field_info(field_name)

                # Gather delayed source data
                source_data = {}
                for source_field in field_info['sources']:
                    source_data[source_field] = getattr(processed_states.data, source_field)

                # Handle extra parameters (e.g., num_targets)
                if 'extra_params' in field_info:
                    for param_name, param_value in field_info['extra_params'].items():
                        # Evaluate self references
                        if isinstance(param_value, str) and param_value.startswith('self.'):
                            attr_name = param_value.split('.')[1]
                            source_data[param_name] = getattr(self, attr_name)
                        else:
                            source_data[param_name] = param_value

                # Get computation function
                compute_fn = getattr(derived_field_computers, field_info['compute_fn'])

                # Compute derived field
                result = compute_fn(**source_data)

                # Handle functions that return tuples (e.g., combined angular velocity)
                if field_info.get('returns_tuple', False):
                    tuple_fields = field_info['tuple_fields']
                    for i, output_field in enumerate(tuple_fields):
                        setattr(processed_states.data, output_field, result[i])
                else:
                    setattr(processed_states.data, field_name, result)

        # ========== Step 3: Copy timestamps ==========

        processed_states.data.timestamp_sim_walltime = states.data.timestamp_sim_walltime.clone()
        processed_states.data.timestamp_motion = states.data.timestamp_motion.clone()
        processed_states.data.timestamp_detection = states.data.timestamp_detection.clone()

        return processed_states

    def _compute_derived_fields_inplace(self, states: AgentStates) -> None:
        """
        Compute derived fields for an AgentStates object IN-PLACE.

        This is used for GT states where we want derived fields computed
        but without any delay processing.

        Args:
            states: AgentStates object to update with derived field values
        """
        # Process in dependency order (level 1, then level 2, etc.)
        derived_by_level = get_derived_fields_by_dependency_level()
        for level in sorted(derived_by_level.keys()):
            for field_name in derived_by_level[level]:
                field_info = get_derived_field_info(field_name)

                # Gather source data from current states
                source_data = {}
                for source_field in field_info['sources']:
                    source_data[source_field] = getattr(states.data, source_field)

                # Handle extra parameters (e.g., num_targets)
                if 'extra_params' in field_info:
                    for param_name, param_value in field_info['extra_params'].items():
                        # Evaluate self references
                        if isinstance(param_value, str) and param_value.startswith('self.'):
                            attr_name = param_value.split('.')[1]
                            source_data[param_name] = getattr(self, attr_name)
                        else:
                            source_data[param_name] = param_value

                # Get computation function
                compute_fn = getattr(derived_field_computers, field_info['compute_fn'])

                # Compute derived field
                result = compute_fn(**source_data)

                # Handle functions that return tuples (e.g., combined angular velocity)
                if field_info.get('returns_tuple', False):
                    tuple_fields = field_info['tuple_fields']
                    for i, output_field in enumerate(tuple_fields):
                        setattr(states.data, output_field, result[i])
                else:
                    setattr(states.data, field_name, result)

    def _apply_communication(
        self, agent_id: AgentID, states: AgentStates, is_ego: bool
    ) -> AgentStates:
        """
        Apply communication delays and recompute derived fields.

        Args:
            agent_id: Agent identifier (used to select per-agent comm samplers)
            states: Base-processed AgentStates
            is_ego: True for ego perspective, False for other agents

        Returns:
            AgentStates with communication delays applied and derived fields recomputed
        """
        comm_states = AgentStates(
            self.num_envs, self.num_joints, self.num_targets, self.device
        )

        # Select appropriate per-agent communication sampler set
        comm_samplers = (
            self.ego_comm_samplers[agent_id] if is_ego
            else self.inter_agent_comm_samplers[agent_id]
        )

        # Apply communication to each field using its dedicated per-agent, per-field sampler
        # CRITICAL: Each agent MUST have its own samplers to avoid state corruption
        # when different agents' data passes through sequentially.

        comm_field_groups = ['motion', 'orientation', 'joints', 'zoom', 'detection_bbox', 'detection_rays']
        for group in comm_field_groups:
            for field_name in FIELD_GROUPS[group]:
                data = getattr(states.data, field_name)
                comm_data, info = comm_samplers[field_name].update(data, self.t_sim)
                setattr(comm_states.data, field_name, comm_data)

        # Static fields don't need communication delay
        for field_name in FIELD_GROUPS['camera_intrinsics']:
            data = getattr(states.data, field_name)
            setattr(comm_states.data, field_name, data.clone())

        # Copy timestamps
        comm_states.data.timestamp_sim_walltime = states.data.timestamp_sim_walltime.clone()
        comm_states.data.timestamp_motion = states.data.timestamp_motion.clone()
        comm_states.data.timestamp_detection = states.data.timestamp_detection.clone()

        # CRITICAL: Recompute derived fields from communication-delayed inputs
        # This ensures fields like camera_position_w are computed from the
        # communication-delayed body_position_w and body_orientation_w, not stale values.
        self._compute_derived_fields_inplace(comm_states)

        return comm_states

    def get_ego_states(self, agent_id: AgentID) -> AgentStates:
        """
        Get ego states for an agent (fast local processing).

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStates with ego perspective delays applied
        """
        return self.delayed_states[agent_id][agent_id]

    def get_other_agent_states(self, ego_agent: AgentID, other_agent: AgentID) -> AgentStates:
        """
        Get other agent's states from ego agent's perspective.

        Args:
            ego_agent: Ego agent identifier
            other_agent: Other agent identifier

        Returns:
            AgentStates with full delays (base + inter-agent comm)
        """
        return self.delayed_states[ego_agent][other_agent]

    def get_all_agent_states_for_ego(self, ego_agent: AgentID) -> Dict[AgentID, AgentStates]:
        """
        Get all agent states from ego agent's perspective.

        Args:
            ego_agent: Ego agent identifier

        Returns:
            Dict mapping agent_id → delayed states
            - ego_agent: Fast ego processing
            - other agents: Slow inter-agent communication
        """
        return self.delayed_states[ego_agent]

    def step(self, dt: Optional[float] | Optional[torch.Tensor] = None):
        """
        Advance time in the delay system.

        Args:
            dt: Time step. If None, uses cfg.dt_sim.
        """
        if dt is None:
            dt = self.cfg.dt_sim

        self.t_sim += dt

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """
        Reset delay system for specified environments.

        Args:
            env_ids: Indices of environments to reset. None means reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset simulation time
        self.t_sim[env_ids] = 0.0

        # Reset all per-agent samplers
        for agent_id in self.agent_ids:
            # Motion samplers
            for sampler in self.motion_samplers[agent_id].values():
                sampler.reset(env_ids)
            # Orientation samplers
            for sampler in self.orientation_samplers[agent_id].values():
                sampler.reset(env_ids)
            # Joint samplers
            for sampler in self.joint_samplers[agent_id].values():
                sampler.reset(env_ids)
            # Zoom sampler
            self.zoom_samplers[agent_id].reset(env_ids)
            # Detection samplers
            self.detection_bbox_samplers[agent_id].reset(env_ids)
            self.detection_rays_samplers[agent_id].reset(env_ids)

        # Passthrough sampler (shared, stateless)
        self.passthrough_sampler.reset(env_ids)

        # Reset per-agent communication samplers
        for agent_id in self.agent_ids:
            for sampler in self.ego_comm_samplers[agent_id].values():
                sampler.reset(env_ids)
            for sampler in self.inter_agent_comm_samplers[agent_id].values():
                sampler.reset(env_ids)

        # Reset timestamp manager
        self.timestamp_manager.reset(env_ids=env_ids)

    # ==================== Curriculum Learning Hooks ====================

    def set_motion_latency_params(
        self,
        time_constant: Optional[float] = None,
    ):
        """
        Update motion sensor parameters for curriculum learning.

        Hook for external curriculum module to dynamically adjust motion sensor
        characteristics during training.

        Args:
            time_constant: New time constant for motion first-order lag filters.
                          If None, parameter is not updated.

        Note:
            - Motion samplers use first-order lag filtering (no stochastic delay)
            - Time constant controls smoothing: smaller = faster response, larger = more filtering
            - Updates apply to all motion and orientation samplers
        """
        if time_constant is not None:
            # Update per-agent motion samplers
            for agent_id in self.agent_ids:
                for sampler in self.motion_samplers[agent_id].values():
                    sampler.update_config(time_constant=time_constant)

            # Update per-agent orientation samplers
            for agent_id in self.agent_ids:
                for sampler in self.orientation_samplers[agent_id].values():
                    sampler.update_config(time_constant=time_constant)

            # Update config for consistency
            self.cfg.motion_time_constant = time_constant
            self.cfg.orientation_time_constant = time_constant

    def set_detection_latency_params(
        self,
        fps_mean: Optional[float] = None,
        fps_std: Optional[float] = None,
        latency_mean: Optional[float] = None,
        latency_std: Optional[float] = None,
        dropout_rate: Optional[float] = None,
    ):
        """
        Update detection sensor parameters for curriculum learning.

        Hook for external curriculum module to dynamically adjust detection sensor
        characteristics during training.

        Args:
            fps_mean: New mean detection rate (Hz). If None, not updated.
            fps_std: New std deviation of detection rate (Hz). If None, not updated.
            latency_mean: New mean processing latency (seconds). If None, not updated.
            latency_std: New std deviation of latency (seconds). If None, not updated.
            dropout_rate: New frame dropout rate [0, 1]. If None, not updated.

        Note:
            - FPS parameters control sampling period: period = 1/fps
            - Updates apply to both bbox and rays detection samplers
            - Parameters use normal distributions internally
        """
        # Convert FPS to period parameters if needed
        period_mean = None
        period_std = None
        if fps_mean is not None:
            period_mean = 1.0 / fps_mean
        if fps_std is not None and fps_mean is not None:
            # Convert FPS std to period std: std_period ≈ std_fps / (fps_mean^2)
            period_std = fps_std / (fps_mean ** 2)

        # Update per-agent detection samplers
        for agent_id in self.agent_ids:
            for sampler in [self.detection_bbox_samplers[agent_id], self.detection_rays_samplers[agent_id]]:
                sampler.update_config(
                    period_mean=period_mean,
                    period_std=period_std,
                    latency_mean=latency_mean,
                    latency_std=latency_std,
                    dropout_prob=dropout_rate,
                )

        # Update config for consistency
        if fps_mean is not None:
            self.cfg.detection_fps_mean = fps_mean
        if fps_std is not None:
            self.cfg.detection_fps_std = fps_std
        if latency_mean is not None:
            self.cfg.detection_latency_mean = latency_mean
        if latency_std is not None:
            self.cfg.detection_latency_std = latency_std
        if dropout_rate is not None:
            self.cfg.detection_dropout_rate = dropout_rate

    def set_comm_latency_params(
        self,
        inter_agent_latency_mean: Optional[float] = None,
        inter_agent_latency_std: Optional[float] = None,
        inter_agent_dropout_rate: Optional[float] = None,
        inter_agent_rate_min: Optional[float] = None,
        inter_agent_rate_max: Optional[float] = None,
    ):
        """
        Update inter-agent communication parameters for curriculum learning.

        Hook for external curriculum module to dynamically adjust inter-agent
        communication characteristics during training.

        Args:
            inter_agent_latency_mean: New mean network latency (seconds). If None, not updated.
            inter_agent_latency_std: New std deviation of latency (seconds). If None, not updated.
            inter_agent_dropout_rate: New packet dropout rate [0, 1]. If None, not updated.
            inter_agent_rate_min: New minimum communication rate (Hz). If None, not updated.
            inter_agent_rate_max: New maximum communication rate (Hz). If None, not updated.

        Note:
            - Ego communication is always fast (sim rate) and not updated
            - Inter-agent communication models network delays and jitter
            - Rate parameters use uniform distribution
            - Latency uses normal distribution
        """
        # Convert rate to period parameters if needed
        period_min = None
        period_max = None
        if inter_agent_rate_min is not None and inter_agent_rate_max is not None:
            period_min = 1.0 / inter_agent_rate_max  # Note: max rate → min period
            period_max = 1.0 / inter_agent_rate_min  # Note: min rate → max period

        # Update per-agent inter-agent communication samplers
        for agent_id in self.agent_ids:
            for sampler in self.inter_agent_comm_samplers[agent_id].values():
                sampler.update_config(
                    period_min=period_min,
                    period_max=period_max,
                    latency_mean=inter_agent_latency_mean,
                    latency_std=inter_agent_latency_std,
                    dropout_prob=inter_agent_dropout_rate,
                )

        # Update config for consistency
        if inter_agent_latency_mean is not None:
            self.cfg.inter_agent_comm_latency_mean = inter_agent_latency_mean
        if inter_agent_latency_std is not None:
            self.cfg.inter_agent_comm_latency_std = inter_agent_latency_std
        if inter_agent_dropout_rate is not None:
            self.cfg.inter_agent_comm_dropout_rate = inter_agent_dropout_rate
        if inter_agent_rate_min is not None:
            self.cfg.inter_agent_comm_rate_min = inter_agent_rate_min
        if inter_agent_rate_max is not None:
            self.cfg.inter_agent_comm_rate_max = inter_agent_rate_max

    def set_time_constants(
        self,
        motion_tc: Optional[float] = None,
        orientation_tc: Optional[float] = None,
        joint_tc: Optional[float] = None,
        zoom_tc: Optional[float] = None,
    ):
        """
        Update time constants for all first-order lag samplers.

        Hook for external curriculum module to dynamically adjust filtering
        characteristics during training.

        Args:
            motion_tc: New time constant for motion samplers (position, velocity). If None, not updated.
            orientation_tc: New time constant for orientation samplers (quaternions). If None, not updated.
            joint_tc: New time constant for joint samplers (joint positions, velocities). If None, not updated.
            zoom_tc: New time constant for zoom sampler (camera zoom level). If None, not updated.

        Note:
            - Time constant controls first-order lag filtering: x_filt = x_filt + alpha * (x_meas - x_filt)
            - Smaller tc → faster response, less filtering
            - Larger tc → slower response, more filtering
            - All samplers use continuous filtering (not sample-and-hold)
        """
        if motion_tc is not None:
            for agent_id in self.agent_ids:
                for sampler in self.motion_samplers[agent_id].values():
                    sampler.update_config(time_constant=motion_tc)
            self.cfg.motion_time_constant = motion_tc

        if orientation_tc is not None:
            for agent_id in self.agent_ids:
                for sampler in self.orientation_samplers[agent_id].values():
                    sampler.update_config(time_constant=orientation_tc)
            self.cfg.orientation_time_constant = orientation_tc

        if joint_tc is not None:
            for agent_id in self.agent_ids:
                for sampler in self.joint_samplers[agent_id].values():
                    sampler.update_config(time_constant=joint_tc)
            self.cfg.joint_time_constant = joint_tc

        if zoom_tc is not None:
            for agent_id in self.agent_ids:
                self.zoom_samplers[agent_id].update_config(time_constant=zoom_tc)
            # No separate config for zoom_tc, it uses joint_tc
