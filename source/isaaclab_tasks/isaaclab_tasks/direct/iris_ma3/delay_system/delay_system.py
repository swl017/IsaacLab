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
from .field_configs import (
    FIELD_GROUPS,
    create_motion_sampler,
    create_orientation_sampler,
    create_joint_sampler,
    create_detection_sampler,
    create_communication_sampler,
    create_passthrough_sampler,
    get_all_groups,
)


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
        """Create base samplers for motion, detection, etc. (shared by all agents)."""
        # Motion samplers (one per motion field)
        self.motion_samplers = {}
        for field in FIELD_GROUPS['motion']:
            self.motion_samplers[field] = create_motion_sampler(
                field_name=field,
                num_envs=self.num_envs,
                time_constant=self.cfg.motion_time_constant,
                dt=self.cfg.dt_sim,
                device=self.device,
            )

        # Orientation samplers
        self.orientation_samplers = {}
        for field in FIELD_GROUPS['orientation']:
            self.orientation_samplers[field] = create_orientation_sampler(
                num_envs=self.num_envs,
                time_constant=self.cfg.orientation_time_constant,
                dt=self.cfg.dt_sim,
                device=self.device,
            )

        # Joint samplers
        self.joint_samplers = {}
        for field in FIELD_GROUPS['joints']:
            self.joint_samplers[field] = create_joint_sampler(
                num_joints=self.num_joints,
                num_envs=self.num_envs,
                time_constant=self.cfg.joint_time_constant,
                dt=self.cfg.dt_sim,
                device=self.device,
            )

        # Zoom sampler (first-order lag for mechanical lens movement)
        self.zoom_sampler = create_motion_sampler(
            field_name='camera_zoom_level',
            num_envs=self.num_envs,
            time_constant=self.cfg.joint_time_constant,  # Similar to joints (mechanical actuator)
            dt=self.cfg.dt_sim,
            device=self.device,
        )

        # Detection samplers (separate for dimension compatibility)
        # detection_bbox: [N, T, 4] - bounding boxes
        self.detection_bbox_sampler = create_detection_sampler(
            fps_mean=self.cfg.detection_fps_mean,
            fps_std=self.cfg.detection_fps_std,
            latency_mean=self.cfg.detection_latency_mean,
            latency_std=self.cfg.detection_latency_std,
            dropout_rate=self.cfg.detection_dropout_rate,
            num_envs=self.num_envs,
            dt=self.cfg.dt_sim,
            device=self.device,
        )

        # detection_rays: [N, T, 3] - ray directions and origins
        self.detection_rays_sampler = create_detection_sampler(
            fps_mean=self.cfg.detection_fps_mean,
            fps_std=self.cfg.detection_fps_std,
            latency_mean=self.cfg.detection_latency_mean,
            latency_std=self.cfg.detection_latency_std,
            dropout_rate=self.cfg.detection_dropout_rate,
            num_envs=self.num_envs,
            dt=self.cfg.dt_sim,
            device=self.device,
        )

        # Passthrough samplers for static fields
        self.passthrough_sampler = create_passthrough_sampler(
            num_envs=self.num_envs,
            device=self.device,
        )

    def _create_communication_samplers(self):
        """Create communication samplers for ego vs. other perspectives."""
        # Ego communication (fast, local)
        self.ego_comm_sampler = create_communication_sampler(
            rate_min=1.0 / self.cfg.dt_sim,  # Run at sim rate
            rate_max=1.0 / self.cfg.dt_sim,
            latency_mean=self.cfg.ego_comm_latency,
            latency_std=0.0,  # Constant
            dropout_rate=0.0,  # No dropout for ego
            num_envs=self.num_envs,
            dt=self.cfg.dt_sim,
            device=self.device,
        )

        # Inter-agent communication (slow, network)
        self.inter_agent_comm_sampler = create_communication_sampler(
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

        # Update delayed states for all perspectives
        self._update_delayed_states_for_agent(agent_id)

    def _update_delayed_states_for_agent(self, agent_id: AgentID):
        """
        Update delayed states for an agent across all perspectives.

        Args:
            agent_id: Agent whose states are being updated
        """
        gt_states = self.gt_states[agent_id]

        # Apply base processing (motion filter, detection)
        base_processed_states = self._apply_base_processing(gt_states)

        # Update for each ego agent's perspective
        for ego_agent in self.agent_ids:
            if ego_agent == agent_id:
                # Ego perspective: fast local communication
                final_states = self._apply_communication(
                    base_processed_states,
                    is_ego=True,
                )
            else:
                # Other agent perspective: slow inter-agent communication
                final_states = self._apply_communication(
                    base_processed_states,
                    is_ego=False,
                )

            # Store delayed states
            self.delayed_states[ego_agent][agent_id] = final_states

    def _apply_base_processing(self, states: AgentStates) -> AgentStates:
        """
        Apply base processing (motion filter, detection) to states.

        Args:
            states: Ground truth AgentStates

        Returns:
            AgentStates with base processing applied
        """
        processed_states = AgentStates(
            self.num_envs, self.num_joints, self.num_targets, self.device
        )

        # Process motion fields
        for field_name, sampler in self.motion_samplers.items():
            data = getattr(states.data, field_name)
            filtered_data, info = sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, filtered_data)

        # Process orientation fields
        for field_name, sampler in self.orientation_samplers.items():
            data = getattr(states.data, field_name)
            filtered_data, info = sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, filtered_data)

        # Process joint fields
        for field_name, sampler in self.joint_samplers.items():
            data = getattr(states.data, field_name)
            filtered_data, info = sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, filtered_data)

        # Process zoom field
        for field_name in FIELD_GROUPS['zoom']:
            data = getattr(states.data, field_name)
            filtered_data, info = self.zoom_sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, filtered_data)

        # Process detection_bbox fields
        for field_name in FIELD_GROUPS['detection_bbox']:
            data = getattr(states.data, field_name)
            delayed_data, info = self.detection_bbox_sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, delayed_data)

        # Process detection_rays fields
        for field_name in FIELD_GROUPS['detection_rays']:
            data = getattr(states.data, field_name)
            delayed_data, info = self.detection_rays_sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, delayed_data)

        # Process static fields (passthrough)
        for field_name in FIELD_GROUPS['camera_intrinsics']:
            data = getattr(states.data, field_name)
            passthrough_data, info = self.passthrough_sampler.update(data, self.t_sim)
            setattr(processed_states.data, field_name, passthrough_data)

        # Copy timestamps (update later)
        processed_states.data.timestamp_sim_walltime = states.data.timestamp_sim_walltime.clone()
        processed_states.data.timestamp_motion = states.data.timestamp_motion.clone()
        processed_states.data.timestamp_detection = states.data.timestamp_detection.clone()

        return processed_states

    def _apply_communication(self, states: AgentStates, is_ego: bool) -> AgentStates:
        """
        Apply communication delays.

        Args:
            states: Base-processed AgentStates
            is_ego: True for ego perspective, False for other agents

        Returns:
            AgentStates with communication delays applied
        """
        comm_states = AgentStates(
            self.num_envs, self.num_joints, self.num_targets, self.device
        )

        # Select appropriate communication sampler
        comm_sampler = self.ego_comm_sampler if is_ego else self.inter_agent_comm_sampler

        # Apply communication to all fields (as a group)
        # For simplicity, we apply to entire state. In practice, could be more selective.
        for field_name in FIELD_GROUPS['motion']:
            data = getattr(states.data, field_name)
            comm_data, info = comm_sampler.update(data, self.t_sim)
            setattr(comm_states.data, field_name, comm_data)

        for field_name in FIELD_GROUPS['orientation']:
            data = getattr(states.data, field_name)
            comm_data, info = comm_sampler.update(data, self.t_sim)
            setattr(comm_states.data, field_name, comm_data)

        for field_name in FIELD_GROUPS['joints']:
            data = getattr(states.data, field_name)
            comm_data, info = comm_sampler.update(data, self.t_sim)
            setattr(comm_states.data, field_name, comm_data)

        for field_name in FIELD_GROUPS['zoom']:
            data = getattr(states.data, field_name)
            comm_data, info = comm_sampler.update(data, self.t_sim)
            setattr(comm_states.data, field_name, comm_data)

        for field_name in FIELD_GROUPS['detection_bbox']:
            data = getattr(states.data, field_name)
            comm_data, info = comm_sampler.update(data, self.t_sim)
            setattr(comm_states.data, field_name, comm_data)

        for field_name in FIELD_GROUPS['detection_rays']:
            data = getattr(states.data, field_name)
            comm_data, info = comm_sampler.update(data, self.t_sim)
            setattr(comm_states.data, field_name, comm_data)

        # Static fields don't need communication delay
        for field_name in FIELD_GROUPS['camera_intrinsics']:
            data = getattr(states.data, field_name)
            setattr(comm_states.data, field_name, data.clone())

        # Copy timestamps
        comm_states.data.timestamp_sim_walltime = states.data.timestamp_sim_walltime.clone()
        comm_states.data.timestamp_motion = states.data.timestamp_motion.clone()
        comm_states.data.timestamp_detection = states.data.timestamp_detection.clone()

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

    def step(self, dt: Optional[float] = None):
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

        # Reset all samplers
        for sampler in self.motion_samplers.values():
            sampler.reset(env_ids)
        for sampler in self.orientation_samplers.values():
            sampler.reset(env_ids)
        for sampler in self.joint_samplers.values():
            sampler.reset(env_ids)
        self.zoom_sampler.reset(env_ids)

        self.detection_bbox_sampler.reset(env_ids)
        self.detection_rays_sampler.reset(env_ids)
        self.ego_comm_sampler.reset(env_ids)
        self.inter_agent_comm_sampler.reset(env_ids)
        self.passthrough_sampler.reset(env_ids)

        # Reset timestamp manager
        self.timestamp_manager.reset(env_ids=env_ids)
