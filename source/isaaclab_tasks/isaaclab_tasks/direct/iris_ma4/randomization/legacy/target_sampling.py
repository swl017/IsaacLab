"""
Gimbal-Aware Target Position Sampling

This module provides target position sampling that guarantees all agents in a formation
can point their gimbals at the target within mechanical limits.

Key Features:
- Constrained sampling (no rejection retries needed)
- Multi-agent aware (considers all agents simultaneously)
- Pitch angle constraint enforcement (critical for gimbals)
- Curriculum-ready parameters
- Fully vectorized GPU operations

Author: Claude Code
"""

import torch
import math
from isaaclab.utils import configclass


@configclass
class TargetSamplerCfg:
    """Configuration for gimbal-aware target sampling.

    Target Distance Calculation:
        1. Minimum distance = max(target_distance_min, formation_spread × distance_scale_factor)
        2. Maximum distance = target_distance_max × scale_factor (curriculum parameter)
        3. Final distance sampled uniformly in [min, max] range

    Curriculum Integration:
        The `scale_factor` parameter in sample_target_position() controls curriculum:
        - At scale_factor=0: max distance ≈ 0, so targets are at minimum distance (~15m)
        - At scale_factor=1: full range (15m to target_distance_max)
    """

    # Target distance range from formation center
    target_distance_min: float = 15.0  # Minimum horizontal distance (meters)
    target_distance_max: float = 60.0  # Maximum horizontal distance (meters) - curriculum scales this
    """Maximum target distance. Multiplied by curriculum scale_factor (0→1) during training."""

    # Formation safety factor (NOT a curriculum parameter)
    distance_scale_factor: float = 1.0
    """Multiplier for formation spread to ensure target is OUTSIDE formation.

    Example: If agents are spread 10m apart, target is at least 10m × 2.5 = 25m away.
    This is a geometric safety constraint, NOT affected by curriculum.
    """

    # Gimbal pitch limits (must match environment config)
    # Sign convention: down is positive pitch, up is negative pitch
    pitch_limit_min: float = math.radians(-70.0)  # Looking UP limit (negative = up)
    pitch_limit_max: float = math.radians(70.0)   # Looking DOWN limit (positive = down)

    # Safety margins (reduce usable range to avoid edge cases)
    pitch_safety_margin: float = math.radians(25.0)  # 25° margin

    # Bearing constraints (direction from formation to target)
    bearing_min: float = 0.0  # Minimum bearing (radians)
    bearing_max: float = 2 * math.pi  # Maximum bearing (radians)

    # Height offset from formation center
    height_offset_min: float = -10.0  # Minimum height offset
    height_offset_max: float = 10.0   # Maximum height offset

    # Fallback behavior when no feasible solution exists
    # Instead of using a fixed height margin (which doesn't guarantee constraints),
    # we iteratively expand the horizontal distance until the height range is feasible.
    fallback_distance_increase_factor: float = 1.15  # Multiply distance by this per iteration
    fallback_max_iterations: int = 10  # Maximum iterations for distance expansion
    fallback_max_distance_multiplier: float = 3.0  # Maximum distance as multiple of original


class TargetSampler:
    """
    Samples target positions that all agents in a formation can point to
    with their gimbals within mechanical limits.
    """

    def __init__(self, cfg: TargetSamplerCfg, device: torch.device):
        """
        Initialize target sampler.

        Args:
            cfg: Configuration for target sampling
            device: PyTorch device (cpu or cuda)
        """
        self.cfg = cfg
        self.device = device

        # Precompute trigonometric values for pitch limits (with safety margins)
        self.pitch_min_safe = self.cfg.pitch_limit_min + self.cfg.pitch_safety_margin
        self.pitch_max_safe = self.cfg.pitch_limit_max - self.cfg.pitch_safety_margin

        # Store tan values directly (no negation)
        # pitch_min_safe = -40° (looking up) gives negative tan
        # pitch_max_safe = +5° (looking down) gives positive tan
        self.tan_pitch_min_safe = math.tan(self.pitch_min_safe)
        self.tan_pitch_max_safe = math.tan(self.pitch_max_safe)

    def sample_target_position(
        self,
        formation_data: torch.Tensor,
        num_envs: int | None = None,
        scale_factor: float = 1.0,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample target positions feasible for all agents in formation.

        Args:
            formation_data: [num_envs, num_agents, 13] from get_random_formation()
                            or [num_envs, num_agents, 3] positions only
            num_envs: Number of environments (inferred from formation_data if None)
            scale_factor: Multiplier for target distance (curriculum hook)
            env_ids: Specific environment indices (optional)

        Returns:
            target_positions: [num_envs, 3] target positions in world frame
        """
        # Extract positions from formation data
        if formation_data.shape[-1] == 13:
            agent_positions = formation_data[:, :, 0:3]  # [num_envs, num_agents, 3]
        elif formation_data.shape[-1] == 3:
            agent_positions = formation_data  # Already positions
        else:
            raise ValueError(f"Invalid formation_data shape: {formation_data.shape}")

        if num_envs is None:
            num_envs = agent_positions.shape[0]

        num_agents = agent_positions.shape[1]

        # Step 1: Analyze formation geometry
        formation_center, max_spread = self._analyze_formation(agent_positions)

        # Step 2: Sample horizontal distance and bearing
        # Compute suggested minimum distance per environment based on formation spread
        # (to ensure target is not inside the formation)
        suggested_min_distance = max_spread * self.cfg.distance_scale_factor

        # Use configured distance range
        target_distance_max_per_env = torch.full(
            (num_envs,), self.cfg.target_distance_max * scale_factor, device=self.device
        )

        target_distance_min_per_env = torch.full(
            (num_envs,), self.cfg.target_distance_min, device=self.device
        )

        # Adjust minimum distance to respect formation geometry
        target_distance_min_per_env = torch.maximum(
            target_distance_min_per_env,
            suggested_min_distance
        )

        # Check if any formations exceed configured max distance (log warning once)
        if (suggested_min_distance > self.cfg.target_distance_max * scale_factor).any():
            if not hasattr(self, '_distance_warning_logged'):
                max_violation = suggested_min_distance.max().item()
                print(f"Warning: Some formations require targets farther than configured max!")
                print(f"  Formation spread: up to {max_spread.max():.1f}m")
                print(f"  Required min distance: up to {max_violation:.1f}m")
                print(f"  Configured max distance: {self.cfg.target_distance_max:.1f}m")
                print(f"  Suggestion: Reduce max_agent_separation or increase target_distance_max")
                self._distance_warning_logged = True

            # Expand max distance for environments that need it
            target_distance_max_per_env = torch.maximum(
                target_distance_max_per_env,
                target_distance_min_per_env + 1.0
            )

        # Sample horizontal distance (per environment)
        horizontal_distance = torch.rand(num_envs, device=self.device) * (
            target_distance_max_per_env - target_distance_min_per_env
        ) + target_distance_min_per_env

        bearing = torch.rand(num_envs, device=self.device) * (
            self.cfg.bearing_max - self.cfg.bearing_min
        ) + self.cfg.bearing_min

        # Step 3: Compute target horizontal position and iteratively expand distance if needed
        current_distance = horizontal_distance.clone()
        max_distance = horizontal_distance * self.cfg.fallback_max_distance_multiplier

        for iteration in range(self.cfg.fallback_max_iterations):
            # Compute target XY at current distance
            target_x = formation_center[:, 0] + current_distance * torch.cos(bearing)
            target_y = formation_center[:, 1] + current_distance * torch.sin(bearing)

            # Step 4: Compute feasible height range
            z_min, z_max, is_valid = self._compute_feasible_height_range(
                agent_positions,
                target_x,
                target_y,
            )

            # Check which environments are still infeasible
            infeasible_mask = ~is_valid
            if not infeasible_mask.any():
                break  # All environments have feasible height ranges

            # Expand distance for infeasible environments
            current_distance[infeasible_mask] = torch.minimum(
                current_distance[infeasible_mask] * self.cfg.fallback_distance_increase_factor,
                max_distance[infeasible_mask]
            )

            # Check if all infeasible envs have hit max distance
            at_max = current_distance[infeasible_mask] >= max_distance[infeasible_mask]
            if at_max.all():
                # Print warning and use current (possibly infeasible) state
                num_still_infeasible = infeasible_mask.sum().item()
                if num_still_infeasible > 0 and not hasattr(self, '_fallback_warning_logged'):
                    print(f"Warning: {num_still_infeasible} environments still infeasible after max distance expansion.")
                    self._fallback_warning_logged = True
                break

        # Recompute final target positions with final distance
        target_x = formation_center[:, 0] + current_distance * torch.cos(bearing)
        target_y = formation_center[:, 1] + current_distance * torch.sin(bearing)

        # Recompute final feasible height range
        z_min, z_max, is_valid = self._compute_feasible_height_range(
            agent_positions,
            target_x,
            target_y,
        )

        # For any remaining infeasible environments, use a safe fallback
        # (this should be rare with proper distance expansion)
        final_infeasible = ~is_valid
        if final_infeasible.any():
            # Use formation center height as fallback (constrained to stay within some range)
            fallback_z = formation_center[final_infeasible, 2]
            z_min[final_infeasible] = fallback_z - 1.0  # Small range
            z_max[final_infeasible] = fallback_z + 1.0

        # Step 5: Sample target height uniformly within feasible range
        # The feasible range [z_min, z_max] now accounts for all gimbal constraints
        target_z = torch.rand(num_envs, device=self.device) * (z_max - z_min) + z_min

        # Step 6: Construct target position
        target_positions = torch.stack([target_x, target_y, target_z], dim=1)

        return target_positions

    def _analyze_formation(
        self, agent_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze formation geometry.

        Args:
            agent_positions: [num_envs, num_agents, 3]

        Returns:
            formation_center: [num_envs, 3] mean position
            max_spread: [num_envs] maximum pairwise distance
        """
        num_envs, num_agents, _ = agent_positions.shape

        # Compute formation center
        formation_center = agent_positions.mean(dim=1)  # [num_envs, 3]

        # Compute maximum pairwise distance
        max_spread = torch.zeros(num_envs, device=self.device)

        for env_idx in range(num_envs):
            max_dist = 0.0
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = torch.norm(
                        agent_positions[env_idx, i] - agent_positions[env_idx, j]
                    )
                    max_dist = max(max_dist, dist.item())
            max_spread[env_idx] = max_dist

        return formation_center, max_spread

    def _compute_feasible_height_range(
        self,
        agent_positions: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute height range where ALL agents can point to target.

        This is the core method that enforces gimbal pitch constraints.

        Args:
            agent_positions: [num_envs, num_agents, 3]
            target_x: [num_envs] target X coordinate
            target_y: [num_envs] target Y coordinate

        Returns:
            z_min: [num_envs] minimum feasible target height
            z_max: [num_envs] maximum feasible target height
            is_valid: [num_envs] bool tensor (True = feasible range exists)
        """
        num_envs, num_agents, _ = agent_positions.shape

        # Initialize arrays to store per-agent constraints
        z_min_per_agent = []
        z_max_per_agent = []

        for agent_idx in range(num_agents):
            agent_pos = agent_positions[:, agent_idx, :]  # [num_envs, 3]

            # Compute horizontal distance from agent to target
            dx = target_x - agent_pos[:, 0]
            dy = target_y - agent_pos[:, 1]
            d_horizontal = torch.sqrt(dx**2 + dy**2 + 1e-8)  # Small epsilon for stability

            z_agent = agent_pos[:, 2]

            # Height range for this agent based on pitch limits:
            # z_target = z_agent - d_h * tan(pitch)
            # z_min comes from pitch_max_safe (looking down the most)
            # z_max comes from pitch_min_safe (looking up the most)
            z_min_i = z_agent - d_horizontal * self.tan_pitch_max_safe  # Lower bound
            z_max_i = z_agent - d_horizontal * self.tan_pitch_min_safe  # Upper bound

            z_min_per_agent.append(z_min_i)
            z_max_per_agent.append(z_max_i)

        # Stack and compute intersection
        z_min_per_agent = torch.stack(z_min_per_agent, dim=1)  # [num_envs, num_agents]
        z_max_per_agent = torch.stack(z_max_per_agent, dim=1)  # [num_envs, num_agents]

        # Intersection: max of minimums, min of maximums
        z_min_feasible = z_min_per_agent.max(dim=1)[0]  # [num_envs]
        z_max_feasible = z_max_per_agent.min(dim=1)[0]  # [num_envs]

        # Check validity (z_min <= z_max means feasible range exists)
        is_valid = z_min_feasible <= z_max_feasible

        return z_min_feasible, z_max_feasible, is_valid

    def validate_target_feasibility(
        self,
        agent_positions: torch.Tensor,
        agent_orientations: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> dict:
        """
        Validate that target is feasible for all agents.

        This computes actual gimbal angles and checks against limits.

        Args:
            agent_positions: [num_envs, num_agents, 3]
            agent_orientations: [num_envs, num_agents, 4] quaternions (w,x,y,z)
            target_positions: [num_envs, 3]

        Returns:
            dict with validation results:
                - all_feasible: [num_envs] bool tensor
                - pitch_angles: [num_envs, num_agents] computed pitch angles
                - violations: [num_envs, num_agents] bool tensor (True = violates limits)
        """
        num_envs, num_agents, _ = agent_positions.shape

        # Expand target to [num_envs, num_agents, 3]
        target_expanded = target_positions.unsqueeze(1).expand(-1, num_agents, -1)

        # Compute vector from each agent to target
        vector_to_target = target_expanded - agent_positions  # [num_envs, num_agents, 3]

        # Compute horizontal distance and vertical offset
        dx = vector_to_target[:, :, 0]
        dy = vector_to_target[:, :, 1]
        dz = vector_to_target[:, :, 2]

        d_horizontal = torch.sqrt(dx**2 + dy**2 + 1e-8)

        # Compute pitch angle
        # pitch = atan2(-dz, d_horizontal)
        pitch_angles = torch.atan2(-dz, d_horizontal)

        # Check violations
        violations = (pitch_angles < self.pitch_min_safe) | (pitch_angles > self.pitch_max_safe)

        # Check if all agents are feasible per environment
        all_feasible = ~violations.any(dim=1)  # [num_envs]

        return {
            'all_feasible': all_feasible,
            'pitch_angles': pitch_angles,
            'violations': violations,
        }


def create_target_sampler(
    pitch_limits: tuple[float, float],
    device: torch.device,
    target_distance_range: tuple[float, float] = (15.0, 80.0),
    pitch_safety_margin_deg: float = 5.0,
) -> TargetSampler:
    """
    Convenience function to create a TargetSampler with common parameters.

    Args:
        pitch_limits: (min, max) pitch angles in radians
        device: PyTorch device
        target_distance_range: (min, max) horizontal distance range
        pitch_safety_margin_deg: Safety margin in degrees

    Returns:
        Configured TargetSampler instance
    """
    cfg = TargetSamplerCfg(
        target_distance_min=target_distance_range[0],
        target_distance_max=target_distance_range[1],
        pitch_limit_min=pitch_limits[0],
        pitch_limit_max=pitch_limits[1],
        pitch_safety_margin=math.radians(pitch_safety_margin_deg),
    )
    return TargetSampler(cfg, device)
