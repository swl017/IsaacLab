import torch
import math

from collections.abc import Sequence
import math
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab.utils import configclass

from isaaclab_tasks.direct.iris_ma3.iris_ma_env3_cfg import IrisMAEnvCfg

@configclass
class InitialStatesRandomizerCfg:
    position_bounds: dict = {
        "x_min": -5.0,
        "x_max": 5.0,
        "y_min": -5.0,
        "y_max": 5.0,
        "z_min": 1.0,
        "z_max": 3.0
    }

    # Assuming facing x direction
    orientation_bounds: dict = {
        "yaw_min": -math.radians(45),
        "yaw_max": math.radians(45)
    }

    # Formation configuration
    formation_center_bounds: dict = {
        "x_min": -5.0,
        "x_max": 5.0,
        "y_min": -10.0,
        "y_max": 10.0,
        "z_min": 15.0,
        "z_max": 20.0
    }

    # Formation types: "random", "line", "grid", "planar"
    # "planar" is a simple horizontal line formation with no height variation
    formation_types: list = ["line", "grid", "planar"]
    default_formation_type: str = "random"  # "random" picks from formation_types

    # Agent separation constraints
    min_agent_separation: float = 3.0  # Minimum distance between agents
    max_agent_separation: float = 12.0  # Maximum distance between agents

    # Formation scaling (hook for future curriculum)
    formation_scale_min: float = 0.8
    formation_scale_max: float = 1.2

    # 3D formation options
    allow_vertical_variation: bool = True
    z_variation_range: tuple = (0.0, 5.0)  # Additional Z variation within formation (legacy)

    # Curriculum-controlled height variation (preferred over z_variation_range)
    z_variation_curriculum_enabled: bool = False  # Enable curriculum-based z variation
    z_variation_min: tuple = (0.0, 0.0)  # Z variation range at scale_factor=0
    z_variation_max: tuple = (0.0, 2.0)  # Z variation range at scale_factor=1

    # Line formation direction constraint
    line_max_z_component: float = 0.5  # Max absolute Z component in line direction (0=horizontal)

    # Orientation options
    orientation_mode: str = "random_yaw"  # "formation_aligned" or "random_yaw"
    orientation_noise_std: float = 0.087  # ~5 degrees in radians

    # Safety constraints
    ground_clearance_min: float = 1.0
    max_collision_retries: int = 10

class InitialStatesRandomizer:
    cfg: InitialStatesRandomizerCfg
    iris_cfg: IrisMAEnvCfg

    def __init__(self, num_envs: int, device: torch.device):
        self.device = device
        self.num_envs = num_envs
        self._ALL_INDICES = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Initialize config instances
        self.cfg = InitialStatesRandomizerCfg()
        self.iris_cfg = IrisMAEnvCfg()

    def get_random_formation(
        self,
        num_agents: int,
        num_envs: int | None = None,
        formation_type: str | None = None,
        scale_factor: float = 1.0,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate random formation positions and orientations for agents.

        Args:
            num_agents: Number of agents per environment
            num_envs: Number of environments (defaults to self.num_envs)
            formation_type: "line", "grid", or None for random selection
            scale_factor: Scaling multiplier for formation size (future curriculum hook)
            env_ids: Specific environment indices to generate for

        Returns:
            torch.Tensor: Root states [num_envs, num_agents, 13]
                [:, :, 0:3] - positions (x, y, z)
                [:, :, 3:7] - orientations (quaternion w, x, y, z)
                [:, :, 7:13] - velocities (linear + angular, all zeros)
        """
        if num_envs is None:
            num_envs = self.num_envs

        if env_ids is None:
            env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        else:
            num_envs = len(env_ids)

        # Determine formation type for each environment
        if formation_type is None or formation_type == "random":
            # Randomly select formation type per environment
            type_indices = torch.randint(0, len(self.cfg.formation_types), (num_envs,), device=self.device)
            formation_types = [self.cfg.formation_types[idx.item()] for idx in type_indices]
        else:
            formation_types = [formation_type] * num_envs

        # Generate formation centers
        formation_centers = self._generate_formation_centers(num_envs)

        # Generate formation yaw rotations
        formation_yaws = self._generate_formation_yaws(num_envs)

        # Initialize output tensor [num_envs, num_agents, 13]
        root_states = torch.zeros(num_envs, num_agents, 13, device=self.device)

        # Generate formations for each type
        for formation_type in set(formation_types):
            # Get environments using this formation type
            type_mask = torch.tensor(
                [ft == formation_type for ft in formation_types],
                dtype=torch.bool,
                device=self.device
            )
            type_envs = torch.where(type_mask)[0]

            if len(type_envs) == 0:
                continue

            # Generate local positions for this formation type
            if formation_type == "line":
                local_positions = self._generate_line_formation(num_agents, len(type_envs), scale_factor)
            elif formation_type == "grid":
                local_positions = self._generate_grid_formation(num_agents, len(type_envs), scale_factor)
            elif formation_type == "planar":
                local_positions = self._generate_planar_formation(num_agents, len(type_envs), scale_factor)
            else:
                raise ValueError(f"Unknown formation type: {formation_type}")

            # Apply collision avoidance
            local_positions = self._ensure_minimum_separation(local_positions)

            # Transform to global coordinates
            global_positions = self._apply_formation_transform(
                local_positions,
                formation_centers[type_envs],
                formation_yaws[type_envs]
            )

            # Generate orientations
            orientations = self._generate_agent_orientations(
                global_positions,
                formation_centers[type_envs],
                formation_yaws[type_envs]
            )

            # Assign to output
            root_states[type_envs, :, 0:3] = global_positions
            root_states[type_envs, :, 3:7] = orientations
            root_states[type_envs, :, 7:10] = math_utils.sample_uniform(
                lower=-self.iris_cfg.max_lin_vel/5 * scale_factor, 
                upper=self.iris_cfg.max_lin_vel/5 * scale_factor,
                size=(len(type_envs), num_agents, 3),
                device=self.device
            )
            root_states[type_envs, :, 12:13] = math_utils.sample_uniform(
                lower=-self.iris_cfg.max_yaw_rate * scale_factor,
                upper=self.iris_cfg.max_yaw_rate * scale_factor,
                size=(len(type_envs), num_agents, 1),
                device=self.device
            )

        # Validate output
        self._validate_formation(root_states)

        return root_states

    def _generate_formation_centers(self, num_envs: int) -> torch.Tensor:
        """Generate random formation center positions."""
        k = torch.rand(num_envs, 3, device=self.device)

        centers = torch.stack([
            self.cfg.formation_center_bounds["x_min"] + k[:, 0] * (
                self.cfg.formation_center_bounds["x_max"] - self.cfg.formation_center_bounds["x_min"]
            ),
            self.cfg.formation_center_bounds["y_min"] + k[:, 1] * (
                self.cfg.formation_center_bounds["y_max"] - self.cfg.formation_center_bounds["y_min"]
            ),
            self.cfg.formation_center_bounds["z_min"] + k[:, 2] * (
                self.cfg.formation_center_bounds["z_max"] - self.cfg.formation_center_bounds["z_min"]
            ),
        ], dim=1)

        return centers

    def _generate_formation_yaws(self, num_envs: int) -> torch.Tensor:
        """Generate random yaw rotations for formations (facing +X by default)."""
        yaw_min = self.cfg.orientation_bounds["yaw_min"]
        yaw_max = self.cfg.orientation_bounds["yaw_max"]
        return torch.rand(num_envs, device=self.device) * (yaw_max - yaw_min) + yaw_min

    def _generate_line_formation(
        self, num_agents: int, num_envs: int, scale_factor: float
    ) -> torch.Tensor:
        """
        Generate line formation along a 3D line.
        Agents placed at random distances from each other along the line.

        The Z component of the line direction is constrained based on curriculum
        (scale_factor), starting horizontal and allowing more vertical as training progresses.

        Returns:
            torch.Tensor: [num_envs, num_agents, 3] local positions
        """
        positions = torch.zeros(num_envs, num_agents, 3, device=self.device)

        for env_idx in range(num_envs):
            # Generate random line direction
            # XY components are random, Z is constrained by curriculum
            xy_dir = torch.randn(2, device=self.device)
            xy_dir = xy_dir / (xy_dir.norm() + 1e-6)

            # Z component scaled by curriculum: at scale_factor=0, Z=0 (horizontal)
            # At scale_factor=1, Z can be up to line_max_z_component
            max_z = self.cfg.line_max_z_component * scale_factor
            z_component = (torch.rand(1, device=self.device).item() * 2 - 1) * max_z

            direction = torch.tensor([xy_dir[0].item(), xy_dir[1].item(), z_component], device=self.device)
            direction = direction / (direction.norm() + 1e-6)

            # Generate random distances along the line for each agent
            # Start from 0 and accumulate random steps
            distances = torch.zeros(num_agents, device=self.device)

            for i in range(1, num_agents):
                # Random distance between min and max separation
                step = torch.rand(1, device=self.device).item() * (
                    self.cfg.max_agent_separation - self.cfg.min_agent_separation
                ) + self.cfg.min_agent_separation
                distances[i] = distances[i - 1] + step * max(scale_factor, 0.3)  # Ensure minimum spacing

            # Center the line around origin
            distances = distances - distances.mean()

            # Place agents along the line
            for i in range(num_agents):
                positions[env_idx, i] = direction * distances[i]

            # Add optional vertical variation (curriculum-controlled if enabled)
            if self.cfg.allow_vertical_variation:
                z_range_min, z_range_max = self._get_z_variation_range(scale_factor)
                z_var = torch.rand(num_agents, device=self.device) * (z_range_max - z_range_min) + z_range_min
                positions[env_idx, :, 2] += z_var

        return positions

    def _get_z_variation_range(self, scale_factor: float) -> tuple[float, float]:
        """
        Get the Z variation range based on curriculum progress.

        Args:
            scale_factor: Curriculum progress (0.0 to 1.0)

        Returns:
            (z_min, z_max) tuple for sampling Z variation
        """
        if self.cfg.z_variation_curriculum_enabled:
            # Interpolate between min and max ranges based on scale_factor
            z_min = self.cfg.z_variation_min[0] + scale_factor * (
                self.cfg.z_variation_max[0] - self.cfg.z_variation_min[0]
            )
            z_max = self.cfg.z_variation_min[1] + scale_factor * (
                self.cfg.z_variation_max[1] - self.cfg.z_variation_min[1]
            )
        else:
            # Use legacy z_variation_range
            z_min = self.cfg.z_variation_range[0]
            z_max = self.cfg.z_variation_range[1]

        return z_min, z_max

    def _generate_grid_formation(
        self, num_agents: int, num_envs: int, scale_factor: float
    ) -> torch.Tensor:
        """
        Generate 2D grid formation in XY plane.
        Grid dimensions automatically determined from num_agents.

        Returns:
            torch.Tensor: [num_envs, num_agents, 3] local positions
        """
        positions = torch.zeros(num_envs, num_agents, 3, device=self.device)

        # Determine grid dimensions (prefer rectangular for better coverage)
        rows = int(math.sqrt(num_agents))
        cols = math.ceil(num_agents / rows)

        for env_idx in range(num_envs):
            # Random spacing for this environment
            spacing_y = torch.rand(1, device=self.device).item() * (
                self.cfg.max_agent_separation - self.cfg.min_agent_separation
            ) + self.cfg.min_agent_separation

            spacing_x = torch.rand(1, device=self.device).item() * (
                self.cfg.max_agent_separation - self.cfg.min_agent_separation
            ) + self.cfg.min_agent_separation

            # Scale spacing by curriculum, with minimum to ensure separation
            spacing_y *= max(scale_factor, 0.3)
            spacing_x *= max(scale_factor, 0.3)

            agent_idx = 0
            for row in range(rows):
                for col in range(cols):
                    if agent_idx >= num_agents:
                        break

                    # Position in grid (centered around origin)
                    x = (col - (cols - 1) / 2) * spacing_x
                    y = (row - (rows - 1) / 2) * spacing_y

                    positions[env_idx, agent_idx, 0] = x
                    positions[env_idx, agent_idx, 1] = y

                    agent_idx += 1

            # Add optional vertical variation (curriculum-controlled if enabled)
            if self.cfg.allow_vertical_variation:
                z_range_min, z_range_max = self._get_z_variation_range(scale_factor)
                z_var = torch.rand(num_agents, device=self.device) * (z_range_max - z_range_min) + z_range_min
                positions[env_idx, :, 2] += z_var

        return positions

    def _generate_planar_formation(
        self, num_agents: int, num_envs: int, scale_factor: float
    ) -> torch.Tensor:
        """
        Generate planar formation in XY plane with NO height variation.
        Agents are placed in a line along a random horizontal direction.

        This is the simplest formation for early curriculum - all agents at the same height.
        It guarantees that any target in front of the formation is gimbal-feasible.

        Args:
            num_agents: Number of agents per environment
            num_envs: Number of environments
            scale_factor: Scaling factor for agent separation (curriculum)

        Returns:
            torch.Tensor: [num_envs, num_agents, 3] local positions (Z = 0 for all)
        """
        positions = torch.zeros(num_envs, num_agents, 3, device=self.device)

        for env_idx in range(num_envs):
            # Random direction in XY plane ONLY (no Z component)
            angle = torch.rand(1, device=self.device).item() * 2 * math.pi
            direction = torch.tensor([math.cos(angle), math.sin(angle), 0.0], device=self.device)

            # Generate cumulative distances along the line
            distances = torch.zeros(num_agents, device=self.device)
            for i in range(1, num_agents):
                step = torch.rand(1, device=self.device).item() * (
                    self.cfg.max_agent_separation - self.cfg.min_agent_separation
                ) + self.cfg.min_agent_separation
                # Ensure minimum spacing even at low scale_factor
                distances[i] = distances[i - 1] + step * max(scale_factor, 0.3)

            # Center around origin
            distances = distances - distances.mean()

            # Place agents along direction (Z stays 0)
            for i in range(num_agents):
                positions[env_idx, i] = direction * distances[i]

        # No vertical variation for planar formation - this is the key feature
        return positions

    def _ensure_minimum_separation(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Ensure agents maintain minimum separation distance.

        Args:
            positions: [num_envs, num_agents, 3]

        Returns:
            Adjusted positions with minimum separation enforced
        """
        num_envs, num_agents, _ = positions.shape
        min_dist = self.cfg.min_agent_separation

        for env_idx in range(num_envs):
            for retry in range(self.cfg.max_collision_retries):
                collision_found = False

                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        dist = (positions[env_idx, i] - positions[env_idx, j]).norm()

                        if dist < min_dist:
                            collision_found = True
                            # Push agents apart along their connecting line
                            direction = positions[env_idx, i] - positions[env_idx, j]
                            direction = direction / (direction.norm() + 1e-6)

                            push_amount = (min_dist - dist) / 2
                            positions[env_idx, i] += direction * push_amount
                            positions[env_idx, j] -= direction * push_amount

                if not collision_found:
                    break

        return positions

    def _apply_formation_transform(
        self,
        local_positions: torch.Tensor,
        centers: torch.Tensor,
        yaws: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform local formation positions to global coordinates.

        Args:
            local_positions: [num_envs, num_agents, 3]
            centers: [num_envs, 3]
            yaws: [num_envs]

        Returns:
            Global positions [num_envs, num_agents, 3]
        """
        num_envs, num_agents, _ = local_positions.shape

        # Create rotation matrices for yaw
        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)

        global_positions = torch.zeros_like(local_positions)

        for env_idx in range(num_envs):
            # Rotation matrix around Z axis
            for agent_idx in range(num_agents):
                x_local = local_positions[env_idx, agent_idx, 0]
                y_local = local_positions[env_idx, agent_idx, 1]
                z_local = local_positions[env_idx, agent_idx, 2]

                # Rotate
                x_rotated = cos_yaw[env_idx] * x_local - sin_yaw[env_idx] * y_local
                y_rotated = sin_yaw[env_idx] * x_local + cos_yaw[env_idx] * y_local

                # Translate
                global_positions[env_idx, agent_idx, 0] = x_rotated + centers[env_idx, 0]
                global_positions[env_idx, agent_idx, 1] = y_rotated + centers[env_idx, 1]
                global_positions[env_idx, agent_idx, 2] = z_local + centers[env_idx, 2]

        return global_positions

    def _generate_agent_orientations(
        self,
        positions: torch.Tensor,
        centers: torch.Tensor,
        formation_yaws: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate orientations for agents based on configuration.

        Args:
            positions: [num_envs, num_agents, 3]
            centers: [num_envs, 3]
            formation_yaws: [num_envs]

        Returns:
            Quaternions [num_envs, num_agents, 4] in (w, x, y, z) format
        """
        num_envs, num_agents, _ = positions.shape
        orientations = torch.zeros(num_envs, num_agents, 4, device=self.device)

        for env_idx in range(num_envs):
            for agent_idx in range(num_agents):
                if self.cfg.orientation_mode == "formation_aligned":
                    # All agents face formation direction (yaw) with small noise
                    yaw = formation_yaws[env_idx]
                    yaw += torch.randn(1, device=self.device).item() * self.cfg.orientation_noise_std

                elif self.cfg.orientation_mode == "random_yaw":
                    # Random yaw orientation
                    yaw_min = self.cfg.orientation_bounds["yaw_min"]
                    yaw_max = self.cfg.orientation_bounds["yaw_max"]
                    yaw = torch.rand(1, device=self.device).item() * (yaw_max - yaw_min) + yaw_min
                else:
                    yaw = 0.0

                # Convert to quaternion (roll=0, pitch=0, yaw)
                quat = quat_from_euler_xyz(
                    torch.tensor([0.0], device=self.device),
                    torch.tensor([0.0], device=self.device),
                    torch.tensor([yaw], device=self.device)
                )
                orientations[env_idx, agent_idx] = quat[0]

        return orientations

    def _validate_formation(self, root_states: torch.Tensor):
        """
        Validate that formation satisfies safety constraints.

        Args:
            root_states: [num_envs, num_agents, 13] full state tensor
        """
        positions = root_states[:, :, 0:3]
        orientations = root_states[:, :, 3:7]
        velocities = root_states[:, :, 7:13]

        num_envs, num_agents, _ = positions.shape

        # Check minimum height
        min_z = positions[:, :, 2].min()
        if min_z < self.cfg.ground_clearance_min:
            print(f"Warning: Formation has agents below minimum height: {min_z:.2f} < {self.cfg.ground_clearance_min:.2f}")

        # Check minimum separation (spot check)
        if num_agents > 1:
            for env_idx in range(min(num_envs, 5)):  # Check first 5 envs
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        dist = (positions[env_idx, i] - positions[env_idx, j]).norm()
                        if dist < self.cfg.min_agent_separation * 0.9:  # 10% tolerance
                            print(f"Warning: Agents too close in env {env_idx}: {dist:.2f}m")

        # Check velocities are within expected bounds
        lin_vel = velocities[:, :, 0:3]
        ang_vel = velocities[:, :, 3:6]

        max_lin_vel = lin_vel.abs().max()
        max_ang_vel = ang_vel.abs().max()

        # Velocities should be small (< max_vel/5 for linear, < max_yaw_rate for angular)
        expected_max_lin = self.iris_cfg.max_lin_vel / 5
        expected_max_ang = self.iris_cfg.max_yaw_rate

        if max_lin_vel > expected_max_lin * 1.1:  # 10% tolerance
            print(f"Warning: Linear velocity too high: {max_lin_vel:.2f} > {expected_max_lin:.2f}")

        if max_ang_vel > expected_max_ang * 1.1:  # 10% tolerance
            print(f"Warning: Angular velocity too high: {max_ang_vel:.2f} > {expected_max_ang:.2f}")

    # Legacy methods for backward compatibility
    def get_random_translation(self, num_envs: int) -> torch.Tensor:
        k = torch.rand(num_envs, 3, device=self.device)

        random_position = torch.stack([
            self.cfg.position_bounds["x_min"] + k[:, 0] * (self.cfg.position_bounds["x_max"] - self.cfg.position_bounds["x_min"]),
            self.cfg.position_bounds["y_min"] + k[:, 1] * (self.cfg.position_bounds["y_max"] - self.cfg.position_bounds["y_min"]),
            self.cfg.position_bounds["z_min"] + k[:, 2] * (self.cfg.position_bounds["z_max"] - self.cfg.position_bounds["z_min"]),
        ], dim=1)

        return random_position

    def get_random_orientation(self, num_envs: int) -> torch.Tensor:
        return math_utils.random_orientation(num_envs, device=self.device)

    def get_random_yaw_orientation(self, num_envs: int) -> torch.Tensor:
        return math_utils.random_yaw_orientation(num_envs, device=self.device)

    def get_default_orientation(self, num_envs: int) -> torch.Tensor:
        return math_utils.default_orientation(num_envs, device=self.device)