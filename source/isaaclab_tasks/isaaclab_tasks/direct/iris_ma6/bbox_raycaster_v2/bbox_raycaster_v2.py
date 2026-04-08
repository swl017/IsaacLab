# file: bbox_raycaster_v2.py
"""Main batched bounding box raycaster V2 with self-occlusion and bbox_empty output."""

from __future__ import annotations

import torch
import warnings
import warp as wp
from typing import TYPE_CHECKING, Dict

import omni.log
import omni.usd
from pxr import UsdGeom

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
import isaaclab.utils.math as math_utils

from .bbox_raycaster_v2_data import BBoxRayCasterV2Data
from .detector_replicator import DetectorReplicator, DetectorReplicatorCfg
from .utils import (
    batch_project_to_image_plane,
    batch_transform_points,
    batch_transform_to_camera_frame,
    batch_check_occlusion_fully_batched,
    bbox_xyxy_to_xywh,
    check_all_corners_visible,
    check_gimbal_lock,
    compute_2d_bbox_from_corners,
    compute_bbox_size,
    get_bbox_corners_local,
    normalize_bboxes,
    apply_inter_target_occlusion,
    generate_occlusion_test_points,
    validate_bbox_sizes,
    validate_projections,
)

if TYPE_CHECKING:
    from .bbox_raycaster_v2_cfg import BBoxRayCasterV2Cfg

import numpy as np

class BBoxRayCasterV2:
    """Batched bounding box raycaster V2 for multi-agent environments.
    
    This class efficiently computes 2D bounding boxes for 3D targets across multiple
    cameras and environments using batched GPU operations. It includes body-local
    occlusion detection via raycasting for both static environment and dynamic agents.
    
    Example:
        >>> # Initialize raycaster
        >>> bbox_raycaster = BBoxRayCasterV2(
        ...     cfg=bbox_cfg,
        ...     num_envs=4096,
        ...     num_targets_per_env=1,
        ...     device="cuda:0"
        ... )
        >>> 
        >>> # Update with current poses
        >>> camera_poses = {
        ...     "drone_0": (pos_tensor, quat_tensor),  # Each (N, 3) and (N, 4)
        ...     "drone_1": (pos_tensor, quat_tensor),
        ... }
        >>> camera_intrinsics = {
        ...     "drone_0": intrinsic_matrix,  # (N, 3, 3)
        ...     "drone_1": intrinsic_matrix,
        ... }
        >>> target_poses = (target_pos, target_quat)  # (N, T, 3) and (N, T, 4)
        >>> 
        >>> # Set agent poses for occlusion checking
        >>> agent_poses = {
        ...     "drone_0": (pos_tensor, quat_tensor),
        ...     "drone_1": (pos_tensor, quat_tensor),
        ... }
        >>> 
        >>> bbox_raycaster.update(camera_poses, camera_intrinsics, target_poses, agent_poses)
        >>> 
        >>> # Access results
        >>> bboxes = bbox_raycaster.data.bboxes  # (N, C, T, 4)
        >>> bbox_empty = bbox_raycaster.data.bbox_empty  # (N, C, T)
    """

    cfg: BBoxRayCasterV2Cfg
    """Configuration for the raycaster."""

    def __init__(
        self,
        cfg: BBoxRayCasterV2Cfg,
        num_envs: int,
        num_targets_per_env: int,
        device: str,
        agent_ids: list[str] | None = None
    ):
        """Initialize batched bbox raycaster.
        
        Args:
            cfg: Configuration instance.
            num_envs: Number of parallel environments.
            num_targets_per_env: Number of targets per environment.
            device: Device for tensor operations (e.g., "cuda:0").
            agent_ids: List of agent IDs for loading agent meshes (e.g., ["drone_0", "drone_1"]).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_targets = num_targets_per_env
        self.num_cameras = cfg.num_cameras_per_env
        self.device = device
        self.agent_ids = agent_ids or []

        # Initialize data container
        self._data = BBoxRayCasterV2Data()

        # Load meshes for raycasting
        self.static_mesh: wp.Mesh | None = None
        self.agent_meshes: Dict[str, wp.Mesh] = {}
        if cfg.enable_occlusion_check:
            self._initialize_meshes()

        # Extract target bounding boxes (one-time, handles all environments)
        self.target_bbox_corners_local: torch.Tensor | None = None
        self.target_bbox_sizes: torch.Tensor | None = None
        self.targets_share_bbox: bool = False
        
        self._extract_target_bboxes()

        # Pre-allocate all buffers
        self._allocate_buffers()

        # Initialize visualization if enabled
        if cfg.debug_vis:
            self._setup_visualization()

        # Memory tracking
        self._memory_tracker = {} if cfg.debug_memory else None

        omni.log.info(
            f"BBoxRayCasterV2 initialized with {num_envs} envs, {self.num_cameras} cameras, "
            f"{num_targets_per_env} targets, {len(self.agent_meshes)} agent meshes"
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"BBoxRayCasterV2:\n"
            f"  Environments: {self.num_envs}\n"
            f"  Cameras per env: {self.num_cameras}\n"
            f"  Targets per env: {self.num_targets}\n"
            f"  Total cameras: {self.num_envs * self.num_cameras}\n"
            f"  Total targets: {self.num_envs * self.num_targets}\n"
            f"  Static mesh loaded: {self.static_mesh is not None}\n"
            f"  Agent meshes loaded: {len(self.agent_meshes)}\n"
            f"  Occlusion checking: {self.cfg.enable_occlusion_check}\n"
            f"  Debug visualization: {self.cfg.debug_vis}"
        )

    """
    Properties
    """

    @property
    def data(self) -> BBoxRayCasterV2Data:
        """Access to the data container."""
        return self._data

    """
    Operations
    """

    def update(
        self,
        camera_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
        camera_intrinsics: dict[str, torch.Tensor],
        target_poses: tuple[torch.Tensor, torch.Tensor],
        agent_poses: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        image_shapes: dict[str, tuple[int, int]] | None = None,
        target_scale: torch.Tensor | None = None
    ):
        """Update bounding boxes for current frame.
        
        This is the main entry point for computing bounding boxes. It performs:
        1. Pose validation and normalization
        2. 3D bbox transformation to world frame
        3. Camera frame transformation
        4. Projection to image planes
        5. 2D bbox computation
        6. Validation (size, FOV, etc.)
        7. Occlusion checking (if enabled) using body-local raycasting
        
        Args:
            camera_poses: Dictionary mapping agent_id to (position, quaternion).
                Position shape: (N, 3), Quaternion shape: (N, 4) in (w, x, y, z) format.
            camera_intrinsics: Dictionary mapping agent_id to intrinsic matrix.
                Intrinsic matrix shape: (N, 3, 3).
            target_poses: Tuple of (position, quaternion) for all targets.
                Position shape: (N, T, 3), Quaternion shape: (N, T, 4).
            agent_poses: Dictionary mapping agent_id to (position, quaternion) for occlusion.
                Position shape: (N, 3), Quaternion shape: (N, 4).
            image_shapes: Optional dict mapping agent_id to (height, width).
                If None, uses shapes from intrinsics stored during initialization.
            target_scale: Optional (N, T, 3) scaling factors (x,y,z) for targets.
        """
        self._track_memory("start")

        # Step 1: Stack camera data from dict to tensors. (N, C, 3), (N, C, 4), (N, C, 3, 3)
        camera_pos_w, camera_quat_w, intrinsic_matrices = self._stack_camera_data(
            camera_poses, camera_intrinsics
        )

        # Step 2: Update image shapes if provided
        if image_shapes is not None:
            self._update_image_shapes(image_shapes)

        # Step 3: Validate and normalize inputs. (N, T, 3), (N, T, 4)
        target_pos_w, target_quat_w = target_poses
        target_quat_w = self._normalize_quaternions(target_quat_w, self.cfg.quat_normalize_epsilon)
        camera_quat_w = self._normalize_quaternions(camera_quat_w, self.cfg.quat_normalize_epsilon)

        # Store poses in data container
        self._data.camera_pos_w = camera_pos_w
        self._data.camera_quat_w = camera_quat_w
        self._data.target_pos_w = target_pos_w
        self._data.target_quat_w = target_quat_w
        self._data.intrinsic_matrices = intrinsic_matrices

        # Check for gimbal lock if warnings enabled
        if self.cfg.warn_gimbal_lock:
            gimbal_lock_mask = check_gimbal_lock(camera_quat_w, self.cfg.gimbal_lock_threshold)
            if gimbal_lock_mask.any():
                warnings.warn(
                    f"Gimbal lock detected in {gimbal_lock_mask.sum().item()} cameras. "
                    "Consider adjusting camera orientations."
                )

        self._track_memory("after_validation")

        # Step 4: Transform bbox corners to world frame
        try:
            self._transform_corners_to_world(target_pos_w, target_quat_w, target_scale)
        except RuntimeError as e:
            warnings.warn(f"Corner transformation failed: {e}")
            self._valid_mask.fill_(False)
            self._sync_public_outputs()
            return

        self._track_memory("after_transform_world")

        # Step 5: Transform corners to camera frames
        self._transform_corners_to_camera(camera_pos_w, camera_quat_w)

        self._track_memory("after_transform_camera")

        # Step 6: Project to image planes
        self._project_corners_to_image(intrinsic_matrices)

        self._track_memory("after_projection")

        # Step 7: Compute 2D bounding boxes
        self._compute_bboxes()

        self._track_memory("after_bbox_computation")

        # Step 8: Validate detections
        self._validate_detections()

        self._track_memory("after_validation_checks")

        # Step 9: Check occlusions (if enabled)
        if self.cfg.enable_occlusion_check:
            self._check_occlusions(agent_poses, camera_poses, target_scale)

        self._track_memory("after_occlusion")

        if self.cfg.enable_inter_target_occlusion:
            self._apply_inter_target_occlusion()

        self._sync_public_outputs()

        # Cleanup intermediate buffers if memory tracking enabled
        if self.cfg.debug_memory:
            torch.cuda.empty_cache()
            self._track_memory("after_cleanup")

    """
    Implementation - Initialization
    """

    def _initialize_meshes(self):
        """Load and convert meshes to Warp format for raycasting.
        
        Loads two types of meshes:
        1. Static environment mesh (ground, obstacles) in world coordinates
        2. Agent meshes (drones) in body-local coordinates for per-env occlusion
        """
        # === Load static environment mesh ===
        if len(self.cfg.mesh_prim_paths) > 0:
            # For now, use first mesh as static environment
            # TODO: Support multiple static meshes or combine into one
            mesh_prim_path = self.cfg.mesh_prim_paths[0]
            
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path,
                lambda prim: prim.GetTypeName() == "Plane"
            )

            if mesh_prim is not None:
                # Create infinite plane
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                self.static_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                omni.log.info(f"Created infinite plane mesh: {mesh_prim_path}")
            else:
                # Try loading as regular mesh
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path,
                    lambda prim: prim.GetTypeName() == "Mesh"
                )

                if mesh_prim is not None and mesh_prim.IsValid():
                    mesh_prim = UsdGeom.Mesh(mesh_prim)
                    points = mesh_prim.GetPointsAttr().Get()
                    indices = mesh_prim.GetFaceVertexIndicesAttr().Get()

                    # Apply world transform
                    transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
                    points = np.asarray(points)
                    points = np.matmul(points, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
                    indices = np.asarray(indices)

                    self.static_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                    omni.log.info(f"Loaded static mesh: {mesh_prim_path} ({len(points)} vertices)")
        
        # === Load agent meshes in body-local coordinates ===
        if self.cfg.load_agent_meshes and len(self.agent_ids) > 0:
            for agent_id in self.agent_ids:
                try:
                    agent_mesh = self._load_agent_mesh(agent_id)
                    if agent_mesh is not None:
                        self.agent_meshes[agent_id] = agent_mesh
                        omni.log.info(f"Loaded body-local mesh for {agent_id}")
                except Exception as e:
                    omni.log.warn(f"Could not load mesh for {agent_id}: {e}")
                    if self.cfg.use_collision_proxy:
                        # Use simple box collision proxy as fallback
                        self.agent_meshes[agent_id] = self._create_box_collision_mesh()
                        omni.log.info(f"Using box collision proxy for {agent_id}")

    def _load_agent_mesh(self, agent_id: str) -> wp.Mesh | None:
        """Load agent mesh in body-local coordinates from first environment.

        Loads ALL mesh components (body, propellers, arms) and combines them
        into a single mesh for comprehensive occlusion detection.

        Args:
            agent_id: Agent identifier (e.g., "drone_0").

        Returns:
            Warp mesh in body-local coordinates, or None if loading failed.
        """
        # Construct path to first environment's agent
        robot_index = agent_id.split("_")[-1]
        robot_prim_path = f"/World/envs/env_0/Robot_{robot_index}"

        # Find ALL meshes under the robot prim (body, propellers, etc.)
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Get the robot root prim to find body position for coordinate transform
        robot_prim = prim_utils.get_prim_at_path(robot_prim_path)
        if robot_prim is None or not robot_prim.IsValid():
            omni.log.warn(f"Could not find robot prim for {agent_id} at {robot_prim_path}")
            return None

        # Get body transform to use as reference frame
        body_prim_path = f"{robot_prim_path}/body"
        body_xform_prim = prim_utils.get_prim_at_path(body_prim_path)
        body_world_transform = None
        if body_xform_prim is not None and body_xform_prim.IsValid():
            body_world_transform = np.array(omni.usd.get_world_transform_matrix(body_xform_prim)).T

        # Recursively find all Mesh prims under the robot
        def find_all_meshes(prim, meshes: list):
            if prim.GetTypeName() == "Mesh":
                meshes.append(prim)
            for child in prim.GetChildren():
                find_all_meshes(child, meshes)

        mesh_prims = []
        find_all_meshes(robot_prim, mesh_prims)

        if len(mesh_prims) == 0:
            omni.log.warn(f"No meshes found for {agent_id} at {robot_prim_path}")
            return None

        omni.log.info(f"Found {len(mesh_prims)} meshes for {agent_id}: {[str(m.GetPath()) for m in mesh_prims]}")

        for mesh_prim in mesh_prims:
            mesh_geom = UsdGeom.Mesh(mesh_prim)
            points = mesh_geom.GetPointsAttr().Get()
            indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

            if points is None or indices is None:
                continue

            points = np.asarray(points, dtype=np.float32)
            indices = np.asarray(indices, dtype=np.int32)

            # Get mesh's world transform
            mesh_world_transform = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T

            # Transform points to world frame
            points_world = np.matmul(points, mesh_world_transform[:3, :3].T) + mesh_world_transform[:3, 3]

            # Transform from world frame to body-local frame
            if body_world_transform is not None:
                # Inverse transform: world -> body local
                body_rotation_inv = body_world_transform[:3, :3].T
                body_translation = body_world_transform[:3, 3]
                points_body_local = np.matmul(points_world - body_translation, body_rotation_inv.T)
            else:
                points_body_local = points_world

            # Handle face indices (USD can have quads, we need triangles)
            # Get face vertex counts to triangulate properly
            face_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
            if face_counts is not None:
                face_counts = np.asarray(face_counts)
                triangulated_indices = []
                idx_ptr = 0
                for count in face_counts:
                    if count == 3:
                        triangulated_indices.append(indices[idx_ptr:idx_ptr+3])
                    elif count == 4:
                        # Triangulate quad into 2 triangles
                        quad = indices[idx_ptr:idx_ptr+4]
                        triangulated_indices.append([quad[0], quad[1], quad[2]])
                        triangulated_indices.append([quad[0], quad[2], quad[3]])
                    else:
                        # Fan triangulation for polygons
                        for i in range(1, count - 1):
                            triangulated_indices.append([indices[idx_ptr], indices[idx_ptr+i], indices[idx_ptr+i+1]])
                    idx_ptr += count
                indices = np.array(triangulated_indices, dtype=np.int32).flatten()

            # Adjust indices for combined mesh
            indices_adjusted = indices + vertex_offset

            all_vertices.append(points_body_local)
            all_faces.append(indices_adjusted.reshape(-1, 3))
            vertex_offset += len(points_body_local)

        if len(all_vertices) == 0:
            omni.log.warn(f"No valid mesh data for {agent_id}")
            return None

        # Combine all meshes
        combined_vertices = np.concatenate(all_vertices, axis=0)
        combined_faces = np.concatenate(all_faces, axis=0).flatten()

        omni.log.info(f"Combined mesh for {agent_id}: {len(combined_vertices)} vertices, {len(combined_faces)//3} triangles")

        # Optional: Apply mesh simplification
        if self.cfg.agent_mesh_simplification < 1.0:
            combined_vertices, combined_faces = self._simplify_mesh(
                combined_vertices, combined_faces, self.cfg.agent_mesh_simplification
            )

        # Convert to warp mesh
        wp_mesh = convert_to_warp_mesh(combined_vertices, combined_faces, device=self.device)

        return wp_mesh

    def _create_box_collision_mesh(self) -> wp.Mesh:
        """Create a simple box collision proxy mesh.
        
        Returns:
            Warp mesh of a box (0.5m x 0.5m x 0.2m) centered at origin.
        """
        import numpy as np
        
        # Box dimensions (body-local frame)
        dx, dy, dz = 0.5, 0.5, 0.2
        
        # 8 corners of box
        vertices = np.array([
            [-dx/2, -dy/2, -dz/2],
            [ dx/2, -dy/2, -dz/2],
            [ dx/2,  dy/2, -dz/2],
            [-dx/2,  dy/2, -dz/2],
            [-dx/2, -dy/2,  dz/2],
            [ dx/2, -dy/2,  dz/2],
            [ dx/2,  dy/2,  dz/2],
            [-dx/2,  dy/2,  dz/2],
        ], dtype=np.float32)
        
        # 12 triangles (2 per face)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ], dtype=np.int32)
        
        return convert_to_warp_mesh(vertices, faces, device=self.device)

    def _simplify_mesh(
        self, 
        vertices: "np.ndarray", 
        faces: "np.ndarray", 
        ratio: float
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Simplify mesh by reducing triangle count.
        
        Args:
            vertices: Vertex array (N, 3).
            faces: Face array (M, 3).
            ratio: Target ratio of triangles to keep (0-1).
            
        Returns:
            Tuple of (simplified_vertices, simplified_faces).
        """
        # Simple decimation: keep every Nth triangle
        # For production, use proper mesh simplification library (e.g., pyfqmr)
        keep_every = max(1, int(1.0 / ratio))
        simplified_faces = faces[::keep_every]
        
        # Remove unused vertices (optional, for memory efficiency)
        used_vertices = np.unique(simplified_faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        simplified_vertices = vertices[used_vertices]
        simplified_faces = np.array([
            [vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]]
            for f in simplified_faces
        ], dtype=np.int32)
        
        return simplified_vertices, simplified_faces

    def _extract_target_bboxes(self):
        """Extract 3D bounding box corners efficiently for all targets."""
        if len(self.cfg.target_prim_paths) == 0:
            raise RuntimeError("No target prim paths provided")
        
        corners_list = []
        bbox_sizes_list = []
        valid_paths = []
        
        for env_path in self.cfg.target_prim_paths:
            if "/target" not in env_path:
                target_path = f"{env_path}/target"
            else:
                target_path = env_path
            
            prim = prim_utils.get_prim_at_path(target_path)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"Invalid target prim path: {target_path}")
            
            try:
                corners = get_bbox_corners_local(target_path, self.device)
                bbox_size = compute_bbox_size(corners)
                
                corners_list.append(corners)
                bbox_sizes_list.append(bbox_size)
                valid_paths.append(target_path)
            except Exception as e:
                raise RuntimeError(f"Failed to extract bbox from {target_path}: {e}")
        
        if len(corners_list) == 0:
            raise RuntimeError("No valid target bboxes extracted")
        
        all_corners = torch.stack(corners_list, dim=0)
        all_sizes = torch.stack(bbox_sizes_list, dim=0)
        
        first_corners = all_corners[0]
        differences = torch.abs(all_corners - first_corners.unsqueeze(0))
        max_difference = differences.max().item()
        all_identical = False  # Conservative: assume different for now
        
        if all_identical:
            self.target_bbox_corners_local = first_corners
            self.target_bbox_sizes = bbox_sizes_list[0]
            self.targets_share_bbox = True
            omni.log.info(f"All {len(corners_list)} targets have identical bbox - using shared storage")
        else:
            self.target_bbox_corners_local = all_corners
            self.target_bbox_sizes = all_sizes
            self.targets_share_bbox = False
            omni.log.info(f"Extracted {len(corners_list)} unique target bboxes")

    def _allocate_buffers(self):
        """Pre-allocate all tensor buffers for efficient memory usage."""
        N, C, T = self.num_envs, self.num_cameras, self.num_targets

        # Camera and target data
        self._data.camera_pos_w = torch.zeros((N, C, 3), device=self.device)
        self._data.camera_quat_w = torch.zeros((N, C, 4), device=self.device)
        self._data.target_pos_w = torch.zeros((N, T, 3), device=self.device)
        self._data.target_quat_w = torch.zeros((N, T, 4), device=self.device)
        self._data.intrinsic_matrices = torch.zeros((N, C, 3, 3), device=self.device)
        self._data.image_shapes = torch.zeros((N, C, 2), device=self.device, dtype=torch.long)

        # Output buffers
        self._data.bboxes = torch.zeros((N, C, T, 4), device=self.device)
        self._data.bboxes_xyxy = torch.zeros((N, C, T, 4), device=self.device)
        self._data.bboxes_normalized = torch.zeros((N, C, T, 4), device=self.device)
        self._valid_mask = torch.zeros((N, C, T), device=self.device, dtype=torch.bool)
        self._data.bbox_empty = torch.ones((N, C, T), device=self.device, dtype=torch.bool)
        self._data.bbox_compute_empty_mask = torch.ones((N, C, T), device=self.device, dtype=torch.bool)
        self._data.bbox_visibility_empty_mask = torch.ones((N, C, T), device=self.device, dtype=torch.bool)
        self._data.bbox_corners_empty_mask = torch.ones((N, C, T), device=self.device, dtype=torch.bool)
        self._data.bbox_size_empty_mask = torch.ones((N, C, T), device=self.device, dtype=torch.bool)
        self._data.bbox_confidence = torch.zeros((N, C, T), device=self.device)
        self._data.occluded = torch.zeros((N, C, T), device=self.device, dtype=torch.bool)
        self._data.self_occlusion_mask = torch.zeros((N, C, T), device=self.device, dtype=torch.bool)

        # Intermediate buffers
        self._corners_world = torch.zeros((N, T, 8, 3), device=self.device)
        self._corners_camera = torch.zeros((N, C, T, 8, 3), device=self.device)
        self._pixels = torch.zeros((N, C, T, 8, 2), device=self.device)
        self._depths = torch.zeros((N, C, T, 8), device=self.device)
        self._corners_valid = torch.zeros((N, C, T, 8), device=self.device, dtype=torch.bool)

        self.targets_share_bbox = False

        # Debug buffers
        if self.cfg.debug_vis or self.cfg.debug_vis_corners:
            self._data.projected_corners_2d = torch.zeros((N, C, T, 8, 2), device=self.device)
            self._data.corners_valid_mask = torch.zeros((N, C, T, 8), device=self.device, dtype=torch.bool)

        # Occlusion buffers
        if self.cfg.enable_occlusion_check:
            num_test_points = self._get_num_occlusion_points()
            self._occlusion_test_points = torch.zeros((N, T, num_test_points, 3), device=self.device)
            self._occlusion_ray_hits = torch.zeros((N, C, T, num_test_points, 3), device=self.device)
            self._data.occlusion_visibility_ratio = torch.zeros((N, C, T), device=self.device)

            if self.cfg.debug_vis or self.cfg.debug_vis_occlusion_rays:
                self._data.occlusion_test_points_w = torch.zeros((N, T, num_test_points, 3), device=self.device)
                self._data.occlusion_ray_hits_w = torch.zeros((N, C, T, num_test_points, 3), device=self.device)

    def _setup_visualization(self):
        """Initialize visualization markers for debugging."""
        if not hasattr(self, "visualizer"):
            self.visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
        self.visualizer.set_visibility(True)

    """
    Implementation - Update Steps
    """

    def _stack_camera_data(
        self,
        camera_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
        camera_intrinsics: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack camera data from dictionary to batched tensors."""
        agent_ids = sorted(camera_poses.keys())

        if len(agent_ids) != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {len(agent_ids)}. "
                f"Agent IDs: {agent_ids}"
            )

        camera_pos_list = []
        camera_quat_list = []
        intrinsic_list = []

        for agent_id in agent_ids:
            pos, quat = camera_poses[agent_id]
            intrinsic = camera_intrinsics[agent_id]

            camera_pos_list.append(pos)
            camera_quat_list.append(quat)
            intrinsic_list.append(intrinsic)

        camera_pos = torch.stack(camera_pos_list, dim=1)
        camera_quat = torch.stack(camera_quat_list, dim=1)
        intrinsic_matrices = torch.stack(intrinsic_list, dim=1)

        return camera_pos, camera_quat, intrinsic_matrices

    def _update_image_shapes(self, image_shapes: dict[str, tuple[int, int]]):
        """Update image shapes from dictionary."""
        agent_ids = sorted(image_shapes.keys())

        for i, agent_id in enumerate(agent_ids):
            h, w = image_shapes[agent_id]
            self._data.image_shapes[:, i, 0] = h
            self._data.image_shapes[:, i, 1] = w

    def _normalize_quaternions(self, quat: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalize quaternions for numerical stability."""
        quat_norm = torch.norm(quat, dim=-1, keepdim=True)
        quat_normalized = quat / torch.clamp(quat_norm, min=eps)
        return quat_normalized

    def _transform_corners_to_world(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        target_scale: torch.Tensor | None = None
    ):
        """Transform target bbox corners from local to world frame."""
        corners_local = self.target_bbox_corners_local
        if target_scale is not None and target_scale.ndim != 3:
            target_scale = None
        
        if self.targets_share_bbox:
            batch_transform_points(
                corners_local,
                target_pos,
                target_quat,
                out=self._corners_world,
                eps=self.cfg.quat_normalize_epsilon
            )
        else:
            N, T = target_pos.shape[:2]
            K = corners_local.shape[1]
            
            corners_expanded = corners_local.unsqueeze(1).expand(N, T, K, 3)
            
            self._batch_transform_points_per_env(
                corners_expanded * target_scale.unsqueeze(2) if target_scale is not None else corners_expanded,
                target_pos,
                target_quat,
                out=self._corners_world
            )
    
    def _batch_transform_points_per_env(
        self,
        points_local: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        out: torch.Tensor
    ):
        """Transform points when each environment has different local points."""
        N, T, K = points_local.shape[:3]
        
        quat_norm = torch.norm(quat, dim=-1, keepdim=True)
        quat_normalized = quat / torch.clamp(quat_norm, min=self.cfg.quat_normalize_epsilon)
        
        points_flat = points_local.reshape(N * T * K, 3)
        quat_flat = quat_normalized.unsqueeze(2).expand(-1, -1, K, -1).reshape(N * T * K, 4)
        
        points_rotated_flat = math_utils.quat_apply(quat_flat, points_flat)
        points_rotated = points_rotated_flat.view(N, T, K, 3)
        
        pos_expanded = pos.unsqueeze(2)
        torch.add(points_rotated, pos_expanded, out=out)

    def _transform_corners_to_camera(
        self,
        camera_pos: torch.Tensor,
        camera_quat: torch.Tensor
    ):
        """Transform corners from world to camera frames."""
        corners_world_expanded = self._corners_world.unsqueeze(1).expand(
            -1, self.num_cameras, -1, -1, -1
        )

        batch_transform_to_camera_frame(
            corners_world_expanded,
            camera_pos,
            camera_quat,
            out=self._corners_camera,
            eps=self.cfg.quat_normalize_epsilon
        )

    def _project_corners_to_image(self, intrinsic_matrices: torch.Tensor):
        """Project corners from camera frame to image plane."""
        pixels, depths, projection_valid = batch_project_to_image_plane(
            self._corners_camera,
            intrinsic_matrices,
            eps=self.cfg.projection_epsilon
        )

        self._pixels = pixels
        self._depths = depths

        self._corners_valid = validate_projections(
            pixels,
            depths,
            projection_valid,
            self._data.image_shapes,
            min_depth=0.01
        )

        if self.cfg.debug_vis or self.cfg.debug_vis_corners:
            self._data.projected_corners_2d = pixels
            self._data.corners_valid_mask = self._corners_valid

    def _compute_bboxes(self):
        """Compute 2D bounding boxes from projected corners."""
        bbox_xyxy, bbox_valid = compute_2d_bbox_from_corners(
            self._pixels,
            self._corners_valid,
            min_area_pixels=self.cfg.min_bbox_area_pixels
        )

        bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)

        self._data.bboxes = bbox_xywh
        self._data.bboxes_xyxy = bbox_xyxy

        bbox_normalized = normalize_bboxes(
            bbox_xywh,
            self._data.image_shapes,
            epsilon=self.cfg.projection_epsilon
        )
        self._data.bboxes_normalized = bbox_normalized

        self._valid_mask = bbox_valid
        self._data.bbox_compute_empty_mask = ~bbox_valid
        self._data.bbox_confidence = bbox_valid.float()

    def _validate_detections(self):
        """Validate bbox detections based on various criteria."""
        corners_ok = check_all_corners_visible(
            self._corners_valid,
            allow_partial=self.cfg.partial_detection_allowed
        )

        size_ok = validate_bbox_sizes(
            self._data.bboxes_normalized,
            self.cfg.min_bbox_size,
            self.cfg.max_bbox_size
        )

        self._valid_mask = self._valid_mask & corners_ok & size_ok
        self._data.bbox_corners_empty_mask = ~corners_ok
        self._data.bbox_size_empty_mask = ~size_ok
        self._data.bbox_confidence = self._data.bbox_confidence * corners_ok.float() * size_ok.float()

    def _check_occlusions(
        self,
        agent_poses: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
        camera_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
        target_scale: torch.Tensor | None = None
    ):
        """Check for occlusions using fully batched body-local raycasting."""
        if self.static_mesh is None and len(self.agent_meshes) == 0:
            warnings.warn("No meshes loaded. Skipping occlusion check.")
            return

        # Generate test points from bbox corners
        generate_occlusion_test_points(
            self._corners_world,
            self.cfg.occlusion_ray_pattern,
            out=self._occlusion_test_points
        )

        # Get target bbox size and ensure shape (N, T, 3) for occlusion checking
        N, T = self.num_envs, self.num_targets
        if self.targets_share_bbox:
            # Shape (3,) - same bbox for all targets across all environments
            target_bbox_size = self.target_bbox_sizes
            if target_scale is not None:
                target_bbox_size = target_bbox_size * target_scale.squeeze(1)[0, :]
            # Expand to (N, T, 3)
            target_bbox_size = target_bbox_size.view(1, 1, 3).expand(N, T, -1)
        else:
            # Shape (num_envs, 3) - stacked from per-environment extraction
            target_bbox_size = self.target_bbox_sizes
            if target_scale is not None:
                target_bbox_size = target_bbox_size * target_scale.squeeze(1)
            # Expand to (N, T, 3) - each environment has same bbox size for all T targets
            target_bbox_size = target_bbox_size.unsqueeze(1).expand(-1, T, -1)

        # Get current agent IDs for cameras
        current_agent_ids = sorted(camera_poses.keys())

        # Check occlusion using fully batched raycasting
        visibility_mask, visibility_ratio, self_occlusion_mask = batch_check_occlusion_fully_batched(
            self._data.camera_pos_w,
            self._data.camera_quat_w,
            self._occlusion_test_points,
            self._data.target_pos_w,
            target_bbox_size,
            agent_poses=agent_poses or {},
            agent_meshes=self.agent_meshes,
            static_mesh=self.static_mesh,
            max_distance=self.cfg.max_distance,
            visibility_threshold=self.cfg.occlusion_visibility_threshold,
            tolerance_scale=self.cfg.occlusion_ray_tolerance,
            current_agent_ids=current_agent_ids,
            enable_self_occlusion=self.cfg.enable_self_occlusion,
            self_occlusion_min_hit_distance_m=self.cfg.self_occlusion_min_hit_distance_m,
        )

        self._data.occlusion_visibility_ratio = visibility_ratio
        self._data.bbox_visibility_empty_mask = ~visibility_mask
        self._data.self_occlusion_mask = self_occlusion_mask
        self._valid_mask = self._valid_mask & visibility_mask
        self._data.bbox_confidence = self._data.bbox_confidence * visibility_ratio.clamp(0.0, 1.0)

        if self.cfg.debug_vis or self.cfg.debug_vis_occlusion_rays:
            self._data.occlusion_test_points_w = self._occlusion_test_points

    def _apply_inter_target_occlusion(self):
        """Apply image-space inter-target occlusion after geometric visibility."""
        bbox_empty, bbox_confidence, occluded = apply_inter_target_occlusion(
            self._data.bboxes_xyxy,
            ~self._valid_mask,
            self._data.bbox_confidence,
            self._data.camera_pos_w,
            self._data.target_pos_w,
            soft_iou_threshold=self.cfg.inter_target_iou_soft_threshold,
            hard_iou_threshold=self.cfg.inter_target_iou_hard_threshold,
            depth_margin_m=self.cfg.inter_target_depth_margin_m,
        )
        self._valid_mask = ~bbox_empty
        self._data.bbox_confidence = bbox_confidence
        self._data.occluded |= occluded

    def _sync_public_outputs(self):
        """Convert internal valid masks to public bbox_empty and zero invalid boxes."""
        self._data.bbox_empty = ~self._valid_mask
        empty_expanded = self._data.bbox_empty.unsqueeze(-1)
        self._data.bboxes = torch.where(empty_expanded, torch.zeros_like(self._data.bboxes), self._data.bboxes)
        self._data.bboxes_xyxy = torch.where(empty_expanded, torch.zeros_like(self._data.bboxes_xyxy), self._data.bboxes_xyxy)
        self._data.bboxes_normalized = torch.where(
            empty_expanded,
            torch.zeros_like(self._data.bboxes_normalized),
            self._data.bboxes_normalized,
        )
        self._data.bbox_confidence = torch.where(
            self._data.bbox_empty,
            torch.zeros_like(self._data.bbox_confidence),
            self._data.bbox_confidence.clamp(0.0, 1.0),
        )

    """
    Helper Methods
    """

    def _get_num_occlusion_points(self) -> int:
        """Get number of occlusion test points based on pattern."""
        if self.cfg.occlusion_ray_pattern == "9point":
            return 9
        elif self.cfg.occlusion_ray_pattern == "corners_only":
            return 4
        elif self.cfg.occlusion_ray_pattern == "center_only":
            return 1
        else:
            raise ValueError(f"Unknown occlusion pattern: {self.cfg.occlusion_ray_pattern}")

    def _track_memory(self, label: str):
        """Track GPU memory usage for profiling."""
        if self._memory_tracker is None:
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        self._memory_tracker[label] = allocated

        if self.cfg.debug_memory:
            print(f"[BBoxRayCasterV2] {label}: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")

    def visualize(self):
        """Update debug visualization markers."""
        if not self.cfg.debug_vis:
            return

        if not hasattr(self, "visualizer"):
            return

        if self.cfg.debug_vis_corners and self._data.projected_corners_2d is not None:
            viz_corners = self._corners_world.reshape(-1, 3)
            valid_corners = ~torch.isinf(viz_corners).any(dim=-1)
            self.visualizer.visualize(viz_corners[valid_corners])

        if self.cfg.debug_vis_occlusion_rays and self._data.occlusion_ray_hits_w is not None:
            viz_hits = self._data.occlusion_ray_hits_w.reshape(-1, 3)
            valid_hits = ~torch.isinf(viz_hits).any(dim=-1)
            self.visualizer.visualize(viz_hits[valid_hits])

    def add_noise(self, noise: torch.Tensor):
        """Add noise to the bounding boxes (for data augmentation).
        
        Args:
            noise: Tensor of shape (N, C, T, 4) representing noise to add to bboxes.
        """
        self._data.bboxes_xyxy += noise
        self._data.bboxes = bbox_xyxy_to_xywh(self._data.bboxes_xyxy)
        self._data.bboxes_normalized = normalize_bboxes(
            self._data.bboxes,
            self._data.image_shapes,
            epsilon=self.cfg.projection_epsilon
        )
        self._validate_detections()

    def setup_detector_replicator(self, cfg: DetectorReplicatorCfg):
        """Create and configure a detector replicator for calibrated noise.

        Args:
            cfg: Detector replicator configuration (enabled, params_path, apply_bias).
        """
        self._detector_replicator = DetectorReplicator(cfg=cfg, device=self.device)
        self._last_replicator_time = -1.0

    def apply_detector_replicator(
        self,
        noise_scale: float = 1.0,
        fp_fn_scale: float = 1.0,
        sim_time: float | None = None,
    ):
        """Apply calibrated noise, miss rate, and FP injection.

        Must be called after update(). Writes results to:
        - data.bboxes_replicated (N, C, T, 4) xywh
        - data.bboxes_xyxy_replicated (N, C, T, 4) xyxy
        - data.bbox_empty_replicated (N, C, T)

        Args:
            noise_scale: Curriculum scale [0, 1] for localization noise.
            fp_fn_scale: Curriculum scale [0, 1] for miss rate and FP.
            sim_time: Current sim time for idempotency guard. If None, no guard.
        """
        # Idempotency guard: skip if already applied this sim step
        if sim_time is not None:
            t = sim_time.item() if isinstance(sim_time, torch.Tensor) and sim_time.numel() == 1 else (
                sim_time[0].item() if isinstance(sim_time, torch.Tensor) else sim_time
            )
            if abs(t - self._last_replicator_time) < 1e-6:
                return
            self._last_replicator_time = t

        if not hasattr(self, "_detector_replicator") or self._detector_replicator is None:
            # No replicator configured — replicated = GT
            self._data.bboxes_replicated = self._data.bboxes.clone()
            self._data.bboxes_xyxy_replicated = self._data.bboxes_xyxy.clone()
            self._data.bbox_empty_replicated = self._data.bbox_empty.clone()
            return

        # Classify background (sky vs ground) for miss rate conditioning
        bg_is_ground = self._classify_background() if fp_fn_scale > 0.0 else None
        self._data.bg_is_ground = bg_is_ground

        bboxes_rep, empty_rep = self._detector_replicator.apply(
            bboxes_xywh=self._data.bboxes,
            bbox_empty=self._data.bbox_empty,
            noise_scale=noise_scale,
            fp_fn_scale=fp_fn_scale,
            bg_is_ground=bg_is_ground,
            image_shapes=self._data.image_shapes,
        )

        self._data.bboxes_replicated = bboxes_rep
        self._data.bbox_empty_replicated = empty_rep
        self._data.bboxes_xyxy_replicated = self._xywh_to_xyxy(bboxes_rep)

    def _classify_background(self) -> torch.Tensor:
        """Classify background as sky or ground for each camera-target pair.

        Uses two criteria (OR):
        1. Raycast from target beyond: if it hits the static mesh → ground.
        2. Ray direction check: if the ray points downward (negative Z),
           the background is terrain/ground even if the mesh raycast misses
           (e.g. flight scene mountains not in the warp mesh).

        Returns:
            bg_is_ground: (N, C, T) bool. True = ground background.
        """
        N, C, T = self._data.bbox_empty.shape

        # Ray from camera through target
        camera_pos = self._data.camera_pos_w  # (N, C, 3)
        target_pos = self._data.target_pos_w  # (N, T, 3)

        # Expand for all camera-target pairs: (N, C, T, 3)
        cam_expanded = camera_pos.unsqueeze(2).expand(N, C, T, 3)
        tgt_expanded = target_pos.unsqueeze(1).expand(N, C, T, 3)

        # Direction: camera → target → beyond (we want what's behind the target)
        ray_dir = tgt_expanded - cam_expanded
        ray_dir = ray_dir / (ray_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # Criterion 2: ray points downward → ground background
        # (catches mountains/terrain not in static_mesh)
        ray_points_down = ray_dir[..., 2] < 0.0  # (N, C, T)

        if self.static_mesh is None:
            return ray_points_down

        # Criterion 1: raycast against static mesh
        batch_size = N * C * T
        ray_starts_flat = tgt_expanded.reshape(batch_size, 3)
        ray_dirs_flat = ray_dir.reshape(batch_size, 3)

        ray_hits_flat, _, _, _ = raycast_mesh(
            ray_starts_flat,
            ray_dirs_flat,
            mesh=self.static_mesh,
            max_dist=self.cfg.max_distance,
            return_distance=False,
            return_normal=False,
        )

        ray_hits = ray_hits_flat.view(N, C, T, 3)
        mesh_hit = ~(torch.isinf(ray_hits).any(dim=-1) | torch.isnan(ray_hits).any(dim=-1))

        return mesh_hit | ray_points_down  # (N, C, T) True = ground

    @staticmethod
    def _xywh_to_xyxy(bboxes_xywh: torch.Tensor) -> torch.Tensor:
        """Convert (cx, cy, w, h) to (x_min, y_min, x_max, y_max)."""
        cx, cy, w, h = bboxes_xywh[..., 0], bboxes_xywh[..., 1], bboxes_xywh[..., 2], bboxes_xywh[..., 3]
        half_w = w / 2.0
        half_h = h / 2.0
        return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)

    def get_normalized_bboxes(self, bboxes_xywh: torch.Tensor) -> torch.Tensor:
        """Get normalized bounding boxes (x_center, y_center, width, height) in [0,1].
        Args:
            bboxes_xywh: Tensor of shape (..., 4)
        Returns:
            Tensor of shape (N, C, T, 4) with normalized bounding boxes.
        """
        return normalize_bboxes(
            bboxes_xywh,
            self._data.image_shapes,
            epsilon=self.cfg.projection_epsilon
        )
