# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main batched bounding box raycaster class."""

from __future__ import annotations

import torch
import warnings
import warp as wp
from typing import TYPE_CHECKING

import omni.log
import omni.usd
from pxr import UsdGeom

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh
import isaaclab.utils.math as math_utils

from .bbox_raycaster_data import BBoxRayCasterData
from .utils import (
    batch_check_occlusion,
    batch_project_to_image_plane,
    batch_transform_points,
    batch_transform_to_camera_frame,
    bbox_xyxy_to_xywh,
    check_all_corners_visible,
    check_gimbal_lock,
    compute_2d_bbox_from_corners,
    compute_bbox_size,
    generate_occlusion_test_points,
    get_bbox_corners_local,
    normalize_bboxes,
    validate_bbox_sizes,
    validate_projections,
)

if TYPE_CHECKING:
    from .bbox_raycaster_cfg import BBoxRayCasterCfg


class BBoxRayCaster:
    """Batched bounding box raycaster for multi-agent environments.
    
    This class efficiently computes 2D bounding boxes for 3D targets across multiple
    cameras and environments using batched GPU operations. It includes occlusion detection
    via raycasting and comprehensive validation.
    
    Example:
        >>> # Initialize raycaster
        >>> bbox_raycaster = BBoxRayCaster(
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
        >>> bbox_raycaster.update(camera_poses, camera_intrinsics, target_poses)
        >>> 
        >>> # Access results
        >>> bboxes = bbox_raycaster.data.bboxes  # (N, C, T, 4)
        >>> valid_mask = bbox_raycaster.data.valid_mask  # (N, C, T)
    """

    cfg: BBoxRayCasterCfg
    """Configuration for the raycaster."""

    def __init__(
        self,
        cfg: BBoxRayCasterCfg,
        num_envs: int,
        num_targets_per_env: int,
        device: str
    ):
        """Initialize batched bbox raycaster.
        
        Args:
            cfg: Configuration instance.
            num_envs: Number of parallel environments.
            num_targets_per_env: Number of targets per environment.
            device: Device for tensor operations (e.g., "cuda:0").
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_targets = num_targets_per_env
        self.num_cameras = cfg.num_cameras_per_env
        self.device = device

        # Initialize data container
        self._data = BBoxRayCasterData()

        # Load meshes for raycasting
        self.meshes: dict[str, wp.Mesh] = {}
        self._initialize_meshes()

        # Extract target bounding boxes (one-time, handles all environments)
        # Will be either:
        #   - (8, 3) if all targets identical (memory efficient)
        #   - (N, 8, 3) if targets differ per environment
        self.target_bbox_corners_local: torch.Tensor | None = None
        self.target_bbox_sizes: torch.Tensor | None = None
        self.targets_share_bbox: bool = True  # Flag set during extraction
        
        self._extract_target_bboxes()

        # Pre-allocate all buffers
        self._allocate_buffers()

        # Initialize visualization if enabled
        if cfg.debug_vis:
            self._setup_visualization()

        # Memory tracking
        self._memory_tracker = {} if cfg.debug_memory else None

        omni.log.info(f"BBoxRayCaster initialized with {num_envs} envs, {self.num_cameras} cameras, {num_targets_per_env} targets")

    def __str__(self) -> str:
        """String representation."""
        return (
            f"BBoxRayCaster:\n"
            f"  Environments: {self.num_envs}\n"
            f"  Cameras per env: {self.num_cameras}\n"
            f"  Targets per env: {self.num_targets}\n"
            f"  Total cameras: {self.num_envs * self.num_cameras}\n"
            f"  Total targets: {self.num_envs * self.num_targets}\n"
            f"  Meshes loaded: {len(self.meshes)}\n"
            f"  Occlusion checking: {self.cfg.enable_occlusion_check}\n"
            f"  Debug visualization: {self.cfg.debug_vis}"
        )

    """
    Properties
    """

    @property
    def data(self) -> BBoxRayCasterData:
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
        image_shapes: dict[str, tuple[int, int]] | None = None
    ):
        """Update bounding boxes for current frame.
        
        This is the main entry point for computing bounding boxes. It performs:
        1. Pose validation and normalization
        2. 3D bbox transformation to world frame
        3. Camera frame transformation
        4. Projection to image planes
        5. 2D bbox computation
        6. Validation (size, FOV, etc.)
        7. Occlusion checking (if enabled)
        
        Args:
            camera_poses: Dictionary mapping agent_id to (position, quaternion).
                Position shape: (N, 3), Quaternion shape: (N, 4) in (w, x, y, z) format.
            camera_intrinsics: Dictionary mapping agent_id to intrinsic matrix.
                Intrinsic matrix shape: (N, 3, 3).
            target_poses: Tuple of (position, quaternion) for all targets.
                Position shape: (N, T, 3), Quaternion shape: (N, T, 4).
            image_shapes: Optional dict mapping agent_id to (height, width).
                If None, uses shapes from intrinsics stored during initialization.
        """
        self._track_memory("start")

        # Step 1: Stack camera data from dict to tensors
        camera_pos_w, camera_quat_w, intrinsic_matrices = self._stack_camera_data(
            camera_poses, camera_intrinsics
        )

        # Step 2: Update image shapes if provided
        if image_shapes is not None:
            self._update_image_shapes(image_shapes)

        # Step 3: Validate and normalize inputs
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
            self._transform_corners_to_world(target_pos_w, target_quat_w)
        except RuntimeError as e:
            warnings.warn(f"Corner transformation failed: {e}")
            self._data.valid_mask.fill_(False)
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
            self._check_occlusions()

        self._track_memory("after_occlusion")

        # Cleanup intermediate buffers if memory tracking enabled
        if self.cfg.debug_memory:
            torch.cuda.empty_cache()
            self._track_memory("after_cleanup")

    """
    Implementation - Initialization
    """

    def _initialize_meshes(self):
        """Load and convert meshes to Warp format for raycasting."""
        if len(self.cfg.mesh_prim_paths) == 0:
            warnings.warn("No mesh prim paths provided. Occlusion checking will not work.")
            return

        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # Check if it's a plane (special case)
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path,
                lambda prim: prim.GetTypeName() == "Plane"
            )

            if mesh_prim is None:
                # Regular mesh
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path,
                    lambda prim: prim.GetTypeName() == "Mesh"
                )

                if mesh_prim is None or not mesh_prim.IsValid():
                    warnings.warn(f"Invalid mesh prim path: {mesh_prim_path}")
                    continue

                # Convert to UsdGeomMesh and extract geometry
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                points = mesh_prim.GetPointsAttr().Get()
                indices = mesh_prim.GetFaceVertexIndicesAttr().Get()

                # Apply world transform
                import numpy as np
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
                points = np.asarray(points)
                points = np.matmul(points, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
                indices = np.asarray(indices)

                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                omni.log.info(f"Loaded mesh: {mesh_prim_path} ({len(points)} vertices, {len(indices)} faces)")

            else:
                # Create infinite plane
                from isaaclab.terrains.trimesh.utils import make_plane
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                omni.log.info(f"Created infinite plane mesh: {mesh_prim_path}")

            self.meshes[mesh_prim_path] = wp_mesh

    def _extract_target_bboxes(self):
        """Extract 3D bounding box corners efficiently for all targets.
        
        Strategy:
        1. Extract bbox from first target
        2. Check if all other targets have identical bbox
        3. If identical: store once (8, 3) - memory efficient
        4. If different: store all (N, 8, 3) - handles heterogeneous targets
        """
        if len(self.cfg.target_prim_paths) == 0:
            raise RuntimeError("No target prim paths provided")
        
        
        # Extract bbox from all environments
        corners_list = []
        bbox_sizes_list = []
        valid_paths = []
        
        for env_path in self.cfg.target_prim_paths:
            # Construct full target path
            if "/target" not in env_path:
                target_path = f"{env_path}/target"
            else:
                target_path = env_path
            
            # Check if path exists
            prim = prim_utils.get_prim_at_path(target_path)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"Invalid target prim path: {target_path}")
            
            try:
                # Extract bbox corners and size
                corners = get_bbox_corners_local(target_path, self.device)
                bbox_size = compute_bbox_size(corners)
                
                corners_list.append(corners)
                bbox_sizes_list.append(bbox_size)
                valid_paths.append(target_path)
            except Exception as e:
                raise RuntimeError(f"Failed to extract bbox from {target_path}: {e}")
        
        if len(corners_list) == 0:
            raise RuntimeError("No valid target bboxes extracted")
        
        # Stack all corners and sizes
        all_corners = torch.stack(corners_list, dim=0)  # (N, 8, 3)
        all_sizes = torch.stack(bbox_sizes_list, dim=0)  # (N, 3)
        
        # Check if all targets have identical bbox (within tolerance)
        first_corners = all_corners[0]
        differences = torch.abs(all_corners - first_corners.unsqueeze(0))
        max_difference = differences.max().item()
        # all_identical = max_difference < 1e-5
        all_identical = False
        
        if all_identical:
            # Optimization: All targets identical - store once
            self.target_bbox_corners_local = first_corners  # (8, 3)
            self.target_bbox_sizes = bbox_sizes_list[0]  # (3,)
            self.targets_share_bbox = True
            
            omni.log.info(
                f"All {len(corners_list)} targets have identical bbox "
                f"(size={bbox_sizes_list[0].tolist()}) - using shared storage"
            )
        else:
            # Different targets - store per-environment
            self.target_bbox_corners_local = all_corners  # (N, 8, 3)
            self.target_bbox_sizes = all_sizes  # (N, 3)
            self.targets_share_bbox = False
            
            omni.log.info(
                f"Extracted {len(corners_list)} unique target bboxes "
                f"(max difference: {max_difference:.6f}m) - using per-env storage"
            )

    def _allocate_buffers(self):
        """Pre-allocate all tensor buffers for efficient memory usage."""
        N, C, T = self.num_envs, self.num_cameras, self.num_targets

        # Camera and target data (stored in data container)
        self._data.camera_pos_w = torch.zeros((N, C, 3), device=self.device)
        self._data.camera_quat_w = torch.zeros((N, C, 4), device=self.device)
        self._data.target_pos_w = torch.zeros((N, T, 3), device=self.device)
        self._data.target_quat_w = torch.zeros((N, T, 4), device=self.device)
        self._data.intrinsic_matrices = torch.zeros((N, C, 3, 3), device=self.device)
        self._data.image_shapes = torch.zeros((N, C, 2), device=self.device, dtype=torch.long)

        # Output buffers
        self._data.bboxes = torch.zeros((N, C, T, 4), device=self.device)
        self._data.bboxes_normalized = torch.zeros((N, C, T, 4), device=self.device)
        self._data.valid_mask = torch.zeros((N, C, T), device=self.device, dtype=torch.bool)

        # Intermediate buffers (reusable across updates)
        self._corners_world = torch.zeros((N, T, 8, 3), device=self.device)
        self._corners_camera = torch.zeros((N, C, T, 8, 3), device=self.device)
        self._pixels = torch.zeros((N, C, T, 8, 2), device=self.device)
        self._depths = torch.zeros((N, C, T, 8), device=self.device)
        self._corners_valid = torch.zeros((N, C, T, 8), device=self.device, dtype=torch.bool)

        # Initialize bbox sharing flag (set properly in _extract_target_bboxes)
        self.targets_share_bbox = False  # Default assumption

        # Debug buffers (optional)
        if self.cfg.debug_vis or self.cfg.debug_vis_corners:
            self._data.projected_corners_2d = torch.zeros((N, C, T, 8, 2), device=self.device)
            self._data.corners_valid_mask = torch.zeros((N, C, T, 8), device=self.device, dtype=torch.bool)

        # Occlusion buffers (conditional)
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
        """Stack camera data from dictionary to batched tensors.
        
        Returns:
            Tuple of (camera_pos, camera_quat, intrinsic_matrices).
        """
        # Sort by agent_id for consistent ordering
        agent_ids = sorted(camera_poses.keys())

        if len(agent_ids) != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {len(agent_ids)}. "
                f"Agent IDs: {agent_ids}"
            )

        # Stack positions and quaternions
        camera_pos_list = []
        camera_quat_list = []
        intrinsic_list = []

        for agent_id in agent_ids:
            pos, quat = camera_poses[agent_id]
            intrinsic = camera_intrinsics[agent_id]

            camera_pos_list.append(pos)
            camera_quat_list.append(quat)
            intrinsic_list.append(intrinsic)

        # Stack into (N, C, ...) format
        camera_pos = torch.stack(camera_pos_list, dim=1)  # (N, C, 3)
        camera_quat = torch.stack(camera_quat_list, dim=1)  # (N, C, 4)
        intrinsic_matrices = torch.stack(intrinsic_list, dim=1)  # (N, C, 3, 3)

        return camera_pos, camera_quat, intrinsic_matrices

    def _update_image_shapes(self, image_shapes: dict[str, tuple[int, int]]):
        """Update image shapes from dictionary."""
        agent_ids = sorted(image_shapes.keys())

        for i, agent_id in enumerate(agent_ids):
            h, w = image_shapes[agent_id]
            self._data.image_shapes[:, i, 0] = h
            self._data.image_shapes[:, i, 1] = w

    def _normalize_quaternions(
        self,
        quat: torch.Tensor,
        eps: float
    ) -> torch.Tensor:
        """Normalize quaternions for numerical stability."""
        quat_norm = torch.norm(quat, dim=-1, keepdim=True)
        quat_normalized = quat / torch.clamp(quat_norm, min=eps)
        return quat_normalized

    def _transform_corners_to_world(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor
    ):
        """Transform target bbox corners from local to world frame.
        
        Handles two cases efficiently:
        1. Shared bbox: All targets identical - corners_local is (8, 3)
        2. Per-env bbox: Different targets - corners_local is (N, 8, 3)
        """
        corners_local = self.target_bbox_corners_local
        
        if self.targets_share_bbox:
            # Case 1: All targets share same bbox (8, 3)
            # batch_transform_points will broadcast to all (N, T, 8, 3)
            batch_transform_points(
                corners_local,
                target_pos,
                target_quat,
                out=self._corners_world,
                eps=self.cfg.quat_normalize_epsilon
            )
        else:
            # Case 2: Different bbox per environment (N, 8, 3)
            # Need to transform each environment's corners separately
            N, T = target_pos.shape[:2]
            K = corners_local.shape[1]  # 8 corners
            
            # Expand corners to include target dimension: (N, 8, 3) -> (N, T, 8, 3)
            corners_expanded = corners_local.unsqueeze(1).expand(N, T, K, 3)
            
            # Transform using per-environment corners
            self._batch_transform_points_per_env(
                corners_expanded,
                target_pos,
                target_quat,
                out=self._corners_world
            )
    
    def _batch_transform_points_per_env(
        self,
        points_local: torch.Tensor,  # (N, T, K, 3) - per-env points
        pos: torch.Tensor,            # (N, T, 3)
        quat: torch.Tensor,           # (N, T, 4)
        out: torch.Tensor             # (N, T, K, 3)
    ):
        """Transform points when each environment has different local points.
        
        This is a specialized version of batch_transform_points for per-env data.
        """
        N, T, K = points_local.shape[:3]
        
        # Normalize quaternions
        quat_norm = torch.norm(quat, dim=-1, keepdim=True)
        quat_normalized = quat / torch.clamp(quat_norm, min=self.cfg.quat_normalize_epsilon)
        
        # Flatten for rotation
        points_flat = points_local.reshape(N * T * K, 3)
        quat_flat = quat_normalized.unsqueeze(2).expand(-1, -1, K, -1).reshape(N * T * K, 4)
        
        # Rotate
        points_rotated_flat = math_utils.quat_apply(quat_flat, points_flat)
        points_rotated = points_rotated_flat.view(N, T, K, 3)
        
        # Add translation
        pos_expanded = pos.unsqueeze(2)  # (N, T, 1, 3)
        torch.add(points_rotated, pos_expanded, out=out)

    def _transform_corners_to_camera(
        self,
        camera_pos: torch.Tensor,
        camera_quat: torch.Tensor
    ):
        """Transform corners from world to camera frames."""
        # Expand corners to include camera dimension
        corners_world_expanded = self._corners_world.unsqueeze(1).expand(
            -1, self.num_cameras, -1, -1, -1
        )  # (N, C, T, 8, 3)

        # Transform to camera frames
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

        # Store in buffers
        self._pixels = pixels
        self._depths = depths

        # Validate projections
        self._corners_valid = validate_projections(
            pixels,
            depths,
            projection_valid,
            self._data.image_shapes,
            min_depth=0.01
        )

        # Store debug data if enabled
        if self.cfg.debug_vis or self.cfg.debug_vis_corners:
            self._data.projected_corners_2d = pixels
            self._data.corners_valid_mask = self._corners_valid

    def _compute_bboxes(self):
        """Compute 2D bounding boxes from projected corners."""
        # Compute bbox in xyxy format
        bbox_xyxy, bbox_valid = compute_2d_bbox_from_corners(
            self._pixels,
            self._corners_valid,
            min_area_pixels=self.cfg.min_bbox_area_pixels
        )

        # Convert to xywh format (center_x, center_y, width, height)
        bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)

        # Store in data container
        self._data.bboxes = bbox_xywh

        # Normalize by image dimensions
        bbox_normalized = normalize_bboxes(
            bbox_xywh,
            self._data.image_shapes,
            epsilon=self.cfg.projection_epsilon
        )
        self._data.bboxes_normalized = bbox_normalized

        # Initialize validity mask with bbox computation result
        self._data.valid_mask = bbox_valid

    def _validate_detections(self):
        """Validate bbox detections based on various criteria."""
        # Check 1: Corner visibility
        corners_ok = check_all_corners_visible(
            self._corners_valid,
            allow_partial=self.cfg.partial_detection_allowed
        )

        # Check 2: Bbox size constraints
        size_ok = validate_bbox_sizes(
            self._data.bboxes_normalized,
            self.cfg.min_bbox_size,
            self.cfg.max_bbox_size
        )

        # Combine with existing validity
        self._data.valid_mask = self._data.valid_mask & corners_ok & size_ok

    def _check_occlusions(self):
        """Check for occlusions using raycasting."""
        if len(self.meshes) == 0:
            warnings.warn("No meshes loaded. Skipping occlusion check.")
            return

        # Generate test points from bbox corners
        generate_occlusion_test_points(
            self._corners_world,
            self.cfg.occlusion_ray_pattern,
            out=self._occlusion_test_points
        )

        # Get target bbox size (handle both shared and per-env cases)
        if self.targets_share_bbox:
            # Single bbox size for all targets
            target_bbox_size = self.target_bbox_sizes  # (3,)
        else:
            # Different bbox size per environment
            target_bbox_size = self.target_bbox_sizes  # (N, 3)

        # Check occlusion for all cameras
        # Use first mesh for now (TODO: support multiple meshes)
        mesh = list(self.meshes.values())[0]

        visibility_mask, visibility_ratio = batch_check_occlusion(
            self._data.camera_pos_w,
            self._occlusion_test_points,
            self._data.target_pos_w,
            target_bbox_size,
            mesh=mesh,
            max_distance=self.cfg.max_distance,
            visibility_threshold=self.cfg.occlusion_visibility_threshold,
            tolerance_scale=self.cfg.occlusion_ray_tolerance,
            out_hits=self._occlusion_ray_hits if self.cfg.debug_vis else None
        )

        # Store visibility ratio
        self._data.occlusion_visibility_ratio = visibility_ratio

        # Update validity mask with occlusion check
        self._data.valid_mask = self._data.valid_mask & visibility_mask

        # Store debug data
        if self.cfg.debug_vis or self.cfg.debug_vis_occlusion_rays:
            self._data.occlusion_test_points_w = self._occlusion_test_points
            self._data.occlusion_ray_hits_w = self._occlusion_ray_hits

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
            print(f"[BBoxRayCaster] {label}: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")

    def visualize(self):
        """Update debug visualization markers."""
        if not self.cfg.debug_vis:
            return

        if not hasattr(self, "visualizer"):
            return

        # Visualize projected corners
        if self.cfg.debug_vis_corners and self._data.projected_corners_2d is not None:
            # This would require converting 2D pixels back to 3D for visualization
            # For now, visualize the 3D corners in world frame
            viz_corners = self._corners_world.reshape(-1, 3)
            valid_corners = ~torch.isinf(viz_corners).any(dim=-1)
            self.visualizer.visualize(viz_corners[valid_corners])

        # Visualize occlusion ray hits
        if self.cfg.debug_vis_occlusion_rays and self._data.occlusion_ray_hits_w is not None:
            viz_hits = self._data.occlusion_ray_hits_w.reshape(-1, 3)
            valid_hits = ~torch.isinf(viz_hits).any(dim=-1)
            self.visualizer.visualize(viz_hits[valid_hits])