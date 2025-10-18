# file: utils/mesh_utils.py

"""Utilities for mesh loading, caching, and management in body-local raycasting."""

import torch
import warp as wp
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle
import hashlib

from pxr import UsdGeom
import omni.usd
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils.warp import convert_to_warp_mesh


class MeshCache:
    """Cache for pre-processed agent meshes to speed up initialization."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize mesh cache.
        
        Args:
            cache_dir: Directory to store cached meshes. If None, uses temp dir.
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "isaaclab" / "bbox_raycaster"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(
        self,
        prim_path: str,
        simplification_ratio: float,
        apply_transform: bool
    ) -> str:
        """Generate cache key for mesh.
        
        Args:
            prim_path: Path to prim in USD stage.
            simplification_ratio: Mesh simplification ratio.
            apply_transform: Whether world transform was applied.
            
        Returns:
            Cache key string.
        """
        key_str = f"{prim_path}_{simplification_ratio}_{apply_transform}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        prim_path: str,
        simplification_ratio: float,
        apply_transform: bool
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached mesh if available.
        
        Returns:
            Tuple of (vertices, faces) if cached, None otherwise.
        """
        cache_key = self.get_cache_key(prim_path, simplification_ratio, apply_transform)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data['vertices'], data['faces']
            except Exception as e:
                print(f"Warning: Failed to load cached mesh: {e}")
                return None
        
        return None
    
    def put(
        self,
        prim_path: str,
        simplification_ratio: float,
        apply_transform: bool,
        vertices: np.ndarray,
        faces: np.ndarray
    ):
        """Cache mesh data.
        
        Args:
            prim_path: Path to prim in USD stage.
            simplification_ratio: Mesh simplification ratio.
            apply_transform: Whether world transform was applied.
            vertices: Mesh vertices array.
            faces: Mesh faces array.
        """
        cache_key = self.get_cache_key(prim_path, simplification_ratio, apply_transform)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'vertices': vertices,
                    'faces': faces,
                    'prim_path': prim_path,
                    'simplification_ratio': simplification_ratio,
                    'apply_transform': apply_transform,
                }, f)
        except Exception as e:
            print(f"Warning: Failed to cache mesh: {e}")
    
    def clear(self):
        """Clear all cached meshes."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class MeshLoader:
    """Utility class for loading and processing meshes."""
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[str] = None):
        """Initialize mesh loader.
        
        Args:
            use_cache: Whether to use mesh caching.
            cache_dir: Directory for cache storage.
        """
        self.cache = MeshCache(cache_dir) if use_cache else None
    
    def load_agent_mesh(
        self,
        prim_path: str,
        device: str,
        simplification_ratio: float = 1.0,
        apply_transform: bool = False
    ) -> Optional[wp.Mesh]:
        """Load agent mesh from USD prim.
        
        Args:
            prim_path: Path to prim (e.g., "/World/envs/env_0/Robot_0").
            device: Device for warp mesh (e.g., "cuda:0").
            simplification_ratio: Mesh simplification ratio [0, 1].
            apply_transform: Whether to apply world transform.
            
        Returns:
            Warp mesh in body-local (or world) coordinates, or None if failed.
        """
        # Try to load from cache
        if self.cache is not None:
            cached = self.cache.get(prim_path, simplification_ratio, apply_transform)
            if cached is not None:
                vertices, faces = cached
                return convert_to_warp_mesh(vertices, faces, device=device)
        
        # Load from USD
        vertices, faces = self._extract_mesh_from_prim(
            prim_path, apply_transform=apply_transform
        )
        
        if vertices is None or faces is None:
            return None
        
        # Simplify if requested
        if simplification_ratio < 1.0:
            vertices, faces = self._simplify_mesh(vertices, faces, simplification_ratio)
        
        # Cache if enabled
        if self.cache is not None:
            self.cache.put(prim_path, simplification_ratio, apply_transform, vertices, faces)
        
        # Convert to warp
        return convert_to_warp_mesh(vertices, faces, device=device)
    
    def _extract_mesh_from_prim(
        self,
        prim_path: str,
        apply_transform: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract mesh geometry from USD prim.
        
        Args:
            prim_path: Path to prim.
            apply_transform: Whether to apply world transform.
            
        Returns:
            Tuple of (vertices, faces) or (None, None) if failed.
        """
        # Try to find body mesh
        body_prim = prim_utils.get_first_matching_child_prim(
            prim_path,
            lambda prim: "body" in str(prim.GetPath()).lower() and prim.GetTypeName() == "Mesh"
        )
        
        if body_prim is None or not body_prim.IsValid():
            print(f"Warning: Could not find body mesh at {prim_path}")
            return None, None
        
        # Extract geometry
        mesh_prim = UsdGeom.Mesh(body_prim)
        points = mesh_prim.GetPointsAttr().Get()
        indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
        
        if points is None or indices is None:
            print(f"Warning: Invalid mesh data at {prim_path}")
            return None, None
        
        vertices = np.asarray(points, dtype=np.float32)
        faces = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        
        # Apply transform if requested
        if apply_transform:
            transform_matrix = np.array(omni.usd.get_world_transform_matrix(body_prim)).T
            vertices = np.matmul(vertices, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
        
        return vertices, faces
    
    def _simplify_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simplify mesh by reducing triangle count.
        
        Args:
            vertices: Vertex array (N, 3).
            faces: Face array (M, 3).
            ratio: Target ratio of triangles to keep (0-1).
            
        Returns:
            Tuple of (simplified_vertices, simplified_faces).
        """
        # Simple decimation: keep every Nth triangle
        keep_every = max(1, int(1.0 / ratio))
        simplified_faces = faces[::keep_every]
        
        # Remove unused vertices
        used_vertices = np.unique(simplified_faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        simplified_vertices = vertices[used_vertices]
        simplified_faces = np.array([
            [vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]]
            for f in simplified_faces
        ], dtype=np.int32)
        
        return simplified_vertices, simplified_faces
    
    def create_box_mesh(
        self,
        size: Tuple[float, float, float] = (0.5, 0.5, 0.2),
        device: str = "cuda:0"
    ) -> wp.Mesh:
        """Create a simple box collision mesh.
        
        Args:
            size: Box dimensions (x, y, z).
            device: Device for warp mesh.
            
        Returns:
            Warp mesh of box centered at origin.
        """
        dx, dy, dz = size
        
        # 8 corners
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
        
        return convert_to_warp_mesh(vertices, faces, device=device)
    
    def create_sphere_mesh(
        self,
        radius: float = 0.3,
        subdivisions: int = 2,
        device: str = "cuda:0"
    ) -> wp.Mesh:
        """Create a sphere collision mesh using icosphere subdivision.
        
        Args:
            radius: Sphere radius.
            subdivisions: Number of subdivision iterations (2-3 recommended).
            device: Device for warp mesh.
            
        Returns:
            Warp mesh of sphere centered at origin.
        """
        # Start with icosahedron
        t = (1.0 + np.sqrt(5.0)) / 2.0
        
        vertices = np.array([
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Subdivide
        for _ in range(subdivisions):
            vertices, faces = self._subdivide_mesh(vertices, faces)
        
        # Scale to desired radius
        vertices *= radius
        
        return convert_to_warp_mesh(vertices, faces, device=device)
    
    def _subdivide_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subdivide mesh by splitting each triangle into 4."""
        edge_map = {}
        new_faces = []
        new_vertices = list(vertices)
        
        def get_midpoint(v1_idx: int, v2_idx: int) -> int:
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge not in edge_map:
                v1 = vertices[v1_idx]
                v2 = vertices[v2_idx]
                mid = (v1 + v2) / 2.0
                mid = mid / np.linalg.norm(mid)  # Project to sphere
                edge_map[edge] = len(new_vertices)
                new_vertices.append(mid)
            return edge_map[edge]
        
        for face in faces:
            v0, v1, v2 = face
            
            # Get midpoints
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            
            # Create 4 new triangles
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20],
            ])
        
        return np.array(new_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int32)


def compute_mesh_properties(mesh: wp.Mesh) -> Dict:
    """Compute useful properties of a mesh.
    
    Args:
        mesh: Warp mesh to analyze.
        
    Returns:
        Dictionary with mesh properties.
    """
    vertices = mesh.points.numpy()
    faces = mesh.indices.numpy().reshape(-1, 3)
    
    # Basic stats
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    
    # Bounding box
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    bbox_size = max_corner - min_corner
    bbox_center = (min_corner + max_corner) / 2.0
    
    # Volume approximation (sum of tetrahedron volumes)
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        volume += np.abs(np.dot(v0, np.cross(v1, v2))) / 6.0
    
    # Surface area
    area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    
    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "bbox_min": min_corner.tolist(),
        "bbox_max": max_corner.tolist(),
        "bbox_size": bbox_size.tolist(),
        "bbox_center": bbox_center.tolist(),
        "volume": float(volume),
        "surface_area": float(area),
    }


def visualize_mesh_stats(agent_meshes: Dict[str, wp.Mesh]):
    """Print mesh statistics for all agents.
    
    Args:
        agent_meshes: Dictionary mapping agent_id to mesh.
    """
    print("=" * 60)
    print("Agent Mesh Statistics")
    print("=" * 60)
    
    for agent_id, mesh in agent_meshes.items():
        props = compute_mesh_properties(mesh)
        print(f"\n{agent_id}:")
        print(f"  Vertices: {props['num_vertices']:,}")
        print(f"  Faces: {props['num_faces']:,}")
        print(f"  Bbox size: [{props['bbox_size'][0]:.3f}, {props['bbox_size'][1]:.3f}, {props['bbox_size'][2]:.3f}] m")
        print(f"  Volume: {props['volume']:.4f} m³")
        print(f"  Surface area: {props['surface_area']:.4f} m²")
    
    print("=" * 60)