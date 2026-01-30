import torch

import omni.usd
from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .camera_frustum import CameraFrustum, create_camera_cfg_tensor, project_2d_to_3d
from .detection_indicator import DetectionIndicator


def is_prim_ready_for_visualization(prim_path: str) -> bool:
    """Check if a USD prim is ready for visualization.

    This helps avoid GPU crashes from Isaac Sim 4.5's timing bug where
    VisualizationMarkers prims aren't fully populated before the Vulkan
    renderer tries to access them.

    Args:
        prim_path: Path to the USD prim to check.

    Returns:
        True if the prim exists and appears ready, False otherwise.
    """
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        # Check if the prim has any children (indicates it's been populated)
        children = prim.GetChildren()
        return len(children) > 0
    except Exception:
        return False


class CustomVisualization:
    def __init__(self, num_envs: int, possible_agents: list[str], device: torch.device | str):
        self.num_envs = num_envs
        self.possible_agents = possible_agents
        self.device = device

        self.camera_frustum = {
            agent_id: CameraFrustum() for agent_id in possible_agents
        }
        self.detection_indicator = {
            agent_id: DetectionIndicator(num_envs, device) for agent_id in possible_agents
        }
        # CRITICAL: Each agent must have a unique prim path to avoid GPU crashes
        # from multiple VisualizationMarkers writing to the same USD prim
        self.tri_cov_visualizer = {}
        self._tri_cov_prim_paths = {}
        self._tri_cov_ready = {}  # Track which visualizers are ready

        for agent_id in possible_agents:
            prim_path = f"/Visuals/TriCovMarkers_{agent_id}"
            self._tri_cov_prim_paths[agent_id] = prim_path
            self._tri_cov_ready[agent_id] = False

            marker_cfg = VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(3/255, 252/255, 248/255),
                        ),
                    ),
                }
            )
            self.tri_cov_visualizer[agent_id] = VisualizationMarkers(marker_cfg)

    def is_tri_cov_visualizer_ready(self, agent_id: str) -> bool:
        """Check if the triangulation covariance visualizer is ready for the given agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            True if the visualizer is ready to use, False otherwise.
        """
        # Once marked ready, stay ready (avoid repeated checks)
        if self._tri_cov_ready.get(agent_id, False):
            return True

        prim_path = self._tri_cov_prim_paths.get(agent_id)
        if prim_path and is_prim_ready_for_visualization(prim_path):
            self._tri_cov_ready[agent_id] = True
            return True

        return False

    def visualize_tri_cov(self, agent_id: str, translations: torch.Tensor, scales: torch.Tensor) -> bool:
        """Safely visualize triangulation covariance markers.

        Args:
            agent_id: The agent identifier.
            translations: Marker positions (N, 3).
            scales: Marker scales (N, 3).

        Returns:
            True if visualization was performed, False if skipped due to prim not ready.
        """
        if not self.is_tri_cov_visualizer_ready(agent_id):
            return False

        try:
            self.tri_cov_visualizer[agent_id].visualize(
                translations=translations,
                scales=scales,
            )
            return True
        except Exception:
            # If visualization fails, mark as not ready and try again later
            self._tri_cov_ready[agent_id] = False
            return False
