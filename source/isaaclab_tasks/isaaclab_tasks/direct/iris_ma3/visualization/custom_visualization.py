import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .camera_frustum import CameraFrustum, create_camera_cfg_tensor, project_2d_to_3d
from .detection_indicator import DetectionIndicator

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
        marker_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/TriCovMarkers",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.5,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(3/255, 252/255, 248/255),
                                ),
                            ),
                        }
                    )
        self.tri_cov_visualizer = {
            agent_id: VisualizationMarkers(marker_cfg) for agent_id in possible_agents
        }
