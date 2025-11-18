from typing import Tuple

import torch

class DetectionIndicator:
    def __init__(self, num_envs: int, device: torch.device):    
        self.num_envs = num_envs
        self.device = device
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
        self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    def get_line_colors(self, detected: torch.Tensor) -> torch.Tensor:
        line_colors_yellow = torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
        line_colors_green = torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
        line_colors_red = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)

        line_colors = torch.where(
            detected,
            line_colors_green,
            line_colors_yellow
        )

        return line_colors

    def draw_indicator(self, start_points: torch.Tensor, end_points: torch.Tensor, detected: torch.Tensor):
        line_colors = self.get_line_colors(detected)
        line_thickness = [1.0] * self.num_envs

        self.draw_interface.draw_lines(
            start_points.tolist(),
            end_points.tolist(),
            line_colors.tolist(),
            line_thickness,
        )