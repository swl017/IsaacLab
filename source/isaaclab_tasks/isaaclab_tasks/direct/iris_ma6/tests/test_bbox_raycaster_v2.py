import importlib
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]
V2_ROOT = REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "iris_ma6" / "bbox_raycaster_v2"
PKG_ROOT = REPO_ROOT / "source" / "isaaclab_tasks"

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


def load_module_from_path(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestBBoxRayCasterV2Static(unittest.TestCase):
    def test_v2_has_no_runtime_import_back_to_v5_bbox_raycaster(self):
        for path in V2_ROOT.rglob("*.py"):
            text = path.read_text()
            self.assertNotIn("iris_ma5.bbox_raycaster", text, msg=f"unexpected V5 runtime import in {path}")


class TestBBoxRayCasterV2TensorLogic(unittest.TestCase):
    def test_zoom_increases_projected_bbox_size(self):
        try:
            projection = load_module_from_path(
                "bbox_raycaster_v2_projection",
                V2_ROOT / "utils" / "projection.py",
            )
        except Exception as exc:
            self.skipTest(f"projection helpers unavailable in current environment: {exc}")

        points_camera = torch.tensor(
            [[[[[-1.0, -1.0, 10.0],
                [1.0, -1.0, 10.0],
                [-1.0, 1.0, 10.0],
                [1.0, 1.0, 10.0],
                [-1.0, -1.0, 12.0],
                [1.0, -1.0, 12.0],
                [-1.0, 1.0, 12.0],
                [1.0, 1.0, 12.0]]]]],
            dtype=torch.float32,
        )  # (1, 1, 1, 8, 3)
        valid_corners = torch.ones((1, 1, 1, 8), dtype=torch.bool)

        k_base = torch.tensor([[[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
        k_zoom = torch.tensor([[[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)

        pixels_base, _, _ = projection.batch_project_to_image_plane(points_camera, k_base.unsqueeze(1))
        pixels_zoom, _, _ = projection.batch_project_to_image_plane(points_camera, k_zoom.unsqueeze(1))

        base_x = pixels_base[0, 0, 0, :, 0]
        base_y = pixels_base[0, 0, 0, :, 1]
        zoom_x = pixels_zoom[0, 0, 0, :, 0]
        zoom_y = pixels_zoom[0, 0, 0, :, 1]

        base_width = (base_x.max() - base_x.min()).item()
        base_height = (base_y.max() - base_y.min()).item()
        zoom_width = (zoom_x.max() - zoom_x.min()).item()
        zoom_height = (zoom_y.max() - zoom_y.min()).item()

        self.assertGreater(zoom_width, base_width)
        self.assertGreater(zoom_height, base_height)

    def test_inter_target_occlusion_hard_and_soft(self):
        inter_target = load_module_from_path(
            "bbox_raycaster_v2_inter_target_occlusion",
            V2_ROOT / "utils" / "inter_target_occlusion.py",
        )

        bboxes_xyxy = torch.tensor(
            [[[
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 24.0, 24.0],
            ]]],
            dtype=torch.float32,
        )
        bbox_empty = torch.tensor([[[False, False, False]]])
        bbox_confidence = torch.ones((1, 1, 3), dtype=torch.float32)
        camera_pos_w = torch.zeros((1, 1, 3), dtype=torch.float32)
        target_pos_w = torch.tensor([[[2.0, 0.0, 5.0], [2.0, 0.0, 8.0], [2.0, 0.0, 6.0]]], dtype=torch.float32)

        updated_empty, updated_confidence, occluded = inter_target.apply_inter_target_occlusion(
            bboxes_xyxy=bboxes_xyxy,
            bbox_empty=bbox_empty,
            bbox_confidence=bbox_confidence,
            camera_pos_w=camera_pos_w,
            target_pos_w=target_pos_w,
            soft_iou_threshold=0.05,
            hard_iou_threshold=0.3,
            depth_margin_m=1.0,
        )

        self.assertFalse(updated_empty[0, 0, 0].item())
        self.assertTrue(updated_empty[0, 0, 1].item())
        self.assertTrue(occluded[0, 0, 1].item())
        self.assertEqual(updated_confidence[0, 0, 1].item(), 0.0)

        soft_boxes = torch.tensor(
            [[[
                [0.0, 0.0, 10.0, 10.0],
                [7.0, 0.0, 17.0, 10.0],
            ]]],
            dtype=torch.float32,
        )
        soft_empty, soft_conf, _ = inter_target.apply_inter_target_occlusion(
            bboxes_xyxy=soft_boxes,
            bbox_empty=torch.tensor([[[False, False]]]),
            bbox_confidence=torch.ones((1, 1, 2), dtype=torch.float32),
            camera_pos_w=camera_pos_w,
            target_pos_w=torch.tensor([[[2.0, 0.0, 5.0], [2.0, 0.0, 8.0]]], dtype=torch.float32),
            soft_iou_threshold=0.05,
            hard_iou_threshold=0.3,
            depth_margin_m=1.0,
        )
        self.assertFalse(soft_empty[0, 0, 1].item())
        self.assertLess(soft_conf[0, 0, 1].item(), 1.0)


class TestBBoxRayCasterV2SelfOcclusion(unittest.TestCase):
    def test_self_occlusion_and_min_hit_distance(self):
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable in current environment: {exc}")

        camera_pos = torch.zeros((1, 1, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
        test_points_world = torch.tensor([[[[0.0, 0.0, 10.0]]]], dtype=torch.float32)
        target_positions = torch.tensor([[[0.0, 0.0, 10.0]]], dtype=torch.float32)
        target_bbox_size = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
        agent_poses = {"drone_0": (torch.zeros((1, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))}
        agent_meshes = {"drone_0": object()}

        far_hit = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        near_hit = torch.tensor([[0.0, 0.0, 0.01]], dtype=torch.float32)

        def fake_raycast_mesh(ray_starts, ray_dirs, mesh, max_dist, return_distance=False, return_normal=False):
            if mesh is agent_meshes["drone_0"]:
                return far_hit.repeat(ray_starts.shape[0], 1), None, None, None
            return torch.full_like(ray_starts, float("inf")), None, None, None

        with mock.patch.object(occlusion, "raycast_mesh", side_effect=fake_raycast_mesh):
            vis_mask, _, self_mask = occlusion.batch_check_occlusion_fully_batched(
                camera_pos=camera_pos,
                camera_quat=camera_quat,
                test_points_world=test_points_world,
                target_positions=target_positions,
                target_bbox_size=target_bbox_size,
                agent_poses=agent_poses,
                agent_meshes=agent_meshes,
                static_mesh=None,
                max_distance=100.0,
                visibility_threshold=1.0,
                current_agent_ids=["drone_0"],
                enable_self_occlusion=False,
            )
            self.assertTrue(vis_mask[0, 0, 0].item())
            self.assertFalse(self_mask[0, 0, 0].item())

        with mock.patch.object(occlusion, "raycast_mesh", side_effect=fake_raycast_mesh):
            vis_mask, _, self_mask = occlusion.batch_check_occlusion_fully_batched(
                camera_pos=camera_pos,
                camera_quat=camera_quat,
                test_points_world=test_points_world,
                target_positions=target_positions,
                target_bbox_size=target_bbox_size,
                agent_poses=agent_poses,
                agent_meshes=agent_meshes,
                static_mesh=None,
                max_distance=100.0,
                visibility_threshold=1.0,
                current_agent_ids=["drone_0"],
                enable_self_occlusion=True,
                self_occlusion_min_hit_distance_m=0.05,
            )
            self.assertFalse(vis_mask[0, 0, 0].item())
            self.assertTrue(self_mask[0, 0, 0].item())

        def fake_raycast_mesh_near(ray_starts, ray_dirs, mesh, max_dist, return_distance=False, return_normal=False):
            if mesh is agent_meshes["drone_0"]:
                return near_hit.repeat(ray_starts.shape[0], 1), None, None, None
            return torch.full_like(ray_starts, float("inf")), None, None, None

        with mock.patch.object(occlusion, "raycast_mesh", side_effect=fake_raycast_mesh_near):
            vis_mask, _, self_mask = occlusion.batch_check_occlusion_fully_batched(
                camera_pos=camera_pos,
                camera_quat=camera_quat,
                test_points_world=test_points_world,
                target_positions=target_positions,
                target_bbox_size=target_bbox_size,
                agent_poses=agent_poses,
                agent_meshes=agent_meshes,
                static_mesh=None,
                max_distance=100.0,
                visibility_threshold=1.0,
                current_agent_ids=["drone_0"],
                enable_self_occlusion=True,
                self_occlusion_min_hit_distance_m=0.05,
            )
            self.assertTrue(vis_mask[0, 0, 0].item())
            self.assertFalse(self_mask[0, 0, 0].item())


if __name__ == "__main__":
    unittest.main()
