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


class TestOcclusionShapeHandling(unittest.TestCase):
    """Tests for shape handling fixes in occlusion detection."""

    def test_target_bbox_size_shape_1d(self):
        """Test that 1D target_bbox_size (3,) is handled correctly."""
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable: {exc}")

        N, C, T, K = 4, 2, 1, 1
        camera_pos = torch.zeros((N, C, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(N, C, 4)
        test_points_world = torch.zeros((N, T, K, 3), dtype=torch.float32)
        test_points_world[..., 2] = 10.0  # Target at z=10
        target_positions = torch.zeros((N, T, 3), dtype=torch.float32)
        target_positions[..., 2] = 10.0

        # 1D shape - this was the bug case
        target_bbox_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        # Should not raise any errors
        vis_mask, vis_ratio, self_mask = occlusion.batch_check_occlusion_fully_batched(
            camera_pos=camera_pos,
            camera_quat=camera_quat,
            test_points_world=test_points_world,
            target_positions=target_positions,
            target_bbox_size=target_bbox_size,
            agent_poses={},
            agent_meshes={},
            static_mesh=None,
            max_distance=100.0,
            visibility_threshold=0.5,
            current_agent_ids=["drone_0", "drone_1"],
            enable_self_occlusion=False,
        )

        self.assertEqual(vis_mask.shape, (N, C, T))
        self.assertEqual(vis_ratio.shape, (N, C, T))
        self.assertEqual(self_mask.shape, (N, C, T))
        # All should be visible (no occlusion)
        self.assertTrue(vis_mask.all())

    def test_target_bbox_size_shape_2d_T3(self):
        """Test that 2D target_bbox_size (T, 3) is handled correctly."""
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable: {exc}")

        N, C, T, K = 4, 2, 3, 1
        camera_pos = torch.zeros((N, C, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(N, C, 4)
        test_points_world = torch.zeros((N, T, K, 3), dtype=torch.float32)
        test_points_world[..., 2] = 10.0
        target_positions = torch.zeros((N, T, 3), dtype=torch.float32)
        target_positions[..., 2] = 10.0

        # 2D shape (T, 3) - one bbox size per target
        target_bbox_size = torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0]], dtype=torch.float32)

        vis_mask, vis_ratio, self_mask = occlusion.batch_check_occlusion_fully_batched(
            camera_pos=camera_pos,
            camera_quat=camera_quat,
            test_points_world=test_points_world,
            target_positions=target_positions,
            target_bbox_size=target_bbox_size,
            agent_poses={},
            agent_meshes={},
            static_mesh=None,
            max_distance=100.0,
            visibility_threshold=0.5,
            current_agent_ids=["drone_0", "drone_1"],
            enable_self_occlusion=False,
        )

        self.assertEqual(vis_mask.shape, (N, C, T))
        self.assertTrue(vis_mask.all())

    def test_target_bbox_size_shape_3d_NT3(self):
        """Test that 3D target_bbox_size (N, T, 3) is handled correctly."""
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable: {exc}")

        N, C, T, K = 4, 2, 2, 1
        camera_pos = torch.zeros((N, C, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(N, C, 4)
        test_points_world = torch.zeros((N, T, K, 3), dtype=torch.float32)
        test_points_world[..., 2] = 10.0
        target_positions = torch.zeros((N, T, 3), dtype=torch.float32)
        target_positions[..., 2] = 10.0

        # 3D shape (N, T, 3) - already expanded
        target_bbox_size = torch.ones((N, T, 3), dtype=torch.float32)

        vis_mask, vis_ratio, self_mask = occlusion.batch_check_occlusion_fully_batched(
            camera_pos=camera_pos,
            camera_quat=camera_quat,
            test_points_world=test_points_world,
            target_positions=target_positions,
            target_bbox_size=target_bbox_size,
            agent_poses={},
            agent_meshes={},
            static_mesh=None,
            max_distance=100.0,
            visibility_threshold=0.5,
            current_agent_ids=["drone_0", "drone_1"],
            enable_self_occlusion=False,
        )

        self.assertEqual(vis_mask.shape, (N, C, T))
        self.assertTrue(vis_mask.all())

    def test_multi_env_agent_occlusion_shapes(self):
        """Test that shapes are correct with multiple environments and agent occlusion."""
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable: {exc}")

        N, C, T, K = 8, 3, 1, 1
        camera_pos = torch.zeros((N, C, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(N, C, 4)
        test_points_world = torch.zeros((N, T, K, 3), dtype=torch.float32)
        test_points_world[..., 2] = 10.0
        target_positions = torch.zeros((N, T, 3), dtype=torch.float32)
        target_positions[..., 2] = 10.0

        # 1D shape with multi-env, multi-camera
        target_bbox_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        agent_poses = {
            "drone_0": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4)),
            "drone_1": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4)),
            "drone_2": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4)),
        }
        agent_meshes = {"drone_0": object(), "drone_1": object(), "drone_2": object()}

        # Mock raycast to return no hits
        def fake_raycast_mesh(ray_starts, ray_dirs, mesh, max_dist, return_distance=False, return_normal=False):
            return torch.full_like(ray_starts, float("inf")), None, None, None

        with mock.patch.object(occlusion, "raycast_mesh", side_effect=fake_raycast_mesh):
            vis_mask, vis_ratio, self_mask = occlusion.batch_check_occlusion_fully_batched(
                camera_pos=camera_pos,
                camera_quat=camera_quat,
                test_points_world=test_points_world,
                target_positions=target_positions,
                target_bbox_size=target_bbox_size,
                agent_poses=agent_poses,
                agent_meshes=agent_meshes,
                static_mesh=None,
                max_distance=100.0,
                visibility_threshold=0.5,
                current_agent_ids=["drone_0", "drone_1", "drone_2"],
                enable_self_occlusion=True,
                self_occlusion_min_hit_distance_m=0.05,
            )

        # Verify shapes
        self.assertEqual(vis_mask.shape, (N, C, T))
        self.assertEqual(vis_ratio.shape, (N, C, T))
        self.assertEqual(self_mask.shape, (N, C, T))
        # All visible since no hits
        self.assertTrue(vis_mask.all())


class TestRaycastCallCount(unittest.TestCase):
    """Test that raycasting is called the expected number of times."""

    def test_single_raycast_per_agent_all_cameras(self):
        """Verify that agent mesh raycasting uses O(1) call per agent (all cameras batched)."""
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable: {exc}")

        N, C, T, K = 16, 3, 1, 9  # 3 cameras
        camera_pos = torch.zeros((N, C, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(N, C, 4).clone()
        test_points_world = torch.zeros((N, T, K, 3), dtype=torch.float32)
        test_points_world[..., 2] = 10.0
        target_positions = torch.zeros((N, T, 3), dtype=torch.float32)
        target_positions[..., 2] = 10.0
        target_bbox_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        # 3 agents
        agent_poses = {
            "drone_0": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()),
            "drone_1": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()),
            "drone_2": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()),
        }
        agent_meshes = {"drone_0": object(), "drone_1": object(), "drone_2": object()}

        raycast_call_count = {"count": 0}

        def counting_fake_raycast_mesh(ray_starts, ray_dirs, mesh, max_dist, return_distance=False, return_normal=False):
            raycast_call_count["count"] += 1
            return torch.full_like(ray_starts, float("inf")), None, None, None

        with mock.patch.object(occlusion, "raycast_mesh", side_effect=counting_fake_raycast_mesh):
            occlusion.batch_check_occlusion_fully_batched(
                camera_pos=camera_pos,
                camera_quat=camera_quat,
                test_points_world=test_points_world,
                target_positions=target_positions,
                target_bbox_size=target_bbox_size,
                agent_poses=agent_poses,
                agent_meshes=agent_meshes,
                static_mesh=None,
                max_distance=100.0,
                visibility_threshold=0.5,
                current_agent_ids=["drone_0", "drone_1", "drone_2"],
                enable_self_occlusion=True,
                self_occlusion_min_hit_distance_m=0.05,
            )

        # With the optimization, we should have exactly num_agents raycast calls (3)
        # Old implementation would have C * num_agents = 9 calls
        # No static mesh, so no static raycast call
        num_agents = len(agent_meshes)
        self.assertEqual(
            raycast_call_count["count"],
            num_agents,
            f"Expected {num_agents} raycast calls (O(1) per agent), got {raycast_call_count['count']}"
        )

    def test_raycast_count_with_static_mesh(self):
        """Verify raycast call count with both static mesh and agent meshes."""
        try:
            occlusion = load_module_from_path(
                "bbox_raycaster_v2_occlusion_fully_batched",
                V2_ROOT / "utils" / "occlusion_fully_batched.py",
            )
        except Exception as exc:
            self.skipTest(f"occlusion module unavailable: {exc}")

        N, C, T, K = 16, 3, 1, 9
        camera_pos = torch.zeros((N, C, 3), dtype=torch.float32)
        camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(N, C, 4).clone()
        test_points_world = torch.zeros((N, T, K, 3), dtype=torch.float32)
        test_points_world[..., 2] = 10.0
        target_positions = torch.zeros((N, T, 3), dtype=torch.float32)
        target_positions[..., 2] = 10.0
        target_bbox_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        # 3 agents
        agent_poses = {
            "drone_0": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()),
            "drone_1": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()),
            "drone_2": (torch.zeros((N, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()),
        }
        agent_meshes = {"drone_0": object(), "drone_1": object(), "drone_2": object()}
        static_mesh = object()

        raycast_call_count = {"count": 0}

        def counting_fake_raycast_mesh(ray_starts, ray_dirs, mesh, max_dist, return_distance=False, return_normal=False):
            raycast_call_count["count"] += 1
            return torch.full_like(ray_starts, float("inf")), None, None, None

        with mock.patch.object(occlusion, "raycast_mesh", side_effect=counting_fake_raycast_mesh):
            occlusion.batch_check_occlusion_fully_batched(
                camera_pos=camera_pos,
                camera_quat=camera_quat,
                test_points_world=test_points_world,
                target_positions=target_positions,
                target_bbox_size=target_bbox_size,
                agent_poses=agent_poses,
                agent_meshes=agent_meshes,
                static_mesh=static_mesh,
                max_distance=100.0,
                visibility_threshold=0.5,
                current_agent_ids=["drone_0", "drone_1", "drone_2"],
                enable_self_occlusion=True,
                self_occlusion_min_hit_distance_m=0.05,
            )

        # Should be: 1 (static) + 3 (agents) = 4 calls
        # Old implementation would have: 1 (static) + 9 (C * agents) = 10 calls
        num_agents = len(agent_meshes)
        expected_calls = 1 + num_agents  # 1 static + num_agents
        self.assertEqual(
            raycast_call_count["count"],
            expected_calls,
            f"Expected {expected_calls} raycast calls (1 static + {num_agents} agents), got {raycast_call_count['count']}"
        )


if __name__ == "__main__":
    unittest.main()
