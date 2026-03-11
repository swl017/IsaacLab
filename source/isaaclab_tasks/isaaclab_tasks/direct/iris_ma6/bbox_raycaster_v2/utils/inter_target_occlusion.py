"""Image-space inter-target occlusion helpers for bbox raycaster V2."""

from __future__ import annotations

import torch


def compute_iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU between two tensors of xyxy boxes with shape (..., 4)."""
    ax1, ay1, ax2, ay2 = boxes_a.unbind(dim=-1)
    bx1, by1, bx2, by2 = boxes_b.unbind(dim=-1)

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)

    inter_w = torch.clamp(ix2 - ix1, min=0.0)
    inter_h = torch.clamp(iy2 - iy1, min=0.0)
    inter_area = inter_w * inter_h

    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = torch.clamp(area_a + area_b - inter_area, min=eps)
    return inter_area / union


def apply_inter_target_occlusion(
    bboxes_xyxy: torch.Tensor,
    bbox_empty: torch.Tensor,
    bbox_confidence: torch.Tensor,
    camera_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    soft_iou_threshold: float,
    hard_iou_threshold: float,
    depth_margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply depth-aware image-space occlusion across targets.

    Args:
        bboxes_xyxy: Shape (N, C, T, 4)
        bbox_empty: Shape (N, C, T), True means empty
        bbox_confidence: Shape (N, C, T)
        camera_pos_w: Shape (N, C, 3)
        target_pos_w: Shape (N, T, 3)
    """
    _, C, T, _ = bboxes_xyxy.shape
    updated_empty = bbox_empty.clone()
    updated_confidence = bbox_confidence.clone()
    occluded = torch.zeros_like(bbox_empty)

    if T < 2:
        return updated_empty, updated_confidence, occluded

    target_pos_exp = target_pos_w.unsqueeze(1).expand(-1, C, -1, -1)
    camera_pos_exp = camera_pos_w.unsqueeze(2).expand(-1, -1, T, -1)
    depths = torch.linalg.norm(target_pos_exp - camera_pos_exp, dim=-1)

    for left_idx in range(T):
        for right_idx in range(left_idx + 1, T):
            boxes_left = bboxes_xyxy[:, :, left_idx, :]
            boxes_right = bboxes_xyxy[:, :, right_idx, :]
            iou = compute_iou_xyxy(boxes_left, boxes_right)

            empty_left = updated_empty[:, :, left_idx]
            empty_right = updated_empty[:, :, right_idx]
            pair_active = (~empty_left) & (~empty_right)

            depth_left = depths[:, :, left_idx]
            depth_right = depths[:, :, right_idx]
            left_near = pair_active & (depth_left + depth_margin_m < depth_right)
            right_near = pair_active & (depth_right + depth_margin_m < depth_left)

            hard = iou > hard_iou_threshold
            soft = (iou > soft_iou_threshold) & (~hard)

            left_occludes_right = left_near & hard
            right_occludes_left = right_near & hard

            if left_occludes_right.any():
                updated_empty[:, :, right_idx] |= left_occludes_right
                occluded[:, :, right_idx] |= left_occludes_right
                updated_confidence[:, :, right_idx] = torch.where(
                    left_occludes_right,
                    torch.zeros_like(updated_confidence[:, :, right_idx]),
                    updated_confidence[:, :, right_idx],
                )

            if right_occludes_left.any():
                updated_empty[:, :, left_idx] |= right_occludes_left
                occluded[:, :, left_idx] |= right_occludes_left
                updated_confidence[:, :, left_idx] = torch.where(
                    right_occludes_left,
                    torch.zeros_like(updated_confidence[:, :, left_idx]),
                    updated_confidence[:, :, left_idx],
                )

            soft_left_occludes_right = left_near & soft & (~updated_empty[:, :, right_idx])
            soft_right_occludes_left = right_near & soft & (~updated_empty[:, :, left_idx])

            if soft_left_occludes_right.any():
                updated_confidence[:, :, right_idx] = torch.where(
                    soft_left_occludes_right,
                    updated_confidence[:, :, right_idx] * (1.0 - 0.5 * iou),
                    updated_confidence[:, :, right_idx],
                )

            if soft_right_occludes_left.any():
                updated_confidence[:, :, left_idx] = torch.where(
                    soft_right_occludes_left,
                    updated_confidence[:, :, left_idx] * (1.0 - 0.5 * iou),
                    updated_confidence[:, :, left_idx],
                )

    return updated_empty, updated_confidence.clamp(0.0, 1.0), occluded
