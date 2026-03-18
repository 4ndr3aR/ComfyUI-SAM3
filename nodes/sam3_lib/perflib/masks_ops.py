# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch

from ..logger import get_logger
logger = get_logger(__name__)

def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        # Sanity check based on callsite for replacement
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        # Based on torchvision masks_to_boxes
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        N, H, W = masks.shape
        device = masks.device
        y = torch.arange(H, device=device).view(1, H)
        x = torch.arange(W, device=device).view(1, W)

        masks_with_obj = masks != 0  # N, H, W
        masks_with_obj_x = masks_with_obj.amax(
            dim=1
        )  # N, H (which columns have objects)
        masks_with_obj_y = masks_with_obj.amax(dim=2)  # N, W (which rows have objects)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        bounding_boxes_0 = torch.amin(
            (masks_without_obj_x * W) + (masks_with_obj_x * x), dim=1
        )
        bounding_boxes_1 = torch.amin(
            (masks_without_obj_y * H) + (masks_with_obj_y * y), dim=1
        )
        bounding_boxes_2 = torch.amax(masks_with_obj_x * x, dim=1)
        bounding_boxes_3 = torch.amax(masks_with_obj_y * y, dim=1)

        bounding_boxes = torch.stack(
            [bounding_boxes_0, bounding_boxes_1, bounding_boxes_2, bounding_boxes_3],
            dim=1,
        ).to(dtype=torch.float)
        assert bounding_boxes.shape == (N, 4)
        assert bounding_boxes.device == masks.device
        assert bounding_boxes.dtype == torch.float
        return bounding_boxes


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor, debug=False) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.
    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks
    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape

    # Flatten masks: (N, 1, H*W) and (1, M, H*W)
    pred_flat = pred_masks.view(N, 1, H * W)
    gt_flat = gt_masks.view(1, M, H * W)
    if debug:
        #print(f'pred_flat: {pred_flat.shape} - gt_flat: {gt_flat.shape}')  # shape: (N, 1, H*W) and (1, M, H*W)
        logger.info(f'pred_flat: {pred_flat.shape} - gt_flat: {gt_flat.shape}')  # shape: (N, 1, H*W) and (1, M, H*W)

    # Compute intersection and union: (N, M)
    intersection = (pred_flat & gt_flat).sum(dim=2).float()
    union = (pred_flat | gt_flat).sum(dim=2).float()
    ious = intersection / union.clamp(min=1)
    return ious  # shape: (N, M)

def mask_iou_chunked(pred_masks: torch.Tensor, gt_masks: torch.Tensor, chunk_size=32, debug=True) -> torch.Tensor:
	"""
	Chunked drop-in replacement for mask_iou.
	Peak extra VRAM: chunk_size × M × H*W instead of N × M × H*W.
	"""
	if debug:
		logger.info(f'mask_iou_chunked() running with chunk_size={chunk_size}')
	assert pred_masks.dtype == gt_masks.dtype == torch.bool, f"Expected boolean tensors, got {pred_masks.dtype} / {gt_masks.dtype}"
	N, H, W = pred_masks.shape
	M       = gt_masks.shape[0]
	HW      = H * W

	pred_flat = pred_masks.view(N, HW)   # (N, H*W) – no broadcast dim yet
	gt_flat   = gt_masks.view(M, HW)     # (M, H*W)

	ious = torch.zeros(N, M, device=pred_masks.device, dtype=torch.float32)
	if debug:
		logger.info(f'pred_flat: {pred_flat.shape} - gt_flat: {gt_flat.shape} - ious: {ious.shape}')

	for row_start in range(0, N, chunk_size):
		row_end   = min(row_start + chunk_size, N)
		p_chunk   = pred_flat[row_start:row_end].unsqueeze(1)  # (cs, 1, H*W)
		g_chunk   = gt_flat.unsqueeze(0)                       # ( 1, M, H*W)

		inter = (p_chunk & g_chunk).sum(dim=2).float()   # (cs, M)
		union = (p_chunk | g_chunk).sum(dim=2).float()   # (cs, M)
		ious[row_start:row_end] = inter / union.clamp(min=1)

		if debug:
			logger.info(f'p_chunk: {p_chunk.shape} - g_chunk: {g_chunk.shape} - inter: {inter.shape} - union: {union.shape} - ious: {ious.shape}')

		# Free the two large intermediates immediately
		del inter, union, p_chunk, g_chunk

	del pred_flat, gt_flat

	return ious

