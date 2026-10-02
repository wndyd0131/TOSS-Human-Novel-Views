from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.image import MaskInput, normalize_mask, to_numpy_rgb
from utils.multiview_selection import estimate_flow, load_raft, warp_with_flow

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor]


def _to_tensor_nchw(image: ImageInput) -> torch.Tensor:
    arr = to_numpy_rgb(image)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _resize_mask_tensor(
    mask: MaskInput,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    mask_arr = normalize_mask(mask)
    mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0).unsqueeze(0)
    if mask_t.shape[-2:] != size:
        mask_t = F.interpolate(mask_t, size=size, mode="nearest")
    return mask_t.to(device)


@torch.no_grad()
def warp_cost_fb(
    left: torch.Tensor,
    right: torch.Tensor,
    raft: torch.nn.Module,
    fg_mask: torch.Tensor | None = None,
    *,
    fb_thresh: float = 1.0,
) -> torch.Tensor:
    """Masked L1 warp error with forward-backward occlusion check. Returns ``[B]``."""
    flow_lr = estimate_flow(right, left, raft)
    flow_rl = estimate_flow(left, right, raft)

    warped_rl_on_right, valid_lr = warp_with_flow(left, flow_lr)
    warped_lr_on_left, _ = warp_with_flow(right, flow_rl)
    fb_flow, _ = warp_with_flow(flow_rl, flow_lr)
    fb_err = (flow_lr + fb_flow).abs().sum(dim=1, keepdim=True)
    fb_valid = (fb_err < fb_thresh).float()

    err = (warped_rl_on_right - right).abs().mean(dim=1, keepdim=True)
    valid = valid_lr * fb_valid

    if fg_mask is not None:
        if fg_mask.shape[-2:] != valid.shape[-2:]:
            fg_mask = F.interpolate(fg_mask, size=valid.shape[-2:], mode="nearest")
        valid = valid * fg_mask

    denom = valid.flatten(1).sum(1).clamp(min=1.0)
    return (err * valid).flatten(1).sum(1) / denom


@torch.no_grad()
def compute_adjacent_warp_errors(
    images: Sequence[ImageInput],
    yaws: Sequence[float],
    raft: torch.nn.Module,
    *,
    device: torch.device | str,
    fg_masks: Sequence[MaskInput] | None = None,
    yaw_key_fn=None,
) -> list[tuple[float, float]]:
    """Return ``[(left_yaw_key, error), ...]`` for sorted adjacent yaw pairs."""
    if len(images) < 2:
        return []

    device = torch.device(device)
    order = sorted(range(len(yaws)), key=lambda idx: float(yaws[idx]))
    sorted_yaws = [float(yaws[idx]) for idx in order]
    sorted_images = [_to_tensor_nchw(images[idx]).to(device) for idx in order]

    fg_tensors: list[torch.Tensor | None]
    if fg_masks is not None:
        if len(fg_masks) != len(images):
            raise ValueError("fg_masks length must match images")
        sorted_fg = [fg_masks[idx] for idx in order]
        fg_tensors = [
            _resize_mask_tensor(mask, sorted_images[0].shape[-2:], device)
            for mask in sorted_fg
        ]
    else:
        fg_tensors = [None] * len(sorted_images)

    key_fn = yaw_key_fn or (lambda yaw: round(float(yaw), 1))
    results: list[tuple[float, float]] = []

    for idx in range(len(sorted_images) - 1):
        left = sorted_images[idx]
        right = sorted_images[idx + 1]
        left_mask = fg_tensors[idx]
        right_mask = fg_tensors[idx + 1]
        if left_mask is not None and right_mask is not None:
            fg_mask = left_mask * right_mask
        else:
            fg_mask = left_mask or right_mask

        error = float(warp_cost_fb(left, right, raft, fg_mask=fg_mask)[0].item())
        results.append((key_fn(sorted_yaws[idx]), error))

    return results


class MultiviewMetricsEvaluator:
    """Lazy-load RAFT and compute multiview consistency warp errors."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        raft_small: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.raft_small = raft_small
        self._raft: Any = None

    def _get_raft(self) -> torch.nn.Module:
        if self._raft is None:
            self._raft = load_raft(self.device, small=self.raft_small)
        return self._raft

    def compute_adjacent_warp_errors(
        self,
        images: Sequence[ImageInput],
        yaws: Sequence[float],
        *,
        fg_masks: Sequence[MaskInput] | None = None,
        yaw_key_fn=None,
    ) -> list[tuple[float, float]]:
        return compute_adjacent_warp_errors(
            images,
            yaws,
            self._get_raft(),
            device=self.device,
            fg_masks=fg_masks,
            yaw_key_fn=yaw_key_fn,
        )
