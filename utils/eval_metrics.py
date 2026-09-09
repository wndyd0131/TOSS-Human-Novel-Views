from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from utils.image import ImageInput, MaskInput, normalize_mask, to_numpy_rgb

DeviceLike = Union[str, torch.device]


def compute_psnr(
    pred: ImageInput,
    target: ImageInput,
    mask: Optional[MaskInput] = None,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute PSNR between pred and target. Higher is better."""
    pred_arr = to_numpy_rgb(pred)
    target_arr = to_numpy_rgb(target)

    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"pred와 target shape이 다릅니다: "
            f"pred={pred_arr.shape}, target={target_arr.shape}"
        )

    squared_error = (pred_arr - target_arr) ** 2

    if mask is not None:
        mask_arr = normalize_mask(mask, ref_shape=pred_arr.shape[:2])
        valid = mask_arr > 0.5
        if not np.any(valid):
            return float("nan")
        mse = float(np.mean(squared_error[valid]))
    else:
        mse = float(np.mean(squared_error))

    if mse <= eps:
        return float("inf")

    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_lpips(
    pred: ImageInput,
    target: ImageInput,
    mask: Optional[MaskInput] = None,
    *,
    lpips_model: Any,
    device: Optional[DeviceLike] = None,
) -> float:
    """Compute LPIPS between pred and target. Lower is better."""

    def to_tensor_nchw_minus1_1(x: ImageInput) -> torch.Tensor:
        arr = to_numpy_rgb(x)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    device = torch.device(device) if device is not None else next(lpips_model.parameters()).device
    pred_t = to_tensor_nchw_minus1_1(pred).to(device)
    target_t = to_tensor_nchw_minus1_1(target).to(device)

    if pred_t.shape != target_t.shape:
        raise ValueError(
            f"pred와 target shape이 다릅니다: "
            f"pred={pred_t.shape}, target={target_t.shape}"
        )

    with torch.no_grad():
        dist_map = lpips_model(pred_t, target_t)

    if mask is None:
        return float(dist_map.mean().item())

    mask_arr = normalize_mask(mask, ref_shape=pred_t.shape[-2:])
    mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0).unsqueeze(0).to(device)

    if mask_t.shape[-2:] != dist_map.shape[-2:]:
        mask_t = F.interpolate(
            mask_t,
            size=dist_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    valid = mask_t > 0.5
    if not valid.any():
        return float("nan")

    return float(dist_map[valid].mean().item())


def compute_identity_similarity(
    pred: ImageInput,
    target: ImageInput,
    *,
    backbone: torch.nn.Module,
    spatial_mode: str = "cover_center",
    device: Optional[DeviceLike] = None,
) -> float:
    """Compute ArcFace cosine similarity. Higher is better."""
    from cldm.arcface_torch_wrapper import preprocess_arcface_input

    def to_tensor_nchw_01(x: ImageInput) -> torch.Tensor:
        arr = to_numpy_rgb(x)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    device = torch.device(device) if device is not None else next(backbone.parameters()).device
    pred_t = to_tensor_nchw_01(pred).to(device)
    target_t = to_tensor_nchw_01(target).to(device)

    if pred_t.shape != target_t.shape:
        raise ValueError(
            f"pred와 target shape이 다릅니다: "
            f"pred={pred_t.shape}, target={target_t.shape}"
        )

    with torch.no_grad():
        pred_arc = preprocess_arcface_input(pred_t, spatial_mode=spatial_mode)
        target_arc = preprocess_arcface_input(target_t, spatial_mode=spatial_mode)
        emb_pred = F.normalize(backbone(pred_arc), dim=-1)
        emb_target = F.normalize(backbone(target_arc), dim=-1)

    return float((emb_pred * emb_target).sum(dim=-1).item())


class ImageMetricsEvaluator:
    """Lazy-load LPIPS / ArcFace and compute selected image metrics."""

    def __init__(
        self,
        device: DeviceLike,
        *,
        arcface_ckpt_path: Optional[str] = None,
        arcface_spatial_mode: str = "cover_center",
        lpips_net: str = "vgg",
    ) -> None:
        self.device = torch.device(device)
        self.arcface_ckpt_path = arcface_ckpt_path
        self.arcface_spatial_mode = arcface_spatial_mode
        self.lpips_net = lpips_net
        self._lpips_model: Any = None
        self._arcface_backbone: Optional[torch.nn.Module] = None

    def _get_lpips_model(self) -> Any:
        if self._lpips_model is None:
            import lpips

            self._lpips_model = (
                lpips.LPIPS(net=self.lpips_net, spatial=True)
                .to(self.device)
                .eval()
            )
        return self._lpips_model

    def _get_arcface_backbone(self) -> torch.nn.Module:
        if self._arcface_backbone is None:
            if not self.arcface_ckpt_path:
                raise ValueError("identity metric requires arcface_ckpt_path")
            from cldm.arcface_torch_wrapper import create_frozen_arcface_backbone

            self._arcface_backbone = (
                create_frozen_arcface_backbone(self.arcface_ckpt_path)
                .to(self.device)
                .eval()
            )
        return self._arcface_backbone

    def compute(
        self,
        pred: ImageInput,
        target: ImageInput,
        fg_mask: Optional[MaskInput] = None,
        *,
        psnr: bool = True,
        fg_psnr: bool = True,
        lpips: bool = True,
        fg_lpips: bool = True,
        identity: bool = True,
    ) -> dict[str, float]:
        results: dict[str, float] = {}

        if psnr:
            results["psnr"] = compute_psnr(pred, target)

        if fg_psnr and fg_mask is not None:
            results["fg_psnr"] = compute_psnr(pred, target, mask=fg_mask)

        if lpips:
            results["lpips"] = compute_lpips(
                pred,
                target,
                lpips_model=self._get_lpips_model(),
                device=self.device,
            )

        if fg_lpips and fg_mask is not None:
            results["fg_lpips"] = compute_lpips(
                pred,
                target,
                mask=fg_mask,
                lpips_model=self._get_lpips_model(),
                device=self.device,
            )

        if identity:
            results["identity"] = compute_identity_similarity(
                pred,
                target,
                backbone=self._get_arcface_backbone(),
                spatial_mode=self.arcface_spatial_mode,
                device=self.device,
            )

        return results


def compute_metrics(
    pred: ImageInput,
    target: ImageInput,
    fg_mask: Optional[MaskInput] = None,
    *,
    evaluator: ImageMetricsEvaluator,
    **metric_flags: bool,
) -> dict[str, float]:
    """Shortcut for ImageMetricsEvaluator.compute()."""
    return evaluator.compute(pred, target, fg_mask=fg_mask, **metric_flags)
