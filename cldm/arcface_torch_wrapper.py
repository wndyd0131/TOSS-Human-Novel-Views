"""
Frozen ArcFace feature backbone for training (differentiable w.r.t. input images only).

**Design choice (pick-backbone):** We vendor a single IResNet implementation
(`cldm.arcface_iresnet`, copied from InsightFace `recognition/arcface_torch`) and
load public ``.pth`` / ``.pt`` weights here. We intentionally do **not** depend on
the `insightface` pip package or ONNXRuntime inference paths, which are often
non-differentiable in training setups.

Checkpoint compatibility:
- Unwraps top-level ``state_dict`` if present (common in training scripts).
- Strips ``module.`` (DataParallel / DDP) and ``backbone.`` prefixes when every
  key uses the latter (partial model exports).
- Uses ``strict=False`` so margin-head-only keys in a full ArcFace checkpoint
  are ignored while the IResNet body still loads.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cldm.arcface_iresnet import IResNet, iresnet100, iresnet50

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ArcFace spatial + normalization (InsightFace arcface_torch-compatible)
# ---------------------------------------------------------------------------
#
# **Channel order:** RGB everywhere for this repo. The VAE / Portrait4D stack uses
# RGB; public ``arcface_torch`` ImageFolder and MX RecordIO decoders also feed RGB.
# Do **not** apply a BGR<->RGB flip unless your tensors are explicitly OpenCV BGR.
#
# **Value range:** ``preprocess_arcface_input`` expects floats in ``[0, 1]`` per channel
# (same as ``torchvision.transforms.ToTensor`` after PIL RGB).
#
# **Normalize:** Matches ``recognition/arcface_torch/dataset.py``:
# ``transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])``, i.e.
# ``(x - 0.5) / 0.5`` which equals ``2*x - 1`` and ``(x * 255 - 127.5) / 127.5``
# on the same [0,1] tensor. (Some docs cite ``/128``; the released training recipe
# uses std ``127.5`` / ``0.5`` on [0,1], which lands on this same map.)
#
# **Spatial (112×112):** Default pipeline for portrait data is ``cover_center``
# (scale until both sides are at least 112, then center-crop). ``resize`` is a
# cheaper baseline that squashes to 112×112 (aspect distortion). For detector-based
# alignment later, use ``face_bbox`` (pixel xyxy on the *same* H×W as ``rgb``).
# ---------------------------------------------------------------------------

ARCFACE_IMAGE_SIZE: int = 112

ArcfaceSpatialMode = Literal["resize", "cover_center", "none"]


def normalize_arcface_torch_rgb(rgb_01: torch.Tensor) -> torch.Tensor:
    """Linear normalize for ``iresnet*`` checkpoints trained with arcface_torch defaults.

    Parameters
    ----------
    rgb_01:
        ``[B, 3, H, W]`` (or ``[3, H, W]``) in ``[0, 1]``, **RGB** order.
    """
    if rgb_01.dim() == 3:
        rgb_01 = rgb_01.unsqueeze(0)
    x = rgb_01.clamp(0.0, 1.0)
    return (x - 0.5) / 0.5


def _resize_cover_center_crop(rgb_01: torch.Tensor, size: int) -> torch.Tensor:
    """Scale so H and W are both >= ``size``, then center-crop to ``size×size``."""
    b, _, h, w = rgb_01.shape
    if h == size and w == size:
        return rgb_01
    scale = max(size / h, size / w)
    new_h = max(int(round(h * scale)), size)
    new_w = max(int(round(w * scale)), size)
    y = F.interpolate(rgb_01, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return y[:, :, top : top + size, left : left + size]


def _spatial_to_112(
    rgb_01: torch.Tensor,
    *,
    mode: ArcfaceSpatialMode,
    size: int,
) -> torch.Tensor:
    if rgb_01.dim() == 3:
        rgb_01 = rgb_01.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False
    if mode == "none":
        out = rgb_01
    elif mode == "resize":
        out = F.interpolate(rgb_01, size=(size, size), mode="bilinear", align_corners=False)
    elif mode == "cover_center":
        out = _resize_cover_center_crop(rgb_01, size)
    else:
        raise ValueError(f"Unknown ArcFace spatial mode: {mode!r}")
    if out.shape[-2] != size or out.shape[-1] != size:
        raise ValueError(
            f"ArcFace spatial mode {mode!r} produced {tuple(out.shape)}, expected *×{size}×{size}"
        )
    return out.squeeze(0) if squeezed else out


def crop_face_bbox_xyxy_then_resize(
    rgb_01: torch.Tensor,
    bbox_xyxy: torch.Tensor,
    *,
    size: int = ARCFACE_IMAGE_SIZE,
    margin: float = 0.2,
) -> torch.Tensor:
    """Crop a square (with margin) from pixel xyxy boxes then resize to ``size``.

    Use when you have a **shared** box from GT or source (detector run offline / frozen):
    apply the **same** ``bbox_xyxy`` to both pred and target tensors so geometry matches.

    Parameters
    ----------
    rgb_01:
        ``[B, 3, H, W]`` in ``[0, 1]``, RGB.
    bbox_xyxy:
        ``[B, 4]`` as ``x1, y1, x2, y2`` in pixel coordinates on that ``H×W``.
    margin:
        Fractional expansion of the box edge length (square is taken from max(w, h)).

    Boxes are fixed w.r.t. gradients; grads flow into ``rgb_01`` via the ROI slice and resize.
    """
    if rgb_01.dim() != 4:
        raise ValueError("rgb_01 must be [B, 3, H, W]")
    if bbox_xyxy.dim() != 2 or bbox_xyxy.shape[1] != 4:
        raise ValueError("bbox_xyxy must be [B, 4] (x1,y1,x2,y2)")

    b, _, h, w = rgb_01.shape
    out = []
    for i in range(b):
        x1, y1, x2, y2 = bbox_xyxy[i].tolist()
        cw = max(x2 - x1, 1e-6)
        ch = max(y2 - y1, 1e-6)
        side = max(cw, ch)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half = side * (1.0 + margin) * 0.5
        nx1 = max(0.0, cx - half)
        ny1 = max(0.0, cy - half)
        nx2 = min(float(w), cx + half)
        ny2 = min(float(h), cy + half)
        x1i = max(0, min(w - 1, int(math.floor(nx1))))
        y1i = max(0, min(h - 1, int(math.floor(ny1))))
        x2i = max(x1i + 1, min(w, int(math.ceil(nx2))))
        y2i = max(y1i + 1, min(h, int(math.ceil(ny2))))
        patch = rgb_01[i : i + 1, :, y1i:y2i, x1i:x2i]
        patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
        out.append(patch)
    return torch.cat(out, dim=0)


def preprocess_arcface_input(
    rgb_01: torch.Tensor,
    *,
    spatial_mode: ArcfaceSpatialMode = "cover_center",
    face_bbox_xyxy: Optional[torch.Tensor] = None,
    bbox_margin: float = 0.2,
) -> torch.Tensor:
    """Resize/crop to 112×112, then apply arcface_torch RGB normalization.

    Returns ``[B, 3, 112, 112]`` (or ``[3,112,112]`` if input was 3D) ready for ``IResNet``.
    """
    batched_3d = rgb_01.dim() == 3
    x = rgb_01.unsqueeze(0) if batched_3d else rgb_01

    if face_bbox_xyxy is not None:
        x = crop_face_bbox_xyxy_then_resize(x, face_bbox_xyxy, size=ARCFACE_IMAGE_SIZE, margin=bbox_margin)
    else:
        x = _spatial_to_112(x, mode=spatial_mode, size=ARCFACE_IMAGE_SIZE)

    x = normalize_arcface_torch_rgb(x)
    return x.squeeze(0) if batched_3d else x


ArchName = str


def _unwrap_checkpoint(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
        # Some exports use 'model' or weights only at top level
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unexpected checkpoint type: {type(raw)}")


def normalize_arcface_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map common training/export key prefixes onto :class:`IResNet` parameter names."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = value

    # If all keys share a single prefix like "backbone." or "model.backbone.", strip it.
    def try_strip_prefix(prefix: str) -> Optional[Dict[str, torch.Tensor]]:
        pref = prefix
        stripped: Dict[str, torch.Tensor] = {}
        for k, v in out.items():
            if not k.startswith(pref):
                return None
            stripped[k[len(pref) :]] = v
        return stripped

    for pattern in ("backbone.", "model.backbone.", "encoder."):
        candidate = try_strip_prefix(pattern)
        if candidate is not None and candidate:
            out = candidate
            break

    return out


def _match_architecture_from_keys(keys: set) -> Optional[ArchName]:
    """Infer iresnet depth from layer4 block naming when metadata is missing."""
    layer4_re = re.compile(r"^layer4\.(\d+)\.")
    idxs = set()
    for k in keys:
        m = layer4_re.match(k)
        if m:
            idxs.add(int(m.group(1)))
    if not idxs:
        return None
    n_blocks = max(idxs) + 1
    if n_blocks == 3:
        return "iresnet50"
    if n_blocks == 30:
        return "iresnet100"
    return None


def build_iresnet(
    arch: ArchName = "iresnet100",
    *,
    dropout: float = 0.0,
    num_features: int = 512,
    fp16: bool = False,
    **kwargs: Any,
) -> IResNet:
    arch = arch.lower().strip()
    if arch == "iresnet100":
        return iresnet100(dropout=dropout, num_features=num_features, fp16=fp16, **kwargs)
    if arch == "iresnet50":
        return iresnet50(dropout=dropout, num_features=num_features, fp16=fp16, **kwargs)
    raise ValueError(f"Unsupported arch '{arch}'. Supported: iresnet50, iresnet100.")


def load_pretrained_into_iresnet(
    model: IResNet,
    checkpoint: Union[str, os.PathLike, Dict[str, Any]],
    *,
    strict: bool = False,
    arch_hint: Optional[ArchName] = None,
) -> tuple[IResNet, Dict[str, Any]]:
    """
    Load weights into an existing IResNet. Returns (model, load_report).

    ``load_report`` contains ``missing_keys`` and ``unexpected_keys`` from PyTorch.
    """
    if isinstance(checkpoint, (str, os.PathLike)):
        path = os.fspath(checkpoint)
        raw = torch.load(path, map_location="cpu")
    else:
        raw = checkpoint

    state_dict = normalize_arcface_state_dict(_unwrap_checkpoint(raw))
    # Drop known non-backbone heads from full ArcFace training checkpoints
    drop_substrings = ("fc7", "margin", "arcface", "head", "cosface", "combined_margin")
    filtered = {k: v for k, v in state_dict.items() if not any(s in k.lower() for s in drop_substrings)}

    incompatible = model.load_state_dict(filtered, strict=strict)
    report: Dict[str, Any] = {
        "missing_keys": list(getattr(incompatible, "missing_keys", []) or ()),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", []) or ()),
        "arch_hint": arch_hint,
    }
    if report["missing_keys"]:
        logger.warning(
            "ArcFace checkpoint missing %d keys (first few): %s",
            len(report["missing_keys"]),
            report["missing_keys"][:8],
        )
    if report["unexpected_keys"]:
        logger.info(
            "ArcFace checkpoint had %d unused keys (first few): %s",
            len(report["unexpected_keys"]),
            report["unexpected_keys"][:8],
        )
    return model, report


def create_frozen_arcface_backbone(
    checkpoint_path: Union[str, os.PathLike],
    *,
    arch: ArchName = "iresnet100",
    dropout: float = 0.0,
    num_features: int = 512,
    fp16_autocast: bool = False,
) -> IResNet:
    """
    Build IResNet, load ``checkpoint_path``, set eval + no grad.

    Recommended public weights (same architecture family): Glint360K / MS1MV2
    iresnet100 checkpoints from the InsightFace arcface_torch recipe (``backbone.pth`` style).
    """
    path = os.fspath(checkpoint_path)
    # Infer arch from filename / checkpoint keys when default might be wrong
    raw = torch.load(path, map_location="cpu")
    state_dict = normalize_arcface_state_dict(_unwrap_checkpoint(raw))
    keys = set(state_dict.keys())
    inferred = _match_architecture_from_keys(keys)
    if inferred and inferred != arch.lower():
        logger.info("Overriding arch %s -> %s based on checkpoint layer4 depth.", arch, inferred)
        arch = inferred

    model = build_iresnet(arch, dropout=dropout, num_features=num_features, fp16=fp16_autocast)
    load_pretrained_into_iresnet(model, raw, strict=False, arch_hint=arch)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
