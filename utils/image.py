from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor]
MaskInput = Union[Image.Image, np.ndarray, torch.Tensor]


def to_numpy_rgb(x: ImageInput) -> np.ndarray:
    """Convert PIL / numpy / torch input to (H, W, 3) float32 in [0, 1]."""
    if isinstance(x, Image.Image):
        arr = np.asarray(x.convert("RGB"), dtype=np.float32) / 255.0
    elif torch.is_tensor(x):
        arr = x.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def normalize_mask(
    mask: MaskInput,
    ref_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Convert mask to (H, W) float32 in [0, 1]."""
    if isinstance(mask, Image.Image):
        mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    elif torch.is_tensor(mask):
        mask_arr = mask.detach().cpu().float().numpy()
    else:
        mask_arr = np.asarray(mask, dtype=np.float32)

    mask_arr = np.squeeze(mask_arr)

    if mask_arr.max() > 1.0:
        mask_arr = mask_arr / 255.0

    if ref_shape is not None and mask_arr.shape != tuple(ref_shape):
        raise ValueError(
            f"mask 크기가 이미지와 다릅니다: "
            f"mask={mask_arr.shape}, image={tuple(ref_shape)}"
        )

    return mask_arr.astype(np.float32)


def preprocess_image(
    input_im: Image.Image,
    fg_mask: Optional[Image.Image] = None,
    size: int = 256,
) -> np.ndarray:
    """
    Resize and composite onto white background.

    Returns (H, W, 3) float32 array in [0, 1].
    """
    input_im = input_im.resize([size, size], Image.Resampling.LANCZOS)
    rgb = np.asarray(input_im.convert("RGB"), dtype=np.float32) / 255.0

    alpha = None
    if fg_mask is not None:
        fg_mask = fg_mask.resize([size, size], Image.Resampling.LANCZOS)
        alpha = np.asarray(fg_mask, dtype=np.float32) / 255.0
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha = alpha[..., None]
    elif input_im.mode == "RGBA":
        rgba = np.asarray(input_im, dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3:4]

    if alpha is not None:
        white_im = np.ones_like(rgb)
        rgb = alpha * rgb + (1.0 - alpha) * white_im

    return rgb
