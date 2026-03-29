"""
TOSS inference API for notebooks — same sampling pipeline as app.py (no Gradio, no CLI).

Example:
    from inference import TossInference
    runner = TossInference(resume_path="ckpt/toss.ckpt")
    out = runner.generate("input.png", prompt="a red shoe", dy=-90)
    out = runner.generate(pil_image, prompt="", dx=0, dy=0, dz=0)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler

from app import get_T_from_relative, load_model, preprocess_image, sample_model

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _to_pil_rgba(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGBA")
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            raise ValueError("grayscale numpy arrays are not supported; use RGB/RGBA")
        if arr.shape[-1] == 4:
            return Image.fromarray(arr, mode="RGBA")
        if arr.shape[-1] == 3:
            return Image.fromarray(arr, mode="RGB").convert("RGBA")
        raise ValueError(f"expected HxWx3 or HxWx4 array, got shape {arr.shape}")
    raise TypeError(f"unsupported image type: {type(image)}")


class TossInference:
    """Load TOSS once, then call ``generate`` with varying inputs (notebook-friendly)."""

    def __init__(
        self,
        model_cfg: str | Path = "models/toss_vae.yaml",
        resume_path: str | Path = "ckpt/toss.ckpt",
        *,
        device: torch.device | str | None = None,
        gpu: int = 0,
        register_scheduler: bool = False,
        lr: float = 1e-4,
        seed: int = 40,
        sd_locked: bool = True,
        only_mid_control: bool = False,
        use_ema_scope: bool = True,
        pose_enc: str = "freq",
        h: int = 256,
        w: int = 256,
    ):
        """
        Args:
            model_cfg: YAML defining the model (e.g. ``models/toss_vae.yaml``).
            resume_path: Checkpoint with ``state_dict`` (e.g. ``ckpt/toss.ckpt``).
            device: Explicit device; if None, uses ``cuda:{gpu}`` when available.
            gpu: CUDA index when ``device`` is None.
            register_scheduler: Passed through to ``load_model`` (same as training flag).
            lr: Required by ``load_model``; not used during inference.
            seed: Fixed at init; call ``set_seed`` to change between runs.
            use_ema_scope / pose_enc / h / w: Defaults for ``generate`` (overridable per call).
        """
        seed_everything(seed, workers=True)
        if device is None:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        hparams = SimpleNamespace(
            resume_path=str(resume_path),
            register_scheduler=register_scheduler,
            lr=lr,
        )
        cfgs = OmegaConf.load(str(model_cfg))
        self.model = load_model(
            self.device, hparams, sd_locked, only_mid_control, cfgs
        )
        self.sampler = DDIMSampler(self.model)

        self._default_use_ema_scope = use_ema_scope
        self._default_pose_enc = pose_enc
        self._default_h = h
        self._default_w = w

    def set_seed(self, seed: int) -> None:
        """Call between ``generate`` runs for reproducible DDIM noise."""
        seed_everything(seed, workers=True)

    @torch.no_grad()
    def generate(
        self,
        image: ImageInput,
        prompt: str = "",
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        *,
        pose_enc: str | None = None,
        h: int | None = None,
        w: int | None = None,
        precision: str = "fp32",
        n_samples: int = 1,
        use_ema_scope: bool | None = None,
        ddim_steps: int = 75,
        ddim_eta: float = 1.0,
        prompt_scale: float = 5.0,
        img_scale: float = 3.0,
        img_ucg: float = 0.05,
    ) -> Image.Image:
        """
        Novel view for one image (same logic as ``app.generate_loop_views`` / ``sample_model``).

        Args:
            image: Path, ``PIL.Image``, or ``HxWx3`` / ``HxWx4`` numpy array (float or uint8).
            prompt: Text conditioning (empty string allowed).
            dx, dy, dz: Relative pose in degrees / distance (see ``app.get_T_from_relative``).
        """
        h = self._default_h if h is None else h
        w = self._default_w if w is None else w
        pose_enc = self._default_pose_enc if pose_enc is None else pose_enc
        if use_ema_scope is None:
            use_ema_scope = self._default_use_ema_scope

        cond_im_pil = _to_pil_rgba(image)
        cond_im = preprocess_image(cond_im_pil)
        cond_im = transforms.ToTensor()(cond_im).unsqueeze(0).to(self.device)
        cond_im = transforms.functional.resize(cond_im, [h, w])

        T = get_T_from_relative(dx, dy, dz, pose_enc)
        x_samples = sample_model(
            cond_im,
            self.model,
            self.sampler,
            precision=precision,
            h=h,
            w=w,
            ddim_steps=ddim_steps,
            n_samples=n_samples,
            prompt_scale=prompt_scale,
            img_scale=img_scale,
            ddim_eta=ddim_eta,
            T=T,
            use_ema_scope=use_ema_scope,
            prompt=prompt,
            img_ucg=img_ucg,
        )
        assert x_samples.shape[0] == 1
        out = x_samples[0].cpu().numpy()
        out = 255.0 * rearrange(out, "c h w -> h w c")
        return Image.fromarray(out.astype(np.uint8))
