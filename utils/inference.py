from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision import transforms

from app import get_T_from_relative, load_model, sample_model
from ldm.models.diffusion.ddim import DDIMSampler
from utils.image import preprocess_image

ImageInputLike = Union[str, Path, Image.Image, np.ndarray]


def _to_pil_rgba(image: ImageInputLike) -> Image.Image:
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


def _tensor_from_pose(pose: Sequence[float]) -> torch.Tensor:
    return torch.tensor(list(pose), dtype=torch.float32)


def _resolve_generate_defaults(
    toss,
    *,
    pose_enc: str | None,
    h: int | None,
    w: int | None,
    use_ema_scope: bool | None,
) -> tuple[int, int, str, bool]:
    h = getattr(toss, "_default_h", 256) if h is None else h
    w = getattr(toss, "_default_w", 256) if w is None else w
    pose_enc = getattr(toss, "_default_pose_enc", "freq") if pose_enc is None else pose_enc
    if use_ema_scope is None:
        use_ema_scope = getattr(toss, "_default_use_ema_scope", True)
    return h, w, pose_enc, use_ema_scope


def _prepare_cond_im_on_toss(
    toss,
    image: ImageInputLike,
    h: int,
    w: int,
) -> torch.Tensor:
    device = toss.device
    cond_im_pil = _to_pil_rgba(image)
    cond_im = preprocess_image(cond_im_pil)
    cond_im = transforms.ToTensor()(cond_im).unsqueeze(0).to(device)
    return transforms.functional.resize(cond_im, [h, w])


def _sample_one_on_toss(
    toss,
    cond_im: torch.Tensor,
    T: torch.Tensor,
    *,
    prompt: str,
    h: int,
    w: int,
    precision: str,
    n_samples: int,
    use_ema_scope: bool,
    ddim_steps: int,
    ddim_eta: float,
    prompt_scale: float,
    img_scale: float,
    img_ucg: float,
) -> Image.Image:
    x_samples = sample_model(
        cond_im,
        toss.model,
        toss.sampler,
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


@torch.no_grad()
def generate_batch(
    toss,
    image: ImageInputLike,
    prompt: str = "",
    *,
    delta_pose_list: Optional[list[np.ndarray]] = None,
    dy_list: Optional[list[float]] = None,
    pose_enc: str | None = None,
    h: int | None = None,
    w: int | None = None,
    precision: str = "fp32",
    n_samples: int = 1,
    use_ema_scope: bool | None = None,
    ddim_steps: int = 100,
    ddim_eta: float = 1.0,
    prompt_scale: float = 5.0,
    img_scale: float = 3.0,
    img_ucg: float = 0.05,
) -> list[Image.Image]:
    """
    Batch novel-view generation for any toss-like object with ``model``,
    ``sampler``, ``device``, and optional ``_default_*`` attrs.

    Works with ``utils.inference.TossInference`` and legacy notebook
    ``TossInference`` instances.
    """
    if delta_pose_list is not None and dy_list is not None:
        raise ValueError("pass only one of delta_pose_list or dy_list")
    if delta_pose_list is None and dy_list is None:
        raise ValueError("pass delta_pose_list or dy_list")

    h, w, pose_enc, use_ema_scope = _resolve_generate_defaults(
        toss,
        pose_enc=pose_enc,
        h=h,
        w=w,
        use_ema_scope=use_ema_scope,
    )
    cond_im = _prepare_cond_im_on_toss(toss, image, h, w)
    sample_kwargs = dict(
        prompt=prompt,
        h=h,
        w=w,
        precision=precision,
        n_samples=n_samples,
        use_ema_scope=use_ema_scope,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        prompt_scale=prompt_scale,
        img_scale=img_scale,
        img_ucg=img_ucg,
    )

    outputs: list[Image.Image] = []
    if delta_pose_list is not None:
        for delta_pose in delta_pose_list:
            delta_pose = np.asarray(delta_pose, dtype=np.float32).reshape(3)
            T = _tensor_from_pose(delta_pose)
            outputs.append(
                _sample_one_on_toss(toss, cond_im, T, **sample_kwargs)
            )
        return outputs

    for yaw_deg in dy_list or []:
        T = _tensor_from_pose((0.0, math.radians(float(yaw_deg)), 0.0))
        outputs.append(_sample_one_on_toss(toss, cond_im, T, **sample_kwargs))
    return outputs


class TossInference:
    """Load TOSS once, then call ``generate`` with varying inputs."""

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
    ) -> None:
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

    def _prepare_cond_im(self, image: ImageInputLike, h: int, w: int) -> torch.Tensor:
        return _prepare_cond_im_on_toss(self, image, h, w)

    def _sample_one(
        self,
        cond_im: torch.Tensor,
        T: torch.Tensor,
        *,
        prompt: str,
        h: int,
        w: int,
        precision: str,
        n_samples: int,
        use_ema_scope: bool,
        ddim_steps: int,
        ddim_eta: float,
        prompt_scale: float,
        img_scale: float,
        img_ucg: float,
    ) -> Image.Image:
        return _sample_one_on_toss(
            self,
            cond_im,
            T,
            prompt=prompt,
            h=h,
            w=w,
            precision=precision,
            n_samples=n_samples,
            use_ema_scope=use_ema_scope,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            prompt_scale=prompt_scale,
            img_scale=img_scale,
            img_ucg=img_ucg,
        )

    @torch.no_grad()
    def generate(
        self,
        image: ImageInputLike,
        prompt: str = "",
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        *,
        delta_pose_list: Optional[list[np.ndarray]] = None,
        dy_list: Optional[list[float]] = None,
        pose_enc: str | None = None,
        h: int | None = None,
        w: int | None = None,
        precision: str = "fp32",
        n_samples: int = 1,
        use_ema_scope: bool | None = None,
        ddim_steps: int = 100,
        ddim_eta: float = 1.0,
        prompt_scale: float = 5.0,
        img_scale: float = 3.0,
        img_ucg: float = 0.05,
    ) -> Union[Image.Image, list[Image.Image]]:
        """
        Novel view synthesis for one image or a batch of poses.

        Single pose: pass ``dx``, ``dy``, ``dz`` (degrees / distance via
        ``get_T_from_relative``).

        Batch reconstruction poses: pass ``delta_pose_list`` of shape-(3,)
        arrays in radians ``[pitch, yaw, distance]`` (training format).

        Batch yaw-only poses: pass ``dy_list`` of yaw degrees; uses
        ``T = [0, rad(dy), 0]``.
        """
        if delta_pose_list is not None:
            return generate_batch(
                self,
                image,
                prompt=prompt,
                delta_pose_list=delta_pose_list,
                pose_enc=pose_enc,
                h=h,
                w=w,
                precision=precision,
                n_samples=n_samples,
                use_ema_scope=use_ema_scope,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                prompt_scale=prompt_scale,
                img_scale=img_scale,
                img_ucg=img_ucg,
            )

        if dy_list is not None:
            return generate_batch(
                self,
                image,
                prompt=prompt,
                dy_list=dy_list,
                pose_enc=pose_enc,
                h=h,
                w=w,
                precision=precision,
                n_samples=n_samples,
                use_ema_scope=use_ema_scope,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                prompt_scale=prompt_scale,
                img_scale=img_scale,
                img_ucg=img_ucg,
            )

        h, w, pose_enc, use_ema_scope = _resolve_generate_defaults(
            self,
            pose_enc=pose_enc,
            h=h,
            w=w,
            use_ema_scope=use_ema_scope,
        )
        cond_im = self._prepare_cond_im(image, h, w)
        sample_kwargs = dict(
            prompt=prompt,
            h=h,
            w=w,
            precision=precision,
            n_samples=n_samples,
            use_ema_scope=use_ema_scope,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            prompt_scale=prompt_scale,
            img_scale=img_scale,
            img_ucg=img_ucg,
        )

        T = get_T_from_relative(dx, dy, dz, pose_enc)
        return self._sample_one(cond_im, T, **sample_kwargs)
