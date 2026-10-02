from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torchvision import transforms

from app import get_T_from_relative, load_model
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


def _resolve_pl_module(toss):
    """Return the module that exposes TOSS sampling APIs (encode/decode/conditioning).

    ``TossInference.model`` is already the Lightning module, while a
    ``LightningModule`` passed directly (e.g. ``TossLoraModule`` during training)
    must be used as-is — ``toss.model`` is only the inner ``DiffusionWrapper``.
    """
    if hasattr(toss, "get_unconditional_conditioning"):
        return toss
    return toss.model


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


def _tensors_to_pils(x_samples: torch.Tensor) -> list[Image.Image]:
    outputs: list[Image.Image] = []
    for sample in x_samples:
        out = sample.cpu().numpy()
        out = 255.0 * rearrange(out, "c h w -> h w c")
        outputs.append(Image.fromarray(out.astype(np.uint8)))
    return outputs


@torch.no_grad()
def _sample_multiview_on_toss(
    toss,
    cond_im: torch.Tensor,
    T_list: list[torch.Tensor],
    *,
    prompt: str,
    h: int,
    w: int,
    precision: str,
    use_ema_scope: bool,
    ddim_steps: int,
    ddim_eta: float,
    prompt_scale: float,
    img_scale: float,
    shared_noise: bool = True,
    x_T: torch.Tensor | None = None,
) -> tuple[list[Image.Image], torch.Tensor | None]:
    """Batched DDIM sampling with optional shared initial noise across poses."""
    if not T_list:
        return [], x_T

    pl_module = _resolve_pl_module(toss)
    sampler = toss.sampler
    n = len(T_list)
    precision_scope = autocast if precision == "autocast" else nullcontext
    ema_scope = pl_module.ema_scope if use_ema_scope else nullcontext

    with precision_scope("cuda"):
        with ema_scope("Sampling..."):
            in_concat = pl_module.encode_first_stage(cond_im * 2 - 1).mode().detach()
            in_concat = in_concat.repeat(n, 1, 1, 1)
            c_cat = cond_im.repeat(n, 1, 1, 1)

            c = pl_module.get_learned_conditioning([prompt] * n)
            uc_cross = pl_module.get_unconditional_conditioning(n)

            delta_pose = torch.stack(
                [t.to(pl_module.device) for t in T_list]
            ).float()

            base = {"delta_pose": delta_pose}
            cond = {
                **base,
                "c_crossattn": [c],
                "c_concat": [c_cat],
                "in_concat": [in_concat],
            }
            uc2 = {
                **base,
                "c_crossattn": [uc_cross],
                "c_concat": [c_cat],
                "in_concat": [in_concat],
            }
            uc = {
                **base,
                "c_crossattn": [uc_cross],
                "c_concat": [c_cat],
                "in_concat": [in_concat * 0],
            }

            if x_T is not None:
                if x_T.shape[0] == 1:
                    noise = x_T.repeat(n, 1, 1, 1)
                elif x_T.shape[0] == n:
                    noise = x_T
                else:
                    raise ValueError(
                        f"x_T batch {x_T.shape[0]} does not match pose count {n}"
                    )
                used_x_T = x_T if x_T.shape[0] == 1 else x_T[:1].clone()
            elif shared_noise:
                used_x_T = torch.randn((1, *in_concat.shape[1:]), device=in_concat.device)
                noise = used_x_T.repeat(n, 1, 1, 1)
            else:
                used_x_T = None
                noise = torch.randn(in_concat.shape, device=in_concat.device)

            samples, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n,
                shape=[4, h // 8, w // 8],
                verbose=False,
                unconditional_guidance_scale=img_scale,
                unconditional_conditioning=uc,
                unconditional_guidance_scale2=prompt_scale,
                unconditional_conditioning2=uc2,
                eta=ddim_eta,
                x_T=noise,
            )
            x = pl_module.decode_first_stage(samples)
            x_samples = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0).cpu()

    return _tensors_to_pils(x_samples), used_x_T


def _build_pose_tensors(
    *,
    delta_pose_list: list[np.ndarray] | None,
    dy_list: list[float] | None,
    pose_enc: str,
) -> list[torch.Tensor]:
    if delta_pose_list is not None:
        return [
            _tensor_from_pose(np.asarray(delta_pose, dtype=np.float32).reshape(3))
            for delta_pose in delta_pose_list
        ]
    if dy_list is not None:
        return [
            get_T_from_relative(0.0, float(yaw_deg), 0.0, pose_enc)
            for yaw_deg in dy_list
        ]
    raise ValueError("pass delta_pose_list or dy_list")


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
    precision: str = "autocast",
    n_samples: int = 1,
    use_ema_scope: bool | None = None,
    ddim_steps: int = 30,
    ddim_eta: float = 0.0,
    prompt_scale: float = 1.0,
    img_scale: float = 3.0,
    img_ucg: float = 0.05,
    shared_noise: bool = True,
    chunk_size: int | None = None,
) -> list[Image.Image]:
    """
    Batch novel-view generation for any toss-like object with ``model``,
    ``sampler``, ``device``, and optional ``_default_*`` attrs.

    Uses batched DDIM with shared ``x_T`` by default (notebook / wandb vis style).
    """
    del n_samples, img_ucg  # kept for API compatibility

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
    T_list = _build_pose_tensors(
        delta_pose_list=delta_pose_list,
        dy_list=dy_list,
        pose_enc=pose_enc,
    )

    sample_kwargs = dict(
        prompt=prompt,
        h=h,
        w=w,
        precision=precision,
        use_ema_scope=use_ema_scope,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        prompt_scale=prompt_scale,
        img_scale=img_scale,
        shared_noise=shared_noise,
    )

    chunk = chunk_size or len(T_list)
    outputs: list[Image.Image] = []
    shared_x_T: torch.Tensor | None = None

    for start in range(0, len(T_list), chunk):
        chunk_T = T_list[start : start + chunk]
        chunk_out, shared_x_T = _sample_multiview_on_toss(
            toss,
            cond_im,
            chunk_T,
            x_T=shared_x_T if shared_noise else None,
            **sample_kwargs,
        )
        outputs.extend(chunk_out)

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
        precision: str = "autocast",
        n_samples: int = 1,
        use_ema_scope: bool | None = None,
        ddim_steps: int = 30,
        ddim_eta: float = 0.0,
        prompt_scale: float = 1.0,
        img_scale: float = 3.0,
        img_ucg: float = 0.05,
        shared_noise: bool = True,
        chunk_size: int | None = None,
    ) -> Union[Image.Image, list[Image.Image]]:
        """
        Novel view synthesis for one image or a batch of poses.

        Single pose: pass ``dx``, ``dy``, ``dz`` (degrees / distance via
        ``get_T_from_relative``).

        Batch reconstruction poses: pass ``delta_pose_list`` of shape-(3,)
        arrays in radians ``[pitch, yaw, distance]`` (training format).

        Batch yaw-only poses: pass ``dy_list`` of yaw degrees.
        """
        if delta_pose_list is not None or dy_list is not None:
            return generate_batch(
                self,
                image,
                prompt=prompt,
                delta_pose_list=delta_pose_list,
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
                shared_noise=shared_noise,
                chunk_size=chunk_size,
            )

        h, w, pose_enc, use_ema_scope = _resolve_generate_defaults(
            self,
            pose_enc=pose_enc,
            h=h,
            w=w,
            use_ema_scope=use_ema_scope,
        )
        cond_im = _prepare_cond_im_on_toss(self, image, h, w)
        T = get_T_from_relative(dx, dy, dz, pose_enc)
        images, _ = _sample_multiview_on_toss(
            self,
            cond_im,
            [T],
            prompt=prompt,
            h=h,
            w=w,
            precision=precision,
            use_ema_scope=use_ema_scope,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            prompt_scale=prompt_scale,
            img_scale=img_scale,
            shared_noise=shared_noise,
        )
        return images[0]
