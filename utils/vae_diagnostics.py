from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from utils.eval_metrics import compute_psnr
from utils.image import ImageInput, preprocess_image, to_numpy_rgb
from utils.inference import generate_batch, _resolve_pl_module
from utils.pose import select_horizontal_views


@dataclass
class VAERoundtripResult:
    original: Image.Image
    reconstruction: Image.Image
    psnr_full: float
    psnr_fg: float
    label: str


@dataclass
class ColorStats:
    view_idx: int
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]
    n_pixels: int


def load_subject_view(
    subject_root: str | Path,
    view_idx: int,
) -> tuple[Image.Image, Image.Image]:
    """Load RGBA frame and alpha mask for one subject view."""
    subject_root = Path(subject_root)
    img = Image.open(subject_root / f"{view_idx:05d}.png").convert("RGBA")
    mask = Image.open(subject_root / f"alpha_maps/{view_idx:05d}.png").convert("L")
    return img, mask


def preprocessed_pil(img: Image.Image, mask: Image.Image, size: int = 256) -> Image.Image:
    """Match batch eval / inference white-background preprocessing."""
    arr = preprocess_image(img, fg_mask=mask, size=size)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def _rgb01_to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def _tensor01_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return _rgb01_to_pil(arr)


@torch.no_grad()
def vae_roundtrip_tensor(
    toss,
    rgb_01: torch.Tensor,
) -> torch.Tensor:
    """
    Encode/decode RGB tensors without UNet sampling.

    ``rgb_01``: ``[B, 3, H, W]`` in ``[0, 1]``.
    Returns reconstructed tensor in ``[0, 1]``.
    """
    pl_module = _resolve_pl_module(toss)
    x = rgb_01 * 2.0 - 1.0
    posterior = pl_module.encode_first_stage(x)
    # Match training/inference: latent is scaled before decode (see get_first_stage_encoding).
    z = pl_module.scale_factor * posterior.mode()
    recon = pl_module.decode_first_stage(z)
    return torch.clamp((recon + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def vae_roundtrip_image(
    toss,
    image: ImageInput,
    *,
    mask: Image.Image | None = None,
    label: str = "image",
    size: int = 256,
) -> VAERoundtripResult:
    """Run VAE roundtrip on one preprocessed RGB image."""
    if isinstance(image, Image.Image):
        pil = preprocessed_pil(image, mask) if mask is not None else image.convert("RGB")
    else:
        pil = _rgb01_to_pil(to_numpy_rgb(image))

    device = toss.device
    h = getattr(toss, "_default_h", size)
    w = getattr(toss, "_default_w", size)
    rgb = preprocess_image(pil, size=size)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    if tensor.shape[-2:] != (h, w):
        tensor = torch.nn.functional.interpolate(tensor, size=(h, w), mode="bilinear")

    recon = vae_roundtrip_tensor(toss, tensor)[0]
    recon_pil = _tensor01_to_pil(recon)
    orig_rgb = to_numpy_rgb(pil)
    recon_rgb = to_numpy_rgb(recon_pil)

    mask_np = None
    if mask is not None:
        mask_np = np.asarray(mask.resize(pil.size, Image.Resampling.LANCZOS)) / 255.0

    return VAERoundtripResult(
        original=pil,
        reconstruction=recon_pil,
        psnr_full=compute_psnr(recon_rgb, orig_rgb),
        psnr_fg=compute_psnr(recon_rgb, orig_rgb, mask=mask_np),
        label=label,
    )


def compute_fg_color_stats(
    image: ImageInput,
    mask: ImageInput,
    *,
    view_idx: int = -1,
) -> ColorStats:
    """Mean/std RGB inside foreground mask."""
    rgb = to_numpy_rgb(image)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    if mask_arr.shape[:2] != rgb.shape[:2]:
        mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8))
        mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0

    valid = mask_arr > 0.5
    if not np.any(valid):
        return ColorStats(
            view_idx=view_idx,
            mean_rgb=(float("nan"),) * 3,
            std_rgb=(float("nan"),) * 3,
            n_pixels=0,
        )

    fg = rgb[valid]
    mean = tuple(float(v) for v in fg.mean(axis=0))
    std = tuple(float(v) for v in fg.std(axis=0))
    return ColorStats(
        view_idx=view_idx,
        mean_rgb=mean,
        std_rgb=std,
        n_pixels=int(valid.sum()),
    )


def compare_subject_vae_roundtrip(
    toss,
    subject_root: str | Path,
    view_indices: Sequence[int],
    *,
    src_view_idx: int | None = None,
) -> list[VAERoundtripResult]:
    """VAE roundtrip on selected views for one subject."""
    subject_root = Path(subject_root)
    results: list[VAERoundtripResult] = []
    for view_idx in view_indices:
        img, mask = load_subject_view(subject_root, view_idx)
        tag = "src" if view_idx == src_view_idx else f"view_{view_idx:02d}"
        results.append(
            vae_roundtrip_image(
                toss,
                img,
                mask=mask,
                label=tag,
            )
        )
    return results


def compare_subject_color_stats(
    subject_root: str | Path,
    view_indices: Sequence[int],
    *,
    src_view_idx: int | None = None,
) -> list[ColorStats]:
    """Foreground RGB mean/std per view (no model)."""
    subject_root = Path(subject_root)
    stats: list[ColorStats] = []
    for view_idx in view_indices:
        img, mask = load_subject_view(subject_root, view_idx)
        pil = preprocessed_pil(img, mask)
        stats.append(
            compute_fg_color_stats(pil, mask, view_idx=view_idx)
        )
    if src_view_idx is not None:
        src = next((s for s in stats if s.view_idx == src_view_idx), None)
        if src is not None:
            for s in stats:
                delta = np.linalg.norm(
                    np.asarray(s.mean_rgb) - np.asarray(src.mean_rgb)
                )
                print(
                    f"view {s.view_idx:02d}: mean RGB={s.mean_rgb} | "
                    f"Δfrom src={delta:.4f}"
                )
    return stats


def sweep_img_scale(
    toss: Any,
    image: ImageInput,
    *,
    dy_deg: float = 0.0,
    img_scales: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0),
    ddim_steps: int = 30,
    seed: int = 40,
) -> list[tuple[float, Image.Image]]:
    """Generate one yaw with multiple CFG ``img_scale`` values."""
    outputs: list[tuple[float, Image.Image]] = []
    for scale in img_scales:
        if hasattr(toss, "set_seed"):
            toss.set_seed(seed)
        batch = generate_batch(
            toss,
            image,
            dy_list=[dy_deg],
            img_scale=float(scale),
            ddim_steps=ddim_steps,
            shared_noise=True,
        )
        outputs.append((float(scale), batch[0]))
    return outputs


def _label_pil(image: Image.Image, text: str, width: int = 256) -> Image.Image:
    canvas = Image.new("RGB", (width, image.height + 28), (255, 255, 255))
    if image.width != width:
        image = image.resize((width, width), Image.Resampling.LANCZOS)
    canvas.paste(image.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), text, fill=(0, 0, 0))
    return canvas


def make_comparison_strip(
    panels: Sequence[tuple[Image.Image, str]],
    *,
    panel_width: int = 256,
) -> Image.Image:
    """Horizontal strip: ``[(image, label), ...]``."""
    labeled = [_label_pil(img, label, width=panel_width) for img, label in panels]
    total_w = sum(img.width for img in labeled)
    max_h = max(img.height for img in labeled)
    strip = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for img in labeled:
        strip.paste(img, (x, 0))
        x += img.width
    return strip


def print_vae_roundtrip_summary(results: Sequence[VAERoundtripResult]) -> None:
    print("=== VAE roundtrip (encode → decode, no UNet) ===")
    for item in results:
        print(
            f"{item.label:>10}: PSNR full={item.psnr_full:.2f} dB | "
            f"FG={item.psnr_fg:.2f} dB"
        )
    print(
        "\nIf FG PSNR is already low (~25–30 dB) and eyes look blurry here, "
        "the bottleneck is mostly VAE/resolution — not LoRA loss choice."
    )


def run_quality_diagnostics(
    toss: Any,
    subject_root: str | Path,
    *,
    src_view_idx: int = 3,
    poses_path: str | Path | None = None,
    dy_deg: float = 0.0,
    img_scales: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    One-shot diagnostic bundle for Colab / local debugging.

    1. VAE roundtrip on source (+ horizontal ring views if ``poses.npy`` given)
    2. FG color stats across views
    3. ``img_scale`` CFG sweep at ``dy_deg``
    """
    subject_root = Path(subject_root)
    outputs: dict[str, Any] = {}

    src_img, src_mask = load_subject_view(subject_root, src_view_idx)
    src_pil = preprocessed_pil(src_img, src_mask)

    view_indices = [src_view_idx]
    if poses_path is None:
        poses_path = subject_root / "poses.npy"
    if Path(poses_path).exists():
        poses = np.load(poses_path)
        # ``select_horizontal_views`` excludes source; prepend it for src roundtrip panels.
        horiz_views = select_horizontal_views(poses, src_view_idx)
        view_indices = [src_view_idx] + horiz_views

    vae_results = compare_subject_vae_roundtrip(
        toss,
        subject_root,
        view_indices,
        src_view_idx=src_view_idx,
    )
    print_vae_roundtrip_summary(vae_results)
    outputs["vae_roundtrip"] = vae_results

    print("\n=== FG color stats (dataset, no model) ===")
    color_stats = compare_subject_color_stats(
        subject_root,
        view_indices,
        src_view_idx=src_view_idx,
    )
    outputs["color_stats"] = color_stats

    print(f"\n=== img_scale sweep at dy={dy_deg:+.1f}° ===")
    cfg_results = sweep_img_scale(
        toss,
        src_pil,
        dy_deg=dy_deg,
        img_scales=img_scales,
    )
    for scale, _ in cfg_results:
        print(f"img_scale={scale:.1f}")
    outputs["cfg_sweep"] = cfg_results

    src_result = next(r for r in vae_results if r.label == "src")
    vae_strip = make_comparison_strip(
        [
            (src_result.original, f"source (view {src_view_idx:02d})"),
            (src_result.reconstruction, "VAE recon (same view)"),
        ]
    )
    cfg_strip = make_comparison_strip(
        [(img, f"scale={scale:.1f}") for scale, img in cfg_results]
    )
    gen_at_zero = cfg_results[0][1] if cfg_results else None
    compare_strip = None
    if gen_at_zero is not None:
        compare_strip = make_comparison_strip(
            [
                (src_result.original, f"source (view {src_view_idx:02d})"),
                (src_result.reconstruction, "VAE recon (same view)"),
                (gen_at_zero, f"gen dy={dy_deg:+.0f}°"),
            ]
        )
    outputs["panels"] = {
        "vae_roundtrip": vae_strip,
        "cfg_sweep": cfg_strip,
        "source_vs_vae_vs_gen": compare_strip,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        vae_strip.save(save_dir / "vae_roundtrip.png")
        cfg_strip.save(save_dir / "cfg_sweep.png")
        if compare_strip is not None:
            compare_strip.save(save_dir / "source_vae_gen.png")
        outputs["saved_dir"] = save_dir
        print(f"\nSaved panels to {save_dir}")

    return outputs
