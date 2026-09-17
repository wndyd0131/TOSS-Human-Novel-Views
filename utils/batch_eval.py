from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from utils.eval_metrics import ImageMetricsEvaluator
from utils.image import preprocess_image, resize_mask
from utils.inference import generate_batch
from utils.pose import (
    DEFAULT_RECON_VIEW_INDICES,
    compute_relative_pose,
    identity_dy_grid,
    select_recon_views,
)


@dataclass
class BatchEvalResults:
    per_yaw: dict[str, dict[float, list[float]]] = field(default_factory=dict)
    per_subject: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    overall: dict[str, list[float]] = field(default_factory=dict)


def _init_metric_buckets() -> dict[str, list[float]]:
    return {
        "psnr": [],
        "fg_psnr": [],
        "lpips": [],
        "fg_lpips": [],
        "identity": [],
    }


_RECON_METRIC_KEYS = ("psnr", "fg_psnr", "lpips", "fg_lpips")


def _reconstruction_enabled(metrics_config: dict[str, bool]) -> bool:
    return any(metrics_config.get(key, True) for key in _RECON_METRIC_KEYS)


def _sorted_yaw_keys(per_yaw: dict[str, dict[float, list[float]]]) -> list[float]:
    for key in ("lpips", "psnr", "identity", "fg_lpips", "fg_psnr"):
        bucket = per_yaw.get(key)
        if bucket:
            return sorted(bucket.keys())
    return []


def _yaw_key(yaw_deg: float) -> float:
    key = round(float(yaw_deg), 1)
    return 0.0 if key == 0 else key


def print_batch_eval_summary(
    results: BatchEvalResults,
    test_subjects: list[Any],
    *,
    recon_view_indices: Sequence[int] = DEFAULT_RECON_VIEW_INDICES,
    identity_num_yaws: int = 45,
    identity_yaw_min_deg: float = -22.0,
    identity_yaw_max_deg: float = 22.0,
) -> None:
    print("\n=== Eval configuration ===")
    print(f"Reconstruction views (GT): {list(recon_view_indices)} (source excluded)")
    print(
        f"Identity grid: {identity_num_yaws} yaws "
        f"from {identity_yaw_min_deg:+.1f}° to {identity_yaw_max_deg:+.1f}° vs source"
    )

    yaw_values = _sorted_yaw_keys(results.per_yaw)

    if results.overall.get("psnr"):
        print("\n=== Per-yaw mean PSNR (reconstruction) ===")
        for yaw in yaw_values:
            if not results.per_yaw["psnr"].get(yaw):
                continue
            print(
                f"yaw {yaw:+.1f}°: "
                f"Full={np.mean(results.per_yaw['psnr'][yaw]):.3f} dB | "
                f"FG={np.mean(results.per_yaw['fg_psnr'][yaw]):.3f} dB | "
                f"n={len(results.per_yaw['psnr'][yaw])}"
            )

        print("\n=== Per-subject mean PSNR (reconstruction) ===")
        for subject in test_subjects:
            subject = str(subject)
            sub = results.per_subject.get(subject, {})
            if sub.get("psnr"):
                print(
                    f"{subject}: "
                    f"Full={np.mean(sub['psnr']):.3f} dB | "
                    f"FG={np.mean(sub['fg_psnr']):.3f} dB"
                )

        print(
            f"\n=== Overall micro-average PSNR (reconstruction) ===\n"
            f"Full image: {np.mean(results.overall['psnr']):.3f} dB "
            f"(n={len(results.overall['psnr'])})\n"
            f"Foreground: {np.mean(results.overall['fg_psnr']):.3f} dB "
            f"(n={len(results.overall['fg_psnr'])})"
        )

    if results.overall.get("lpips"):
        print("\n=== Per-yaw mean LPIPS (reconstruction) ===")
        for yaw in yaw_values:
            if not results.per_yaw["lpips"].get(yaw):
                continue
            print(
                f"yaw {yaw:+.1f}°: "
                f"Full={np.mean(results.per_yaw['lpips'][yaw]):.4f} | "
                f"FG={np.mean(results.per_yaw['fg_lpips'][yaw]):.4f} | "
                f"n={len(results.per_yaw['lpips'][yaw])}"
            )

        print("\n=== Per-subject mean LPIPS (reconstruction) ===")
        for subject in test_subjects:
            subject = str(subject)
            sub = results.per_subject.get(subject, {})
            if sub.get("lpips"):
                print(
                    f"{subject}: "
                    f"Full={np.mean(sub['lpips']):.4f} | "
                    f"FG={np.mean(sub['fg_lpips']):.4f}"
                )

        print(
            f"\n=== Overall micro-average LPIPS (reconstruction) ===\n"
            f"Full image: {np.mean(results.overall['lpips']):.4f} "
            f"(n={len(results.overall['lpips'])})\n"
            f"Foreground: {np.mean(results.overall['fg_lpips']):.4f} "
            f"(n={len(results.overall['fg_lpips'])})"
        )

    if results.overall.get("identity"):
        print(
            "\n=== Per-yaw mean Identity Similarity vs source "
            "(higher is better) ==="
        )
        for yaw in yaw_values:
            if not results.per_yaw["identity"].get(yaw):
                continue
            print(
                f"yaw {yaw:+.1f}°: "
                f"IdSim={np.mean(results.per_yaw['identity'][yaw]):.4f} | "
                f"n={len(results.per_yaw['identity'][yaw])}"
            )

        print(
            "\n=== Per-subject mean Identity Similarity vs source "
            "(higher is better) ==="
        )
        for subject in test_subjects:
            subject = str(subject)
            sub = results.per_subject.get(subject, {})
            if sub.get("identity"):
                print(f"{subject}: IdSim={np.mean(sub['identity']):.4f}")

        print(
            f"\n=== Overall micro-average Identity Similarity vs source "
            f"(higher is better) ===\n"
            f"IdSim={np.mean(results.overall['identity']):.4f} "
            f"(n={len(results.overall['identity'])})"
        )


def _generate_batch(
    toss,
    src_input: Image.Image,
    *,
    delta_pose_list: list[np.ndarray] | None = None,
    dy_list: list[float] | None = None,
    eval_gen_batch_size: int | None = None,
) -> list[Image.Image]:
    pose_list = delta_pose_list if delta_pose_list is not None else dy_list
    if pose_list is None:
        raise ValueError("pass delta_pose_list or dy_list")

    gen_pils: list[Image.Image] = []
    chunk = eval_gen_batch_size or len(pose_list)
    with torch.no_grad():
        for start in range(0, len(pose_list), chunk):
            batch = pose_list[start : start + chunk]
            if delta_pose_list is not None:
                outs = generate_batch(
                    toss,
                    src_input,
                    prompt="",
                    delta_pose_list=batch,
                )
            else:
                outs = generate_batch(
                    toss,
                    src_input,
                    prompt="",
                    dy_list=batch,
                )
            gen_pils.extend(outs)
    return gen_pils


def _record_metrics(
    results: BatchEvalResults,
    subject: str,
    yaw_key: float,
    metrics: dict[str, float],
) -> None:
    for key, value in metrics.items():
        results.per_yaw[key][yaw_key].append(value)
        results.per_subject[subject][key].append(value)
        results.overall[key].append(value)


def run_batch_eval(
    toss,
    metrics_evaluator: ImageMetricsEvaluator,
    test_root: str,
    test_subjects: list[Any],
    *,
    src_view_idx: int = 3,
    recon_view_indices: Sequence[int] = DEFAULT_RECON_VIEW_INDICES,
    identity_yaw_min_deg: float = -22.0,
    identity_yaw_max_deg: float = 22.0,
    identity_num_yaws: int = 45,
    eval_gen_batch_size: Optional[int] = None,
    metrics_config: Optional[dict[str, bool]] = None,
    verbose: bool = True,
) -> BatchEvalResults:
    """
    Generate novel views per subject and evaluate with metrics_evaluator.

    Requires a toss-like object with ``model``, ``sampler``, and ``device``.
    Batch generation uses ``utils.inference.generate_batch``.

    Reconstruction track (PSNR / LPIPS):
        Fixed upper-ring GT view indices (``recon_view_indices``), excluding
        ``src_view_idx``. Full 3DOF ``compute_relative_pose`` per view vs GT.
        Overall mean is a micro-average over all eval (subject, view) pairs.

    Identity track:
        Uniform yaw grid (``identity_num_yaws`` from ``identity_yaw_min_deg`` to
        ``identity_yaw_max_deg``). Yaw-only generation vs preprocessed source.
        Independent of reconstruction view selection.

    Pred and GT are both passed through ``preprocess_image`` with the GT
    ``alpha_maps`` mask so background pixels are composited onto white equally.
    """
    metrics_config = metrics_config or {}
    run_reconstruction = _reconstruction_enabled(metrics_config)
    run_identity = metrics_config.get("identity", True)

    dy_grid = identity_dy_grid(
        identity_yaw_min_deg,
        identity_yaw_max_deg,
        identity_num_yaws,
    )

    results = BatchEvalResults(
        per_yaw={key: defaultdict(list) for key in _init_metric_buckets()},
        per_subject={
            str(subject): _init_metric_buckets() for subject in test_subjects
        },
        overall=_init_metric_buckets(),
    )

    for subject in test_subjects:
        subject = str(subject)
        sub_path = os.path.join(test_root, subject)

        poses = np.load(os.path.join(sub_path, "poses.npy")).reshape(-1, 4, 4)
        recon_views = select_recon_views(
            len(poses),
            src_view_idx,
            recon_view_indices,
        )

        src_img = Image.open(
            os.path.join(sub_path, f"{src_view_idx:05d}.png")
        ).convert("RGBA")
        src_mask = Image.open(
            os.path.join(sub_path, f"alpha_maps/{src_view_idx:05d}.png")
        ).convert("L")

        src_np = preprocess_image(src_img, fg_mask=src_mask)
        src_input = Image.fromarray(
            np.clip(src_np * 255.0, 0, 255).astype(np.uint8)
        )

        if run_reconstruction:
            if not recon_views:
                if verbose:
                    print(f"subject {subject}: no reconstruction views selected")
            else:
                recon_meta: list[tuple[int, float, np.ndarray]] = []
                for view_idx in recon_views:
                    delta = compute_relative_pose(
                        poses[src_view_idx],
                        poses[view_idx],
                    )
                    yaw_key = _yaw_key(float(np.degrees(delta[1])))
                    recon_meta.append((view_idx, yaw_key, delta.astype(np.float32)))

                delta_pose_list = [delta for _, _, delta in recon_meta]
                recon_pils = _generate_batch(
                    toss,
                    src_input,
                    delta_pose_list=delta_pose_list,
                    eval_gen_batch_size=eval_gen_batch_size,
                )

                for gen_pil, (view_idx, yaw_key, _) in zip(recon_pils, recon_meta):
                    gt_img = Image.open(
                        os.path.join(sub_path, f"{view_idx:05d}.png")
                    ).convert("RGBA")
                    gt_mask = Image.open(
                        os.path.join(sub_path, f"alpha_maps/{view_idx:05d}.png")
                    ).convert("L")

                    gt_mask_np = resize_mask(gt_mask)
                    gt_np = preprocess_image(gt_img, fg_mask=gt_mask)
                    gen_np = preprocess_image(
                        gen_pil.convert("RGB"),
                        fg_mask=gt_mask,
                    )

                    recon_flags = {
                        key: metrics_config.get(key, True)
                        for key in _RECON_METRIC_KEYS
                    }
                    m = metrics_evaluator.compute_metrics(
                        gen_np,
                        gt_np,
                        fg_mask=gt_mask_np,
                        identity=False,
                        **recon_flags,
                    )
                    _record_metrics(results, subject, yaw_key, m)

                    if verbose:
                        print(
                            f"subject {subject}, "
                            f"yaw={yaw_key:+.1f}°, "
                            f"view={view_idx:02d} [recon] | "
                            f"Full PSNR={m.get('psnr', float('nan')):.3f} dB | "
                            f"FG PSNR={m.get('fg_psnr', float('nan')):.3f} dB | "
                            f"Full LPIPS={m.get('lpips', float('nan')):.4f} | "
                            f"FG LPIPS={m.get('fg_lpips', float('nan')):.4f}"
                        )

        if run_identity:
            id_pils = _generate_batch(
                toss,
                src_input,
                dy_list=dy_grid,
                eval_gen_batch_size=eval_gen_batch_size,
            )

            for gen_pil, dy_deg in zip(id_pils, dy_grid):
                yaw_key = _yaw_key(dy_deg)
                gen_np = preprocess_image(
                    gen_pil.convert("RGB"),
                    fg_mask=src_mask,
                )
                id_sim = metrics_evaluator.compute_identity_metric(gen_np, src_np)
                m = {"identity": id_sim}
                _record_metrics(results, subject, yaw_key, m)

                if verbose:
                    print(
                        f"subject {subject}, "
                        f"dy={yaw_key:+.1f}° [identity vs source] | "
                        f"IdSim={id_sim:.4f}"
                    )

    if verbose:
        print_batch_eval_summary(
            results,
            test_subjects,
            recon_view_indices=recon_view_indices,
            identity_num_yaws=identity_num_yaws,
            identity_yaw_min_deg=identity_yaw_min_deg,
            identity_yaw_max_deg=identity_yaw_max_deg,
        )

    return results
