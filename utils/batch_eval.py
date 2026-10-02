from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from utils.eval_metrics import ImageMetricsEvaluator
from utils.head_pose import (
    HeadPoseEstimator,
    calibrate_pose_estimator_on_gt,
    compute_pose_errors,
)
from utils.image import preprocess_image, resize_mask
from utils.inference import generate_batch
from utils.multiview_metrics import MultiviewMetricsEvaluator
from utils.pose import (
    DEFAULT_RECON_VIEW_INDICES,
    compute_relative_pose,
    identity_dy_grid,
    select_horizontal_views,
    select_recon_views,
)


@dataclass
class BatchEvalResults:
    per_yaw: dict[str, dict[float, list[float]]] = field(default_factory=dict)
    per_subject: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    overall: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _init_metric_buckets() -> dict[str, list[float]]:
    return {
        "psnr": [],
        "fg_psnr": [],
        "lpips": [],
        "fg_lpips": [],
        "identity": [],
        "mv_warp_error": [],
        "gt_mv_warp_error": [],
        "pose_yaw_error": [],
        "pose_pitch_leak": [],
    }


_RECON_METRIC_KEYS = ("psnr", "fg_psnr", "lpips", "fg_lpips")
_IDENTITY_METRIC_KEYS = ("identity",)
_MV_METRIC_KEYS = ("mv_warp_error", "gt_mv_warp_error")
_POSE_METRIC_KEYS = ("pose_yaw_error", "pose_pitch_leak")

_CSV_SUMMARY_HEADER = (
    "timestamp",
    "checkpoint",
    "num_subjects",
    "recon_n",
    "psnr",
    "fg_psnr",
    "lpips",
    "fg_lpips",
    "identity_n",
    "identity",
    "mv_warp_error",
    "gt_mv_warp_error",
    "pose_yaw_error",
    "pose_pitch_leak",
    "src_view_idx",
    "recon_view_indices",
    "identity_num_yaws",
)
_EVAL_STATE_VERSION = 2
_EVAL_STATE_FILENAME = "eval_state.json"


def _reconstruction_enabled(metrics_config: dict[str, bool]) -> bool:
    return any(metrics_config.get(key, True) for key in _RECON_METRIC_KEYS)


def _sorted_yaw_keys(per_yaw: dict[str, dict[float, list[float]]]) -> list[float]:
    for key in (
        "lpips",
        "psnr",
        "identity",
        "pose_yaw_error",
        "pose_pitch_leak",
        "fg_lpips",
        "fg_psnr",
        "mv_warp_error",
        "gt_mv_warp_error",
    ):
        bucket = per_yaw.get(key)
        if bucket:
            return sorted(bucket.keys())
    return []


def _yaw_key(yaw_deg: float) -> float:
    key = round(float(yaw_deg), 1)
    return 0.0 if key == 0 else key


def format_batch_eval_summary(
    results: BatchEvalResults,
    test_subjects: list[Any],
    *,
    recon_view_indices: Sequence[int] = DEFAULT_RECON_VIEW_INDICES,
    identity_num_yaws: int = 45,
    identity_yaw_min_deg: float = -22.0,
    identity_yaw_max_deg: float = 22.0,
) -> str:
    """Return the same text that ``print_batch_eval_summary`` prints."""
    lines: list[str] = []
    lines.append("\n=== Eval configuration ===")
    lines.append(
        f"Reconstruction views (GT): {list(recon_view_indices)} (source excluded)"
    )
    lines.append(
        f"Identity grid: {identity_num_yaws} yaws "
        f"from {identity_yaw_min_deg:+.1f}° to {identity_yaw_max_deg:+.1f}° vs source"
    )

    yaw_values = _sorted_yaw_keys(results.per_yaw)

    if results.overall.get("psnr"):
        lines.append("\n=== Per-yaw mean PSNR (reconstruction) ===")
        for yaw in yaw_values:
            if not results.per_yaw["psnr"].get(yaw):
                continue
            lines.append(
                f"yaw {yaw:+.1f}°: "
                f"Full={np.mean(results.per_yaw['psnr'][yaw]):.3f} dB | "
                f"FG={np.mean(results.per_yaw['fg_psnr'][yaw]):.3f} dB | "
                f"n={len(results.per_yaw['psnr'][yaw])}"
            )

        lines.append("\n=== Per-subject mean PSNR (reconstruction) ===")
        for subject in test_subjects:
            subject = str(subject)
            sub = results.per_subject.get(subject, {})
            if sub.get("psnr"):
                lines.append(
                    f"{subject}: "
                    f"Full={np.mean(sub['psnr']):.3f} dB | "
                    f"FG={np.mean(sub['fg_psnr']):.3f} dB"
                )

        lines.append(
            f"\n=== Overall micro-average PSNR (reconstruction) ===\n"
            f"Full image: {np.mean(results.overall['psnr']):.3f} dB "
            f"(n={len(results.overall['psnr'])})\n"
            f"Foreground: {np.mean(results.overall['fg_psnr']):.3f} dB "
            f"(n={len(results.overall['fg_psnr'])})"
        )

    if results.overall.get("lpips"):
        lines.append("\n=== Per-yaw mean LPIPS (reconstruction) ===")
        for yaw in yaw_values:
            if not results.per_yaw["lpips"].get(yaw):
                continue
            lines.append(
                f"yaw {yaw:+.1f}°: "
                f"Full={np.mean(results.per_yaw['lpips'][yaw]):.4f} | "
                f"FG={np.mean(results.per_yaw['fg_lpips'][yaw]):.4f} | "
                f"n={len(results.per_yaw['lpips'][yaw])}"
            )

        lines.append("\n=== Per-subject mean LPIPS (reconstruction) ===")
        for subject in test_subjects:
            subject = str(subject)
            sub = results.per_subject.get(subject, {})
            if sub.get("lpips"):
                lines.append(
                    f"{subject}: "
                    f"Full={np.mean(sub['lpips']):.4f} | "
                    f"FG={np.mean(sub['fg_lpips']):.4f}"
                )

        lines.append(
            f"\n=== Overall micro-average LPIPS (reconstruction) ===\n"
            f"Full image: {np.mean(results.overall['lpips']):.4f} "
            f"(n={len(results.overall['lpips'])})\n"
            f"Foreground: {np.mean(results.overall['fg_lpips']):.4f} "
            f"(n={len(results.overall['fg_lpips'])})"
        )

    if results.overall.get("identity"):
        lines.append(
            "\n=== Per-yaw mean Identity Similarity vs source "
            "(higher is better) ==="
        )
        for yaw in yaw_values:
            if not results.per_yaw["identity"].get(yaw):
                continue
            lines.append(
                f"yaw {yaw:+.1f}°: "
                f"IdSim={np.mean(results.per_yaw['identity'][yaw]):.4f} | "
                f"n={len(results.per_yaw['identity'][yaw])}"
            )

        lines.append(
            "\n=== Per-subject mean Identity Similarity vs source "
            "(higher is better) ==="
        )
        for subject in test_subjects:
            subject = str(subject)
            sub = results.per_subject.get(subject, {})
            if sub.get("identity"):
                lines.append(f"{subject}: IdSim={np.mean(sub['identity']):.4f}")

        lines.append(
            f"\n=== Overall micro-average Identity Similarity vs source "
            f"(higher is better) ===\n"
            f"IdSim={np.mean(results.overall['identity']):.4f} "
            f"(n={len(results.overall['identity'])})"
        )

    if results.overall.get("mv_warp_error"):
        lines.append(
            "\n=== Per-pair mean Multiview warp error (lower is better) ==="
        )
        for yaw in _sorted_yaw_keys(results.per_yaw):
            if not results.per_yaw["mv_warp_error"].get(yaw):
                continue
            gt_vals = results.per_yaw["gt_mv_warp_error"].get(yaw, [])
            gt_mean = np.mean(gt_vals) if gt_vals else float("nan")
            pred_mean = np.mean(results.per_yaw["mv_warp_error"][yaw])
            ratio = pred_mean / gt_mean if gt_vals and gt_mean > 0 else float("nan")
            lines.append(
                f"pair left yaw {yaw:+.1f}°: "
                f"pred={pred_mean:.4f} | GT ref={gt_mean:.4f} | "
                f"ratio={ratio:.3f} | n={len(results.per_yaw['mv_warp_error'][yaw])}"
            )
        lines.append(
            f"\n=== Overall micro-average Multiview warp error ===\n"
            f"pred={np.mean(results.overall['mv_warp_error']):.4f} "
            f"(n={len(results.overall['mv_warp_error'])})"
        )
        if results.overall.get("gt_mv_warp_error"):
            lines.append(
                f"GT reference={np.mean(results.overall['gt_mv_warp_error']):.4f} "
                f"(n={len(results.overall['gt_mv_warp_error'])})"
            )

    if results.overall.get("pose_yaw_error"):
        lines.append(
            "\n=== Per-yaw mean Pose yaw error (lower is better) ==="
        )
        for yaw in yaw_values:
            if not results.per_yaw["pose_yaw_error"].get(yaw):
                continue
            lines.append(
                f"yaw {yaw:+.1f}°: "
                f"yaw_err={np.mean(results.per_yaw['pose_yaw_error'][yaw]):.2f}° | "
                f"pitch_leak={np.mean(results.per_yaw['pose_pitch_leak'][yaw]):.2f}° | "
                f"n={len(results.per_yaw['pose_yaw_error'][yaw])}"
            )
        lines.append(
            f"\n=== Overall micro-average Pose errors ===\n"
            f"yaw_err={np.mean(results.overall['pose_yaw_error']):.2f}° "
            f"(n={len(results.overall['pose_yaw_error'])})\n"
            f"pitch_leak={np.mean(results.overall['pose_pitch_leak']):.2f}° "
            f"(n={len(results.overall['pose_pitch_leak'])})"
        )

    if results.metadata.get("pose_estimator_gt_mae_deg") is not None:
        lines.append(
            "\n=== Pose estimator GT calibration (reference) ==="
        )
        lines.append(
            f"GT MAE={results.metadata['pose_estimator_gt_mae_deg']:.2f}° | "
            f"slope={results.metadata.get('pose_estimator_gt_slope', float('nan')):.3f} | "
            f"n={results.metadata.get('pose_estimator_gt_n', 0)}"
        )

    return "\n".join(lines)


def print_batch_eval_summary(
    results: BatchEvalResults,
    test_subjects: list[Any],
    *,
    recon_view_indices: Sequence[int] = DEFAULT_RECON_VIEW_INDICES,
    identity_num_yaws: int = 45,
    identity_yaw_min_deg: float = -22.0,
    identity_yaw_max_deg: float = 22.0,
) -> None:
    print(
        format_batch_eval_summary(
            results,
            test_subjects,
            recon_view_indices=recon_view_indices,
            identity_num_yaws=identity_num_yaws,
            identity_yaw_min_deg=identity_yaw_min_deg,
            identity_yaw_max_deg=identity_yaw_max_deg,
        )
    )


def _format_yaw_json_key(yaw_deg: float) -> str:
    return f"{float(yaw_deg):+.1f}"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _parse_yaw_json_key(key: str) -> float:
    return _yaw_key(float(key))


def _results_to_serializable(results: BatchEvalResults) -> dict[str, Any]:
    per_yaw: dict[str, dict[str, list[float]]] = {}
    for metric_key, yaw_bucket in results.per_yaw.items():
        per_yaw[metric_key] = {
            _format_yaw_json_key(yaw): list(values)
            for yaw, values in yaw_bucket.items()
        }
    return {
        "overall": {
            key: list(values) for key, values in results.overall.items()
        },
        "per_subject": {
            subject: {key: list(values) for key, values in metrics.items()}
            for subject, metrics in results.per_subject.items()
        },
        "per_yaw": per_yaw,
        "metadata": dict(results.metadata),
    }


def _results_from_serializable(data: dict[str, Any]) -> BatchEvalResults:
    per_yaw = {
        key: defaultdict(list) for key in _init_metric_buckets()
    }
    for metric_key in _init_metric_buckets():
        for yaw_str, values in data.get("per_yaw", {}).get(metric_key, {}).items():
            per_yaw[metric_key][_parse_yaw_json_key(yaw_str)] = list(values)

    per_subject: dict[str, dict[str, list[float]]] = {}
    for subject, metrics in data.get("per_subject", {}).items():
        per_subject[str(subject)] = {
            key: list(values) for key, values in metrics.items()
        }

    overall = _init_metric_buckets()
    for key, values in data.get("overall", {}).items():
        if key in overall:
            overall[key] = list(values)

    return BatchEvalResults(
        per_yaw=per_yaw,
        per_subject=per_subject,
        overall=overall,
        metadata=dict(data.get("metadata", {})),
    )


def _normalize_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    recon_views = normalized.get("recon_view_indices")
    if recon_views is not None and isinstance(recon_views, Sequence) and not isinstance(
        recon_views, (str, bytes)
    ):
        normalized["recon_view_indices"] = list(recon_views)
    metrics_config = normalized.get("metrics_config")
    if metrics_config is not None:
        normalized["metrics_config"] = dict(metrics_config)
    return normalized


def _eval_configs_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(_normalize_eval_config(left), sort_keys=True) == json.dumps(
        _normalize_eval_config(right), sort_keys=True
    )


def _build_run_eval_config(
    *,
    src_view_idx: int,
    recon_view_indices: Sequence[int],
    identity_yaw_min_deg: float,
    identity_yaw_max_deg: float,
    identity_num_yaws: int,
    metrics_config: dict[str, bool],
) -> dict[str, Any]:
    return _normalize_eval_config(
        {
            "src_view_idx": src_view_idx,
            "recon_view_indices": list(recon_view_indices),
            "identity_yaw_min_deg": identity_yaw_min_deg,
            "identity_yaw_max_deg": identity_yaw_max_deg,
            "identity_num_yaws": identity_num_yaws,
            "metrics_config": dict(metrics_config),
        }
    )


def _init_batch_eval_results(test_subjects: list[Any]) -> BatchEvalResults:
    return BatchEvalResults(
        per_yaw={key: defaultdict(list) for key in _init_metric_buckets()},
        per_subject={
            str(subject): _init_metric_buckets() for subject in test_subjects
        },
        overall=_init_metric_buckets(),
        metadata={},
    )


def _ensure_subject_buckets(
    results: BatchEvalResults,
    test_subjects: list[Any],
) -> None:
    for subject in test_subjects:
        subject = str(subject)
        if subject not in results.per_subject:
            results.per_subject[subject] = _init_metric_buckets()


def save_eval_checkpoint(
    checkpoint_dir: str | Path,
    *,
    results: BatchEvalResults,
    completed_subjects: list[str],
    checkpoint: str,
    test_subjects: list[Any],
    eval_config: dict[str, Any],
) -> Path:
    """Write rolling ``eval_state.json`` after each completed subject."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / _EVAL_STATE_FILENAME
    payload = {
        "version": _EVAL_STATE_VERSION,
        "checkpoint": checkpoint,
        "eval_config": _normalize_eval_config(eval_config),
        "test_subjects": [str(subject) for subject in test_subjects],
        "completed_subjects": list(completed_subjects),
        "results": _results_to_serializable(results),
    }
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return state_path


def load_eval_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Load ``eval_state.json`` if present."""
    state_path = Path(checkpoint_dir) / _EVAL_STATE_FILENAME
    if not state_path.exists():
        return None
    with state_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate_track(
    results: BatchEvalResults,
    metric_keys: Sequence[str],
    *,
    aggregation: str,
) -> dict[str, Any]:
    overall: dict[str, float] = {}
    for key in metric_keys:
        values = results.overall.get(key, [])
        if values:
            overall[key] = float(np.mean(values))

    primary = metric_keys[0] if metric_keys else None
    n_samples = len(results.overall.get(primary, [])) if primary else 0

    per_subject: dict[str, Any] = {}
    for subject, sub in results.per_subject.items():
        entry: dict[str, Any] = {}
        n = 0
        for key in metric_keys:
            values = sub.get(key, [])
            if values:
                entry[key] = float(np.mean(values))
                n = len(values)
        if entry:
            entry["n"] = n
            per_subject[subject] = entry

    yaw_keys: set[float] = set()
    for key in metric_keys:
        yaw_keys.update(results.per_yaw.get(key, {}).keys())

    per_yaw: dict[str, Any] = {}
    for yaw in sorted(yaw_keys):
        entry: dict[str, Any] = {}
        n = 0
        for key in metric_keys:
            values = results.per_yaw.get(key, {}).get(yaw, [])
            if values:
                entry[key] = float(np.mean(values))
                n = len(values)
        if entry:
            entry["n"] = n
            per_yaw[_format_yaw_json_key(yaw)] = entry

    return {
        "aggregation": aggregation,
        "n_samples": n_samples,
        "overall": overall,
        "per_subject": per_subject,
        "per_yaw": per_yaw,
    }


def _build_eval_json_payload(
    results: BatchEvalResults,
    *,
    checkpoint: str,
    timestamp: str,
    test_subjects: list[Any],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint": checkpoint,
        "timestamp": timestamp,
        "eval_config": eval_config,
        "test_subjects": [str(subject) for subject in test_subjects],
    }

    if any(results.overall.get(key) for key in _RECON_METRIC_KEYS):
        payload["reconstruction"] = _aggregate_track(
            results,
            _RECON_METRIC_KEYS,
            aggregation="micro-average over (subject, gt_view)",
        )

    if results.overall.get("identity"):
        payload["identity"] = _aggregate_track(
            results,
            _IDENTITY_METRIC_KEYS,
            aggregation="micro-average over (subject, dy_grid) vs source",
        )

    if any(results.overall.get(key) for key in _MV_METRIC_KEYS):
        payload["multiview_consistency"] = _aggregate_track(
            results,
            _MV_METRIC_KEYS,
            aggregation="micro-average over (subject, adjacent yaw pair)",
        )

    if any(results.overall.get(key) for key in _POSE_METRIC_KEYS):
        payload["pose_accuracy"] = _aggregate_track(
            results,
            _POSE_METRIC_KEYS,
            aggregation="micro-average over (subject, dy_grid)",
        )

    if results.metadata:
        payload["gt_reference"] = dict(results.metadata)

    return payload


def _resolve_eval_timestamp(timestamp: str | None) -> tuple[str, str]:
    """Return ISO UTC timestamp and ``YYYYMMDD_HHMM`` filename prefix."""
    if timestamp is None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return now.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y%m%d_%H%M")


def _append_csv_summary_row(
    csv_path: Path,
    row: dict[str, Any],
) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_SUMMARY_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in _CSV_SUMMARY_HEADER})


def save_batch_eval_logs(
    results: BatchEvalResults,
    log_dir: str | Path,
    *,
    checkpoint: str,
    test_subjects: list[Any],
    eval_config: dict[str, Any] | None = None,
    timestamp: str | None = None,
    include_timestamp_in_filename: bool = True,
) -> dict[str, Path]:
    """
    Persist eval results for one run.

    Writes:
        ``{YYYYMMDD_HHMM}_{checkpoint_stem}.json`` — full archive (default)
        ``{YYYYMMDD_HHMM}_{checkpoint_stem}.txt`` — human-readable summary
        ``summary.csv`` — append one comparison row (created if missing)

    Set ``include_timestamp_in_filename=False`` for ``{checkpoint_stem}.*`` only.

    Log files are intended for Drive/local storage, not the git repo.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_config = dict(eval_config or {})
    timestamp, time_prefix = _resolve_eval_timestamp(timestamp)
    checkpoint_stem = Path(checkpoint).stem
    run_stem = (
        f"{time_prefix}_{checkpoint_stem}"
        if include_timestamp_in_filename
        else checkpoint_stem
    )

    json_path = log_dir / f"{run_stem}.json"
    txt_path = log_dir / f"{run_stem}.txt"
    csv_path = log_dir / "summary.csv"

    payload = _build_eval_json_payload(
        results,
        checkpoint=checkpoint,
        timestamp=timestamp,
        test_subjects=test_subjects,
        eval_config=eval_config,
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    summary_kwargs = {
        "recon_view_indices": eval_config.get(
            "recon_view_indices", DEFAULT_RECON_VIEW_INDICES
        ),
        "identity_num_yaws": eval_config.get("identity_num_yaws", 45),
        "identity_yaw_min_deg": eval_config.get("identity_yaw_min_deg", -22.0),
        "identity_yaw_max_deg": eval_config.get("identity_yaw_max_deg", 22.0),
    }
    summary_body = format_batch_eval_summary(
        results,
        test_subjects,
        **summary_kwargs,
    )
    txt_content = (
        f"checkpoint: {checkpoint}\n"
        f"timestamp: {timestamp}\n"
        f"{summary_body}\n"
    )
    txt_path.write_text(txt_content, encoding="utf-8")

    recon_n = len(results.overall.get("psnr", []))
    identity_n = len(results.overall.get("identity", []))
    recon_view_indices = eval_config.get("recon_view_indices", DEFAULT_RECON_VIEW_INDICES)
    if isinstance(recon_view_indices, Sequence) and not isinstance(
        recon_view_indices, (str, bytes)
    ):
        recon_view_indices_str = json.dumps(list(recon_view_indices))
    else:
        recon_view_indices_str = json.dumps(recon_view_indices)

    csv_row = {
        "timestamp": timestamp,
        "checkpoint": checkpoint,
        "num_subjects": len(test_subjects),
        "recon_n": recon_n,
        "psnr": _mean_or_none(results.overall.get("psnr", [])),
        "fg_psnr": _mean_or_none(results.overall.get("fg_psnr", [])),
        "lpips": _mean_or_none(results.overall.get("lpips", [])),
        "fg_lpips": _mean_or_none(results.overall.get("fg_lpips", [])),
        "identity_n": identity_n,
        "identity": _mean_or_none(results.overall.get("identity", [])),
        "mv_warp_error": _mean_or_none(results.overall.get("mv_warp_error", [])),
        "gt_mv_warp_error": _mean_or_none(results.overall.get("gt_mv_warp_error", [])),
        "pose_yaw_error": _mean_or_none(results.overall.get("pose_yaw_error", [])),
        "pose_pitch_leak": _mean_or_none(results.overall.get("pose_pitch_leak", [])),
        "src_view_idx": eval_config.get("src_view_idx", ""),
        "recon_view_indices": recon_view_indices_str,
        "identity_num_yaws": eval_config.get("identity_num_yaws", ""),
    }
    _append_csv_summary_row(csv_path, csv_row)

    return {"json": json_path, "txt": txt_path, "csv": csv_path}


def _generate_batch(
    toss,
    src_input: Image.Image,
    *,
    delta_pose_list: list[np.ndarray] | None = None,
    dy_list: list[float] | None = None,
    eval_gen_batch_size: int | None = None,
) -> list[Image.Image]:
    if delta_pose_list is not None:
        return generate_batch(
            toss,
            src_input,
            prompt="",
            delta_pose_list=delta_pose_list,
            shared_noise=True,
            chunk_size=eval_gen_batch_size,
        )
    if dy_list is not None:
        return generate_batch(
            toss,
            src_input,
            prompt="",
            dy_list=dy_list,
            shared_noise=True,
            chunk_size=eval_gen_batch_size,
        )
    raise ValueError("pass delta_pose_list or dy_list")


def _load_subject_rgba(
    sub_path: str,
    view_idx: int,
) -> tuple[Image.Image, Image.Image]:
    img = Image.open(os.path.join(sub_path, f"{view_idx:05d}.png")).convert("RGBA")
    mask = Image.open(
        os.path.join(sub_path, f"alpha_maps/{view_idx:05d}.png")
    ).convert("L")
    return img, mask


def _preprocessed_pil(img: Image.Image, mask: Image.Image) -> Image.Image:
    arr = preprocess_image(img, fg_mask=mask)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def _record_gt_references(
    results: BatchEvalResults,
    subject: str,
    sub_path: str,
    poses: np.ndarray,
    src_view_idx: int,
    src_input: Image.Image,
    mv_evaluator: MultiviewMetricsEvaluator,
    gt_calib_samples: list[tuple[np.ndarray, Image.Image, Image.Image, int, int]],
) -> None:
    horiz_views = select_horizontal_views(poses, src_view_idx)
    if len(horiz_views) < 2:
        return

    gt_images: list[Image.Image] = []
    gt_masks: list[Image.Image] = []
    gt_yaws: list[float] = []

    for view_idx in horiz_views:
        gt_img, gt_mask = _load_subject_rgba(sub_path, view_idx)
        gt_pil = _preprocessed_pil(gt_img, gt_mask)
        delta = compute_relative_pose(poses[src_view_idx], poses[view_idx])
        gt_yaws.append(float(np.degrees(delta[1])))
        gt_images.append(gt_pil)
        gt_masks.append(gt_mask)
        gt_calib_samples.append((poses, src_input, gt_pil, src_view_idx, view_idx))

    pair_errors = mv_evaluator.compute_adjacent_warp_errors(
        gt_images,
        gt_yaws,
        fg_masks=gt_masks,
        yaw_key_fn=_yaw_key,
    )
    for left_yaw, err in pair_errors:
        _record_metrics(results, subject, left_yaw, {"gt_mv_warp_error": err})


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
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    checkpoint: str | None = None,
    eval_config: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
    mv_evaluator: MultiviewMetricsEvaluator | None = None,
    pose_estimator: HeadPoseEstimator | None = None,
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

    Checkpointing (Colab):
        Set ``checkpoint_dir`` and ``checkpoint`` to save ``eval_state.json`` after
        each subject. Pass ``resume=True`` to skip completed subjects. When all
        subjects finish, ``save_batch_eval_logs`` runs automatically into
        ``log_dir`` (defaults to ``checkpoint_dir``). If interrupted, re-run with
        the same args and ``resume=True``; call ``save_batch_eval_logs`` manually
        only if you need logs without finishing all subjects.
    """
    metrics_config = metrics_config or {}
    run_reconstruction = _reconstruction_enabled(metrics_config)
    run_identity = metrics_config.get("identity", True)
    run_mv = metrics_config.get("mv_consistency", True)
    run_pose = metrics_config.get("pose_accuracy", True)
    run_gt_refs = metrics_config.get("gt_references", True)
    run_sweep = run_identity or run_mv or run_pose

    device = getattr(toss, "device", torch.device("cpu"))
    if (run_mv or run_gt_refs) and mv_evaluator is None:
        mv_evaluator = MultiviewMetricsEvaluator(device=device)
    if (run_pose or run_gt_refs) and pose_estimator is None:
        pose_estimator = HeadPoseEstimator(device=str(device))

    gt_calib_samples: list[tuple[np.ndarray, Image.Image, Image.Image, int, int]] = []

    run_eval_config = _build_run_eval_config(
        src_view_idx=src_view_idx,
        recon_view_indices=recon_view_indices,
        identity_yaw_min_deg=identity_yaw_min_deg,
        identity_yaw_max_deg=identity_yaw_max_deg,
        identity_num_yaws=identity_num_yaws,
        metrics_config=metrics_config,
    )
    if eval_config is not None:
        run_eval_config.update(_normalize_eval_config(eval_config))

    if checkpoint_dir is not None and not checkpoint:
        raise ValueError("checkpoint_dir requires checkpoint name for metadata/logs")

    subject_ids = [str(subject) for subject in test_subjects]
    completed_subjects: list[str] = []
    completed_set: set[str] = set()
    results = _init_batch_eval_results(test_subjects)

    if checkpoint_dir is not None:
        state = load_eval_checkpoint(checkpoint_dir) if resume else None
        if state is not None:
            if state.get("version") != _EVAL_STATE_VERSION:
                raise ValueError(
                    f"unsupported eval checkpoint version: {state.get('version')}"
                )
            if state.get("test_subjects") != subject_ids:
                raise ValueError(
                    "checkpoint test_subjects mismatch; use a fresh checkpoint_dir "
                    "or pass the same test_subjects list"
                )
            if not _eval_configs_match(
                state.get("eval_config", {}),
                run_eval_config,
            ):
                raise ValueError(
                    "checkpoint eval_config mismatch; use a fresh checkpoint_dir "
                    "or pass the same eval settings"
                )
            results = _results_from_serializable(state["results"])
            completed_subjects = [str(s) for s in state.get("completed_subjects", [])]
            completed_set = set(completed_subjects)
            _ensure_subject_buckets(results, test_subjects)
            if verbose:
                print(
                    f"Resumed checkpoint: {len(completed_subjects)}/"
                    f"{len(subject_ids)} subjects complete"
                )
        elif resume and verbose:
            print("No checkpoint found; starting fresh eval")

    dy_grid = identity_dy_grid(
        identity_yaw_min_deg,
        identity_yaw_max_deg,
        identity_num_yaws,
    )

    for subject in subject_ids:
        if subject in completed_set:
            if verbose:
                print(f"subject {subject}: skip (checkpoint)")
            continue

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

        if run_sweep:
            id_pils = _generate_batch(
                toss,
                src_input,
                dy_list=dy_grid,
                eval_gen_batch_size=eval_gen_batch_size,
            )

            if run_mv and mv_evaluator is not None:
                pair_errors = mv_evaluator.compute_adjacent_warp_errors(
                    id_pils,
                    dy_grid,
                    fg_masks=[src_mask] * len(id_pils),
                    yaw_key_fn=_yaw_key,
                )
                for left_yaw, err in pair_errors:
                    _record_metrics(
                        results,
                        subject,
                        left_yaw,
                        {"mv_warp_error": err},
                    )
                    if verbose:
                        print(
                            f"subject {subject}, "
                            f"pair left yaw={left_yaw:+.1f}° [mv] | "
                            f"warp_err={err:.4f}"
                        )

            for gen_pil, dy_deg in zip(id_pils, dy_grid):
                yaw_key = _yaw_key(dy_deg)
                gen_np = preprocess_image(
                    gen_pil.convert("RGB"),
                    fg_mask=src_mask,
                )

                metrics: dict[str, float] = {}
                if run_identity:
                    metrics["identity"] = metrics_evaluator.compute_identity_metric(
                        gen_np, src_np
                    )
                if run_pose and pose_estimator is not None:
                    metrics.update(
                        compute_pose_errors(
                            dy_deg,
                            gen_pil.convert("RGB"),
                            src_input.convert("RGB"),
                            pose_estimator,
                        )
                    )

                if metrics:
                    _record_metrics(results, subject, yaw_key, metrics)

                if verbose and run_identity and "identity" in metrics:
                    msg = (
                        f"subject {subject}, "
                        f"dy={yaw_key:+.1f}° [identity vs source] | "
                        f"IdSim={metrics['identity']:.4f}"
                    )
                    if run_pose and "pose_yaw_error" in metrics:
                        msg += (
                            f" | pose_yaw_err={metrics['pose_yaw_error']:.2f}°"
                            f" | pitch_leak={metrics['pose_pitch_leak']:.2f}°"
                        )
                    print(msg)

        if run_gt_refs and mv_evaluator is not None and pose_estimator is not None:
            _record_gt_references(
                results,
                subject,
                sub_path,
                poses,
                src_view_idx,
                src_input,
                mv_evaluator,
                gt_calib_samples,
            )

        completed_subjects.append(subject)
        completed_set.add(subject)
        if checkpoint_dir is not None:
            assert checkpoint is not None
            save_eval_checkpoint(
                checkpoint_dir,
                results=results,
                completed_subjects=completed_subjects,
                checkpoint=checkpoint,
                test_subjects=test_subjects,
                eval_config=run_eval_config,
            )
            if verbose:
                print(
                    f"subject {subject}: checkpoint saved "
                    f"({len(completed_subjects)}/{len(subject_ids)})"
                )

    if run_gt_refs and gt_calib_samples and pose_estimator is not None:
        calib = calibrate_pose_estimator_on_gt(pose_estimator, gt_calib_samples)
        results.metadata["pose_estimator_gt_mae_deg"] = calib.mae_deg
        results.metadata["pose_estimator_gt_slope"] = calib.slope
        results.metadata["pose_estimator_gt_intercept"] = calib.intercept
        results.metadata["pose_estimator_gt_n"] = calib.n_samples

    if verbose:
        print_batch_eval_summary(
            results,
            test_subjects,
            recon_view_indices=recon_view_indices,
            identity_num_yaws=identity_num_yaws,
            identity_yaw_min_deg=identity_yaw_min_deg,
            identity_yaw_max_deg=identity_yaw_max_deg,
        )

    if (
        checkpoint_dir is not None
        and checkpoint is not None
        and len(completed_subjects) == len(subject_ids)
    ):
        log_paths = save_batch_eval_logs(
            results,
            log_dir or checkpoint_dir,
            checkpoint=checkpoint,
            test_subjects=test_subjects,
            eval_config=run_eval_config,
        )
        if verbose:
            print("\n=== Eval logs saved ===")
            for key, path in log_paths.items():
                print(f"{key}: {path}")

    return results
