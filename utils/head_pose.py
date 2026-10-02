from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from utils.image import ImageInput, to_numpy_rgb
from utils.pose import compute_relative_pose


@dataclass
class PoseCalibrationStats:
    mae_deg: float
    slope: float
    intercept: float
    n_samples: int


class HeadPoseEstimator:
    """Lazy-load 6DRepNet for relative head-pose measurements."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        gpu_id = -1
        if isinstance(device, str) and device.startswith("cuda"):
            gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        self.gpu_id = gpu_id
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sixdrepnet import SixDRepNet

            self._model = SixDRepNet(gpu_id=self.gpu_id)
        return self._model

    def _to_bgr(self, image: ImageInput) -> np.ndarray:
        rgb = (to_numpy_rgb(image) * 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def estimate(self, image: ImageInput) -> tuple[float, float, float]:
        """Return pitch, yaw, roll in degrees."""
        bgr = self._to_bgr(image)
        pitch, yaw, roll = self._get_model().predict(bgr)
        return float(pitch[0]), float(yaw[0]), float(roll[0])

    def relative_yaw(self, pred: ImageInput, src: ImageInput) -> float:
        _, yaw_pred, _ = self.estimate(pred)
        _, yaw_src, _ = self.estimate(src)
        return yaw_pred - yaw_src


def compute_pose_errors(
    requested_dy_deg: float,
    pred: ImageInput,
    src: ImageInput,
    estimator: HeadPoseEstimator,
) -> dict[str, float]:
    pitch_pred, _, _ = estimator.estimate(pred)
    pitch_src, _, _ = estimator.estimate(src)
    rel_yaw = estimator.relative_yaw(pred, src)
    return {
        "pose_yaw_error": abs(rel_yaw - float(requested_dy_deg)),
        "pose_pitch_leak": abs(pitch_pred - pitch_src),
    }


def calibrate_pose_estimator_on_gt(
    estimator: HeadPoseEstimator,
    samples: Sequence[tuple[np.ndarray, ImageInput, ImageInput, int, int]],
) -> PoseCalibrationStats:
    """
    Fit estimator yaw deltas against ``poses.npy`` GT deltas.

    Each sample is ``(poses, src_img, tgt_img, src_view_idx, tgt_view_idx)``.
    """
    gt_deltas: list[float] = []
    est_deltas: list[float] = []

    for poses, src_img, tgt_img, src_idx, tgt_idx in samples:
        delta = compute_relative_pose(poses[src_idx], poses[tgt_idx])
        gt_deltas.append(float(np.degrees(delta[1])))
        est_deltas.append(estimator.relative_yaw(tgt_img, src_img))

    if not gt_deltas:
        return PoseCalibrationStats(
            mae_deg=float("nan"),
            slope=float("nan"),
            intercept=float("nan"),
            n_samples=0,
        )

    gt_arr = np.asarray(gt_deltas, dtype=np.float64)
    est_arr = np.asarray(est_deltas, dtype=np.float64)
    mae = float(np.mean(np.abs(est_arr - gt_arr)))

    if len(gt_arr) >= 2 and np.std(gt_arr) > 1e-6:
        slope, intercept = np.polyfit(gt_arr, est_arr, 1)
    else:
        slope, intercept = float("nan"), float("nan")

    return PoseCalibrationStats(
        mae_deg=mae,
        slope=float(slope),
        intercept=float(intercept),
        n_samples=len(gt_deltas),
    )
