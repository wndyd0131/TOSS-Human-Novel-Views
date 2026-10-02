from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from utils.image import ImageInput, to_numpy_rgb
from utils.pose import compute_relative_pose


def _import_sixdrepnet():
    """
    Import SixDRepNet without resolving ``import utils`` to this repo.

    sixdrepnet uses bare ``import utils`` (see 6DRepNet#53). Because this
    project also has a top-level ``utils`` package on ``sys.path``, simply
    evicting ``sys.modules`` is not enough: Python would re-import our package
    from disk. Bind ``sixdrepnet.utils`` as ``utils`` before loading sixdrepnet
    submodules and patch their module-level ``utils`` references explicitly.
    """
    import importlib
    import sys

    project_utils = {
        name: module
        for name, module in sys.modules.items()
        if name == "utils" or name.startswith("utils.")
    }
    for name in project_utils:
        del sys.modules[name]

    for name in list(sys.modules):
        if name.startswith("sixdrepnet"):
            del sys.modules[name]

    sixdrepnet_utils = importlib.import_module("sixdrepnet.utils")
    sys.modules["utils"] = sixdrepnet_utils

    try:
        sixdrepnet_pkg = importlib.import_module("sixdrepnet")
        for module_name in ("sixdrepnet.model", "sixdrepnet.regressor"):
            module = sys.modules.get(module_name)
            if module is not None:
                module.utils = sixdrepnet_utils

        model_module = sys.modules.get("sixdrepnet.model")
        if model_module is None or not hasattr(
            model_module.utils, "compute_rotation_matrix_from_ortho6d"
        ):
            raise RuntimeError(
                "Failed to bind sixdrepnet.utils; restart the runtime and retry."
            )

        return sixdrepnet_pkg.SixDRepNet
    finally:
        if sys.modules.get("utils") is sixdrepnet_utils:
            del sys.modules["utils"]
        sys.modules.update(project_utils)


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
        if self._model is not None:
            import sys

            model_module = sys.modules.get("sixdrepnet.model")
            if model_module is None or not hasattr(
                model_module.utils, "compute_rotation_matrix_from_ortho6d"
            ):
                self._model = None

        if self._model is None:
            SixDRepNet = _import_sixdrepnet()
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
