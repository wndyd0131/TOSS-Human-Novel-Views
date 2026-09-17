from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_RECON_VIEW_INDICES = (0, 1, 2, 3, 8, 12, 13, 15)


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float]:
    """
    Extract pitch (x-rotation) and yaw (y-rotation) from a 3x3 rotation matrix.
    Returns angles in radians.

    Assumes rotation order: R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    """
    sy = np.clip(R[0, 2], -1.0, 1.0)
    yaw = np.arcsin(sy)

    if np.abs(sy) < 0.99999:
        pitch = np.arctan2(-R[1, 2], R[2, 2])
    else:
        pitch = np.arctan2(R[2, 1], R[1, 1])

    return float(pitch), float(yaw)


def pose_matrix_to_toss_format(pose_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose matrix to TOSS format: [pitch, yaw, distance]."""
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]

    pitch, yaw = rotation_matrix_to_euler(R)
    distance = np.linalg.norm(t)

    return np.array([pitch, yaw, distance], dtype=np.float32)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def compute_relative_pose(
    src_pose_4x4: np.ndarray,
    tgt_pose_4x4: np.ndarray,
) -> np.ndarray:
    """
    Relative [delta_pitch, delta_yaw, delta_distance] in radians.
    Nersemble yaw sign is flipped to match Portrait4D/TOSS convention.
    """
    src_pitch, src_yaw = rotation_matrix_to_euler(src_pose_4x4[:3, :3])
    tgt_pitch, tgt_yaw = rotation_matrix_to_euler(tgt_pose_4x4[:3, :3])

    delta_pitch = _wrap_angle(tgt_pitch - src_pitch)
    delta_yaw = -_wrap_angle(tgt_yaw - src_yaw)

    src_dist = float(np.linalg.norm(src_pose_4x4[:3, 3]))
    tgt_dist = float(np.linalg.norm(tgt_pose_4x4[:3, 3]))

    return np.array(
        [delta_pitch, delta_yaw, tgt_dist - src_dist],
        dtype=np.float32,
    )


def select_recon_views(
    num_views: int,
    src_view_idx: int,
    view_indices: Sequence[int] = DEFAULT_RECON_VIEW_INDICES,
) -> list[int]:
    """Return sorted unique view indices for recon eval, excluding source."""
    selected: list[int] = []
    seen: set[int] = set()
    for view_idx in view_indices:
        view_idx = int(view_idx)
        if view_idx == src_view_idx:
            continue
        if view_idx < 0 or view_idx >= num_views:
            continue
        if view_idx in seen:
            continue
        seen.add(view_idx)
        selected.append(view_idx)
    return sorted(selected)


def identity_dy_grid(
    yaw_min: float = -22.0,
    yaw_max: float = 22.0,
    n: int = 45,
) -> list[float]:
    """Uniform yaw grid in degrees for identity eval."""
    if n < 1:
        raise ValueError("identity grid size must be at least 1")
    grid = np.linspace(float(yaw_min), float(yaw_max), int(n), dtype=np.float64)
    return [float(yaw) for yaw in grid]
