from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_RECON_VIEW_INDICES = (0, 1, 2, 3, 8, 12, 13, 15)


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float]:
    """
    Extract pitch (x-rotation) and yaw (y-rotation) from a 3x3 rotation matrix.
    Returns angles in radians.

    Assumes R = Ry(yaw) @ Rx(pitch) @ Rz(roll). Roll is ignored; pitch and yaw
    are read from the third column, which is independent of roll:
      R[:, 2] = [sin(y)*cos(p), -sin(p), cos(y)*cos(p)]
    """
    R = np.asarray(R, dtype=np.float64)
    sin_pitch = np.clip(-R[1, 2], -1.0, 1.0)
    pitch = float(np.arcsin(sin_pitch))
    cos_pitch = float(np.cos(pitch))
    if cos_pitch > 1e-6:
        yaw = float(np.arctan2(R[0, 2], R[2, 2]))
    else:
        yaw = 0.0  # gimbal lock: pitch ≈ ±90°
    return pitch, yaw


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
    delta_yaw = _wrap_angle(tgt_yaw - src_yaw)

    src_dist = float(np.linalg.norm(src_pose_4x4[:3, 3]))
    tgt_dist = float(np.linalg.norm(tgt_pose_4x4[:3, 3]))

    return np.array(
        [delta_pitch, delta_yaw, tgt_dist - src_dist],
        dtype=np.float32,
    )


def _rotation_from_pitch_yaw(pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for ``R = Ry(yaw) @ Rx(pitch)`` (roll = 0)."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    return ry @ rx


def synthesize_target_pose(
    src_pose_4x4: np.ndarray,
    delta_pitch_deg: float = 0.0,
    delta_yaw_deg: float = 0.0,
    delta_dist: float = 0.0,
) -> np.ndarray:
    """Target 4×4 from relative pitch/yaw (degrees) and distance offset vs source."""
    src_pose_4x4 = np.asarray(src_pose_4x4, dtype=np.float64)
    src_pitch, src_yaw = rotation_matrix_to_euler(src_pose_4x4[:3, :3])
    src_dist = float(np.linalg.norm(src_pose_4x4[:3, 3]))

    tgt_pitch = _wrap_angle(src_pitch + np.radians(float(delta_pitch_deg)))
    tgt_yaw = _wrap_angle(src_yaw + np.radians(float(delta_yaw_deg)))
    tgt_dist = max(src_dist + float(delta_dist), 1e-6)

    tgt = np.eye(4, dtype=np.float32)
    tgt[:3, :3] = _rotation_from_pitch_yaw(tgt_pitch, tgt_yaw).astype(np.float32)
    t_src = src_pose_4x4[:3, 3]
    if src_dist > 1e-8:
        tgt[:3, 3] = (t_src / src_dist * tgt_dist).astype(np.float32)
    else:
        tgt[:3, 3] = np.array([0.0, 0.0, tgt_dist], dtype=np.float32)
    return tgt


def delta_pose_for_yaw_degrees(
    src_pose_4x4: np.ndarray,
    yaw_deg: float,
    delta_pitch_deg: float = 0.0,
    delta_dist: float = 0.0,
) -> np.ndarray:
    """Training-format ``[Δpitch, Δyaw, Δdist]`` (rad) for a yaw sweep from ``src_pose``."""
    src_pose_4x4 = np.asarray(src_pose_4x4, dtype=np.float32)
    tgt_pose = synthesize_target_pose(
        src_pose_4x4,
        delta_pitch_deg=delta_pitch_deg,
        delta_yaw_deg=yaw_deg,
        delta_dist=delta_dist,
    )
    return compute_relative_pose(src_pose_4x4, tgt_pose)


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


def select_horizontal_views(
    poses: np.ndarray,
    src_view_idx: int,
    *,
    pitch_threshold_deg: float = 15.0,
) -> list[int]:
    """Return non-source views with small relative pitch, sorted by relative yaw."""
    num_views = len(poses)
    selected: list[tuple[float, int]] = []

    for view_idx in range(num_views):
        if view_idx == src_view_idx:
            continue
        delta = compute_relative_pose(poses[src_view_idx], poses[view_idx])
        pitch_deg = abs(float(np.degrees(delta[0])))
        if pitch_deg <= pitch_threshold_deg:
            yaw_deg = float(np.degrees(delta[1]))
            selected.append((yaw_deg, view_idx))

    selected.sort(key=lambda item: item[0])
    return [view_idx for _, view_idx in selected]
