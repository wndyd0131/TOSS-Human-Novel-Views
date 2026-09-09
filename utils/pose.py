from __future__ import annotations

import numpy as np


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
