"""Tests for pose_delta_mode (rotation | spherical)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POSE_PATH = _REPO_ROOT / "utils" / "pose.py"


def _load_pose_module():
    spec = importlib.util.spec_from_file_location("_pose_under_test", _POSE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["_pose_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


pose = _load_pose_module()


def test_validate_pose_delta_mode_rejects_unknown():
    try:
        pose.validate_pose_delta_mode("euler")
    except ValueError as exc:
        assert "pose_delta_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_spherical_yaw_sweep_negates_yaw_vs_from_relative_deg():
    delta = pose.delta_pose_for_yaw_degrees(
        np.eye(4, dtype=np.float32),
        15.0,
        mode="spherical",
    )
    raw = pose.get_T_from_relative_deg(0.0, 15.0, 0.0)
    expected = pose._finalize_spherical_delta(raw, pose_enc="freq")
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-6)
    assert float(delta[1]) < 0.0


def test_relative_delta_pose_modes_on_c2w_poses():
    poses_path = _REPO_ROOT / "datasets" / "nersemble_to_portrait4d_v2" / "017" / "poses.npy"
    if not poses_path.is_file():
        return

    poses = np.load(poses_path).reshape(-1, 4, 4)
    src, tgt = 0, 1
    rot = pose.relative_delta_pose(poses[src], poses[tgt], mode="rotation")
    sph = pose.relative_delta_pose(poses[src], poses[tgt], mode="spherical")
    assert rot.shape == (3,) and sph.shape == (3,)
    assert not np.allclose(rot, sph, atol=1e-4)
    assert -math.pi <= float(sph[1]) <= math.pi
    if float(rot[1]) != 0.0:
        assert math.copysign(1.0, float(rot[1])) == math.copysign(1.0, float(sph[1]))


def test_c2w_matches_inv_world2cam():
    json_path = (
        _REPO_ROOT
        / "datasets"
        / "nersemble_images"
        / "017"
        / "calibration"
        / "camera_params.json"
    )
    poses_path = _REPO_ROOT / "datasets" / "nersemble_to_portrait4d_v2" / "017" / "poses.npy"
    if not json_path.is_file() or not poses_path.is_file():
        return

    import json

    with open(json_path, encoding="utf-8") as f:
        w2c_dict = json.load(f)["world_2_cam"]
    serials = sorted(w2c_dict.keys())
    w2c = np.stack([np.array(w2c_dict[s], dtype=np.float64) for s in serials])
    c2w = np.linalg.inv(w2c)
    poses = np.load(poses_path).reshape(-1, 4, 4)
    max_diff = float(np.max(np.abs(poses - c2w)))
    assert max_diff < 1e-5


if __name__ == "__main__":
    test_validate_pose_delta_mode_rejects_unknown()
    test_spherical_yaw_sweep_negates_yaw_vs_from_relative_deg()
    test_relative_delta_pose_modes_on_c2w_poses()
    test_c2w_matches_inv_world2cam()
    print("tests/test_pose_delta_mode.py: ok")
