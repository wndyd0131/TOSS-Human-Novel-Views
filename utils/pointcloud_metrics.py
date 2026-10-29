"""Point cloud metrics (Chamfer L2, F1@τ) for COLMAP fused.ply comparison."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

DeviceLike = Union[str, torch.device]

DEFAULT_MAX_POINTS = 100_000
FUSED_PLY_RELATIVE = "dense/fused.ply"


def resolve_fused_ply(
    root: str | Path,
    identity: str,
    colmap_workspace: str = "colmap_256",
) -> Path:
    """Return path to ``{root}/{identity}/{colmap_workspace}/dense/fused.ply``."""
    return Path(root) / identity / colmap_workspace / FUSED_PLY_RELATIVE


def load_ply_vertices(path: str | Path) -> np.ndarray:
    """
    Load vertex xyz from ASCII or binary PLY (float/double x, y, z).

    Returns:
        ``(N, 3)`` float64 array.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PLY not found: {path}")

    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY (no end_header): {path}")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        fmt = "ascii"
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False

        for line in header_lines:
            if line.startswith("format "):
                parts = line.split()
                if len(parts) >= 2:
                    fmt = parts[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
                in_vertex = True
            elif line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))

        if vertex_count <= 0:
            raise ValueError(f"No vertices in PLY: {path}")

        prop_names = [name for _, name in properties]
        try:
            ix = prop_names.index("x")
            iy = prop_names.index("y")
            iz = prop_names.index("z")
        except ValueError as exc:
            raise ValueError(f"PLY missing x/y/z properties: {path}") from exc

        prop_types = [properties[i][0] for i in (ix, iy, iz)]

        if fmt == "ascii":
            vertices = np.zeros((vertex_count, 3), dtype=np.float64)
            for i in range(vertex_count):
                row = handle.readline()
                if not row:
                    raise ValueError(f"Unexpected EOF reading vertex {i}: {path}")
                parts = row.decode("ascii", errors="replace").split()
                if len(parts) < len(prop_names):
                    raise ValueError(f"Malformed vertex line {i}: {path}")
                vertices[i, 0] = float(parts[ix])
                vertices[i, 1] = float(parts[iy])
                vertices[i, 2] = float(parts[iz])
            return vertices

        if fmt.startswith("binary"):
            little = fmt == "binary_little_endian"
            endian = "<" if little else ">"
            type_map = {
                "char": ("b", 1),
                "uchar": ("B", 1),
                "short": ("h", 2),
                "ushort": ("H", 2),
                "int": ("i", 4),
                "uint": ("I", 4),
                "float": ("f", 4),
                "double": ("d", 8),
            }

            row_struct = struct.Struct(
                "".join(
                    (endian + type_map[t][0] if t in type_map else endian + "f")
                    for t, _ in properties
                )
            )
            if any(t not in type_map for t, _ in properties):
                raise ValueError(f"Unsupported PLY property types: {path}")

            vertices = np.zeros((vertex_count, 3), dtype=np.float64)
            for i in range(vertex_count):
                raw = handle.read(row_struct.size)
                if len(raw) != row_struct.size:
                    raise ValueError(f"Unexpected EOF at vertex {i}: {path}")
                values = row_struct.unpack(raw)
                vertices[i, 0] = float(values[ix])
                vertices[i, 1] = float(values[iy])
                vertices[i, 2] = float(values[iz])
            return vertices

        raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")


def subsample_points(
    points: np.ndarray,
    max_points: int,
    seed: int = 0,
) -> np.ndarray:
    """Uniform random subsample to at most ``max_points`` rows."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def bbox_diagonal(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return 0.0
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _as_tensor(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)


@torch.no_grad()
def _min_distances_query(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    For each row in ``query`` (M, 3), min L2 distance to ``reference`` (N, 3).

    Returns ``(M,)`` on the same device as ``query``.
    """
    if reference.numel() == 0:
        return torch.full((query.shape[0],), float("inf"), device=query.device)
    if query.numel() == 0:
        return torch.empty((0,), device=query.device)

    mins: list[torch.Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        dist = torch.cdist(chunk, reference, p=2)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins, dim=0)


@torch.no_grad()
def compute_chamfer_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
    device: Optional[DeviceLike] = None,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """
    Symmetric Chamfer L2 (lower is better).

    Subsamples both clouds with the same ``seed`` for comparable pair sizes.
    """
    device = torch.device(device or "cpu")
    pred_s = subsample_points(pred, max_points, seed=seed)
    gt_s = subsample_points(gt, max_points, seed=seed + 1)

    pred_t = _as_tensor(pred_s, device)
    gt_t = _as_tensor(gt_s, device)

    d_pred = _min_distances_query(pred_t, gt_t, chunk_size=chunk_size).mean().item()
    d_gt = _min_distances_query(gt_t, pred_t, chunk_size=chunk_size).mean().item()

    return {
        "chamfer_l2_pred_to_gt": float(d_pred),
        "chamfer_l2_gt_to_pred": float(d_gt),
        "chamfer_l2": float((d_pred + d_gt) * 0.5),
    }


@torch.no_grad()
def compute_pointcloud_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    threshold: float,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
    device: Optional[DeviceLike] = None,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Precision / recall / F1 at distance threshold ``threshold`` (higher F1 is better)."""
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    device = torch.device(device or "cpu")
    pred_s = subsample_points(pred, max_points, seed=seed)
    gt_s = subsample_points(gt, max_points, seed=seed + 1)

    pred_t = _as_tensor(pred_s, device)
    gt_t = _as_tensor(gt_s, device)

    if pred_t.shape[0] == 0 or gt_t.shape[0] == 0:
        return {
            "pointcloud_precision": float("nan"),
            "pointcloud_recall": float("nan"),
            "pointcloud_f1": float("nan"),
        }

    d_pred = _min_distances_query(pred_t, gt_t, chunk_size=chunk_size)
    d_gt = _min_distances_query(gt_t, pred_t, chunk_size=chunk_size)

    precision = (d_pred < threshold).float().mean().item()
    recall = (d_gt < threshold).float().mean().item()

    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "pointcloud_precision": float(precision),
        "pointcloud_recall": float(recall),
        "pointcloud_f1": float(f1),
    }


def resolve_f1_threshold(
    gt: np.ndarray,
    *,
    f1_threshold: Optional[float] = None,
    f1_threshold_rel: float = 0.01,
) -> float:
    if f1_threshold is not None:
        return float(f1_threshold)
    diag = bbox_diagonal(gt)
    if diag <= 0:
        raise ValueError("Cannot set relative F1 threshold on empty GT cloud")
    return float(f1_threshold_rel * diag)


def compare_fused_ply_files(
    pred_path: str | Path,
    gt_path: str | Path,
    *,
    compute_chamfer: bool = True,
    compute_f1: bool = True,
    max_points: int = DEFAULT_MAX_POINTS,
    f1_threshold: Optional[float] = None,
    f1_threshold_rel: float = 0.01,
    seed: int = 0,
    device: Optional[DeviceLike] = None,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Load two PLY files and compute selected point cloud metrics."""
    pred = load_ply_vertices(pred_path)
    gt = load_ply_vertices(gt_path)

    out: dict[str, float] = {}
    if compute_chamfer:
        out.update(
            compute_chamfer_distance(
                pred,
                gt,
                max_points=max_points,
                seed=seed,
                device=device,
                chunk_size=chunk_size,
            )
        )
    if compute_f1:
        tau = resolve_f1_threshold(
            gt,
            f1_threshold=f1_threshold,
            f1_threshold_rel=f1_threshold_rel,
        )
        out["f1_threshold"] = tau
        out.update(
            compute_pointcloud_f1(
                pred,
                gt,
                threshold=tau,
                max_points=max_points,
                seed=seed,
                device=device,
                chunk_size=chunk_size,
            )
        )
    return out


def _smoke_test() -> None:
    """Synthetic identical / shifted clouds."""
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    shift = 0.05
    shifted = pts + np.array([shift, 0.0, 0.0])

    cd_same = compute_chamfer_distance(pts, pts, max_points=10, seed=0)
    assert cd_same["chamfer_l2"] < 1e-5

    cd_shift = compute_chamfer_distance(shifted, pts, max_points=10, seed=0)
    assert abs(cd_shift["chamfer_l2"] - shift) < 1e-4

    f1_same = compute_pointcloud_f1(pts, pts, threshold=0.01, max_points=10, seed=0)
    assert f1_same["pointcloud_f1"] > 0.99

    f1_shift = compute_pointcloud_f1(
        shifted, pts, threshold=0.01, max_points=10, seed=0
    )
    assert f1_shift["pointcloud_f1"] < f1_same["pointcloud_f1"]

    tmp = Path("/tmp/toss_ply_smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    ply_a = tmp / "a.ply"
    ply_b = tmp / "b.ply"
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    body = "".join(f"{x} {y} {z}\n" for x, y, z in pts)
    ply_a.write_text(header + body, encoding="ascii")
    ply_b.write_text(
        header
        + "".join(f"{x} {y} {z}\n" for x, y, z in shifted),
        encoding="ascii",
    )
    loaded = load_ply_vertices(ply_a)
    assert loaded.shape == (4, 3)
    metrics = compare_fused_ply_files(ply_b, ply_a, max_points=10, seed=0)
    assert "chamfer_l2" in metrics and "pointcloud_f1" in metrics


if __name__ == "__main__":
    _smoke_test()
    print("pointcloud_metrics smoke test OK")
