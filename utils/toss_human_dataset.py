from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.image import preprocess_image
from utils.pose import compute_relative_pose


def preprocess_normal_for_cosine(
    normal_rgb,
    fg_mask=None,
    bg_color=None,
    bg_eps=0.02,
):
    """
    Preprocess normal map for cosine similarity loss.

    The normal PNG has two invalid regions:
      (1) uncropping padding outside the pixel3dmm bbox (constant color)
      (2) around-head pixels inside the bbox that pixel3dmm wrote garbage into
          (excluded via the FG/rembg silhouette mask)

    Args:
        normal_rgb: [H, W, 3] or [3, H, W], values in [0, 1] (standard normal map: RGB = xyz)
        fg_mask: optional [1, H, W] or [H, W] foreground (rembg) mask in [0, 1].
        bg_color: optional [3] iterable in [0, 1] specifying the uncropping padding color.
        bg_eps: max L_inf distance (in [0,1] RGB space) from bg_color to count as padding.

    Returns:
        normal: [3, H, W] L2-normalized, in [-1, 1] range (zeroed where invalid)
        valid_mask: [1, H, W] float, 1 where valid (head region), 0 elsewhere
    """
    if isinstance(normal_rgb, np.ndarray):
        normal = torch.from_numpy(normal_rgb).float()
    else:
        normal = normal_rgb.float()

    if normal.ndim == 3 and normal.shape[0] != 3 and normal.shape[-1] == 3:
        normal = normal.permute(2, 0, 1)  # HWC -> CHW

    if bg_color is None:
        corners = torch.stack([
            normal[:, 0, 0],
            normal[:, 0, -1],
            normal[:, -1, 0],
            normal[:, -1, -1],
        ], dim=0)
        bg_color = corners.median(dim=0).values
    else:
        bg_color = torch.as_tensor(bg_color, dtype=normal.dtype)

    diff = (normal - bg_color.view(3, 1, 1)).abs().amax(dim=0, keepdim=True)
    in_bbox_mask = (diff > bg_eps).float()

    if fg_mask is not None:
        if isinstance(fg_mask, np.ndarray):
            fg_mask = torch.from_numpy(fg_mask).float()
        fg_mask = fg_mask.float()
        if fg_mask.ndim == 2:
            fg_mask = fg_mask.unsqueeze(0)
        fg_bin = (fg_mask > 0.5).float()
        valid_mask = in_bbox_mask * fg_bin
    else:
        valid_mask = in_bbox_mask

    normal = normal * 2.0 - 1.0
    normal = normal / normal.norm(dim=0, keepdim=True).clamp(min=1e-8)
    normal = normal * valid_mask

    return normal, valid_mask


class _SubjectCache:
    """Lazy per-subject cache for poses.npy and correlation arrays."""

    def __init__(
        self,
        load_corr: bool,
        corr_file: str,
        valid_file: str,
        flow_file: str,
    ):
        self.load_corr = load_corr
        self.corr_file = corr_file
        self.valid_file = valid_file
        self.flow_file = flow_file
        self._poses: dict[str, np.ndarray] = {}
        self._corr: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _load_poses(self, sub_path: str) -> np.ndarray:
        poses = np.load(os.path.join(sub_path, "poses.npy"))
        if poses.ndim == 2 and poses.shape[1] == 16:
            poses = poses.reshape(-1, 4, 4)
        return poses

    def get_poses(self, sub_path: str) -> np.ndarray:
        if sub_path not in self._poses:
            self._poses[sub_path] = self._load_poses(sub_path)
        return self._poses[sub_path]

    def get_corr(self, sub_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sub_path not in self._corr:
            corr_arr = np.load(os.path.join(sub_path, self.corr_file))
            valid_arr = np.load(os.path.join(sub_path, self.valid_file))
            flow_arr = np.load(os.path.join(sub_path, self.flow_file))
            self._corr[sub_path] = (corr_arr, valid_arr, flow_arr)
        return self._corr[sub_path]

    def preload(self, sub_paths: list[str]) -> None:
        for sub_path in sub_paths:
            self.get_poses(sub_path)
            if self.load_corr:
                self.get_corr(sub_path)


class TossHumanDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        src_view_idx=3,
        skip_identity=True,
        load_normals=False,
        load_depth=False,
        load_corr=True,
        preload_subject_arrays=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.src_view_idx = src_view_idx
        self.skip_identity = skip_identity
        self.load_normals = load_normals
        self.load_depth = load_depth
        self.load_corr = load_corr

        self.corr_file = "correlation.npy"
        self.valid_file = "valid.npy"
        self.flow_file = "flow_tgt2src.npy"

        self._cache = _SubjectCache(
            load_corr=self.load_corr,
            corr_file=self.corr_file,
            valid_file=self.valid_file,
            flow_file=self.flow_file,
        )

        self.subjects = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.samples = []
        skipped = 0

        for sub in self.subjects:
            sub_path = os.path.join(root_dir, sub)
            poses_path = os.path.join(sub_path, "poses.npy")
            if not os.path.exists(poses_path):
                skipped += 1
                continue

            views = sorted([
                f for f in os.listdir(sub_path)
                if os.path.isfile(os.path.join(sub_path, f))
                and os.path.splitext(f)[1].lower() in [".jpg", ".png"]
                and os.path.splitext(f)[0].isdigit()
            ])

            corr_index_map = {}
            if self.load_corr:
                non_src_sorted = [
                    int(os.path.splitext(f)[0]) for f in views
                    if int(os.path.splitext(f)[0]) != self.src_view_idx
                ]
                corr_index_map = {v: i for i, v in enumerate(non_src_sorted)}

            for view_file in views:
                view_id = view_file.split(".")[0]
                view_index = int(view_id)
                if self.skip_identity and view_index == self.src_view_idx:
                    continue

                mask_file = os.path.join("alpha_maps", f"{view_id}.png")
                normal_file = os.path.join("normals_uncropped", f"{view_id}.png")
                normal_path = os.path.join(sub_path, normal_file)
                depth_file = os.path.join("depth", f"{view_id}.npy")
                depth_path = os.path.join(sub_path, depth_file)

                has_mask = os.path.exists(os.path.join(sub_path, mask_file))
                has_normal = os.path.exists(normal_path) if self.load_normals else True
                has_depth = os.path.exists(depth_path) if self.load_depth else True
                if self.load_corr:
                    has_corr = (
                        os.path.exists(os.path.join(sub_path, self.corr_file))
                        and os.path.exists(os.path.join(sub_path, self.valid_file))
                        and os.path.exists(os.path.join(sub_path, self.flow_file))
                    )
                else:
                    has_corr = True

                if has_mask and has_normal and has_depth and has_corr:
                    self.samples.append({
                        "sub_path": sub_path,
                        "view_file": view_file,
                        "mask_file": mask_file,
                        "normal_file": normal_file,
                        "depth_file": depth_file,
                        "view_index": view_index,
                        "corr_index": corr_index_map.get(view_index),
                    })

        parts = []
        if self.load_normals:
            parts.append("normals")
        if self.load_depth:
            parts.append("depth")
        if self.load_corr:
            parts.append("correlation")
        extra_str = f" (with {', '.join(parts)})" if parts else ""
        print(
            f"[Dataset] Loaded {len(self.samples)} samples{extra_str}, "
            f"skipped {skipped} incomplete subjects"
        )

        if preload_subject_arrays:
            unique_sub_paths = sorted({s["sub_path"] for s in self.samples})
            print(f"[Dataset] Preloading arrays for {len(unique_sub_paths)} subjects...")
            self._cache.preload(unique_sub_paths)
            print("[Dataset] Preload complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sub_path = sample["sub_path"]

        src_path = os.path.join(sub_path, f"{self.src_view_idx:05d}.png")
        src = Image.open(src_path).convert("RGBA")
        src_mask_file = os.path.join("alpha_maps", f"{self.src_view_idx:05d}.png")
        src_mask_path = os.path.join(sub_path, src_mask_file)
        src_mask = Image.open(src_mask_path).convert("L")

        img_path = os.path.join(sub_path, sample["view_file"])
        image = Image.open(img_path).convert("RGBA")

        mask_path = os.path.join(sub_path, sample["mask_file"])
        mask = Image.open(mask_path).convert("L")

        poses = self._cache.get_poses(sub_path)

        delta_pose = compute_relative_pose(
            poses[self.src_view_idx],
            poses[sample["view_index"]],
        )
        assert abs(delta_pose[1]) < 1.0, (
            f"Yaw {delta_pose[1]} seems too large - check units!"
        )
        delta_pose = torch.from_numpy(delta_pose).float()

        src = preprocess_image(src, fg_mask=src_mask)
        image = preprocess_image(image, fg_mask=mask)

        src = torch.from_numpy(src).permute(2, 0, 1).float()
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        if self.transform:
            src = self.transform(src)
            image = self.transform(image)

        mask_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        mask = mask_transform(mask)

        out = {
            "jpg": image,
            "hint": src,
            "mask": mask,
            "delta_pose": delta_pose,
            "subject_id": os.path.basename(sub_path),
            "txt": "",
        }

        if self.load_normals:
            normal_path = os.path.join(sub_path, sample["normal_file"])
            normal_img = Image.open(normal_path).convert("RGB")
            normal_img = normal_img.resize((256, 256), Image.NEAREST)
            normal_np = np.array(normal_img).astype(np.float32) / 255.0
            fg_for_normal = mask[0].numpy()
            normal, normal_valid_mask = preprocess_normal_for_cosine(
                normal_np, fg_mask=fg_for_normal
            )
            out["normal"] = normal
            out["normal_mask"] = normal_valid_mask

        if self.load_depth:
            depth_path = os.path.join(sub_path, sample["depth_file"])
            depth_np = np.load(depth_path).astype(np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np.squeeze()
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            depth_t = torch.from_numpy(depth_np).float().unsqueeze(0)
            depth_t = F.interpolate(
                depth_t.unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            depth_t = torch.clamp(depth_t, min=0.0)
            valid = (depth_t > 1e-6) & (depth_t < 1e6)
            depth_mask = valid.float()
            out["depth"] = depth_t
            out["depth_mask"] = depth_mask

        if self.load_corr:
            ci = sample["corr_index"]
            assert ci is not None, (
                f"No correlation row for view_index {sample['view_index']} in {sub_path} "
                f"(is it the source view?)"
            )
            corr_arr, valid_arr, flow_arr = self._cache.get_corr(sub_path)
            assert 0 <= ci < corr_arr.shape[0], (
                f"corr_index {ci} (view {sample['view_index']}) out of range for "
                f"correlation array {corr_arr.shape} in {sub_path}"
            )
            out["corr_gt"] = torch.from_numpy(corr_arr[ci]).float().unsqueeze(0)
            out["valid_mask"] = torch.from_numpy(
                valid_arr[ci].astype(np.float32)
            ).unsqueeze(0)
            out["flow_tgt2src"] = torch.from_numpy(flow_arr[ci]).float()

        return out
