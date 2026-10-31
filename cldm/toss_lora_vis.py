"""Training-time multiview / GT reconstruction visualization for WandB."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import einops
import numpy as np
import torch
import wandb
from PIL import Image

if TYPE_CHECKING:
    from cldm.toss_lora import TossLoraModule

TRAINING_VIS_DEFAULTS = {
    "every_n_steps": 50,
    "yaws_deg": (-15, -5, 5, 15),
    "ddim_steps": 30,
    "img_scale": 5.0,
    "prompt_scale": 1.0,
    "seed": 42,
    "image_size": 256,
    "precision": "autocast",
    "use_ema_scope": False,
    "source_image": None,
    "source_mask": None,
    "data_root": None,
    "subject_ids": None,
    "gt_view_indices": (2, 3, 15),
    "src_view_idx": 3,
    "gt_enabled": True,
    "pose_delta_mode": None,
}


def install_training_vis_config(module, training_vis: dict | None) -> None:
    """Apply nested ``training_vis`` YAML dict to flat ``module.vis_*`` attributes."""
    overrides = training_vis or {}
    unknown = set(overrides) - set(TRAINING_VIS_DEFAULTS)
    if unknown:
        allowed = ", ".join(sorted(TRAINING_VIS_DEFAULTS))
        raise ValueError(
            f"Unknown training_vis key(s): {sorted(unknown)}. Allowed: {allowed}"
        )

    cfg = {**TRAINING_VIS_DEFAULTS, **overrides}
    module.training_vis = cfg

    module.vis_every_n_steps = int(cfg["every_n_steps"])
    module.vis_yaws_deg = [float(y) for y in cfg["yaws_deg"]]
    module.vis_ddim_steps = int(cfg["ddim_steps"])
    module.vis_img_scale = float(cfg["img_scale"])
    module.vis_prompt_scale = float(cfg["prompt_scale"])
    module.vis_seed = int(cfg["seed"])
    module.vis_image_size = int(cfg["image_size"])
    module.vis_precision = str(cfg["precision"])
    module.vis_use_ema_scope = bool(cfg["use_ema_scope"])
    module.vis_source_image = cfg["source_image"]
    module.vis_source_mask = cfg["source_mask"]
    module.vis_data_root = cfg["data_root"]

    subj_ids = cfg["subject_ids"]
    module.vis_subject_ids = [str(s) for s in subj_ids] if subj_ids else None

    gt_indices = cfg["gt_view_indices"]
    module.vis_gt_view_indices = [int(v) for v in gt_indices] if gt_indices else []
    module.vis_src_view_idx = int(cfg["src_view_idx"])
    module.vis_gt_enabled = bool(cfg["gt_enabled"])
    vis_mode = cfg["pose_delta_mode"]
    if vis_mode is None:
        vis_mode = getattr(module, "pose_delta_mode", "rotation")
    from utils.pose import validate_pose_delta_mode

    module.vis_pose_delta_mode = validate_pose_delta_mode(vis_mode)


def _normal_to_rgb_vis(norm):
    """Map unit normal field to RGB in [0,1] for visualization. norm: [B,3,H,W] or [3,H,W]."""
    if norm.dim() == 4:
        norm = norm[0]
    return torch.clamp((norm + 1) / 2, 0, 1)


class TossLoraTrainingVis:
    def __init__(self, module: TossLoraModule):
        self.module = module

    @property
    def m(self) -> TossLoraModule:
        return self.module

    @staticmethod
    def _hint_tensor_to_pil(hint_tensor):
        """Convert batch hint ``[C,H,W]`` in [0,1] to PIL RGB for inference."""
        arr = hint_tensor.detach().cpu().permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _load_vis_source_from_paths(self):
        """Load fixed RGBA + mask paths (same preprocessing as batch_eval / Colab)."""
        from utils.image import preprocess_image

        m = self.m
        if not m.vis_source_image:
            return
        img = Image.open(m.vis_source_image).convert("RGBA")
        mask = None
        if m.vis_source_mask:
            mask = Image.open(m.vis_source_mask).convert("L")
        arr = preprocess_image(img, fg_mask=mask, size=m.vis_image_size)
        pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        name = str(m.vis_source_image)
        m._vis_source_pils = [pil]
        m._vis_names = [name]
        hint = torch.from_numpy(arr).permute(2, 0, 1).float()
        m._vis_sources = hint.unsqueeze(0)
        print(f"[VIS] Fixed source from disk: {name}")

    @staticmethod
    def _load_poses_npy(subject_dir: str):
        poses_path = os.path.join(subject_dir, "poses.npy")
        if not os.path.isfile(poses_path):
            return None
        poses = np.load(poses_path)
        if poses.ndim == 2 and poses.shape[1] == 16:
            poses = poses.reshape(-1, 4, 4)
        return poses

    def _gt_items_for_subject_dir(self, subject_dir: str):
        """Return list of (view_idx, delta, gt_pil, gt_mask) for one subject folder."""
        from utils.image import preprocess_image
        from utils.pose import relative_delta_pose

        m = self.m
        poses = self._load_poses_npy(subject_dir)
        if poses is None:
            print(f"[VIS][WARN] Missing poses.npy under {subject_dir}; skip GT for this subject.")
            return []

        src_idx = m.vis_src_view_idx
        if src_idx < 0 or src_idx >= len(poses):
            print(
                f"[VIS][WARN] vis_src_view_idx={src_idx} out of range for {subject_dir} "
                f"(num_views={len(poses)})."
            )
            return []

        items = []
        for view_idx in m.vis_gt_view_indices:
            view_idx = int(view_idx)
            img_path = os.path.join(subject_dir, f"{view_idx:05d}.png")
            mask_path = os.path.join(subject_dir, "alpha_maps", f"{view_idx:05d}.png")
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                print(
                    f"[VIS][WARN] Missing GT view {view_idx:05d} under {subject_dir}; skip that view."
                )
                continue
            if view_idx < 0 or view_idx >= len(poses):
                print(f"[VIS][WARN] view_idx={view_idx} out of range for {subject_dir}; skip.")
                continue

            delta = relative_delta_pose(
                poses[src_idx],
                poses[view_idx],
                mode=m.vis_pose_delta_mode,
            ).astype(np.float32)
            gt_img = Image.open(img_path).convert("RGBA")
            gt_mask = Image.open(mask_path).convert("L")
            gt_arr = preprocess_image(gt_img, fg_mask=gt_mask, size=m.vis_image_size)
            gt_pil = Image.fromarray(np.clip(gt_arr * 255.0, 0, 255).astype(np.uint8))
            items.append((view_idx, delta, gt_pil, gt_mask))
        return items

    def _subject_dir_for_id(self, subject_id: str) -> str:
        root = os.path.abspath(self.m.vis_data_root)
        return os.path.join(root, str(subject_id))

    def _subject_dir_for_vis_name(self, name: str) -> str | None:
        """Subject folder containing ``poses.npy`` for multiview vis (matches GT recon)."""
        m = self.m
        if m.vis_data_root:
            candidate = self._subject_dir_for_id(name)
            if os.path.isdir(candidate) and os.path.isfile(
                os.path.join(candidate, "poses.npy")
            ):
                return candidate
            if os.path.isfile(name):
                subject_dir = os.path.dirname(os.path.abspath(name))
                if os.path.isfile(os.path.join(subject_dir, "poses.npy")):
                    return subject_dir
            basename = os.path.basename(str(name).rstrip(os.sep))
            candidate = self._subject_dir_for_id(basename)
            if os.path.isdir(candidate) and os.path.isfile(
                os.path.join(candidate, "poses.npy")
            ):
                return candidate
        if m.vis_source_image:
            subject_dir = os.path.dirname(os.path.abspath(m.vis_source_image))
            if os.path.isfile(os.path.join(subject_dir, "poses.npy")):
                return subject_dir
        return None

    def _vis_gt_subject_id_list(self):
        m = self.m
        if m.vis_subject_ids:
            return list(m.vis_subject_ids)
        if m._vis_subject_ids:
            return list(m._vis_subject_ids)
        if m.vis_data_root and m._vis_names:
            return [str(n) for n in m._vis_names]
        return []

    def _build_vis_gt_bundles(self):
        """One bundle per subject: {subject_name, items}. Same subjects as yaw vis when possible."""
        m = self.m
        if not m.vis_gt_enabled or not m.vis_gt_view_indices:
            return []

        bundles = []
        for subject_id in self._vis_gt_subject_id_list():
            subject_dir = self._subject_dir_for_id(subject_id)
            if not os.path.isdir(subject_dir):
                print(f"[VIS][WARN] Subject dir not found: {subject_dir}; skip GT.")
                continue
            items = self._gt_items_for_subject_dir(subject_dir)
            if items:
                view_ids = [v for v, _, _, _ in items]
                print(
                    f"[VIS] GT recon subject {subject_id} (src view {m.vis_src_view_idx}): "
                    f"targets {view_ids}"
                )
                bundles.append({"subject_name": str(subject_id), "items": items})

        if bundles:
            return bundles

        if m.vis_source_image:
            subject_dir = os.path.dirname(os.path.abspath(m.vis_source_image))
            items = self._gt_items_for_subject_dir(subject_dir)
            if items:
                name = os.path.basename(subject_dir.rstrip(os.sep))
                view_ids = [v for v, _, _, _ in items]
                print(
                    f"[VIS] GT recon from vis_source_image subject {name} "
                    f"(src view {m.vis_src_view_idx}): targets {view_ids}"
                )
                return [{"subject_name": name, "items": items}]

        if not self._vis_gt_subject_id_list() and not m.vis_source_image:
            print(
                "[VIS][WARN] GT recon needs vis_data_root + vis_subject_ids (or set_vis_sources "
                "subject_ids / names), or vis_source_image."
            )
        return []

    def _ensure_vis_gt_bundles(self):
        m = self.m
        if m._vis_gt_bundles is None:
            m._vis_gt_bundles = self._build_vis_gt_bundles()
        elif not m._vis_gt_bundles and self._vis_gt_subject_id_list():
            m._vis_gt_bundles = self._build_vis_gt_bundles()
        return m._vis_gt_bundles

    def _load_vis_sources_from_dataset_root(self):
        """Load fixed yaw sources from ``vis_data_root`` / ``vis_subject_ids`` (same as GT subjects)."""
        from utils.image import preprocess_image

        m = self.m
        if not m.vis_data_root or not m.vis_subject_ids:
            return
        if m._vis_source_pils is not None:
            return

        hints = []
        names = []
        src_idx = m.vis_src_view_idx
        for subject_id in m.vis_subject_ids:
            subject_dir = self._subject_dir_for_id(subject_id)
            img_path = os.path.join(subject_dir, f"{src_idx:05d}.png")
            mask_path = os.path.join(subject_dir, "alpha_maps", f"{src_idx:05d}.png")
            if not os.path.isfile(img_path):
                print(f"[VIS][WARN] Missing source {img_path}; skip vis subject {subject_id}.")
                continue
            mask = Image.open(mask_path).convert("L") if os.path.isfile(mask_path) else None
            img = Image.open(img_path).convert("RGBA")
            arr = preprocess_image(img, fg_mask=mask, size=m.vis_image_size)
            hints.append(torch.from_numpy(arr).permute(2, 0, 1).float())
            names.append(str(subject_id))

        if not hints:
            print("[VIS][WARN] No vis sources loaded from vis_data_root / vis_subject_ids.")
            return

        self.set_vis_sources(torch.stack(hints, dim=0), names=names, subject_ids=names)
        print(f"[VIS] Loaded {len(names)} yaw/GT source(s) from {m.vis_data_root}")

    def prepare(self):
        """Called from ``TossLoraModule.on_train_start`` — load vis sources and GT bundles."""
        self._load_vis_source_from_paths()
        self._load_vis_sources_from_dataset_root()
        self._ensure_vis_gt_bundles()

    def set_vis_sources(self, images, names=None, subject_ids=None):
        """Fix the source faces used for training-time multiview visualization."""
        m = self.m
        if images.ndim == 4 and images.shape[-1] == 3:
            images = einops.rearrange(images, "b h w c -> b c h w")
        m._vis_sources = images.detach().float().cpu().clone()
        m._vis_names = list(names) if names is not None else [str(i) for i in range(images.shape[0])]
        if subject_ids is not None:
            m._vis_subject_ids = [str(s) for s in subject_ids]
        elif m.vis_data_root:
            m._vis_subject_ids = list(m._vis_names)
        else:
            m._vis_subject_ids = None
        m._vis_source_pils = [
            self._hint_tensor_to_pil(m._vis_sources[i]) for i in range(m._vis_sources.shape[0])
        ]
        m._vis_gt_bundles = None
        print(f"[VIS] Fixed {len(m._vis_names)} visualization source(s): {m._vis_names}")

    def log_multiview_vis(self, batch, wandb_log):
        """Sample fixed sources via ``utils.inference.generate_batch`` (Colab / eval path)."""
        from utils.inference import generate_batch
        from utils.pose import delta_pose_for_yaw_degrees

        m = self.m
        if m._vis_sources is None and m._vis_source_pils is None:
            src = batch[m.control_key][:1]
            self.set_vis_sources(
                src,
                names=[str(batch["subject_id"][0])] if "subject_id" in batch else None,
            )
            print("[VIS][WARN] vis_source_image / set_vis_sources not set; using first training batch hint.")

        step = int(m.global_step)
        source_pils = m._vis_source_pils
        if source_pils is None:
            source_pils = [self._hint_tensor_to_pil(t) for t in m._vis_sources]

        unet = m.model.diffusion_model
        was_training = unet.training
        unet.eval()
        normal_gt_and_preds = []
        try:
            with torch.no_grad():
                wandb_images = []
                for src_pil, name in zip(source_pils, m._vis_names):
                    src_show = np.asarray(src_pil.convert("RGB"), dtype=np.float32) / 255.0
                    src_show = torch.from_numpy(src_show).permute(2, 0, 1)
                    wandb_images.append(wandb.Image(
                        src_show,
                        caption=f"Step {step} | {name} | SOURCE",
                    ))

                    subject_dir = self._subject_dir_for_vis_name(name)
                    if subject_dir is None:
                        print(
                            "[VIS][WARN] No poses.npy for multiview vis "
                            f"(name={name}); set vis_data_root or vis_source_image. Skip preds."
                        )
                        continue
                    poses = self._load_poses_npy(subject_dir)
                    if poses is None:
                        print(
                            f"[VIS][WARN] Missing poses.npy under {subject_dir}; skip preds."
                        )
                        continue
                    src_idx = m.vis_src_view_idx
                    if src_idx < 0 or src_idx >= len(poses):
                        print(
                            f"[VIS][WARN] vis_src_view_idx={src_idx} out of range "
                            f"(num_views={len(poses)}); skip preds."
                        )
                        continue
                    src_pose = poses[src_idx]
                    delta_pose_list = [
                        delta_pose_for_yaw_degrees(
                            src_pose,
                            yaw_deg,
                            mode=m.vis_pose_delta_mode,
                        )
                        for yaw_deg in m.vis_yaws_deg
                    ]
                    preds = generate_batch(
                        m,
                        src_pil,
                        prompt="",
                        delta_pose_list=delta_pose_list,
                        h=m.vis_image_size,
                        w=m.vis_image_size,
                        ddim_steps=m.vis_ddim_steps,
                        ddim_eta=0.0,
                        img_scale=m.vis_img_scale,
                        prompt_scale=m.vis_prompt_scale,
                        precision=m.vis_precision,
                        use_ema_scope=m.vis_use_ema_scope,
                        shared_noise=True,
                        seed=m.vis_seed,
                    )
                    for yaw_deg, pred in zip(m.vis_yaws_deg, preds):
                        pred_t = torch.from_numpy(
                            np.asarray(pred.convert("RGB"), dtype=np.float32) / 255.0
                        ).permute(2, 0, 1)
                        wandb_images.append(wandb.Image(
                            pred_t,
                            caption=(
                                f"Step {step} | {name} | Yaw: {yaw_deg:g}° | "
                                f"cfg={m.vis_img_scale:g} | seed={m.vis_seed}"
                            ),
                        ))

                if "normal" in batch and "normal_mask" in batch:
                    gt_n = batch["normal"][:1].to(m.device)
                    normal_gt_and_preds.append(wandb.Image(
                        _normal_to_rgb_vis(gt_n),
                        caption=f"Step {step} | GT normal (target view)",
                    ))
                    nm = batch["normal_mask"][:1].to(m.device).float()
                    if nm.ndim == 4:
                        nm = nm[0]
                    mask_row = nm[0] if nm.ndim == 3 else nm
                    m_vis = torch.clamp(mask_row, 0, 1).unsqueeze(0).expand(3, -1, -1)
                    normal_gt_and_preds.append(wandb.Image(
                        m_vis,
                        caption=f"Step {step} | GT normal_mask",
                    ))
        finally:
            unet.train(was_training)

        wandb_log["multiview_predictions"] = wandb_images
        if normal_gt_and_preds:
            wandb_log["normal_gt_and_preds"] = normal_gt_and_preds
        print(f"[VIS] Logged multiview (+ normals) at step {step}")

    def log_gt_recon_vis(self, wandb_log):
        """GT-aligned reconstructions via ``generate_batch(delta_pose_list=...)``."""
        bundles = self._ensure_vis_gt_bundles()
        if not bundles:
            return

        from utils.eval_metrics import compute_psnr
        from utils.image import preprocess_image, resize_mask
        from utils.inference import generate_batch

        m = self.m
        source_pils = m._vis_source_pils
        if source_pils is None and m._vis_sources is not None:
            source_pils = [self._hint_tensor_to_pil(t) for t in m._vis_sources]
        if not source_pils:
            print("[VIS][WARN] No vis source for GT recon; skipping.")
            return

        step = int(m.global_step)
        unet = m.model.diffusion_model
        was_training = unet.training
        unet.eval()
        wandb_images = []
        n_views = 0
        try:
            with torch.no_grad():
                for bundle_idx, bundle in enumerate(bundles):
                    subject = bundle["subject_name"]
                    items = bundle["items"]
                    if bundle_idx >= len(source_pils):
                        print(
                            f"[VIS][WARN] No yaw source for GT subject {subject} "
                            f"(index {bundle_idx}); skip."
                        )
                        continue
                    src_pil = source_pils[bundle_idx]
                    delta_pose_list = [d for _, d, _, _ in items]
                    preds = generate_batch(
                        m,
                        src_pil,
                        prompt="",
                        delta_pose_list=delta_pose_list,
                        h=m.vis_image_size,
                        w=m.vis_image_size,
                        ddim_steps=m.vis_ddim_steps,
                        ddim_eta=0.0,
                        img_scale=m.vis_img_scale,
                        prompt_scale=m.vis_prompt_scale,
                        precision=m.vis_precision,
                        use_ema_scope=m.vis_use_ema_scope,
                        shared_noise=True,
                        seed=m.vis_seed,
                    )

                    src_show = np.asarray(src_pil.convert("RGB"), dtype=np.float32) / 255.0
                    src_show = torch.from_numpy(src_show).permute(2, 0, 1)
                    wandb_images.append(wandb.Image(
                        src_show,
                        caption=(
                            f"Step {step} | {subject} | SOURCE (view {m.vis_src_view_idx})"
                        ),
                    ))

                    for (view_idx, delta, gt_pil, gt_mask_pil), pred in zip(items, preds):
                        gt_t = torch.from_numpy(
                            np.asarray(gt_pil.convert("RGB"), dtype=np.float32) / 255.0
                        ).permute(2, 0, 1)
                        pred_t = torch.from_numpy(
                            np.asarray(pred.convert("RGB"), dtype=np.float32) / 255.0
                        ).permute(2, 0, 1)
                        pitch_deg = float(np.degrees(delta[0]))
                        yaw_deg = float(np.degrees(delta[1]))
                        dist = float(delta[2])
                        pose_str = (
                            f"Δpitch={pitch_deg:.2f}° Δyaw={yaw_deg:.2f}° Δdist={dist:.4f}"
                        )
                        wandb_images.append(wandb.Image(
                            gt_t,
                            caption=(
                                f"Step {step} | {subject} | GT view {view_idx:05d} | {pose_str}"
                            ),
                        ))
                        wandb_images.append(wandb.Image(
                            pred_t,
                            caption=(
                                f"Step {step} | {subject} | PRED view {view_idx:05d} | "
                                f"{pose_str} | cfg={m.vis_img_scale:g} | seed={m.vis_seed}"
                            ),
                        ))

                        gt_np = np.asarray(gt_pil.convert("RGB"), dtype=np.float32) / 255.0
                        pred_np = preprocess_image(
                            pred.convert("RGB"),
                            fg_mask=gt_mask_pil,
                            size=m.vis_image_size,
                        )
                        mask_np = resize_mask(gt_mask_pil, size=m.vis_image_size)
                        fg_psnr = compute_psnr(pred_np, gt_np, mask=mask_np)
                        wandb_log[f"vis_gt/fg_psnr_{subject}_view_{view_idx}"] = fg_psnr
                        n_views += 1
        finally:
            unet.train(was_training)

        wandb_log["gt_recon_predictions"] = wandb_images
        print(
            f"[VIS] Logged GT recon ({len(bundles)} subject(s), {n_views} view(s)) at step {step}"
        )


def maybe_log_training_vis(
    training_vis: TossLoraTrainingVis,
    batch,
    wandb_log: dict,
    step: int,
) -> None:
    m = training_vis.m
    if m.vis_every_n_steps <= 0 or step % m.vis_every_n_steps != 0:
        return
    training_vis.log_multiview_vis(batch, wandb_log)
    training_vis.log_gt_recon_vis(wandb_log)
