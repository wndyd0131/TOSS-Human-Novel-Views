import math
import torch
from cldm.toss import TOSS
from ldm.models.diffusion.ddim import DDIMSampler
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb
import lpips
import pytz
from datetime import datetime
from contextlib import nullcontext

import einops
import numpy as np
from PIL import Image

from cldm.arcface_torch_wrapper import create_frozen_arcface_backbone, preprocess_arcface_input
# CRITICAL FIX: Completely disable gradient checkpointing to fix LoRA gradient flow
# The custom CheckpointFunction doesn't properly handle PEFT's dynamically added parameters
import ldm.modules.diffusionmodules.util as ldm_util
_original_checkpoint = ldm_util.checkpoint
def _no_checkpoint(func, inputs, params, flag):
    """Always bypass checkpointing - just run the function directly"""
    return func(*inputs)
ldm_util.checkpoint = _no_checkpoint
print("[PATCH] Disabled gradient checkpointing in ldm_util.checkpoint")

run = None

def _cosine_similarity_loss(pred_normals, gt_normals, mask, eps=1e-8):
    """Masked cosine similarity loss. pred, gt: [B,3,H,W] L2-normalized. mask: [B,1,H,W]."""
    cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True).clamp(-1, 1)
    loss = 1 - cos_sim  # [B,1,H,W]
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + eps)
    return loss.mean()

def _normal_to_rgb_vis(norm):
    """Map unit normal field to RGB in [0,1] for visualization. norm: [B,3,H,W] or [3,H,W]. Returns [3,H,W]."""
    if norm.dim() == 4:
        norm = norm[0]
    return torch.clamp((norm + 1) / 2, 0, 1)

def _grayscale_sobel_3ch(rgb_01):
    """rgb_01: [B, 3, H, W] in [0, 1]. Returns 3-channel Sobel magnitude in [0, 1]."""
    from kornia.color import rgb_to_grayscale
    from kornia.filters import sobel
    gray = rgb_to_grayscale(rgb_01)         # [B, 1, H, W]
    edge = sobel(gray)                      # [B, 1, H, W] L2 gradient magnitude
    return edge.expand(-1, 3, -1, -1).clamp(0.0, 1.0)

def _grad_norm_l2(loss, params):
    """L2 norm of gradients of ``loss`` w.r.t. ``params`` without modifying ``.grad``."""
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    sq = sum(g.detach().pow(2).sum() for g in grads if g is not None)
    return sq.sqrt()

def _grad_probe(loss, named_tensors):
    """Per-tensor grad norms of ``loss``.

    Keeps three outcomes distinct that a plain norm collapses into ``0``:
    ``no_requires_grad`` (never tracked), ``None`` (tracked but unreached), and an exact ``0.0000e+00``.
    """
    live = [(n, t) for n, t in named_tensors if t.requires_grad]
    result = {n: "no_requires_grad" for n, t in named_tensors if not t.requires_grad}
    if live:
        grads = torch.autograd.grad(
            loss, [t for _, t in live],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        for (n, _), g in zip(live, grads):
            result[n] = "None" if g is None else f"{g.detach().norm().item():.4e}"
    return " ".join(f"{n}={result[n]}" for n, _ in named_tensors)

class TossLoraModule(TOSS):
    def __init__(
        self,
        lora_config_params,
        *args,
        normal_estimator_path="hf:clay3d/omnidata",
        geometry_loss_weight=0.0,
        identity_loss_weight=0.0,
        identity_t_cut=200,
        arcface_ckpt_path=None,
        arcface_spatial_mode="cover_center",
        dists_loss_weight=0.0,
        dists_t_cut=200,
        lambda_corr=0.01,
        corr_embed_dim=128,
        corr_proj_trainable=True,
        corr_debug_steps=200,
        corr_flow_src_size=32,
        vis_every_n_steps=50,
        vis_yaws_deg=(-15, -5, 5, 15),
        vis_ddim_steps=30,
        vis_img_scale=3.0,
        vis_prompt_scale=1.0,
        vis_seed=40,
        vis_image_size=256,
        vis_precision="autocast",
        vis_use_ema_scope=False,
        vis_source_image=None,
        vis_source_mask=None,
        **kwargs,
    ):
        kwargs.pop("lora_config_params", None)  # consumed by us
        self.normal_estimator_path = kwargs.pop("normal_estimator_path", normal_estimator_path)
        self.geometry_loss_weight = kwargs.pop("geometry_loss_weight", geometry_loss_weight)
        self.identity_loss_weight = float(kwargs.pop("identity_loss_weight", identity_loss_weight))
        self.identity_t_cut = int(kwargs.pop("identity_t_cut", identity_t_cut))
        self.arcface_ckpt_path = kwargs.pop("arcface_ckpt_path", arcface_ckpt_path)
        self.arcface_spatial_mode = kwargs.pop("arcface_spatial_mode", arcface_spatial_mode)
        self.dists_loss_weight = float(kwargs.pop("dists_loss_weight", dists_loss_weight))
        self.dists_t_cut = int(kwargs.pop("dists_t_cut", dists_t_cut))
        # Correlation feature regularization config
        self.lambda_corr = float(kwargs.pop("lambda_corr", lambda_corr))
        self.corr_embed_dim = int(kwargs.pop("corr_embed_dim", corr_embed_dim))
        self.corr_proj_trainable = bool(kwargs.pop("corr_proj_trainable", corr_proj_trainable))
        self.corr_debug_steps = int(kwargs.pop("corr_debug_steps", corr_debug_steps))
        # corr_flow_src_size not None: flow_tgt2src is source latent indices (see correlation_gt_2).
        # None: flow is already a grid_sample grid in [-1, 1].
        src_size = kwargs.pop("corr_flow_src_size", corr_flow_src_size)
        if src_size is None:
            self.corr_flow_src_size = None
        elif isinstance(src_size, int):
            self.corr_flow_src_size = (src_size, src_size)
        else:
            self.corr_flow_src_size = (int(src_size[0]), int(src_size[1]))
        self._corr_flow_warned = False
        # Training-time multiview visualization (defaults match utils.inference.generate_batch)
        self.vis_every_n_steps = int(kwargs.pop("vis_every_n_steps", vis_every_n_steps))
        self.vis_yaws_deg = [float(y) for y in kwargs.pop("vis_yaws_deg", vis_yaws_deg)]
        self.vis_ddim_steps = int(kwargs.pop("vis_ddim_steps", vis_ddim_steps))
        self.vis_img_scale = float(kwargs.pop("vis_img_scale", vis_img_scale))
        self.vis_prompt_scale = float(kwargs.pop("vis_prompt_scale", vis_prompt_scale))
        self.vis_seed = int(kwargs.pop("vis_seed", vis_seed))
        self.vis_image_size = int(kwargs.pop("vis_image_size", vis_image_size))
        self.vis_precision = str(kwargs.pop("vis_precision", vis_precision))
        self.vis_use_ema_scope = bool(kwargs.pop("vis_use_ema_scope", vis_use_ema_scope))
        self.vis_source_image = kwargs.pop("vis_source_image", vis_source_image)
        self.vis_source_mask = kwargs.pop("vis_source_mask", vis_source_mask)
        super().__init__(*args, **kwargs)
        self._default_h = self.vis_image_size
        self._default_w = self.vis_image_size
        self._default_use_ema_scope = self.vis_use_ema_scope
        self._default_pose_enc = "freq"
        self._normal_estimator = None
        self._arcface_backbone = None
        self._dists_model = None
        self._sampler = None
        # Plain attributes (not buffers) so they are never written into checkpoints.
        self._vis_sources = None
        self._vis_names = None
        self._vis_source_pils = None

        global run
        kst = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity="wndyd0131-sungkyunkwan-university",
            project="toss-lora",
            name=f"toss-lora_{now_kst}",
            config={
                # LoRA config
                "lora_r": lora_config_params.get("r", 16),
                "lora_alpha": lora_config_params.get("lora_alpha", 16),
                "lora_dropout": lora_config_params.get("lora_dropout", 0.0),
                "target_modules": lora_config_params.get("target_modules", []),
                # Model config
                "architecture": "TOSS + LoRA",
                "base_model": "Stable Diffusion UNet",
                # Identity loss (ArcFace)
                "identity_loss_weight": self.identity_loss_weight,
                "identity_t_cut": self.identity_t_cut,
                "arcface_ckpt_path": self.arcface_ckpt_path,
                "arcface_spatial_mode": self.arcface_spatial_mode,
                # DISTS perceptual loss (image + Sobel)
                "dists_loss_weight": self.dists_loss_weight,
                "dists_t_cut": self.dists_t_cut,
                # Correlation feature regularization
                "lambda_corr": self.lambda_corr,
                "corr_embed_dim": self.corr_embed_dim,
                "corr_proj_trainable": self.corr_proj_trainable,
                "corr_flow_src_size": self.corr_flow_src_size,
            },
        )

        unet = self.model.diffusion_model
        def disable_all_ckpt(m):
            for attr in ["use_checkpoint", "checkpoint", "use_checkpointing"]:
                if hasattr(m, attr):
                    setattr(m, attr, False)
        unet.apply(disable_all_ckpt)

        # 1. Freeze the base model
        self.requires_grad_(False)

        # 2. Configure LoRA (Critical)
        peft_config = LoraConfig(**lora_config_params)
        self.model.diffusion_model = get_peft_model(self.model.diffusion_model, peft_config)

        # CRITICAL: Disable checkpointing AGAIN after PEFT wrapping
        # PEFT changes module hierarchy, must ensure checkpointing is disabled
        self.model.diffusion_model.apply(disable_all_ckpt)

        # 3. Unfreeze PoseNet params
        pose_net_count = 0
        for n, p in self.model.diffusion_model.named_parameters():
            if "pose_net" in n:
                p.requires_grad = True
                pose_net_count += 1
                print(f"[INIT] Unfreezing pose_net param: {n}, shape={p.shape}")
            if "vae_proj" in n:
                p.requires_grad = True
        print(f"[INIT] Unfroze {pose_net_count} pose_net parameters")
        print(f"[INIT] Unfroze vae_proj parameters")
        
        # 4. Explicitly enable gradients for LoRA parameters (in case PEFT didn't)
        lora_count = 0
        lora_params_info = []
        for n, p in self.model.diffusion_model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
                lora_count += 1
                lora_params_info.append((n, p.shape, p.requires_grad))

            # Also enable output layers if needed
            if "base_model.model.out." in n:
                p.requires_grad = True

        print(f"[INIT] Enabled requires_grad for {lora_count} LoRA parameters")
        for name, shape, req_grad in lora_params_info:
            print(f"  -> {name}: shape={shape}, requires_grad={req_grad}")

        self.model.diffusion_model.print_trainable_parameters()
        
        # Loss weights for hybrid loss
        # self.perceptual_weight = 0.0  # Weight for perceptual loss
        self.mse_weight = 1.0  # Small MSE component for stability
        self.mask_min_weight = 0.2  # Soft mask: background contributes 20%, face contributes 100%

        # 5. Correlation feature regularization
        # Shared 1x1 projection: last output-block feature (320ch @ 32x32) -> low-dim corr embedding.
        # Created AFTER requires_grad_(False), so it is trainable by default. Ablate by freezing.
        self.corr_proj = nn.Conv2d(320, self.corr_embed_dim, kernel_size=1)
        self.corr_proj.requires_grad_(self.corr_proj_trainable)
        print(f"[INIT] corr_proj: 320 -> {self.corr_embed_dim} (trainable={self.corr_proj_trainable}, lambda_corr={self.lambda_corr})")
        print(
            f"[INIT] corr flow: {self.corr_flow_src_size} "
            f"(None = pre-normalized grid; else latent-index -> grid like correlation_gt_2)"
        )

        # Forward hook captures the last output-block feature [2B, 320, 32, 32] each forward.
        # [:B] = target (noised) branch, [B:] = source branch (see apply_model batch-doubling).
        self._corr_feat = {}

        def _corr_hook(module, inp, out):
            self._corr_feat["feat"] = out

        unet = self.model.diffusion_model
        if hasattr(unet, "base_model"):  # unwrap PEFT wrapper
            unet = unet.base_model.model
        unet.output_blocks[-1].register_forward_hook(_corr_hook)
        print(f"[INIT] Registered corr feature hook on output_blocks[-1]")

    def _decode_first_stage_train(self, z):
        """Like ``decode_first_stage`` but without grad-disabled decorator so x0 gradients reach RGB."""
        z = (1.0 / self.scale_factor) * z
        return self.first_stage_model.decode(z)

    def _gt_rgb_01_from_batch(self, batch):
        """Target view RGB in [0, 1], CHW — same layout as ``TOSS.get_input`` before ``*2-1`` encode."""
        x = batch[self.first_stage_key]
        x = x.to(self.device)
        if x.ndim == 4 and x.shape[-1] == 3:
            x = einops.rearrange(x, "b h w c -> b c h w")
        return x.clamp(0.0, 1.0)

    def _ensure_arcface_backbone(self):
        if self._arcface_backbone is None:
            if not self.arcface_ckpt_path:
                raise ValueError("identity_loss_weight > 0 requires arcface_ckpt_path to a backbone .pth")
            self._arcface_backbone = create_frozen_arcface_backbone(self.arcface_ckpt_path).to(self.device)
        self._arcface_backbone.eval()
        return self._arcface_backbone

    def _ensure_dists(self):
        if self._dists_model is None:
            from DISTS_pytorch import DISTS
            m = DISTS().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
            self._dists_model = m
        self._dists_model.eval()
        return self._dists_model

    def _identity_autocast_ctx(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    @property
    def normal_estimator(self):
        """Lazy-load frozen DPT-Hybrid normal estimator."""
        if self._normal_estimator is None:
            from ldm.modules.midas.api import DPTNormalInference
            self._normal_estimator = DPTNormalInference(self.normal_estimator_path).to(self.device)
            print(f"[INIT] Loaded DPT-Hybrid normal estimator from {self.normal_estimator_path}")
        return self._normal_estimator

    @property
    def sampler(self):
        """Lazy DDIM sampler for ``generate_batch`` (WandB preview)."""
        if self._sampler is None:
            from ldm.models.diffusion.ddim import DDIMSampler
            self._sampler = DDIMSampler(self)
        return self._sampler

    @staticmethod
    def _hint_tensor_to_pil(hint_tensor):
        """Convert batch hint ``[C,H,W]`` in [0,1] to PIL RGB for inference."""
        arr = hint_tensor.detach().cpu().permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def on_save_checkpoint(self, checkpoint):
        # We override this to prevent the parent class from 
        # trying to call self.embedding_manager.save()
        
        # If you want to keep the checkpoint small (LoRA only),
        # you can clear the base model weights from the checkpoint:
        # checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if "lora" in k or "pose_net" in k}
        pass

    def _load_vis_source_from_paths(self):
        """Load fixed RGBA + mask paths (same preprocessing as batch_eval / Colab)."""
        from utils.image import preprocess_image

        if not self.vis_source_image:
            return
        img = Image.open(self.vis_source_image).convert("RGBA")
        mask = None
        if self.vis_source_mask:
            mask = Image.open(self.vis_source_mask).convert("L")
        arr = preprocess_image(img, fg_mask=mask, size=self.vis_image_size)
        pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        name = str(self.vis_source_image)
        self._vis_source_pils = [pil]
        self._vis_names = [name]
        hint = torch.from_numpy(arr).permute(2, 0, 1).float()
        self._vis_sources = hint.unsqueeze(0)
        print(f"[VIS] Fixed source from disk: {name}")

    def on_train_start(self):
        """Called by PyTorch Lightning when training starts - log trainer config"""
        self._trainable_params = [p for p in self.parameters() if p.requires_grad]
        self._load_vis_source_from_paths()
        if run is not None and self.trainer is not None:
            run.config.update({
                "max_epochs": self.trainer.max_epochs,
                "max_steps": self.trainer.max_steps,
                "batch_size": self.trainer.datamodule.batch_size if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule else None,
                "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
                "gradient_clip_val": self.trainer.gradient_clip_val,
                "learning_rate": self.learning_rate,
                # Training config from kwargs if available
                "image_size": self.image_size,
                "timesteps": self.num_timesteps,
                # Loss config
                "loss_type": "mse",
                "mse_weight": self.mse_weight,
                "identity_loss_weight": self.identity_loss_weight,
                "identity_t_cut": self.identity_t_cut,
                "dists_loss_weight": self.dists_loss_weight,
                "dists_t_cut": self.dists_t_cut,
                "lambda_corr": self.lambda_corr,
                "corr_embed_dim": self.corr_embed_dim,
                "corr_proj_trainable": self.corr_proj_trainable,
                "corr_flow_src_size": self.corr_flow_src_size,
            }, allow_val_change=True)

    @staticmethod
    def _as_bchw(t):
        """Ensure a per-location map is [B, 1, H, W]. Accepts [B, H, W] or [B, 1, H, W]."""
        if t.dim() == 3:
            t = t.unsqueeze(1)
        return t

    def _debug_corr_grad(self, loss_corr, feat, corr_pred, p_tgt, p_src, valid_mask, numer):
        """Walk the corr gradient chain (loss -> corr_pred -> p_tgt/p_src -> corr_proj) to find where it dies."""
        step = int(self.global_step)
        with torch.no_grad():
            print(
                f"[CORR-GRAD step={step}] "
                f"feat(requires_grad={feat.requires_grad},grad_fn={feat.grad_fn is not None}) | "
                f"loss(requires_grad={loss_corr.requires_grad},grad_fn={type(loss_corr.grad_fn).__name__ if loss_corr.grad_fn is not None else None}) | "
                f"numer={numer.item():.6e} denom={valid_mask.sum().item():.4f} | "
                f"valid_mask(n>0={int((valid_mask > 0).sum())},n>0.5={int((valid_mask > 0.5).sum())},"
                f"numel={valid_mask.numel()},min={valid_mask.min().item():.4f},max={valid_mask.max().item():.4f})"
            )

        if not loss_corr.requires_grad:
            print(f"[CORR-GRAD step={step}] loss_corr is detached -> no gradient path at all")
            return

        probe = [
            ("d/dcorr_pred", corr_pred),
            ("d/dp_tgt", p_tgt),
            ("d/dp_src", p_src),
            ("d/dfeat", feat),
            ("d/dproj_w", self.corr_proj.weight),
        ]
        if self.corr_proj.bias is not None:
            probe.append(("d/dproj_b", self.corr_proj.bias))
        print(f"[CORR-GRAD step={step}] {_grad_probe(loss_corr, probe)}")

    def _compute_corr_loss(self, batch, x, src_kept=None, debug=False):
        """Correlation feature regularization on the captured last-output-block feature.

        Returns (loss_corr, log_dict) or (None, {}) if inputs are unavailable.
        ``src_kept`` masks out UCG-dropped samples where ``in_concat`` was zeroed.
        """
        feat = self._corr_feat.get("feat")
        if feat is None:
            return None, {}
        if "corr_gt" not in batch or "valid_mask" not in batch or "flow_tgt2src" not in batch:
            return None, {}

        B = x.shape[0]
        f_tgt = feat[:B]   # target (noised) branch, [B, 320, Hf, Wf]
        f_src = feat[B:]   # source branch,          [B, 320, Hf, Wf]

        p_tgt = F.normalize(self.corr_proj(f_tgt), dim=1)  # [B, D, Hf, Wf]
        p_src = F.normalize(self.corr_proj(f_src), dim=1)  # [B, D, Hf, Wf]
        feat_hw = p_tgt.shape[-2:]

        # Warp source features into the target frame (same ops as correlation_gt_2.correlation_for_pair).
        flow = batch["flow_tgt2src"].to(device=p_src.device, dtype=p_src.dtype)
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)

        Hf, Wf = feat_hw
        latent_denom = max(float(Hf), float(Wf)) - 1.0
        if latent_denom < 1.0:
            latent_denom = 1.0

        if self.corr_flow_src_size is None:
            grid = flow
        else:
            grid = flow / latent_denom * 2.0 - 1.0

        if grid.shape[1:3] != feat_hw:
            grid = F.interpolate(
                grid.permute(0, 3, 1, 2),
                size=feat_hw,
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1).contiguous()

        in_bounds = (grid.abs() <= 1.0).all(dim=-1).unsqueeze(1).float()  # [B, 1, Hf, Wf]

        warped_src = F.grid_sample(
            p_src,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        warped_src = F.normalize(warped_src, dim=1)

        corr_pred = (warped_src * p_tgt).sum(dim=1, keepdim=True)  # [B, 1, Hf, Wf]

        corr_gt = self._as_bchw(batch["corr_gt"].to(self.device).float())
        valid_mask = self._as_bchw(batch["valid_mask"].to(self.device).float())

        # Resize GT/mask to the feature resolution if they differ.
        if corr_gt.shape[-2:] != feat_hw:
            corr_gt = F.interpolate(corr_gt, size=feat_hw, mode="bilinear", align_corners=False)
        if valid_mask.shape[-2:] != feat_hw:
            valid_mask = F.interpolate(valid_mask, size=feat_hw, mode="nearest")

        if src_kept is not None:
            src_kept = src_kept.to(valid_mask.device)
            if not src_kept.any():
                return None, {}
            valid_mask = valid_mask * src_kept.view(-1, 1, 1, 1).float()

        numer = (((corr_pred - corr_gt) ** 2) * valid_mask).sum()
        loss_corr = numer / (valid_mask.sum() + 1e-8)

        in_bounds_ratio = (in_bounds * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        log = {"corr_loss": loss_corr, "corr_flow_in_bounds": in_bounds_ratio}
        if src_kept is not None:
            log["corr_src_kept_ratio"] = src_kept.float().mean()

        with torch.no_grad():
            vm = valid_mask > 0.5
            n_valid = int(vm.sum())
            if n_valid >= 2:
                pred_v = corr_pred[vm]
                gt_v = corr_gt[vm]
                pred_std = pred_v.std(unbiased=True)
                gt_std = gt_v.std(unbiased=True)
                log["corr_pred_valid_std"] = pred_std
                log["corr_gt_valid_std"] = gt_std
                denom = pred_std * gt_std
                if denom > 1e-8:
                    pred_c = pred_v - pred_v.mean()
                    gt_c = gt_v - gt_v.mean()
                    log["corr_pred_gt_pearson"] = (pred_c * gt_c).sum() / ((n_valid - 1) * denom)
                else:
                    log["corr_pred_gt_pearson"] = pred_v.new_tensor(float("nan"))
            else:
                nan = corr_pred.new_tensor(float("nan"))
                log["corr_pred_valid_std"] = nan
                log["corr_gt_valid_std"] = nan
                log["corr_pred_gt_pearson"] = nan

        if not self._corr_flow_warned and in_bounds_ratio.item() < 0.5:
            self._corr_flow_warned = True
            print(
                f"[WARN] Only {in_bounds_ratio.item():.1%} of supervised grid coords land inside [-1, 1] "
                f"(grid min={grid.min().item():.3f}, max={grid.max().item():.3f}). "
                f"Check flow_tgt2src convention (corr_flow_src_size={self.corr_flow_src_size}) "
                f"and (x=width, y=height) channel order."
            )

        if debug:
            with torch.no_grad():
                vm = valid_mask > 0.5
                if vm.any():
                    gt_valid = corr_gt[vm]
                    gt_stats = (gt_valid.min().item(), gt_valid.mean().item(), gt_valid.max().item())
                else:
                    gt_stats = (float("nan"), float("nan"), float("nan"))
                warped_norm = warped_src.norm(dim=1)  # [B, Hf, Wf]
                print(
                    f"[CORR-DEBUG step={int(self.global_step)}] "
                    f"feat={tuple(feat.shape)} | "
                    f"f_tgt(mean={f_tgt.mean().item():.4f},std={f_tgt.std().item():.4f}) "
                    f"f_src(mean={f_src.mean().item():.4f},std={f_src.std().item():.4f}) | "
                    f"grid(min={grid.min().item():.3f},max={grid.max().item():.3f},"
                    f"in_bounds@valid={in_bounds_ratio.item():.4f}) | "
                    f"warped_src_norm(mean={warped_norm.mean().item():.4f}) | "
                    f"corr_pred(min={corr_pred.min().item():.4f},mean={corr_pred.mean().item():.4f},max={corr_pred.max().item():.4f}) | "
                    f"corr_gt@valid(min={gt_stats[0]:.4f},mean={gt_stats[1]:.4f},max={gt_stats[2]:.4f}) | "
                    f"valid_ratio={valid_mask.mean().item():.4f} | "
                    f"loss_corr={loss_corr.item():.6f}"
                )
            self._debug_corr_grad(loss_corr, feat, corr_pred, p_tgt, p_src, valid_mask, numer)

        return loss_corr, log

    def set_vis_sources(self, images, names=None):
        """Fix the source faces used for training-time multiview visualization.

        images: [N, 3, H, W] (or [N, H, W, 3]) in [0, 1], same layout as the dataset's "hint".
        """
        if images.ndim == 4 and images.shape[-1] == 3:
            images = einops.rearrange(images, "b h w c -> b c h w")
        self._vis_sources = images.detach().float().cpu().clone()
        self._vis_names = list(names) if names is not None else [str(i) for i in range(images.shape[0])]
        self._vis_source_pils = [
            self._hint_tensor_to_pil(self._vis_sources[i]) for i in range(self._vis_sources.shape[0])
        ]
        print(f"[VIS] Fixed {len(self._vis_names)} visualization source(s): {self._vis_names}")

    def _log_multiview_vis(self, batch, wandb_log):
        """Sample fixed sources via ``utils.inference.generate_batch`` (Colab / eval path)."""
        from utils.inference import generate_batch

        if self._vis_sources is None and self._vis_source_pils is None:
            src = batch[self.control_key][:1]
            self.set_vis_sources(
                src,
                names=[str(batch["subject_id"][0])] if "subject_id" in batch else None,
            )
            print("[VIS][WARN] vis_source_image / set_vis_sources not set; using first training batch hint.")

        step = int(self.global_step)
        source_pils = self._vis_source_pils
        if source_pils is None:
            source_pils = [self._hint_tensor_to_pil(t) for t in self._vis_sources]

        unet = self.model.diffusion_model
        was_training = unet.training
        unet.eval()
        normal_gt_and_preds = []
        try:
            with torch.no_grad():
                wandb_images = []
                for src_pil, name in zip(source_pils, self._vis_names):
                    preds = generate_batch(
                        self,
                        src_pil,
                        prompt="",
                        dy_list=list(self.vis_yaws_deg),
                        h=self.vis_image_size,
                        w=self.vis_image_size,
                        ddim_steps=self.vis_ddim_steps,
                        ddim_eta=0.0,
                        img_scale=self.vis_img_scale,
                        prompt_scale=self.vis_prompt_scale,
                        precision=self.vis_precision,
                        use_ema_scope=self.vis_use_ema_scope,
                        shared_noise=True,
                        seed=self.vis_seed,
                    )
                    src_show = np.asarray(src_pil.convert("RGB"), dtype=np.float32) / 255.0
                    src_show = torch.from_numpy(src_show).permute(2, 0, 1)
                    wandb_images.append(wandb.Image(
                        src_show,
                        caption=f"Step {step} | {name} | SOURCE",
                    ))
                    for yaw_deg, pred in zip(self.vis_yaws_deg, preds):
                        pred_t = torch.from_numpy(
                            np.asarray(pred.convert("RGB"), dtype=np.float32) / 255.0
                        ).permute(2, 0, 1)
                        wandb_images.append(wandb.Image(
                            pred_t,
                            caption=(
                                f"Step {step} | {name} | Yaw: {yaw_deg:g}° | "
                                f"cfg={self.vis_img_scale:g} | seed={self.vis_seed}"
                            ),
                        ))

                if "normal" in batch and "normal_mask" in batch:
                    gt_n = batch["normal"][:1].to(self.device)
                    normal_gt_and_preds.append(wandb.Image(
                        _normal_to_rgb_vis(gt_n),
                        caption=f"Step {step} | GT normal (target view)",
                    ))
                    nm = batch["normal_mask"][:1].to(self.device).float()
                    if nm.ndim == 4:
                        nm = nm[0]
                    m = nm[0] if nm.ndim == 3 else nm
                    m_vis = torch.clamp(m, 0, 1).unsqueeze(0).expand(3, -1, -1)
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

    def _compute_batch_loss(self, batch, *, decode_mode="train"):
        """Shared train/val loss: masked MSE + optional corr, identity, DISTS."""
        x, cond = self.get_input(batch, self.first_stage_key)
        src_kept = cond["in_concat"][0].flatten(1).abs().sum(dim=1) > 0  # [B] bool

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        mse_loss = F.mse_loss(model_output, noise, reduction="mean")

        mask = batch.get("mask")
        masked_mse_loss = None
        if mask is not None:
            mask = mask.to(self.device)
            soft_mask = mask * (1.0 - self.mask_min_weight) + self.mask_min_weight
            latent_mask = F.interpolate(soft_mask, size=model_output.shape[-2:], mode="area")
            masked_mse_loss = (
                F.mse_loss(model_output, noise, reduction="none") * latent_mask
            ).sum() / latent_mask.sum()
            loss = self.mse_weight * masked_mse_loss
        else:
            loss = mse_loss

        identity_loss = None
        dists_loss = None
        d_img = None
        d_sobel = None
        corr_loss = None
        corr_log = {}

        if self.lambda_corr > 0.0:
            debug_corr = (
                decode_mode == "train"
                and int(self.global_step) < self.corr_debug_steps
            )
            corr_loss, corr_log = self._compute_corr_loss(
                batch, x, src_kept=src_kept, debug=debug_corr
            )
            if corr_loss is not None:
                loss = loss + self.lambda_corr * corr_loss

        need_identity = self.identity_loss_weight > 0.0
        need_dists = self.dists_loss_weight > 0.0

        if need_identity or need_dists:
            union_cut = max(
                self.identity_t_cut if need_identity else 0,
                self.dists_t_cut if need_dists else 0,
            )
            sel = t < union_cut
            if torch.any(sel):
                x0_pred = self.predict_start_from_noise(
                    x_noisy[sel], t[sel], model_output[sel]
                )
                if decode_mode == "train":
                    pred_img = self._decode_first_stage_train(x0_pred)
                else:
                    pred_img = self.decode_first_stage(x0_pred)
                pred_rgb = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
                gt_rgb = self._gt_rgb_01_from_batch(batch)[sel]
                t_sel = t[sel]

                if need_identity:
                    sub = t_sel < self.identity_t_cut
                    if torch.any(sub):
                        pred_rgb_id = pred_rgb[sub]
                        gt_rgb_id = gt_rgb[sub]

                        backbone = self._ensure_arcface_backbone()
                        with self._identity_autocast_ctx():
                            pred_arc = preprocess_arcface_input(
                                pred_rgb_id.float(),
                                spatial_mode=self.arcface_spatial_mode,
                            )
                            gt_arc = preprocess_arcface_input(
                                gt_rgb_id.float().detach(),
                                spatial_mode=self.arcface_spatial_mode,
                            )
                            emb_pred = F.normalize(backbone(pred_arc), dim=-1)
                            emb_gt = F.normalize(backbone(gt_arc), dim=-1).detach()
                        identity_loss = (1.0 - (emb_pred * emb_gt).sum(dim=-1)).mean()
                        loss = loss + self.identity_loss_weight * identity_loss

                if need_dists:
                    sub = t_sel < self.dists_t_cut
                    if torch.any(sub):
                        pred_rgb_d = pred_rgb[sub].float()
                        gt_rgb_d = gt_rgb[sub].float().detach()

                        dists_model = self._ensure_dists()
                        require_grad = decode_mode == "train"
                        d_img = dists_model(
                            pred_rgb_d, gt_rgb_d,
                            require_grad=require_grad,
                            batch_average=True,
                        )

                        pred_sobel = _grayscale_sobel_3ch(pred_rgb_d)
                        gt_sobel = _grayscale_sobel_3ch(gt_rgb_d).detach()
                        d_sobel = dists_model(
                            pred_sobel, gt_sobel,
                            require_grad=require_grad,
                            batch_average=True,
                        )

                        dists_loss = d_img + d_sobel
                        loss = loss + self.dists_loss_weight * dists_loss

        mse_for_grad = masked_mse_loss if masked_mse_loss is not None else mse_loss

        log_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
        }
        if masked_mse_loss is not None:
            log_dict["masked_mse_loss"] = masked_mse_loss
        if identity_loss is not None:
            log_dict["identity_loss"] = identity_loss
        if dists_loss is not None:
            log_dict["dists_loss"] = dists_loss
            log_dict["dists_loss_img"] = d_img
            log_dict["dists_loss_sobel"] = d_sobel
        if corr_loss is not None:
            log_dict.update(corr_log)
            log_dict["weighted_corr_loss"] = self.lambda_corr * corr_loss

        return loss, log_dict, mse_for_grad, corr_loss, dists_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, log_dict, _, _, _ = self._compute_batch_loss(batch, decode_mode="eval")
        val_logs = {f"val/{k}": v for k, v in log_dict.items()}
        self.log_dict(
            val_logs,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self):
        if run is None or self.trainer is None:
            return
        metrics = {}
        for key, value in self.trainer.callback_metrics.items():
            if not str(key).startswith("val/"):
                continue
            if torch.is_tensor(value):
                metrics[key] = value.detach().item()
            else:
                metrics[key] = value
        if metrics:
            run.log(metrics, step=int(self.global_step))

    def training_step(self, batch, batch_idx):
        self.model.diffusion_model.train()

        if hasattr(self.model.diffusion_model, "enable_adapters"):
            self.model.diffusion_model.enable_adapters()

        loss, log_dict, mse_for_grad, corr_loss, dists_loss = self._compute_batch_loss(
            batch, decode_mode="train"
        )

        wandb_log = dict(log_dict)

        if self.global_step % 50 == 0:

            # Gradient 진단용 parameter groups
            lora_params = [
                p for n, p in self.model.diffusion_model.named_parameters()
                if p.requires_grad and "lora" in n.lower()
            ]

            finetune_params = [
                p for n, p in self.model.diffusion_model.named_parameters()
                if p.requires_grad and (
                    "pose_net" in n
                    or "vae_proj" in n
                    or "base_model.model.out." in n
                )
            ]

            corr_proj_params = [
                p for p in self.corr_proj.parameters()
                if p.requires_grad
            ]

            # MSE -> LoRA gradient (uses masked MSE when enabled)
            wandb_log["mse_grad_lora"] = _grad_norm_l2(
                mse_for_grad, lora_params
            )

            if corr_loss is not None:
                weighted_corr = self.lambda_corr * corr_loss

                # CORR -> 각각의 parameter group gradient
                wandb_log["corr_grad_lora"] = _grad_norm_l2(
                    weighted_corr, lora_params
                )

                wandb_log["corr_grad_finetune"] = _grad_norm_l2(
                    weighted_corr, finetune_params
                )

                wandb_log["corr_grad_proj"] = _grad_norm_l2(
                    weighted_corr, corr_proj_params
                )

                # CORR이 LoRA에 주는 영향 / MSE가 LoRA에 주는 영향
                mse_g = wandb_log["mse_grad_lora"]
                corr_g = wandb_log["corr_grad_lora"]

                wandb_log["corr_mse_grad_ratio_lora"] = (
                    corr_g / (mse_g + 1e-12))

            if dists_loss is not None:
                weighted_dists = self.dists_loss_weight * dists_loss
                wandb_log["dists_grad_lora"] = _grad_norm_l2(
                    weighted_dists, lora_params
                )
                wandb_log["dists_grad_finetune"] = _grad_norm_l2(
                    weighted_dists, finetune_params
                )
                wandb_log["dists_mse_grad_ratio_lora"] = (
                    wandb_log["dists_grad_lora"]
                    / (wandb_log["mse_grad_lora"] + 1e-12)
                )

        if self.vis_every_n_steps > 0 and self.global_step % self.vis_every_n_steps == 0:
            self._log_multiview_vis(batch, wandb_log)

        print(f"LOSS logged: total={loss.item():.4f}")
        run.log(wandb_log, step=int(self.global_step))

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Explicitly collect LoRA and pose_net params separately
        lora_params = []
        finetune_params = []
        other_params = []
        
        # Use named_parameters to ensure we get the actual parameter objects
        for n, p in self.model.diffusion_model.named_parameters():
            if p.requires_grad:
                if "lora" in n.lower():
                    lora_params.append(p)
                    print(f"[OPT] LoRA param: {n}, shape={p.shape}")
                elif "pose_net" in n:
                    finetune_params.append(p)
                    print(f"[OPT] pose_net param: {n}, shape={p.shape}")
                elif "vae_proj" in n:
                    finetune_params.append(p)
                elif "base_model.model.out." in n:
                    finetune_params.append(p)
                else:
                    other_params.append(p)
                    print(f"[OPT] Other param: {n}, shape={p.shape}")
        
        print(f"\n[OPT] Summary:")
        print(f"  LoRA params: {len(lora_params)}")
        print(f"  finetune params: {len(finetune_params)}")
        print(f"  Other trainable params: {len(other_params)}")
        print(f"  Total params in optimizer: {len(lora_params) + len(finetune_params) + len(other_params)}")
        
        if len(lora_params) == 0:
            print("[WARNING] No LoRA params found! Check if PEFT is properly configured.")
            # Fallback: try to get params differently
            for n, p in self.named_parameters():
                if "lora" in n.lower() and p.requires_grad:
                    lora_params.append(p)
                    print(f"[OPT-FALLBACK] Found LoRA param: {n}")
        
        # Use separate param groups with different learning rates
        # pose_net uses lower LR (0.1x) for gentle fine-tuning
        param_groups = [
            {"params": lora_params, "lr": self.learning_rate, "name": "lora"},
        ]

        if len(finetune_params) > 0:
            param_groups.append({"params": finetune_params, "lr": self.learning_rate * 0.1, "name": "finetune"})
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": self.learning_rate, "name": "other"})

        # corr_proj lives on the LightningModule (not under diffusion_model), so add it explicitly.
        # Skip when frozen so we can ablate the projection layer without touching this loss.
        corr_params = [p for p in self.corr_proj.parameters() if p.requires_grad]
        if len(corr_params) > 0:
            param_groups.append({"params": corr_params, "lr": self.learning_rate, "name": "corr_proj"})
            print(f"[OPT] corr_proj params: {len(corr_params)}")
        else:
            print("[OPT] corr_proj frozen (not added to optimizer)")

        optimizer = torch.optim.AdamW(param_groups)
        
        return optimizer