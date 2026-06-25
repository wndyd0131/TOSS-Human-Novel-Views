import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb
import lpips
import pytz
from datetime import datetime
from contextlib import nullcontext

import einops

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


def _normal_gradient_loss(pred_normals, gt_normals, mask, eps=1e-8):
    """Masked L1 loss on finite-difference spatial gradients of normal components."""
    dx_pred = pred_normals[:, :, :, 1:] - pred_normals[:, :, :, :-1]
    dy_pred = pred_normals[:, :, 1:, :] - pred_normals[:, :, :-1, :]
    dx_gt = gt_normals[:, :, :, 1:] - gt_normals[:, :, :, :-1]
    dy_gt = gt_normals[:, :, 1:, :] - gt_normals[:, :, :-1, :]

    loss_x = (dx_pred - dx_gt).abs().mean(dim=1, keepdim=True)
    loss_y = (dy_pred - dy_gt).abs().mean(dim=1, keepdim=True)

    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss_x = loss_x * mask_x
        loss_y = loss_y * mask_y
        denom = mask_x.sum() + mask_y.sum()
        return (loss_x.sum() + loss_y.sum()) / (denom + eps)
    return (loss_x.mean() + loss_y.mean()) / 2

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


def _set_expose_dec_feat(module, value=True):
    """Set expose_dec_feat on the underlying UNetModel_toss (PEFT version-agnostic)."""
    if hasattr(module, "get_base_model"):
        base = module.get_base_model()
        # get_base_model() may return UNetModel_toss directly or a wrapper with .model
        target = base.model if hasattr(base, "model") else base
        target.expose_dec_feat = value
    else:
        module.expose_dec_feat = value


class DepthHead(nn.Module):
    """Predicts a disparity map from UNet decoder features.

    Input:  decoder feature [B, in_ch, 32, 32] (in_ch = model_channels, 320).
    Output: disparity [B, 1, 256, 256], non-negative (softplus).

    Three x2 upsample+conv stages take 32 -> 64 -> 128 -> 256.
    """

    def __init__(self, in_ch=320, base_ch=128):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
            )

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        self.up1 = block(base_ch, base_ch)          # 32 -> 64
        self.up2 = block(base_ch, base_ch // 2)     # 64 -> 128
        self.up3 = block(base_ch // 2, base_ch // 4)  # 128 -> 256
        self.out_conv = nn.Conv2d(base_ch // 4, 1, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.in_conv(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out_conv(x)
        return F.softplus(x)  # disparity >= 0


class NormalHead(nn.Module):
    """Predicts a normal map from UNet decoder features.

    Input:  decoder feature [B, in_ch, 32, 32] (in_ch = model_channels, 320).
    Output: raw normal [B, 3, 256, 256] (L2-normalized at loss time).

    Three x2 upsample+conv stages take 32 -> 64 -> 128 -> 256.
    """

    def __init__(self, in_ch=320, base_ch=128):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
            )

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        self.up1 = block(base_ch, base_ch)          # 32 -> 64
        self.up2 = block(base_ch, base_ch // 2)     # 64 -> 128
        self.up3 = block(base_ch // 2, base_ch // 4)  # 128 -> 256
        self.out_conv = nn.Conv2d(base_ch // 4, 3, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.in_conv(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.out_conv(x)


class NormalRefineHead(nn.Module):
    """Stronger fusion head from decoded target-view RGB + coarse normal."""

    def __init__(self, in_ch=6, base_ch=64):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
            )

        self.enc1 = block(in_ch, base_ch)          # 256
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 128
        self.enc2 = block(base_ch, base_ch * 2)

        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 64
        self.mid = block(base_ch * 2, base_ch * 4)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
        )
        self.dec1 = block(base_ch * 4, base_ch * 2)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        self.dec2 = block(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, pred_rgb, n_coarse):
        n_coarse = F.normalize(n_coarse, dim=1, eps=1e-8)
        x = torch.cat([pred_rgb, n_coarse], dim=1)

        e1 = self.enc1(x)          # 256
        e2 = self.enc2(self.down1(e1))  # 128
        m = self.mid(self.down2(e2))    # 64

        d1 = self.up1(m)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.out(d2)
        return F.normalize(out, dim=1, eps=1e-8)


class TossLoraModule(TOSS):
    def __init__(
        self,
        lora_config_params,
        *args,
        normal_head_loss_weight=0.0,
        normal_head_t_cut=200,
        use_normal_head=False,
        use_normal_refine_head=False,
        normal_refine_loss_weight=0.005,
        normal_refine_t_cut=100,
        coarse_grad_loss_weight=0.0,
        refine_grad_loss_weight=0.0,
        identity_loss_weight=0.0,
        identity_t_cut=200,
        arcface_ckpt_path=None,
        arcface_spatial_mode="cover_center",
        dists_loss_weight=0.0,
        dists_t_cut=200,
        depth_loss_weight=0.0,
        depth_t_cut=200,
        use_depth_head=False,
        **kwargs,
    ):
        kwargs.pop("lora_config_params", None)  # consumed by us
        kwargs.pop("normal_estimator_path", None)
        kwargs.pop("geometry_loss_weight", None)
        kwargs.pop("geometry_t_cut", None)
        self.use_normal_head = bool(kwargs.pop("use_normal_head", use_normal_head))
        self.normal_head_loss_weight = float(kwargs.pop("normal_head_loss_weight", normal_head_loss_weight))
        self.normal_head_t_cut = int(kwargs.pop("normal_head_t_cut", normal_head_t_cut))
        self.use_normal_refine_head = bool(kwargs.pop("use_normal_refine_head", use_normal_refine_head))
        self.normal_refine_loss_weight = float(kwargs.pop("normal_refine_loss_weight", normal_refine_loss_weight))
        self.normal_refine_t_cut = int(kwargs.pop("normal_refine_t_cut", normal_refine_t_cut))
        self.coarse_grad_loss_weight = float(kwargs.pop("coarse_grad_loss_weight", coarse_grad_loss_weight))
        self.refine_grad_loss_weight = float(kwargs.pop("refine_grad_loss_weight", refine_grad_loss_weight))
        self.identity_loss_weight = float(kwargs.pop("identity_loss_weight", identity_loss_weight))
        self.identity_t_cut = int(kwargs.pop("identity_t_cut", identity_t_cut))
        self.arcface_ckpt_path = kwargs.pop("arcface_ckpt_path", arcface_ckpt_path)
        self.arcface_spatial_mode = kwargs.pop("arcface_spatial_mode", arcface_spatial_mode)
        self.dists_loss_weight = float(kwargs.pop("dists_loss_weight", dists_loss_weight))
        self.dists_t_cut = int(kwargs.pop("dists_t_cut", dists_t_cut))
        self.depth_loss_weight = float(kwargs.pop("depth_loss_weight", depth_loss_weight))
        self.depth_t_cut = int(kwargs.pop("depth_t_cut", depth_t_cut))
        self.use_depth_head = bool(kwargs.pop("use_depth_head", use_depth_head))
        super().__init__(*args, **kwargs)
        self._arcface_backbone = None
        self._dists_model = None

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
                # Normal head (decoder feature) loss
                "use_normal_head": self.use_normal_head,
                "normal_head_loss_weight": self.normal_head_loss_weight,
                "normal_head_t_cut": self.normal_head_t_cut,
                # Normal refine head (decoded x0) loss
                "use_normal_refine_head": self.use_normal_refine_head,
                "normal_refine_loss_weight": self.normal_refine_loss_weight,
                "normal_refine_t_cut": self.normal_refine_t_cut,
                "coarse_grad_loss_weight": self.coarse_grad_loss_weight,
                "refine_grad_loss_weight": self.refine_grad_loss_weight,
                # Depth head (decoder feature) loss
                "use_depth_head": self.use_depth_head,
                "depth_loss_weight": self.depth_loss_weight,
                "depth_t_cut": self.depth_t_cut,
            },
        )

        unet = self.model.diffusion_model
        model_channels = unet.model_channels
        self._need_dec_feat = self.use_normal_head or self.use_depth_head
        if self._need_dec_feat:
            unet.expose_dec_feat = True
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

        # Ensure expose_dec_feat is set on the underlying UNet after PEFT wrapping.
        if self._need_dec_feat:
            _set_expose_dec_feat(self.model.diffusion_model, True)

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

        # Normal/depth heads: predict from UNet decoder features (not x0 latent).
        # Created AFTER self.requires_grad_(False) so their params stay trainable.
        self.normal_head = None
        self.normal_refine_head = None
        self.depth_head = None
        if self.use_normal_head:
            self.normal_head = NormalHead(in_ch=model_channels)
            for p in self.normal_head.parameters():
                p.requires_grad = True
            print(f"[INIT] Created normal_head (in_ch={model_channels}), trainable")
        else:
            print("[INIT] normal_head disabled (use_normal_head=False)")

        if self.use_normal_refine_head:
            if self.normal_head is None:
                raise ValueError("use_normal_refine_head requires use_normal_head=True (coarse head provides n_coarse)")
            self.normal_refine_head = NormalRefineHead()
            for p in self.normal_refine_head.parameters():
                p.requires_grad = True
            refine_params = sum(p.numel() for p in self.normal_refine_head.parameters())
            print(f"[INIT] Created normal_refine_head, trainable ({refine_params} params)")
        else:
            print("[INIT] normal_refine_head disabled (use_normal_refine_head=False)")

        if self.use_depth_head:
            self.depth_head = DepthHead(in_ch=model_channels)
            for p in self.depth_head.parameters():
                p.requires_grad = True
            print(f"[INIT] Created depth_head (in_ch={model_channels}), trainable")
        else:
            print("[INIT] depth_head disabled (use_depth_head=False)")

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

    def _prepare_normal_gt(self, batch, sel, target_hw):
        """Return L2-normalized GT normals and mask for the given batch selection."""
        gt_normals = batch["normal"].to(self.device)
        if gt_normals.ndim == 4 and gt_normals.shape[-1] == 3:
            gt_normals = gt_normals.permute(0, 3, 1, 2)
        gt_normals = gt_normals[sel]
        if gt_normals.shape[-2:] != target_hw:
            gt_normals = F.interpolate(
                gt_normals,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        gt_normals = gt_normals / gt_normals.norm(dim=1, keepdim=True).clamp(min=1e-8)

        normal_mask = batch["normal_mask"].to(self.device).float()
        if normal_mask.ndim == 3:
            normal_mask = normal_mask.unsqueeze(1)
        normal_mask = normal_mask[sel]
        if normal_mask.shape[-2:] != target_hw:
            normal_mask = F.interpolate(
                normal_mask,
                size=target_hw,
                mode="area",
            )
        return gt_normals, normal_mask

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

    def on_save_checkpoint(self, checkpoint):
        # We override this to prevent the parent class from 
        # trying to call self.embedding_manager.save()
        
        # If you want to keep the checkpoint small (LoRA only),
        # you can clear the base model weights from the checkpoint:
        # checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if "lora" in k or "pose_net" in k}
        pass

    def on_train_start(self):
        """Called by PyTorch Lightning when training starts - log trainer config"""
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
                "use_normal_head": self.use_normal_head,
                "normal_head_loss_weight": self.normal_head_loss_weight,
                "normal_head_t_cut": self.normal_head_t_cut,
                "use_normal_refine_head": self.use_normal_refine_head,
                "normal_refine_loss_weight": self.normal_refine_loss_weight,
                "normal_refine_t_cut": self.normal_refine_t_cut,
                "coarse_grad_loss_weight": self.coarse_grad_loss_weight,
                "refine_grad_loss_weight": self.refine_grad_loss_weight,
                "use_depth_head": self.use_depth_head,
                "depth_loss_weight": self.depth_loss_weight,
                "depth_t_cut": self.depth_t_cut,
            }, allow_val_change=True)

    def training_step(self, batch, batch_idx):

        # Ensure model is in training mode
        self.model.diffusion_model.train()
        
        # CRITICAL: Explicitly enable LoRA adapters for PEFT
        if hasattr(self.model.diffusion_model, 'enable_adapters'):
            self.model.diffusion_model.enable_adapters()
        
        x, cond = self.get_input(batch, self.first_stage_key)

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        '''Forward'''
        need_aux = self._need_dec_feat
        if need_aux:
            model_output, aux = self.apply_model(x_noisy, t, cond, return_aux=True)
            dec_feat = aux.get("dec_feat")
        else:
            model_output = self.apply_model(x_noisy, t, cond)
            dec_feat = None

        mse_loss = F.mse_loss(model_output, noise, reduction="mean")

        '''Masked Loss'''
        # mask = None
        mask = batch.get("mask")  # Original mask [B, 1, 256, 256]

        # if mask is not None:
        #     mask = mask.to(self.device)
        #     # Soft mask: mask=1 (face) -> weight=1.0, mask=0 (background) -> weight=min_weight
        #     soft_mask = mask * (1.0 - self.mask_min_weight) + self.mask_min_weight

        #     # Masked latent MSE, normalized by mask sum to avoid diluting head signal
        #     latent_mask = F.interpolate(soft_mask, size=model_output.shape[-2:], mode="area")
        #     masked_mse_loss = (F.mse_loss(model_output, noise, reduction="none") * latent_mask).sum() / latent_mask.sum()

        #     loss = self.mse_weight * masked_mse_loss
        #     print(f"MASKED LOSS: mse={masked_mse_loss.item():.4f}")
        # else:
        loss = mse_loss

        identity_loss = None
        dists_loss = None
        normal_coarse_loss = None
        normal_coarse_grad_loss = None
        normal_refine_loss = None
        normal_refine_grad_loss = None
        depth_loss = None
        d_img = None
        d_sobel = None

        need_identity = self.identity_loss_weight > 0.0
        need_dists    = self.dists_loss_weight    > 0.0
        need_normal = (
            self.use_normal_head
            and (
                self.normal_head_loss_weight > 0.0
                or self.coarse_grad_loss_weight > 0.0
            )
            and "normal" in batch
            and "normal_mask" in batch
        )
        need_normal_refine = (
            self.use_normal_refine_head
            and (
                self.normal_refine_loss_weight > 0.0
                or self.refine_grad_loss_weight > 0.0
            )
            and self.normal_head is not None
            and self.normal_refine_head is not None
            and "normal" in batch
            and "normal_mask" in batch
        )
        need_depth = (
            self.use_depth_head
            and self.depth_loss_weight > 0.0
            and "depth" in batch
            and "depth_mask" in batch
        )

        pred_rgb = None
        sel_decode = None

        # Decode pred_rgb / gt_rgb for identity, DISTS, and normal refine (requires VAE decode).
        need_decode = need_identity or need_dists or need_normal_refine
        if need_decode:
            union_cut = max(
                self.identity_t_cut if need_identity else 0,
                self.dists_t_cut if need_dists else 0,
                self.normal_refine_t_cut if need_normal_refine else 0,
            )
            sel_decode = t < union_cut
            if torch.any(sel_decode):
                x0_pred = self.predict_start_from_noise(x_noisy[sel_decode], t[sel_decode], model_output[sel_decode])
                pred_img = self._decode_first_stage_train(x0_pred)
                pred_rgb = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
                gt_rgb = self._gt_rgb_01_from_batch(batch)[sel_decode]
                t_sel = t[sel_decode]

                if need_identity:
                    sub = t_sel < self.identity_t_cut
                    if torch.any(sub):
                        pred_rgb_id = pred_rgb[sub]
                        gt_rgb_id   = gt_rgb[sub]

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
                        identity_loss = (1.0 - (emb_pred * emb_gt).sum(dim=-1)).mean() # cosine similarity loss
                        loss = loss + self.identity_loss_weight * identity_loss

                if need_dists:
                    sub = t_sel < self.dists_t_cut
                    if torch.any(sub):
                        pred_rgb_d = pred_rgb[sub].float()
                        gt_rgb_d   = gt_rgb[sub].float().detach()

                        dists_model = self._ensure_dists()
                        d_img = dists_model(pred_rgb_d, gt_rgb_d, require_grad=True, batch_average=True)

                        pred_sobel = _grayscale_sobel_3ch(pred_rgb_d)
                        gt_sobel   = _grayscale_sobel_3ch(gt_rgb_d).detach()
                        d_sobel = dists_model(pred_sobel, gt_sobel, require_grad=True, batch_average=True)

                        dists_loss = d_img + d_sobel
                        loss = loss + self.dists_loss_weight * dists_loss

        # Normal head loss from UNet decoder features (no VAE decode).
        if need_normal and dec_feat is not None:
            sel_coarse = t < self.normal_head_t_cut
            if torch.any(sel_coarse):
                pred_normals = self.normal_head(dec_feat[sel_coarse])
                pred_normals = pred_normals / pred_normals.norm(dim=1, keepdim=True).clamp(min=1e-8)

                gt_normals, normal_mask = self._prepare_normal_gt(
                    batch, sel_coarse, pred_normals.shape[-2:]
                )

                gt_normals_det = gt_normals.detach()
                normal_mask_det = normal_mask.detach()

                if self.normal_head_loss_weight > 0.0:
                    normal_coarse_loss = _cosine_similarity_loss(
                        pred_normals,
                        gt_normals_det,
                        normal_mask_det,
                    )
                    loss = loss + self.normal_head_loss_weight * normal_coarse_loss

                if self.coarse_grad_loss_weight > 0.0:
                    normal_coarse_grad_loss = _normal_gradient_loss(
                        pred_normals,
                        gt_normals_det,
                        normal_mask_det,
                    )
                    loss = loss + self.coarse_grad_loss_weight * normal_coarse_grad_loss

                with torch.no_grad():
                    self._wandb_pred_normals = pred_normals[:1].detach()
                    self._wandb_gt_normals = gt_normals[:1].detach()
                    self._wandb_normal_mask = normal_mask[:1].detach()
                    self._wandb_pred_normals_step = int(self.global_step)

        # Normal refine head loss from decoded x0 RGB + coarse normals.
        if need_normal_refine and dec_feat is not None and pred_rgb is not None and sel_decode is not None:
            sel_refine = t < self.normal_refine_t_cut
            refine_sub = sel_refine[sel_decode]
            if torch.any(refine_sub):
                n_coarse = self.normal_head(dec_feat[sel_refine])

                pred_rgb_ref = pred_rgb[refine_sub]
                n_refined = self.normal_refine_head(pred_rgb_ref, n_coarse)

                gt_normals, normal_mask = self._prepare_normal_gt(
                    batch, sel_refine, n_refined.shape[-2:]
                )

                gt_normals_det = gt_normals.detach()
                normal_mask_det = normal_mask.detach()

                if self.normal_refine_loss_weight > 0.0:
                    normal_refine_loss = _cosine_similarity_loss(
                        n_refined,
                        gt_normals_det,
                        normal_mask_det,
                    )
                    loss = loss + self.normal_refine_loss_weight * normal_refine_loss

                if self.refine_grad_loss_weight > 0.0:
                    normal_refine_grad_loss = _normal_gradient_loss(
                        n_refined,
                        gt_normals_det,
                        normal_mask_det,
                    )
                    loss = loss + self.refine_grad_loss_weight * normal_refine_grad_loss

                with torch.no_grad():
                    self._wandb_pred_normals_refined = n_refined[:1].detach()
                    self._wandb_pred_normals_refined_step = int(self.global_step)

        # Depth head loss from UNet decoder features (no VAE decode).
        if need_depth and dec_feat is not None:
            sel = t < self.depth_t_cut
            if torch.any(sel):
                eps = 1e-6
                pred_disp = self.depth_head(dec_feat[sel])

                gt_depth = batch["depth"].to(self.device).float()
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                gt_depth = gt_depth[sel]

                depth_mask = batch["depth_mask"].to(self.device).float()
                if depth_mask.ndim == 3:
                    depth_mask = depth_mask.unsqueeze(1)
                depth_mask = depth_mask[sel]

                fg_mask = batch.get("mask")
                if fg_mask is not None:
                    fg_mask = fg_mask.to(self.device).float()
                    if fg_mask.ndim == 3:
                        fg_mask = fg_mask.unsqueeze(1)
                    fg_mask = (fg_mask[sel] > 0.5).float()
                    d_mask = depth_mask * fg_mask
                else:
                    d_mask = depth_mask

                if gt_depth.shape[-2:] != pred_disp.shape[-2:]:
                    gt_depth = F.interpolate(
                        gt_depth, size=pred_disp.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                if d_mask.shape[-2:] != pred_disp.shape[-2:]:
                    d_mask = F.interpolate(
                        d_mask, size=pred_disp.shape[-2:], mode="area",
                    )

                gt_disp = 1.0 / gt_depth.clamp(min=eps)
                depth_loss = (d_mask * (pred_disp - gt_disp).abs()).sum() / (d_mask.sum() + eps)
                loss = loss + self.depth_loss_weight * depth_loss

                with torch.no_grad():
                    self._wandb_pred_disp = pred_disp[:1].detach()
                    self._wandb_gt_disp = gt_disp[:1].detach()
                    self._wandb_depth_mask = d_mask[:1].detach()
                    self._wandb_depth_step = int(self.global_step)

        wandb_log = {
            "loss": loss,
            "mse_loss": mse_loss
        }
        if identity_loss is not None:
            wandb_log["identity_loss"] = identity_loss
        if dists_loss is not None:
            wandb_log["dists_loss"] = dists_loss
            wandb_log["dists_loss_img"] = d_img
            wandb_log["dists_loss_sobel"] = d_sobel
        if normal_coarse_loss is not None:
            wandb_log["normal_coarse_loss"] = normal_coarse_loss
        if normal_coarse_grad_loss is not None:
            wandb_log["normal_coarse_grad_loss"] = normal_coarse_grad_loss
        if normal_refine_loss is not None:
            wandb_log["normal_refine_loss"] = normal_refine_loss
        if normal_refine_grad_loss is not None:
            wandb_log["normal_refine_grad_loss"] = normal_refine_grad_loss
        if depth_loss is not None:
            wandb_log["depth_loss"] = depth_loss

        '''WanDB logging'''
        if self.global_step % 50 == 0:
            # Generate 4 multiview predictions from a single source image
            with torch.no_grad():
                import math
                
                # Get one source image from batch
                source_img = batch[self.control_key][:1].to(self.device)  # [1, C, H, W]
                if source_img.ndim == 4 and source_img.shape[-1] == 3:
                    source_img = source_img.permute(0, 3, 1, 2)
                source_img_display = torch.clamp(source_img, 0, 1)
                
                # Encode source image to latent
                source_latent = self.encode_first_stage(source_img * 2 - 1).mode().detach()
                
                # Get text conditioning (empty)
                c_text = self.get_learned_conditioning([""])
                
                # Define 4 different yaw angles for multiview (in radians)
                # e.g., -15°, -5°, +5°, +15°
                yaw_angles_deg = [-15, -5, 5, 15]
                
                wandb_images = []
                normal_gt_and_preds = []
                
                wandb_images.append(wandb.Image( # Add source image first
                    source_img_display[0],
                    caption=f"Step {self.global_step} | SOURCE"
                ))

                if "normal" in batch and "normal_mask" in batch:
                    gt_n = batch["normal"][:1].to(self.device)
                    vis_gt = _normal_to_rgb_vis(gt_n)
                    normal_gt_and_preds.append(
                        wandb.Image(
                            vis_gt,
                            caption=f"Step {self.global_step} | GT normal (target view)",
                        )
                    )

                    if (
                        self.use_normal_head
                        and self.normal_head_loss_weight > 0.0
                        and getattr(self, "_wandb_pred_normals_step", -1) == int(self.global_step)
                        and getattr(self, "_wandb_pred_normals", None) is not None
                    ):
                        pred_normals_vis = self._wandb_pred_normals
                        gt_vis = getattr(self, "_wandb_gt_normals", None)
                        nm_pred = self._wandb_normal_mask
                        if nm_pred.ndim == 3:
                            nm_pred = nm_pred.unsqueeze(1)

                        m_rgb = nm_pred[0] if nm_pred.ndim == 4 else nm_pred

                        vis_pred = _normal_to_rgb_vis(pred_normals_vis * nm_pred) * m_rgb
                        normal_gt_and_preds.append(
                            wandb.Image(
                                vis_pred,
                                caption=f"Step {self.global_step} | Pred normal (coarse) * mask",
                            )
                        )

                        if (
                            self.use_normal_refine_head
                            and self.normal_refine_loss_weight > 0.0
                            and getattr(self, "_wandb_pred_normals_refined_step", -1) == int(self.global_step)
                            and getattr(self, "_wandb_pred_normals_refined", None) is not None
                        ):
                            pred_refined_vis = self._wandb_pred_normals_refined
                            vis_refined = _normal_to_rgb_vis(pred_refined_vis * nm_pred) * m_rgb
                            normal_gt_and_preds.append(
                                wandb.Image(
                                    vis_refined,
                                    caption=f"Step {self.global_step} | Pred normal (refined) * mask",
                                )
                            )

                        if gt_vis is not None:
                            vis_gt_masked = _normal_to_rgb_vis(gt_vis * nm_pred) * m_rgb
                            normal_gt_and_preds.append(
                                wandb.Image(
                                    vis_gt_masked,
                                    caption=f"Step {self.global_step} | GT normal * mask (loss target)",
                                )
                            )

                    nm = batch["normal_mask"][:1].to(self.device).float()
                    if nm.ndim == 4:
                        nm = nm[0]
                    m = nm[0] if nm.ndim == 3 else nm
                    m = torch.clamp(m, 0, 1)
                    m_vis = m.unsqueeze(0).expand(3, -1, -1)
                    normal_gt_and_preds.append(
                        wandb.Image(
                            m_vis,
                            caption=f"Step {self.global_step} | GT normal_mask",
                        )
                    )
                
                # Generate prediction for each pose
                for yaw_deg in yaw_angles_deg:
                    yaw_rad = math.radians(yaw_deg)
                    delta_pose_mv = torch.tensor([[0.0, yaw_rad, 0.0]], device=self.device) # Create pose: [pitch, yaw, distance]
                    
                    cond_mv = { # Create conditioning dict for this pose
                        'c_crossattn': [c_text],
                        'c_concat': [source_img],
                        'in_concat': [source_latent],
                        'delta_pose': delta_pose_mv
                    }
                    
                    from ldm.models.diffusion.ddim import DDIMSampler # Sample using DDIM for faster inference
                    sampler = DDIMSampler(self)
                    
                    shape = [4, source_img.shape[2] // 8, source_img.shape[3] // 8]
                    
                    samples, _ = sampler.sample( # Use fewer steps for visualization (faster)
                        S=20,  # Quick sampling
                        batch_size=1,
                        shape=shape,
                        conditioning=cond_mv,
                        verbose=False,
                        unconditional_guidance_scale=1.0,
                        eta=0.0
                    )
                    
                    pred_img = self.decode_first_stage(samples) # Decode to image
                    pred_img = torch.clamp((pred_img + 1) / 2, 0, 1)
                    
                    wandb_images.append(wandb.Image(
                        pred_img[0],
                        caption=f"Step {self.global_step} | Yaw: {yaw_deg}°"
                    ))
                
                wandb_log["multiview_predictions"] = wandb_images
                if normal_gt_and_preds:
                    wandb_log["normal_gt_and_preds"] = normal_gt_and_preds

                if (
                    self.use_depth_head
                    and self.depth_loss_weight > 0.0
                    and getattr(self, "_wandb_depth_step", -1) == int(self.global_step)
                    and getattr(self, "_wandb_pred_disp", None) is not None
                ):
                    def _disp_to_rgb_vis(disp, mask):
                        # Normalize disparity into [0,1] using the valid (masked) range.
                        d = disp[0]  # [1, H, W]
                        m = mask[0] > 0.5
                        if m.any():
                            vals = d[m]
                            lo, hi = vals.min(), vals.max()
                        else:
                            lo, hi = d.min(), d.max()
                        d_norm = ((d - lo) / (hi - lo + 1e-8)).clamp(0, 1)
                        d_norm = d_norm * mask[0]  # zero out invalid pixels
                        return d_norm.expand(3, -1, -1)

                    depth_gt_and_preds = [
                        wandb.Image(
                            _disp_to_rgb_vis(self._wandb_gt_disp, self._wandb_depth_mask),
                            caption=f"Step {self.global_step} | GT disparity (target view)",
                        ),
                        wandb.Image(
                            _disp_to_rgb_vis(self._wandb_pred_disp, self._wandb_depth_mask),
                            caption=f"Step {self.global_step} | Pred disparity (depth_head)",
                        ),
                        wandb.Image(
                            self._wandb_depth_mask[0].expand(3, -1, -1).clamp(0, 1),
                            caption=f"Step {self.global_step} | depth valid mask",
                        ),
                    ]
                    wandb_log["depth_gt_and_preds"] = depth_gt_and_preds

                print(f"[VIS] Logged multiview (+ normals) at step {self.global_step}")

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

        # Normal/depth heads live on self (not inside self.model.diffusion_model).
        if self.normal_head is not None:
            normal_head_params = [p for p in self.normal_head.parameters() if p.requires_grad]
            if len(normal_head_params) > 0:
                param_groups.append({"params": normal_head_params, "lr": self.learning_rate, "name": "normal_head"})
                print(f"[OPT] normal_head params: {len(normal_head_params)}")

        if self.normal_refine_head is not None:
            normal_refine_params = [p for p in self.normal_refine_head.parameters() if p.requires_grad]
            if len(normal_refine_params) > 0:
                param_groups.append({"params": normal_refine_params, "lr": self.learning_rate, "name": "normal_refine_head"})
                print(f"[OPT] normal_refine_head params: {len(normal_refine_params)}")

        if self.depth_head is not None:
            depth_head_params = [p for p in self.depth_head.parameters() if p.requires_grad]
            if len(depth_head_params) > 0:
                param_groups.append({"params": depth_head_params, "lr": self.learning_rate, "name": "depth_head"})
                print(f"[OPT] depth_head params: {len(depth_head_params)}")

        optimizer = torch.optim.AdamW(param_groups)
        
        return optimizer