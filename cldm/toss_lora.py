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
        corr_debug_steps=5,
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
        super().__init__(*args, **kwargs)
        self._normal_estimator = None
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
            }, allow_val_change=True)

    @staticmethod
    def _as_bchw(t):
        """Ensure a per-location map is [B, 1, H, W]. Accepts [B, H, W] or [B, 1, H, W]."""
        if t.dim() == 3:
            t = t.unsqueeze(1)
        return t

    def _compute_corr_loss(self, batch, x, debug=False):
        """Correlation feature regularization on the captured last-output-block feature.

        Returns (loss_corr, log_dict) or (None, {}) if inputs are unavailable.
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

        # Warp projected source features into the target coordinate frame.
        # flow_tgt2src is a normalized grid_sample grid: [B, Hf, Wf, 2], last dim = (x=width, y=height) in [-1, 1].
        flow = batch["flow_tgt2src"].to(device=p_src.device, dtype=p_src.dtype)
        if flow.shape[1:3] != feat_hw:
            # normalized coords are resolution-independent -> spatially resize the grid
            flow = F.interpolate(flow.permute(0, 3, 1, 2), size=feat_hw, mode="bilinear", align_corners=False)
            flow = flow.permute(0, 2, 3, 1).contiguous()

        warped_src = F.grid_sample(p_src, flow, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B, D, Hf, Wf]

        corr_pred = (warped_src * p_tgt).sum(dim=1, keepdim=True)  # [B, 1, Hf, Wf]

        corr_gt = self._as_bchw(batch["corr_gt"].to(self.device).float())
        valid_mask = self._as_bchw(batch["valid_mask"].to(self.device).float())

        # Resize GT/mask to the feature resolution if they differ.
        if corr_gt.shape[-2:] != feat_hw:
            corr_gt = F.interpolate(corr_gt, size=feat_hw, mode="bilinear", align_corners=False)
        if valid_mask.shape[-2:] != feat_hw:
            valid_mask = F.interpolate(valid_mask, size=feat_hw, mode="nearest")

        loss_corr = (((corr_pred - corr_gt) ** 2) * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        log = {"corr_loss": loss_corr}

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
                    f"flow(min={flow.min().item():.3f},max={flow.max().item():.3f}) | "
                    f"warped_src_norm(mean={warped_norm.mean().item():.4f}) | "
                    f"corr_pred(min={corr_pred.min().item():.4f},mean={corr_pred.mean().item():.4f},max={corr_pred.max().item():.4f}) | "
                    f"corr_gt@valid(min={gt_stats[0]:.4f},mean={gt_stats[1]:.4f},max={gt_stats[2]:.4f}) | "
                    f"valid_ratio={valid_mask.mean().item():.4f} | "
                    f"loss_corr={loss_corr.item():.6f}"
                )

        return loss_corr, log

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
        model_output = self.apply_model(x_noisy, t, cond)

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
        d_img = None
        d_sobel = None
        corr_loss = None

        '''Correlation feature regularization'''
        if self.lambda_corr > 0.0:
            debug_corr = int(self.global_step) < self.corr_debug_steps
            corr_loss, corr_log = self._compute_corr_loss(batch, x, debug=debug_corr)
            if corr_loss is not None:
                loss = loss + self.lambda_corr * corr_loss

        need_identity = self.identity_loss_weight > 0.0
        need_dists    = self.dists_loss_weight    > 0.0

        # Decode pred_rgb / gt_rgb ONCE on the union mask so identity and DISTS
        # share a single VAE-decoder forward + backward graph (memory win:
        # avoids running the decoder twice). t가 높을 경우 너무 noisy하기 때문에
        # 예측이 불안정하여, timestep가 높은 경우에는 비교하지 않음.
        if need_identity or need_dists:
            union_cut = max(
                self.identity_t_cut if need_identity else 0,
                self.dists_t_cut    if need_dists    else 0,
            )
            sel = t < union_cut
            if torch.any(sel):
                x0_pred = self.predict_start_from_noise(x_noisy[sel], t[sel], model_output[sel]) # clean latent
                pred_img = self._decode_first_stage_train(x0_pred) # latent to image, gradients reach RGB
                pred_rgb = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0) # [-1, 1] to [0, 1]
                gt_rgb = self._gt_rgb_01_from_batch(batch)[sel] # rgb to [0, 1]
                t_sel = t[sel]

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
        if corr_loss is not None:
            wandb_log["corr_loss"] = corr_loss

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