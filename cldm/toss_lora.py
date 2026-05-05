import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb
import lpips

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

class TossLoraModule(TOSS):
    def __init__(self, lora_config_params, *args, normal_estimator_path="hf:clay3d/omnidata", geometry_loss_weight=0.1, **kwargs):
        kwargs.pop("lora_config_params", None)  # consumed by us
        self.normal_estimator_path = kwargs.pop("normal_estimator_path", normal_estimator_path)
        self.geometry_loss_weight = kwargs.pop("geometry_loss_weight", geometry_loss_weight)
        super().__init__(*args, **kwargs)
        self._normal_estimator = None

        global run
        run = wandb.init(
            entity="wndyd0131-sungkyunkwan-university",
            project="toss-lora",
            config={
                # LoRA config
                "lora_r": lora_config_params.get("r", 16),
                "lora_alpha": lora_config_params.get("lora_alpha", 16),
                "lora_dropout": lora_config_params.get("lora_dropout", 0.0),
                "target_modules": lora_config_params.get("target_modules", []),
                # Model config
                "architecture": "TOSS + LoRA",
                "base_model": "Stable Diffusion UNet"
            },
        )

        # Change pose_net in_feature channel from 51 to 16
        # self.model.diffusion_model.pose_net = nn.Sequential(
        #     nn.Linear(16, 320), # 51 -> 16
        #     nn.SiLU(),
        #     nn.Linear(320, 320)
        # )

        # Disable checkpoints
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
        
        # Verify checkpointing is disabled
        ckpt_enabled_count = 0
        for name, module in self.model.diffusion_model.named_modules():
            if hasattr(module, 'checkpoint') and module.checkpoint:
                ckpt_enabled_count += 1
                print(f"[WARNING] Checkpointing still enabled on: {name}")
        print(f"[INIT] Modules with checkpointing enabled: {ckpt_enabled_count}")

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

        print(f"[INIT] Enabled requires_grad for {lora_count} LoRA parameters")
        for name, shape, req_grad in lora_params_info:
            print(f"  -> {name}: shape={shape}, requires_grad={req_grad}")

        self.model.diffusion_model.print_trainable_parameters()
        
        # Verify PEFT is properly active
        if hasattr(self.model.diffusion_model, 'peft_config'):
            print(f"[INIT] PEFT config active: {self.model.diffusion_model.peft_config}")
        else:
            print("[WARNING] No peft_config found - PEFT may not be properly initialized!")

        # Initialize perceptual loss (LPIPS)
        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        self.lpips_loss.requires_grad_(False)  # Freeze LPIPS network
        print("[INIT] Initialized LPIPS perceptual loss (VGG backbone)")
        
        # Loss weights for hybrid loss
        self.perceptual_weight = 1.0  # Weight for perceptual loss
        self.mse_weight = 0.1  # Small MSE component for stability
        self.mask_min_weight = 0.2  # Soft mask: background contributes 20%, face contributes 100%

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
                "loss_type": "perceptual + mse",
                "perceptual_weight": self.perceptual_weight,
                "mse_weight": self.mse_weight,
                "mask_min_weight": self.mask_min_weight,
                "lpips_backbone": "vgg",
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
        model_output = self.apply_model(x_noisy, t, cond)

        target = noise

        mse_loss = F.mse_loss(model_output, target, reduction="mean")

        '''Perceptual Loss'''
        perceptual_loss = None
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        pred_x0 = (x_noisy - sqrt_one_minus_alphas_cumprod * model_output) / sqrt_alphas_cumprod

        gt_x0 = x  # The clean latent we started with
        
        # Decode to image space for perceptual loss
        # Use torch.no_grad for decoder to save memory (only need gradients through encoder path)
        pred_img = self.decode_first_stage(pred_x0)  # [-1, 1] range
        gt_img = self.decode_first_stage(gt_x0)  # [-1, 1] range
        
        # Compute perceptual loss (LPIPS expects [-1, 1] range)
        # Move LPIPS to same device as images
        self.lpips_loss = self.lpips_loss.to(pred_img.device)
        perceptual_loss = self.lpips_loss(pred_img, gt_img).mean()

        '''Masked Loss'''
        # mask = None
        mask = batch.get("mask")  # Original mask [B, 1, 256, 256]

        if mask is not None:
            mask = mask.to(self.device)
            # Soft mask: mask=1 (face) -> weight=1.0, mask=0 (background) -> weight=min_weight
            soft_mask = mask * (1.0 - self.mask_min_weight) + self.mask_min_weight

            # Masked perceptual loss: mask images before LPIPS so it focuses on head
            img_mask = F.interpolate(soft_mask, size=pred_img.shape[-2:], mode="bilinear", align_corners=False)
            masked_perceptual_loss = self.lpips_loss(pred_img * img_mask, gt_img * img_mask).mean()

            # Masked latent MSE, normalized by mask sum to avoid diluting head signal
            latent_mask = F.interpolate(soft_mask, size=model_output.shape[-2:], mode="area")
            masked_mse_loss = (F.mse_loss(model_output, noise, reduction="none") * latent_mask).sum() / latent_mask.sum()

            loss = self.perceptual_weight * masked_perceptual_loss + self.mse_weight * masked_mse_loss
            print(f"MASKED LOSS: perceptual={masked_perceptual_loss.item():.4f}, mse={masked_mse_loss.item():.4f}")
        elif perceptual_loss is not None and mse_loss is not None:
            loss = self.perceptual_weight * perceptual_loss + self.mse_weight * mse_loss
            print(f"LOSS: perceptual={perceptual_loss.item():.4f}, mse={mse_loss.item():.4f}, total={loss.item():.4f}")
        else:
            loss = mse_loss

        ''' Geometry loss (frozen DPT-Hybrid proxy) '''
        geom_loss = torch.tensor(0.0, device=self.device)
        if self.geometry_loss_weight > 0 and "normal" in batch and "normal_mask" in batch:
            pred_imgs = torch.clamp((pred_img + 1) / 2, 0, 1)
            if pred_imgs.ndim == 4 and pred_imgs.shape[-1] == 3:
                pred_imgs = pred_imgs.permute(0, 3, 1, 2)
            pred_normals = self.normal_estimator(pred_imgs)
            gt_normals = batch["normal"].to(self.device)
            normal_mask = batch["normal_mask"].to(self.device)
            geom_loss = _cosine_similarity_loss(pred_normals, gt_normals, normal_mask)
            loss = loss + self.geometry_loss_weight * geom_loss

        wandb_log = {
            "loss": loss,
            "perceptual_loss": perceptual_loss,
            "mse_loss": mse_loss,
            "geometry_loss": geom_loss,
        }

        '''WanDB logging'''
        if batch_idx % 50 == 0:
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
                    
                    pred_imgs_nchw = pred_img
                    if pred_imgs_nchw.ndim == 4 and pred_imgs_nchw.shape[-1] == 3:
                        pred_imgs_nchw = pred_imgs_nchw.permute(0, 3, 1, 2)
                    pred_normals_dpt = self.normal_estimator(pred_imgs_nchw)
                    vis_n = _normal_to_rgb_vis(pred_normals_dpt)
                    normal_gt_and_preds.append(
                        wandb.Image(
                            vis_n,
                            caption=f"Step {self.global_step} | Normal pred (DPT) | Pred",
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
        
        optimizer = torch.optim.AdamW(param_groups)
        
        return optimizer