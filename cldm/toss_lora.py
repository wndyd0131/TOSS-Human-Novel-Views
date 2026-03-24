import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb

run = None


def _cosine_similarity_loss(pred_normals, gt_normals, mask, eps=1e-8):
    """Masked cosine similarity loss. pred, gt: [B,3,H,W] L2-normalized. mask: [B,1,H,W]."""
    cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True).clamp(-1, 1)
    loss = 1 - cos_sim  # [B,1,H,W]
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + eps)
    return loss.mean()


def _geometry_perceptual_loss(pred_features, gt_features, mask=None, eps=1e-8):
    """L2 loss between geometry features. pred_features, gt_features: [B,C,H,W]. mask: [B,1,H,W] optional."""
    if pred_features.shape != gt_features.shape:
        gt_features = F.interpolate(gt_features, size=pred_features.shape[2:], mode="bilinear", align_corners=False)
    if mask is not None:
        mask = F.interpolate(mask.float(), size=pred_features.shape[2:], mode="area")
        diff = (pred_features - gt_features) ** 2
        diff = diff.mean(dim=1, keepdim=True) * mask
        return diff.sum() / (mask.sum() + eps)
    return F.mse_loss(pred_features, gt_features)


class TossLoraModule(TOSS):
    def __init__(self, lora_config_params, *args, normal_estimator_path="hf:clay3d/omnidata", geometry_loss_weight=0.1, geometry_perceptual_loss_weight=0.1, **kwargs):
        kwargs.pop("lora_config_params", None)  # consumed by us
        self.normal_estimator_path = kwargs.pop("normal_estimator_path", normal_estimator_path)
        self.geometry_loss_weight = kwargs.pop("geometry_loss_weight", geometry_loss_weight)
        self.geometry_perceptual_loss_weight = kwargs.pop("geometry_perceptual_loss_weight", geometry_perceptual_loss_weight)
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

        # internal_unet = self.model.diffusion_model

        # # Turn off checkpointing manually
        # internal_unet.use_checkpoint = False

        # Change pose_net in_feature channel from 51 to 16
        # self.model.diffusion_model.pose_net = nn.Sequential(
        #     nn.Linear(16, 320), # 51 -> 16
        #     nn.SiLU(),
        #     nn.Linear(320, 320)
        # )

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

        # 2b. Unfreeze pose_net for human face adaptation
        # The pose_net was trained on objects - human head poses have different distributions
        pose_net_count = 0
        for n, p in self.model.diffusion_model.named_parameters():
            if "pose_net" in n:
                p.requires_grad = True
                pose_net_count += 1
                print(f"[INIT] Unfreezing pose_net param: {n}, shape={p.shape}")
        print(f"[INIT] Unfroze {pose_net_count} pose_net parameters")
        
        # 3. Explicitly enable gradients for LoRA parameters (in case PEFT didn't)
        lora_count = 0
        lora_params_info = []
        for n, p in self.model.diffusion_model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
                lora_count += 1
                lora_params_info.append((n, p.shape, p.requires_grad))

                # CRITICAL FIX: Initialize lora_B with small non-zero values
                # This prevents gradient attribution issues when lora_B is all zeros
                if "lora_B" in n and p.abs().sum() == 0:
                    print(f"[INIT] Fixing zero-initialized lora_B: {n}")
                    with torch.no_grad():
                        # Use larger initialization for better gradient flow
                        p.normal_(mean=0.0, std=1e-3)
                        
            # Also enable output layers if needed
            if "base_model.model.out." in n:
                p.requires_grad = True

        print(f"[INIT] Enabled requires_grad for {lora_count} LoRA parameters")
        for name, shape, req_grad in lora_params_info:
            print(f"  -> {name}: shape={shape}, requires_grad={req_grad}")

        self.model.diffusion_model.print_trainable_parameters()
        
        # Verify PEFT is properly active
        if hasattr(self.model.diffusion_model, 'peft_config'):
            print(f"[INIT] PEFT config active: {self.model.diffusion_model.peft_config}")
        else:
            print("[WARNING] No peft_config found - PEFT may not be properly initialized!")
        
        # Debug: Check LoRA layer scaling and adapter status
        self._debug_lora_setup()
        
        # Debug: Check pose_net weights
        self._debug_pose_net_weights()

    @property
    def normal_estimator(self):
        """Lazy-load frozen DPT-Hybrid normal estimator."""
        if self._normal_estimator is None:
            from ldm.modules.midas.api import DPTNormalInference
            self._normal_estimator = DPTNormalInference(self.normal_estimator_path).to(self.device)
            print(f"[INIT] Loaded DPT-Hybrid normal estimator from {self.normal_estimator_path}")
        return self._normal_estimator

    def _register_lora_debug_hook(self):
        """Register hooks to debug LoRA forward pass"""
        from peft.tuners.lora import Linear as LoraLinear
        
        def make_hook(name):
            def hook(module, input, output):
                # Check if LoRA is actually contributing
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    x = input[0]
                    for adapter_name in module.lora_A.keys():
                        lora_A = module.lora_A[adapter_name]
                        lora_B = module.lora_B[adapter_name]
                        scaling = module.scaling.get(adapter_name, 1.0)
                        
                        # Compute LoRA contribution manually
                        lora_out = lora_B(lora_A(x)) * scaling
                        
                        print(f"\n[HOOK] {name}:")
                        print(f"  input mean: {x.abs().mean().item():.6f}")
                        print(f"  lora_A(x) mean: {lora_A(x).abs().mean().item():.6f}")
                        print(f"  lora_B(lora_A(x)) mean: {lora_B(lora_A(x)).abs().mean().item():.6f}")
                        print(f"  scaling: {scaling}")
                        print(f"  lora_contribution mean: {lora_out.abs().mean().item():.6f}")
                        print(f"  output mean: {output.abs().mean().item():.6f}")
                        break
            return hook
        
        # Find and hook one LoRA layer
        for name, module in self.model.diffusion_model.named_modules():
            if isinstance(module, LoraLinear) and 'to_q' in name:
                module.register_forward_hook(make_hook(name))
                print(f"[DEBUG] Registered forward hook on {name}")
                
                # Also register backward hook on the lora_B weight
                for adapter_name in module.lora_B.keys():
                    lora_B = module.lora_B[adapter_name]
                    def backward_hook(grad):
                        print(f"\n[BACKWARD HOOK] lora_B gradient received!")
                        print(f"  grad shape: {grad.shape if grad is not None else None}")
                        print(f"  grad mean: {grad.abs().mean().item() if grad is not None else None}")
                        return grad
                    lora_B.weight.register_hook(backward_hook)
                    print(f"[DEBUG] Registered backward hook on lora_B.weight")
                    break
                break

    def _debug_lora_setup(self):
        """Debug helper to verify LoRA is properly configured"""
        from peft.tuners.lora import LoraLayer
        
        print("\n[DEBUG] LoRA Layer Analysis:")
        for name, module in self.model.diffusion_model.named_modules():
            if isinstance(module, LoraLayer):
                # Check scaling
                scaling = getattr(module, 'scaling', {})
                # Check if adapter is disabled
                disable_adapters = getattr(module, 'disable_adapters', False)
                # Check merged status  
                merged = getattr(module, 'merged', False)
                
                print(f"  {name}:")
                print(f"    scaling: {scaling}")
                print(f"    disable_adapters: {disable_adapters}")
                print(f"    merged: {merged}")
                
                # Check lora_A and lora_B
                if hasattr(module, 'lora_A'):
                    for adapter_name, lora_a in module.lora_A.items():
                        print(f"    lora_A[{adapter_name}]: shape={lora_a.weight.shape}, requires_grad={lora_a.weight.requires_grad}")
                if hasattr(module, 'lora_B'):
                    for adapter_name, lora_b in module.lora_B.items():
                        print(f"    lora_B[{adapter_name}]: shape={lora_b.weight.shape}, requires_grad={lora_b.weight.requires_grad}")
                break  # Only check one to avoid spam
        
        # Check active adapter
        if hasattr(self.model.diffusion_model, 'active_adapter'):
            print(f"\n[DEBUG] Active adapter: {self.model.diffusion_model.active_adapter}")
        if hasattr(self.model.diffusion_model, 'active_adapters'):
            print(f"[DEBUG] Active adapters: {self.model.diffusion_model.active_adapters}")

    def _debug_pose_net_weights(self):
        """Debug helper to inspect pose_net weights"""
        print("\n[DEBUG] Pose Net Weight Analysis:")
        
        # Access pose_net (may be wrapped by PEFT)
        unet = self.model.diffusion_model
        pose_net = None
        
        # Try to find pose_net in the model
        if hasattr(unet, 'base_model'):
            # PEFT wrapped model
            if hasattr(unet.base_model, 'model') and hasattr(unet.base_model.model, 'pose_net'):
                pose_net = unet.base_model.model.pose_net
        elif hasattr(unet, 'pose_net'):
            pose_net = unet.pose_net
        
        if pose_net is None:
            print("  [WARNING] pose_net not found!")
            return
        
        print(f"  pose_net structure: {pose_net}")
        print(f"  pose_enc type: {getattr(unet.base_model.model if hasattr(unet, 'base_model') else unet, 'pose_enc', 'unknown')}")
        
        # Iterate through pose_net layers
        for name, param in pose_net.named_parameters():
            print(f"\n  Layer: pose_net.{name}")
            print(f"    shape: {param.shape}")
            print(f"    requires_grad: {param.requires_grad}")
            print(f"    mean: {param.data.mean().item():.6f}")
            print(f"    std: {param.data.std().item():.6f}")
            print(f"    min: {param.data.min().item():.6f}")
            print(f"    max: {param.data.max().item():.6f}")
            print(f"    abs_mean: {param.data.abs().mean().item():.6f}")
            
            # Check if weights look initialized (not all zeros)
            if param.data.abs().sum() == 0:
                print(f"    [WARNING] All zeros - may not be loaded properly!")
            
        # Print input/output dimensions
        if hasattr(pose_net, '0') and hasattr(pose_net[0], 'in_features'):
            print(f"\n  Input features (pose_net[0].in_features): {pose_net[0].in_features}")
        if hasattr(pose_net, '2') and hasattr(pose_net[2], 'out_features'):
            print(f"  Output features (pose_net[2].out_features): {pose_net[2].out_features}")

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
            }, allow_val_change=True)

    def training_step(self, batch, batch_idx):
        # use of delta pose
        # masking
        # masked loss

        # Ensure model is in training mode
        self.model.diffusion_model.train()
        
        # CRITICAL: Explicitly enable LoRA adapters for PEFT
        if hasattr(self.model.diffusion_model, 'enable_adapters'):
            self.model.diffusion_model.enable_adapters()
        
        # Debug: verify LoRA is active on first step
        if batch_idx == 0 and self.global_step == 0:
            print(f"[TRAIN] Training step 0, verifying LoRA setup...")
            lora_active = 0
            for n, m in self.model.diffusion_model.named_modules():
                if 'lora' in n.lower():
                    lora_active += 1
            print(f"[TRAIN] Found {lora_active} LoRA modules in forward path")
            
            # Register a hook to check LoRA layer output
            self._register_lora_debug_hook()

        x, cond = self.get_input(batch, self.first_stage_key)

        print("DEBUG_DELTA_POSE:", cond['delta_pose'][0])

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        # Note: x_noisy doesn't need requires_grad - we only need gradients for model params
        model_output = self.apply_model(x_noisy, t, cond)

        target = noise

        # MSE
        loss = (model_output - target) ** 2

        # if torch.rand(1) < 0.1:
        #     cond['in_concat'][0] = torch.zeros_like(cond['in_concat'][0])

        # loss, loss_dict = self.forward(x, cond)


        mask = batch.get("mask") # Original mask [B, 1, 256, 256]
        # Latent mask
        if mask is not None:
            latent_mask = F.interpolate(mask, size=loss.shape[-2:], mode="area")
            
            loss = loss * latent_mask
            
            loss = loss.sum() / (latent_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        # Geometry losses (frozen DPT-Hybrid proxy)
        if (self.geometry_loss_weight > 0 or self.geometry_perceptual_loss_weight > 0) and "normal" in batch and "normal_mask" in batch:
            x0_pred = x - model_output
            pred_imgs = self.decode_first_stage(x0_pred)
            pred_imgs = torch.clamp((pred_imgs + 1) / 2, 0, 1)
            if pred_imgs.ndim == 4 and pred_imgs.shape[-1] == 3:
                pred_imgs = pred_imgs.permute(0, 3, 1, 2)
            gt_imgs = batch[self.first_stage_key].to(self.device)
            if gt_imgs.ndim == 4 and gt_imgs.shape[-1] == 3:
                gt_imgs = gt_imgs.permute(0, 3, 1, 2)
            gt_imgs = torch.clamp(gt_imgs, 0, 1)
            normal_mask = batch["normal_mask"].to(self.device)

            if self.geometry_loss_weight > 0:
                pred_normals = self.normal_estimator(pred_imgs)
                gt_normals = batch["normal"].to(self.device)
                geom_loss = _cosine_similarity_loss(pred_normals, gt_normals, normal_mask)
                loss = loss + self.geometry_loss_weight * geom_loss
                if run is not None:
                    run.log({"geometry_loss": geom_loss})

            if self.geometry_perceptual_loss_weight > 0:
                pred_feats = self.normal_estimator.forward_features(pred_imgs)
                gt_feats = self.normal_estimator.forward_features(gt_imgs)
                geom_perc_loss = _geometry_perceptual_loss(pred_feats, gt_feats, normal_mask)
                loss = loss + self.geometry_perceptual_loss_weight * geom_perc_loss
                if run is not None:
                    run.log({"geometry_perceptual_loss": geom_perc_loss})

        run.log({"loss": loss})
        print("LOSS_SHAPE:", loss.shape)

        if batch_idx % 500 == 0:
            # Decode
            source_img = batch[self.control_key].to(self.device)
            print("SOURCE_IMG_BATCH:", source_img.shape)
            with torch.no_grad():
                # Get source images (hint/control) from batch
                source_imgs = batch[self.control_key].to(self.device)
                if source_imgs.ndim == 4 and source_imgs.shape[-1] == 3:
                    source_imgs = source_imgs.permute(0, 3, 1, 2)  # HWC -> CHW
                source_imgs = torch.clamp(source_imgs, 0, 1)
                
                # Get target images (ground truth)
                target_imgs = batch[self.first_stage_key].to(self.device)
                if target_imgs.ndim == 4 and target_imgs.shape[-1] == 3:
                    target_imgs = target_imgs.permute(0, 3, 1, 2)  # HWC -> CHW
                target_imgs = torch.clamp(target_imgs, 0, 1)

                # Decode model predictions
                pred_imgs = self.decode_first_stage(x - model_output)
                pred_imgs = torch.clamp((pred_imgs + 1) / 2, 0, 1)

                print("PRED_IMG_SHAPE:", pred_imgs.shape)

                # Get pose information
                delta_pose = cond['delta_pose']  # [B, pose_dim]

                # Wandb
                # Create comparison images for each sample in batch
                wandb_images = []
                for i in range(min(pred_imgs.shape[0], 4)):  # Limit to 4 samples
                    # Format pose info for caption
                    pose_vals = delta_pose[i].cpu().numpy()
                    pose_str = f"Pose: [{pose_vals[0]:.2f}, {pose_vals[1]:.2f}, {pose_vals[2]:.2f}, ...]" if len(pose_vals) > 3 else f"Pose: {pose_vals}"
                    
                    # Log source image
                    wandb_images.append(wandb.Image(
                        source_imgs[i], 
                        caption=f"Step {self.global_step} | Sample {i} | SOURCE"
                    ))
                    
                    # Log target (ground truth)
                    wandb_images.append(wandb.Image(
                        target_imgs[i], 
                        caption=f"Step {self.global_step} | Sample {i} | TARGET | {pose_str}"
                    ))
                    
                    # Log prediction
                    wandb_images.append(wandb.Image(
                        pred_imgs[i], 
                        caption=f"Step {self.global_step} | Sample {i} | PREDICTION | {pose_str}"
                    ))

                # 5. Log to the active run
                run.log({"visual_predictions": wandb_images})

        # loss, loss_dict = self.shared_step(batch)

        # if mask is not None:
        #     error = (loss_dict["pred_noise"] - loss_dict["target_noise"]) ** 2
        #     loss = (error * mask).mean()

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_after_backward(self):
        """Called after loss.backward() - check if LoRA gets gradients"""
        if self.global_step % 100 == 0:
            lora_grads = []
            pose_net_grads = []
            for n, p in self.model.diffusion_model.named_parameters():
                if "lora" in n.lower():
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        grad_mean = p.grad.abs().mean().item()
                        grad_max = p.grad.abs().max().item()
                        lora_grads.append((n, grad_norm, grad_mean, grad_max))
                    else:
                        lora_grads.append((n, None, None, None))
                        
                elif "pose_net" in n:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        pose_net_grads.append((n, grad_norm))
                    else:
                        pose_net_grads.append((n, None))
            
            # Print LoRA gradients
            print(f"  LoRA params ({len(lora_grads)}):")
            has_nonzero_grad = False
            for name, norm, mean, max_val in lora_grads:
                if norm is not None:
                    if norm > 1e-10:
                        has_nonzero_grad = True
                    print(f"    {name.split('.')[-3]}: norm={norm:.2e}, mean={mean:.2e}, max={max_val:.2e}")
                else:
                    print(f"    {name.split('.')[-3]}: grad=None!")
            
            if not has_nonzero_grad:
                print("  [WARNING] All LoRA gradients are zero or None!")
            
            # Print pose_net gradients  
            print(f"  pose_net params ({len(pose_net_grads)}):")
            for name, norm in pose_net_grads:
                if norm is not None:
                    print(f"    {name}: norm={norm:.2e}")
                else:
                    print(f"    {name}: grad=None!")


    def configure_optimizers(self):
        # Explicitly collect LoRA and pose_net params separately
        lora_params = []
        pose_net_params = []
        other_params = []
        
        # Use named_parameters to ensure we get the actual parameter objects
        for n, p in self.model.diffusion_model.named_parameters():
            if p.requires_grad:
                if "lora" in n.lower():
                    lora_params.append(p)
                    print(f"[OPT] LoRA param: {n}, shape={p.shape}")
                elif "pose_net" in n:
                    pose_net_params.append(p)
                    print(f"[OPT] pose_net param: {n}, shape={p.shape}")
                else:
                    other_params.append(p)
                    print(f"[OPT] Other param: {n}, shape={p.shape}")
        
        print(f"\n[OPT] Summary:")
        print(f"  LoRA params: {len(lora_params)}")
        print(f"  pose_net params: {len(pose_net_params)}")
        print(f"  Other trainable params: {len(other_params)}")
        print(f"  Total params in optimizer: {len(lora_params) + len(pose_net_params) + len(other_params)}")
        
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
        
        if len(pose_net_params) > 0:
            pose_net_lr = self.learning_rate * 0.1  # 10x lower than LoRA
            param_groups.append({"params": pose_net_params, "lr": pose_net_lr, "name": "pose_net"})
            print(f"[OPT] pose_net learning rate: {pose_net_lr} (0.1x of LoRA lr: {self.learning_rate})")
        
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": self.learning_rate, "name": "other"})
        
        optimizer = torch.optim.AdamW(param_groups)
        
        return optimizer