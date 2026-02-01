import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="wndyd0131-sungkyunkwan-university",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

class TossLoraModule(TOSS):
    def __init__(self, lora_config_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # # PoseNet
        # for n, p in self.model.diffusion_model.named_parameters():
        #     if "pose_net" in n:
        #         p.requires_grad = True

        # # Posenet
        # for param in self.model.diffusion_model.pose_net.parameters():
        #     param.requires_grad = True
        
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
                        # Initialize with small random values (like 1e-6 scale)
                        p.normal_(mean=0.0, std=1e-6)
                        
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

    def on_save_checkpoint(self, checkpoint):
        # We override this to prevent the parent class from 
        # trying to call self.embedding_manager.save()
        
        # If you want to keep the checkpoint small (LoRA only),
        # you can clear the base model weights from the checkpoint:
        # checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if "lora" in k or "pose_net" in k}
        pass

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

        run.log({"loss": loss})

        if batch_idx % 500 == 0:
            # Dec`ode
            with torch.no_grad():
                out = self.decode_first_stage(model_output)
                out = torch.clamp((out + 1) / 2, 0, 1)
                wandb_images = [
                    wandb.Image(img, caption=f"Step {self.global_step}_Idx_{i}") 
                    for i, img in enumerate(out)
                ]
                
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
            found_lora = False
            for n, p in self.model.diffusion_model.named_parameters():
                if "lora" in n.lower():
                    found_lora = True
                    if p.grad is not None:
                        grad_norm = p.grad.abs().mean().item()
                        weight_norm = p.abs().mean().item()
                        print(f"[GRAD] {n}: grad={grad_norm:.6f}, weight={weight_norm:.6f}")
                    else:
                        print(f"[GRAD] {n}: grad=None, weight={p.abs().mean().item():.6f}")
            if not found_lora:
                print("[GRAD] No LoRA parameters found in model!")


    def configure_optimizers(self):
        # Explicitly collect LoRA and other trainable params by name
        lora_params = []
        other_params = []
        
        # Use named_parameters to ensure we get the actual parameter objects
        param_dict = {}
        for n, p in self.model.diffusion_model.named_parameters():
            if p.requires_grad:
                param_dict[n] = p
                if "lora" in n.lower():
                    lora_params.append(p)
                    print(f"[OPT] LoRA param: {n}, shape={p.shape}, requires_grad={p.requires_grad}")
                else:
                    other_params.append(p)
        
        print(f"[OPT] LoRA params: {len(lora_params)}, Other trainable params: {len(other_params)}")
        print(f"[OPT] Total params in optimizer: {len(lora_params) + len(other_params)}")
        
        if len(lora_params) == 0:
            print("[WARNING] No LoRA params found! Check if PEFT is properly configured.")
            # Fallback: try to get params differently
            for n, p in self.named_parameters():
                if "lora" in n.lower() and p.requires_grad:
                    lora_params.append(p)
                    print(f"[OPT-FALLBACK] Found LoRA param: {n}")
        
        all_params = lora_params + other_params
        
        # Use separate param groups with explicit lr for LoRA
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": self.learning_rate, "name": "lora"},
            {"params": other_params, "lr": self.learning_rate, "name": "other"}
        ])
        
        return optimizer

        # lora_params = [p for n, p in self.named_parameters() if "lora" in n and p.requires_grad]
        # pose_params = [p for n, p in self.named_parameters() if "pose_net" in n and p.requires_grad]

        # return torch.optim.AdamW([
        #     {"params": lora_params, "lr": 1e-4}, 
        #     {"params": pose_params, "lr": 1e-3} # 10x faster learning for 3D parameters
        # ])