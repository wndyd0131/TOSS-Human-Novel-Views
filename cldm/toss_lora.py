import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn

class TossLoraModule(TOSS):
    def __init__(self, lora_config_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # internal_unet = self.model.diffusion_model

        # # Turn off checkpointing manually
        # internal_unet.use_checkpoint = False

        # Change pose_net in_feature channel from 51 to 16
        self.model.diffusion_model.pose_net = nn.Sequential(
            nn.Linear(16, 320), # 51 -> 16
            nn.SiLU(),
            nn.Linear(320, 320)
        )

        unet = self.model.diffusion_model
        def disable_all_ckpt(m):
            for attr in ["use_checkpoint", "checkpoint", "use_checkpointing"]:
                if hasattr(m, attr):
                    setattr(m, attr, False)

        unet.apply(disable_all_ckpt)

        # 1. Freeze the base model
        self.requires_grad_(False)

        # 2. Configure LoRA
        peft_config = LoraConfig(**lora_config_params)
        self.model.diffusion_model = get_peft_model(self.model.diffusion_model, peft_config)

        for n, p in self.model.diffusion_model.named_parameters():
            if "pose_net" in n:
                p.requires_grad = True

        for n, p in self.model.diffusion_model.named_parameters():
            # print(n)
            if "base_model.model.out." in n:
                p.requires_grad = True

        # Posenet
        for param in self.model.diffusion_model.pose_net.parameters():
            param.requires_grad = True

        self.model.diffusion_model.print_trainable_parameters()

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

        x, cond = self.get_input(batch, self.first_stage_key)

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

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

        # loss, loss_dict = self.shared_step(batch)

        # if mask is not None:
        #     error = (loss_dict["pred_noise"] - loss_dict["target_noise"]) ** 2
        #     loss = (error * mask).mean()

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.diffusion_model.parameters() if p.requires_grad]

        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        # lora_params = [p for n, p in self.named_parameters() if "lora" in n and p.requires_grad]
        # pose_params = [p for n, p in self.named_parameters() if "pose_net" in n and p.requires_grad]

        # return torch.optim.AdamW([
        #     {"params": lora_params, "lr": 1e-4}, 
        #     {"params": pose_params, "lr": 1e-3} # 10x faster learning for 3D parameters
        # ])