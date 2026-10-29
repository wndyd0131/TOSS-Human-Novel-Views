"""WandB session, scalar logging, and training-step gradient diagnostics for TossLoraModule."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

import pytz
import torch
import wandb

if TYPE_CHECKING:
    from cldm.toss_lora import TossLoraModule

DEFAULT_WANDB_ENTITY = "wndyd0131-sungkyunkwan-university"
DEFAULT_WANDB_PROJECT = "toss-lora"


def make_run_name_kst() -> str:
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    return f"toss-lora_{now_kst}"


def build_init_config(module: TossLoraModule, lora_config_params: dict) -> dict[str, Any]:
    return {
        "lora_r": lora_config_params.get("r", 16),
        "lora_alpha": lora_config_params.get("lora_alpha", 16),
        "lora_dropout": lora_config_params.get("lora_dropout", 0.0),
        "target_modules": lora_config_params.get("target_modules", []),
        "architecture": "TOSS + LoRA",
        "base_model": "Stable Diffusion UNet",
        "identity_loss_weight": module.identity_loss_weight,
        "identity_t_cut": module.identity_t_cut,
        "arcface_ckpt_path": module.arcface_ckpt_path,
        "arcface_spatial_mode": module.arcface_spatial_mode,
        "dists_loss_weight": module.dists_loss_weight,
        "dists_t_cut": module.dists_t_cut,
        "lambda_corr": module.lambda_corr,
        "corr_embed_dim": module.corr_embed_dim,
        "corr_proj_trainable": module.corr_proj_trainable,
        "corr_flow_src_size": module.corr_flow_src_size,
    }


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


class WandbSession:
    def __init__(self, run):
        self.run = run

    @classmethod
    def start(cls, module: TossLoraModule, lora_config_params: dict) -> WandbSession:
        run = wandb.init(
            entity=DEFAULT_WANDB_ENTITY,
            project=DEFAULT_WANDB_PROJECT,
            name=make_run_name_kst(),
            config=build_init_config(module, lora_config_params),
        )
        return cls(run)

    def update_trainer(self, module: TossLoraModule, trainer) -> None:
        if self.run is None or trainer is None:
            return
        self.run.config.update({
            "max_epochs": trainer.max_epochs,
            "max_steps": trainer.max_steps,
            "batch_size": (
                trainer.datamodule.batch_size
                if hasattr(trainer, "datamodule") and trainer.datamodule
                else None
            ),
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "gradient_clip_val": trainer.gradient_clip_val,
            "learning_rate": module.learning_rate,
            "image_size": module.image_size,
            "timesteps": module.num_timesteps,
            "loss_type": "mse",
            "mse_weight": module.mse_weight,
            "identity_loss_weight": module.identity_loss_weight,
            "identity_t_cut": module.identity_t_cut,
            "dists_loss_weight": module.dists_loss_weight,
            "dists_t_cut": module.dists_t_cut,
            "lambda_corr": module.lambda_corr,
            "corr_embed_dim": module.corr_embed_dim,
            "corr_proj_trainable": module.corr_proj_trainable,
            "corr_flow_src_size": module.corr_flow_src_size,
        }, allow_val_change=True)

    def log_train(self, log_dict: dict, step: int, total_loss_item: float | None = None) -> None:
        if self.run is None:
            return
        if total_loss_item is not None:
            print(f"LOSS logged: total={total_loss_item:.4f}")
        self.run.log(log_dict, step=step)

    def log_validation(self, callback_metrics, step: int) -> None:
        if self.run is None:
            return
        metrics = {}
        for key, value in callback_metrics.items():
            if not str(key).startswith("val/"):
                continue
            if torch.is_tensor(value):
                metrics[key] = value.detach().item()
            else:
                metrics[key] = value
        if metrics:
            self.run.log(metrics, step=step)


def append_grad_diagnostics(
    wandb_log: dict,
    module: TossLoraModule,
    mse_for_grad,
    corr_loss,
    dists_loss,
) -> None:
    if int(module.global_step) % 50 != 0:
        return

    lora_params = [
        p for n, p in module.model.diffusion_model.named_parameters()
        if p.requires_grad and "lora" in n.lower()
    ]
    finetune_params = [
        p for n, p in module.model.diffusion_model.named_parameters()
        if p.requires_grad and (
            "pose_net" in n
            or "vae_proj" in n
            or "base_model.model.out." in n
        )
    ]
    corr_proj_params = [p for p in module.corr_proj.parameters() if p.requires_grad]

    wandb_log["mse_grad_lora"] = _grad_norm_l2(mse_for_grad, lora_params)

    if corr_loss is not None:
        weighted_corr = module.lambda_corr * corr_loss
        wandb_log["corr_grad_lora"] = _grad_norm_l2(weighted_corr, lora_params)
        wandb_log["corr_grad_finetune"] = _grad_norm_l2(weighted_corr, finetune_params)
        wandb_log["corr_grad_proj"] = _grad_norm_l2(weighted_corr, corr_proj_params)
        mse_g = wandb_log["mse_grad_lora"]
        corr_g = wandb_log["corr_grad_lora"]
        wandb_log["corr_mse_grad_ratio_lora"] = corr_g / (mse_g + 1e-12)

    if dists_loss is not None:
        weighted_dists = module.dists_loss_weight * dists_loss
        wandb_log["dists_grad_lora"] = _grad_norm_l2(weighted_dists, lora_params)
        wandb_log["dists_grad_finetune"] = _grad_norm_l2(weighted_dists, finetune_params)
        wandb_log["dists_mse_grad_ratio_lora"] = (
            wandb_log["dists_grad_lora"] / (wandb_log["mse_grad_lora"] + 1e-12)
        )
