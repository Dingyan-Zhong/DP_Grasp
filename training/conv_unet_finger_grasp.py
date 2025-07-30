from typing import Optional
import attr
import datetime
import wandb
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from DP_Grasp.training.utils import get_channel_fusion_module, get_resnet, load_checkpoint, replace_bn_with_gn, save_checkpoint
from DP_Grasp.data.grasp_dataset import RGBD_R7_Dataset
from DP_Grasp.model.conv_unet import ConditionalUnet1D


@attr.s
class ConvUnetTrainingConfig:
    dataset_path: str = attr.ib()
    project_name: str = attr.ib()
    learning_rate: float = attr.ib()
    num_warmup_steps: int = attr.ib()
    ema_power: float = attr.ib()
    batch_size: int = attr.ib()
    save_interval: int = attr.ib()
    save_directory: str = attr.ib()
    epochs: int = attr.ib()
    wandb_run_id: Optional[str] = attr.ib()
    local_wandb_run_file: Optional[str] = attr.ib()
    checkpoint_path: Optional[str] = attr.ib()
    use_wandb: bool = attr.ib()


def train_epoch(
        model: nn.Module,
        ema_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        noise_scheduler: DDIMScheduler,
        num_train_timesteps: int,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        is_train: bool = True,
    )->float:

    epoch_loss = []

    for batch in train_loader:
        depth_map = batch['obj_depth_map'].to(device)
        obj_mask = batch['obj_rgb'].to(device)
        top_grasp = batch['top_grasp_r7'].to(device)

        if is_train:
            optimizer.zero_grad()

        # encode the observation
        obs_cond = torch.cat([depth_map, obj_mask], dim=1)
        obs_cond = model['channel_fusion_module'](obs_cond)
        obs_cond = model['vision_encoder'](obs_cond)

        B = top_grasp.shape[0]
        timesteps = torch.randint(
                    0, num_train_timesteps,
                    (B,), device=device
                ).long().to(device)

        noise = torch.randn(top_grasp.shape, device=device, dtype=torch.float32)
        noisy_actions = noise_scheduler.add_noise(top_grasp, noise, timesteps).to(torch.float32)

        # predict the noise residual
        noise_pred = model['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

        loss = nn.functional.mse_loss(noise_pred, noise)
        
        if is_train:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema_model.step(model.parameters())

        epoch_loss.append(loss.item())

    return np.mean(epoch_loss)


def main(config: ConvUnetTrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RGBD_R7_Dataset(config.dataset_path, split="train")
    val_dataset = RGBD_R7_Dataset(config.dataset_path, split="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    channel_fusion_module = get_channel_fusion_module(4, 3).to(device)
    vision_encoder = get_resnet('resnet18').to(device)
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(input_dim=7, global_cond_dim=512).to(device)

    model = nn.ModuleDict({
        'vision_encoder': vision_encoder.to(device),
        'channel_fusion_module': channel_fusion_module.to(device),
        'noise_pred_net': noise_pred_net.to(device)
    }).to(device)

    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    ema_model = EMAModel(model, power=config.ema_power)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=len(train_loader) * config.epochs
    )

    if config.checkpoint_path is not None:
        start_step = load_checkpoint(model, ema_model, optimizer, lr_scheduler, config.checkpoint_path, device)
    else:
        start_step = 0

    if config.use_wandb:
        if config.wandb_run_id is not None:
            wandb.init(project=config.project_name, id=config.wandb_run_id, config=config, resume="allow")
        else:
            wandb.init(project=config.project_name, config=config)

    with tqdm(range(config.epochs), desc='Epoch') as tglobal:
        for epoch in tglobal:
            model.train()
            train_loss = train_epoch(model, 
                                     ema_model, 
                                     optimizer, 
                                     lr_scheduler, 
                                     noise_scheduler, 
                                     noise_scheduler.config.num_train_timesteps, 
                                     train_loader, 
                                     device, 
                                     is_train=True)

            model.eval()
            with torch.no_grad():
                val_loss = train_epoch(model, 
                                       ema_model, 
                                       optimizer, 
                                       lr_scheduler, 
                                       noise_scheduler, 
                                       noise_scheduler.config.num_train_timesteps, 
                                       val_loader, device, 
                                       is_train=False)

            if config.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })

            if config.save_interval is not None:
                if (epoch + 1) % config.save_interval == 0:
                    save_checkpoint(model, ema_model, optimizer, lr_scheduler, epoch, config.save_directory+f"/{config.project_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.epochs}.pth", config.use_wandb, attr.asdict(config))
        
            tglobal.set_postfix(train_loss=train_loss, val_loss=val_loss)

        save_checkpoint(model, ema_model, optimizer, lr_scheduler, config.epochs, config.save_directory+f"/{config.project_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.epochs}.pth", config.use_wandb, attr.asdict(config))

        print(f"Training complete. Saved checkpoint to {config.save_directory}")


if __name__ == "__main__":
    config = ConvUnetTrainingConfig(
        dataset_path="s3://covariant-datasets-prod/dp_finger_grasp_dataset_small_test_2025_07_24_09_40",
        project_name="conv_unet_finger_grasp",
        learning_rate=1e-4,
        num_warmup_steps=100,
        ema_power=0.75,
        batch_size=32,
        save_interval=100,
        save_directory="/home/ubuntu/DP_grasp/checkpoints",
        epochs=1000,
        wandb_run_id=None,
        local_wandb_run_file=None,
        checkpoint_path=None,
        use_wandb=True,
    )

    main(config)

            
    