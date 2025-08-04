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
from training.utils import get_channel_fusion_module, get_resnet, load_checkpoint, replace_bn_with_gn, save_checkpoint
from data.grasp_dataset import PC_R10_Dataset
from model.conv_unet import ConditionalUnet1D
from training.utils import load_config_from_yaml, parse_args


@attr.s
class ConvUnetTrainingConfig:
    dataset_path: str = attr.ib()
    project_name: str = attr.ib()
    learning_rate: float = attr.ib(converter=float)
    num_warmup_steps: int = attr.ib(converter=int)
    ema_power: float = attr.ib(converter=float)
    batch_size: int = attr.ib(converter=int)
    save_interval: int = attr.ib(converter=int)
    save_directory: str = attr.ib()
    epochs: int = attr.ib(converter=int)
    wandb_run_id: Optional[str] = attr.ib()
    local_wandb_run_file: Optional[str] = attr.ib()
    checkpoint_path: Optional[str] = attr.ib()
    use_wandb: bool = attr.ib(converter=bool)


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
        point_map = batch['obj_point_map_normalized'].to(device)
        top_grasp = batch['top_grasp_r10'].to(device)

        if is_train:
            optimizer.zero_grad()

        # encode the observation
        #obs_cond = torch.cat([depth_map, obj_mask], dim=1)
        #obs_cond = model['channel_fusion_module'](obs_cond)
        obs_cond = model['vision_encoder'](point_map)

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
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    train_dataset = PC_R10_Dataset(config.dataset_path, split="train")
    val_dataset = PC_R10_Dataset(config.dataset_path, split="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    #channel_fusion_module = get_channel_fusion_module(4, 3).to(device)
    vision_encoder = get_resnet('resnet18').to(device)
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(input_dim=10, global_cond_dim=512).to(device)

    model = nn.ModuleDict({
        'vision_encoder': vision_encoder.to(device),
        #'channel_fusion_module': channel_fusion_module.to(device),
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
                    save_checkpoint(model, ema_model, optimizer, lr_scheduler, epoch, config.save_directory+f"/{config.project_name}_{config.wandb_run_id}_{start_time}/{epoch}_epoch.pth", config.use_wandb, attr.asdict(config))
        
            tglobal.set_postfix(train_loss=train_loss, val_loss=val_loss)

        save_checkpoint(model, ema_model, optimizer, lr_scheduler, config.epochs, config.save_directory+f"/{config.project_name}_{config.wandb_run_id}_{start_time}/{config.epochs}_epoch.pth", config.use_wandb, attr.asdict(config))

        print(f"Training complete. Saved checkpoint to {config.save_directory}")


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_config_from_yaml(args.yaml_path)

    # Override with command line arguments if provided
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_dict['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.save_directory is not None:
        config_dict['save_directory'] = args.save_directory
    if args.save_interval is not None:
        config_dict['save_interval'] = args.save_interval
    if args.use_wandb is not None:
        config_dict['use_wandb'] = args.use_wandb
    if args.wandb_run_id is not None:
        config_dict['wandb_run_id'] = args.wandb_run_id
    if args.local_wandb_run_file is not None:
        config_dict['local_wandb_run_file'] = args.local_wandb_run_file
    if args.checkpoint_path is not None:
        config_dict['checkpoint_path'] = args.checkpoint_path
    
    # Create config object
    config = ConvUnetTrainingConfig(**config_dict)

    main(config)

            
    