import argparse
import os
import json

import yaml
import wandb
import torch
import torch.nn as nn
import torchvision
from typing import Callable

def get_channel_fusion_module(in_channels: int, out_channels: int, kernel_size: int = 1, padding: int = 0, stride: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def save_checkpoint(model: nn.Module, 
                    ema_model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    step: int, 
                    filepath: str, 
                    training_config: dict,
                    use_wandb: bool=False)->None:
    """
    Saves a checkpoint locally and optionally logs a metadata artifact to W&B.
    """
    # 1. Save the full checkpoint file locally
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved locally to '{filepath}' at step {step}.")

    # 2. If using W&B, log a metadata artifact
    if use_wandb:
        metadata = {
            'step': step,
            'local_checkpoint_path': os.path.abspath(filepath),
            'training_config': training_config,
        }
        meta_filepath = "checkpoint_meta.json"
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)

        artifact = wandb.Artifact(
            name=f"checkpoint-meta-{wandb.run.id}",
            type="checkpoint_metadata",
            metadata={"step": step}
        )
        artifact.add_file(meta_filepath)
        wandb.log_artifact(artifact)
        os.remove(meta_filepath)
        print("Checkpoint metadata logged as a W&B Artifact.")


def load_checkpoint(model: nn.Module, 
                    ema_model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    filepath: str, 
                    device: torch.device)->int:
    """
    Loads the full checkpoint from a local filepath.
    """
    if not os.path.exists(filepath):
        print("Checkpoint file not found. Starting from scratch.")
        return 0

    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_step = checkpoint['step']
        print(f"Checkpoint loaded successfully from '{filepath}'. Resuming from step {start_step}.")
        return start_step
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        return 0
    
def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert None strings to actual None for optional fields
    for key in ['wandb_run_id', 'local_wandb_run_file', 'checkpoint_path']:
        if key in config_dict and config_dict[key] == 'null':
            config_dict[key] = None
    
    return config_dict

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ConvUNet for finger grasp')
    parser.add_argument('--yaml_path', type=str, help='Path to YAML config file')
    # Add command line overrides for common parameters
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--save_interval', type=int, help='Override save interval')
    parser.add_argument('--save_directory', type=str, help='Override save directory')
    parser.add_argument('--use_wandb', type=bool, help='Override use wandb')
    parser.add_argument('--wandb_run_id', type=str, help='Override wandb run id')
    parser.add_argument('--local_wandb_run_file', type=str, help='Override local wandb run file')
    parser.add_argument('--checkpoint_path', type=str, help='Override checkpoint path')

    
    return parser.parse_args()