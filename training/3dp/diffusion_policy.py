import argparse
import io
import math
import os
import datetime
from typing import Tuple, Optional, Dict, Any, List, Union

import click

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from diffusers import DDPMScheduler, DDIMScheduler
from scipy.ndimage import zoom
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader
import yaml

# PointNet hyperparameters for point cloud encoding
in_channels_point_net: int = 3
out_channels_point_net: int = 128
use_layer_norm_point_net: bool = True
final_norm_point_net: str = 'layernorm'


def axis_angle_to_rot6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle rotation (3D) to 6D rotation representation.
    Args:
        axis_angle: Tensor of shape (..., 3) containing axis-angle rotations
    Returns:
        rot6d: Tensor of shape (..., 6) containing first two columns of rotation matrix
    """

    def normalize(v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)

    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = normalize(axis_angle)

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1 - cos

    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]

    r11 = cos + x * x * one_minus_cos
    r12 = x * y * one_minus_cos - z * sin
    r21 = x * y * one_minus_cos + z * sin
    r22 = cos + y * y * one_minus_cos

    rot6d = torch.cat([r11, r12, torch.zeros_like(r11),
                       r21, r22, torch.zeros_like(r11)], dim=-1)
    return rot6d


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Args:
        rot6d: Tensor of shape (..., 6) containing first two columns of rotation matrix
    Returns:
        rot_matrix: Tensor of shape (..., 3, 3) containing full rotation matrix
    """
    x = rot6d[..., :3]
    y = rot6d[..., 3:6]

    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-10)
    z = torch.cross(x, y, dim=-1)

    rot_matrix = torch.stack([x, y, z], dim=-2)
    return rot_matrix


def rot6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to axis-angle rotation.
    Args:
        rot6d: Tensor of shape (..., 6) containing first two columns of rotation matrix
    Returns:
        axis_angle: Tensor of shape (..., 3) containing axis-angle rotation
    """
    rot_matrix = rot6d_to_rotation_matrix(rot6d)

    trace = rot_matrix[..., 0, 0] + rot_matrix[..., 1, 1] + rot_matrix[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-10, 1 - 1e-10)
    angle = torch.acos(cos_angle)

    sin_angle = torch.sin(angle)
    axis = torch.zeros_like(rot_matrix[..., 0])
    mask = sin_angle > 1e-10
    axis[mask, 0] = rot_matrix[mask, 2, 1] - rot_matrix[mask, 1, 2]
    axis[mask, 1] = rot_matrix[mask, 0, 2] - rot_matrix[mask, 2, 0]
    axis[mask, 2] = rot_matrix[mask, 1, 0] - rot_matrix[mask, 0, 1]
    axis[mask] = axis[mask] / (2 * sin_angle[mask].unsqueeze(-1))

    small_angle_mask = sin_angle <= 1e-10
    axis[small_angle_mask] = torch.tensor([1.0, 0.0, 0.0], device=rot6d.device).repeat(small_angle_mask.sum(), 1)

    axis_angle = axis * angle.unsqueeze(-1)
    return axis_angle


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PointNetEncoderXYZ(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 128,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs: Any
                 ) -> None:
        super().__init__()
        block_channel: List[int] = [64, 128, 256]
        cprint(f"[PointNetEncoderXYZ] use_layernorm: {use_layernorm}", 'cyan')
        cprint(f"[PointNetEncoderXYZ] use_final_norm: {final_norm}", 'cyan')

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
        self.in_channels: int = in_channels

        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection: nn.Module = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection: nn.Module = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection: bool = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[-1] == self.in_channels, \
            f"Expected input shape (B, N, {self.in_channels}), got {x.shape}"
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class ConditionalResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 n_groups=8,
                 condition_type='film'):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.GroupNorm(n_groups, out_channels) if n_groups > 1 else nn.Identity(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.GroupNorm(n_groups, out_channels) if n_groups > 1 else nn.Identity(),
                nn.ReLU(),
            ),
        ])

        self.condition_type = condition_type
        self.out_channels = out_channels

        if condition_type == 'film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")

        self.residual_linear = nn.Linear(in_channels, out_channels) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond=None):
        out = self.blocks[0](x)
        if cond is not None and self.condition_type == 'film':
            embed = self.cond_encoder(cond)
            scale, bias = embed.chunk(2, dim=-1)
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_linear(x)
        return out


class ConditionalUnet(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=[512, 1024, 2048],
                 n_groups=8,
                 condition_type='film',
                 use_down_condition=True,
                 use_mid_condition=True,
                 use_up_condition=True):
        super().__init__()
        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim if global_cond_dim is not None else dsed

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock(
                mid_dim, mid_dim, cond_dim=cond_dim,
                n_groups=n_groups, condition_type=condition_type
            ),
            ConditionalResidualBlock(
                mid_dim, mid_dim, cond_dim=cond_dim,
                n_groups=n_groups, condition_type=condition_type
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock(
                    dim_in, dim_out, cond_dim=cond_dim,
                    n_groups=n_groups, condition_type=condition_type),
                ConditionalResidualBlock(
                    dim_out, dim_out, cond_dim=cond_dim,
                    n_groups=n_groups, condition_type=condition_type),
                Downsample(dim_out, dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    n_groups=n_groups, condition_type=condition_type),
                ConditionalResidualBlock(
                    dim_in, dim_in, cond_dim=cond_dim,
                    n_groups=n_groups, condition_type=condition_type),
                Upsample(dim_in, dim_in) if not is_last else nn.Identity()
            ]))

        self.final_block = nn.Sequential(
            nn.Linear(start_dim, start_dim),
            nn.ReLU(),
            nn.Linear(start_dim, input_dim),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None, **kwargs):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([timestep_embed, global_cond], dim=-1) if global_cond is not None else timestep_embed

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            if self.use_down_condition:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature if self.use_mid_condition else None)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                x = resnet2(x)
            x = upsample(x)

        x = self.final_block(x)
        return x


class PointCloudGraspNormalizer:
    def __init__(self, width_min: float = 0.03, width_max: float = 0.07):
        self.range_eps = 1e-4
        self.width_min = width_min  # 3cm
        self.width_max = width_max  # 7cm

    def compute_normalization_params(self, point_cloud: np.ndarray, output_min: float = 0.0, output_max: float = 1.0) -> \
    Dict[str, torch.Tensor]:
        """
        Compute per-sample normalization parameters based on point cloud bounding box.
        Args:
            point_cloud: Shape (N, 3) or (C, H, W).
            output_min, output_max: Target range for normalization.
        Returns:
            params_dict: Dictionary containing scale and offset for xyz and width.
        """
        if len(point_cloud.shape) > 2:
            point_cloud = point_cloud.reshape(-1, 3)

        # Compute bounding box for xyz
        input_min = np.min(point_cloud, axis=0)
        input_max = np.max(point_cloud, axis=0)
        input_range = input_max - input_min
        ignore_dims = input_range < self.range_eps
        input_range[ignore_dims] = output_max - output_min

        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dims] = (output_max + output_min) / 2 - input_min[ignore_dims]

        # Compute width normalization parameters
        width_range = self.width_max - self.width_min
        width_scale = (output_max - output_min) / width_range if width_range >= self.range_eps else 1.0
        width_offset = output_min - width_scale * self.width_min

        return {
            'xyz': {
                'scale': torch.tensor(scale, dtype=torch.float32),
                'offset': torch.tensor(offset, dtype=torch.float32),
                'input_stats': {
                    'min': torch.tensor(input_min, dtype=torch.float32),
                    'max': torch.tensor(input_max, dtype=torch.float32)
                }
            },
            'width': {
                'scale': torch.tensor(width_scale, dtype=torch.float32),
                'offset': torch.tensor(width_offset, dtype=torch.float32),
                'input_stats': {
                    'min': torch.tensor(self.width_min, dtype=torch.float32),
                    'max': torch.tensor(self.width_max, dtype=torch.float32)
                }
            }
        }

    def normalize(self, point_cloud: np.ndarray, grasp: np.ndarray, use_position_only: bool = False) -> Tuple[
        np.ndarray, np.ndarray, Dict[str, torch.Tensor]]:
        """
        Normalize point cloud and grasp dynamically based on point cloud bounding box.
        Args:
            point_cloud: Shape (C, H, W) or (N, 3).
            grasp: Shape (10,) [x, y, z, r1, r2, r3, r4, r5, r6, width] or (3,) [x, y, z].
            use_position_only: If True, only normalize the position (x, y, z).
        Returns:
            normalized_point_cloud, normalized_grasp, params_dict
        """
        input_shape = point_cloud.shape
        if len(input_shape) > 2:
            point_cloud = point_cloud.reshape(-1, 3)

        # Compute normalization parameters
        params_dict = self.compute_normalization_params(point_cloud)

        # Normalize point cloud
        xyz_params = params_dict['xyz']
        point_cloud = torch.from_numpy(point_cloud).to(dtype=xyz_params['scale'].dtype,
                                                       device=xyz_params['scale'].device)
        normalized_point_cloud = point_cloud * xyz_params['scale'] + xyz_params['offset']
        normalized_point_cloud = normalized_point_cloud.cpu().numpy()
        if len(input_shape) > 2:
            normalized_point_cloud = normalized_point_cloud.reshape(input_shape)

        # Normalize grasp position
        normalized_grasp = grasp.copy()
        grasp_tensor = torch.from_numpy(grasp[:3]).to(dtype=xyz_params['scale'].dtype,
                                                      device=xyz_params['scale'].device)
        normalized_grasp[:3] = (grasp_tensor * xyz_params['scale'] + xyz_params['offset']).cpu().numpy()

        if not use_position_only:
            # Normalize rotation (rot6d) with Gram-Schmidt orthogonalization
            rot6d = torch.from_numpy(grasp[3:9]).float()
            v1 = rot6d[:3]
            v2 = rot6d[3:6]
            v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-10)
            v2 = v2 - torch.sum(v2 * v1_norm) * v1_norm
            v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-10)
            normalized_rot6d = torch.cat([v1_norm, v2_norm], dim=-1)
            normalized_grasp[3:9] = normalized_rot6d.cpu().numpy()

            # Normalize width
            width_params = params_dict['width']
            grasp_tensor = torch.from_numpy(grasp[9:10]).to(dtype=width_params['scale'].dtype,
                                                            device=width_params['scale'].device)
            normalized_grasp[9] = (grasp_tensor * width_params['scale'] + width_params['offset']).cpu().numpy()[0]

        return normalized_point_cloud, normalized_grasp, params_dict

    def unnormalize(self, normalized_grasp: Union[np.ndarray, torch.Tensor], params_dict: Dict[str, torch.Tensor],
                    use_position_only: bool = False) -> np.ndarray:
        """
        Unnormalize grasp point(s) using provided normalization parameters.
        Args:
            normalized_grasp: Shape (10,) or (3,) or (batch_size, 10) or (batch_size, 3).
            params_dict: Dictionary containing scale and offset for xyz and width.
            use_position_only: If True, only unnormalize the position (x, y, z).
        Returns:
            denormalized_grasp: Same shape as input, in original coordinates.
        """
        if isinstance(normalized_grasp, np.ndarray):
            normalized_grasp = torch.from_numpy(normalized_grasp)

        if normalized_grasp.dim() == 1:
            normalized_grasp = normalized_grasp.unsqueeze(0)
        batch_size = normalized_grasp.shape[0]

        denormalized_grasp = normalized_grasp.clone()
        xyz_params = params_dict['xyz']
        scale_xyz = xyz_params['scale'].to(normalized_grasp.device)
        offset_xyz = xyz_params['offset'].to(normalized_grasp.device)
        denormalized_grasp[:, :3] = (normalized_grasp[:, :3] - offset_xyz) / scale_xyz

        if not use_position_only:
            # Re-orthogonalize rot6d
            rot6d = normalized_grasp[:, 3:9]
            v1 = rot6d[:, :3]
            v2 = rot6d[:, 3:6]
            v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-10)
            v2 = v2 - torch.sum(v2 * v1_norm, dim=-1, keepdim=True) * v1_norm
            v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-10)
            denormalized_grasp[:, 3:9] = torch.cat([v1_norm, v2_norm], dim=-1)

            # Unnormalize width
            width_params = params_dict['width']
            scale_width = width_params['scale'].to(normalized_grasp.device)
            offset_width = width_params['offset'].to(normalized_grasp.device)
            denormalized_grasp[:, 9] = (normalized_grasp[:, 9] - offset_width) / scale_width

        if batch_size == 1:
            denormalized_grasp = denormalized_grasp.squeeze(0)

        return denormalized_grasp.cpu().numpy()


class ParquetPointCloudDataset(Dataset):
    def __init__(self, parquet_path: str, split: str = "train", target_size: Tuple[int, int] = (32, 32),
                 use_top_grasp: bool = False, use_position_only: bool = False, max_samples: Optional[int] = None) -> None:
        self.data: pd.DataFrame = pd.read_parquet(parquet_path, engine="pyarrow")
        print(self.data.columns)
        required_columns: List[str] = ['obj_point_map_unfiltered', 'split']
        grasp_column: str = 'top_grasp_r7' if use_top_grasp else 'all_grasps_on_top_obj_r7'
        required_columns.append(grasp_column)
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Parquet file must contain columns: {required_columns}")
        self.data = self.data[self.data['split'] == split]
        self.target_size: Tuple[int, int] = target_size
        self.use_top_grasp: bool = use_top_grasp
        self.use_position_only: bool = use_position_only
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.grasp_dim: int = 3 if use_position_only else 10
        self.normalizer = PointCloudGraspNormalizer()

        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        # Process samples
        for idx in range(min(len(self.data), max_samples) if max_samples is not None else len(self.data)):
            row: pd.Series = self.data.iloc[idx]
            point_map: np.ndarray = bytes_to_numpy(row['obj_point_map_unfiltered'])
            grasp_data: np.ndarray = bytes_to_numpy(row[grasp_column])
            if grasp_data is not None:
                if self.use_top_grasp:
                    if len(grasp_data) > 0:
                        grasp = grasp_data[:3] if use_position_only else grasp_data
                        if not use_position_only:
                            axis_angle = torch.from_numpy(grasp[3:6]).float()
                            rot6d = axis_angle_to_rot6d(axis_angle).numpy()
                            grasp = np.concatenate([grasp[:3], rot6d, grasp[6:7]])
                        self.samples.append((point_map, grasp))
                    else:
                        cprint(f"Sample {idx} has no valid top grasp, skipping", "yellow")
                else:
                    if len(grasp_data) > 0:
                        for grasp in grasp_data:
                            grasp = grasp[:3] if use_position_only else grasp
                            if not use_position_only:
                                axis_angle = torch.from_numpy(grasp[3:6]).float()
                                rot6d = axis_angle_to_rot6d(axis_angle).numpy()
                                grasp = np.concatenate([grasp[:3], rot6d, grasp[6:7]])
                            self.samples.append((point_map, grasp))
                    else:
                        cprint(f"Sample {idx} has no valid grasps, skipping", "yellow")
            else:
                cprint(f"Sample {idx} has no valid grasp data, skipping", "yellow")

        # Analyze rotation distribution if not position-only
        if self.samples and not use_position_only:
            grasp_stats = np.stack([sample[1] for sample in self.samples])
            rotation_stats = grasp_stats[:, 3:9]
            cprint(f"Rotation (rot6d) mean: {np.mean(rotation_stats, axis=0)}", "cyan")
            cprint(f"Rotation (rot6d) std: {np.std(rotation_stats, axis=0)}", "cyan")
            cprint(f"Grasp point mean: {np.mean(grasp_stats, axis=0)}", "cyan")
            cprint(f"Grasp point std: {np.std(grasp_stats, axis=0)}", "cyan")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        try:
            point_map, grasp = self.samples[idx]
            normalized_point_map, normalized_grasp, params_dict = self.normalizer.normalize(point_map, grasp,
                                                                                            self.use_position_only)
            normalized_point_map = get_resized_point_cloud(normalized_point_map, self.target_size)
            if normalized_point_map is None:
                raise ValueError(
                    f"Sample {idx} point cloud resize failed, size does not match expected {self.target_size}")
            normalized_point_map = normalized_point_map.transpose(1, 2, 0)
            valid_points = normalized_point_map.reshape(-1, 3)
            if len(valid_points) == 0:
                raise ValueError(f"Sample {idx} point cloud is empty")
            return {
                'point_cloud': torch.FloatTensor(valid_points),
                'top_grasp_r7': torch.FloatTensor(normalized_grasp),
                'normalizer_params': params_dict
            }
        except ValueError as e:
            cprint(str(e), "red")
            return None


def custom_collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def train(args: argparse.Namespace, max_samples: Optional[int] = None) -> None:
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    target_size = tuple(args.target_size)
    grasp_dim = args.grasp_dim
    num_train_timesteps = args.num_train_timesteps
    beta_schedule = args.beta_schedule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parquet_path = args.parquet_path
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if args.use_wandb:
        try:
            wandb.init(project="3d_diffusion_grasp",
                       config={"lr": lr, "batch_size": batch_size, "beta_schedule": beta_schedule,
                               "use_position_only": args.use_position_only})
        except ImportError:
            cprint("wandb not installed, logging disabled", "yellow")
            args.use_wandb = False

    train_dataset = ParquetPointCloudDataset(parquet_path, split="train", target_size=target_size,
                                             use_top_grasp=args.use_top_grasp, use_position_only=args.use_position_only, max_samples=max_samples)
    val_dataset = ParquetPointCloudDataset(
        parquet_path, split="val", target_size=target_size,
        use_top_grasp=args.use_top_grasp, use_position_only=args.use_position_only, #max_samples=max_samples
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=custom_collate_fn)

    pointnet_encoder = PointNetEncoderXYZ(
        in_channels=in_channels_point_net,
        out_channels=out_channels_point_net,
        use_layernorm=use_layer_norm_point_net,
        final_norm=final_norm_point_net
    ).to(device)
    model = ConditionalUnet(input_dim=grasp_dim, global_cond_dim=out_channels_point_net).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    optimizer = optim.Adam(list(model.parameters()) + list(pointnet_encoder.parameters()), lr=lr)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_grasp_error = float('inf')
    best_train_loss_filename = None
    best_val_loss_filename = None
    best_grasp_error_filename = None

    for epoch in range(num_epochs):
        model.train()
        pointnet_encoder.train()
        train_loss = 0.0
        train_loss_xyz = 0.0
        train_loss_rot6d = 0.0
        train_loss_width = 0.0
        train_batch_count = 0
        for batch in train_dataloader:
            if batch is None:
                continue
            point_cloud = batch['point_cloud'].to(device)
            grasp_point = batch['top_grasp_r7'].to(device)
            normalizer_params = batch['normalizer_params']

            condition = pointnet_encoder(point_cloud)
            actual_batch_size = grasp_point.shape[0]
            noise = torch.randn_like(grasp_point).to(device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (actual_batch_size,),
                                      device=device).long()
            noisy_grasp_point = scheduler.add_noise(grasp_point, noise, timesteps)

            optimizer.zero_grad()
            pred_noise = model(noisy_grasp_point, timesteps, condition)
            loss = nn.MSELoss()(pred_noise, noise)

            # Calculate component-wise losses
            loss_xyz = nn.MSELoss()(pred_noise[:, :3], noise[:, :3])
            if not args.use_position_only:
                loss_rot6d = nn.MSELoss()(pred_noise[:, 3:9], noise[:, 3:9])
                loss_width = nn.MSELoss()(pred_noise[:, 9:], noise[:, 9:])
            else:
                loss_rot6d = torch.tensor(0.0, device=device)
                loss_width = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_xyz += loss_xyz.item()
            train_loss_rot6d += loss_rot6d.item()
            train_loss_width += loss_width.item()
            train_batch_count += 1

        model.eval()
        pointnet_encoder.eval()
        val_loss = 0.0
        val_loss_xyz = 0.0
        val_loss_rot6d = 0.0
        val_loss_width = 0.0
        val_grasp_error = 0.0
        val_grasp_error_xyz = 0.0
        val_grasp_error_rot = 0.0
        val_grasp_error_width = 0.0
        val_batch_count = 0
        normalizer = PointCloudGraspNormalizer()
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue
                point_cloud = batch['point_cloud'].to(device)
                grasp_point = batch['top_grasp_r7'].to(device)
                normalizer_params = batch['normalizer_params']

                condition = pointnet_encoder(point_cloud)
                actual_batch_size = grasp_point.shape[0]
                noise = torch.randn_like(grasp_point).to(device)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (actual_batch_size,),
                                          device=device).long()
                noisy_grasp_point = scheduler.add_noise(grasp_point, noise, timesteps)

                pred_noise = model(noisy_grasp_point, timesteps, condition)
                loss = nn.MSELoss()(pred_noise, noise)
                loss_xyz = nn.MSELoss()(pred_noise[:, :3], noise[:, :3])
                if not args.use_position_only:
                    loss_rot6d = nn.MSELoss()(pred_noise[:, 3:9], noise[:, 3:9])
                    loss_width = nn.MSELoss()(pred_noise[:, 9:], noise[:, 9:])
                else:
                    loss_rot6d = torch.tensor(0.0, device=device)
                    loss_width = torch.tensor(0.0, device=device)

                x = torch.randn_like(grasp_point).to(device)
                for t in reversed(range(scheduler.config.num_train_timesteps)):
                    timesteps_t = torch.tensor([t], device=device).long()
                    pred_noise = model(x, timesteps_t, condition)
                    x = scheduler.step(pred_noise, timesteps_t, x).prev_sample

                x_denorm = normalizer.unnormalize(x, normalizer_params, args.use_position_only)
                grasp_point_denorm = normalizer.unnormalize(grasp_point, normalizer_params, args.use_position_only)
                if not args.use_position_only:
                    pred_rot_matrix = rot6d_to_rotation_matrix(torch.from_numpy(x_denorm[:, 3:9]).float())
                    gt_rot_matrix = rot6d_to_rotation_matrix(torch.from_numpy(grasp_point_denorm[:, 3:9]).float())
                    rot_error = torch.norm(pred_rot_matrix - gt_rot_matrix, dim=(-2, -1)).numpy()
                    pos_error = np.mean((x_denorm[:, :3] - grasp_point_denorm[:, :3]) ** 2, axis=1)
                    width_error = np.mean((x_denorm[:, 9] - grasp_point_denorm[:, 9]) ** 2)
                    grasp_error = np.mean(pos_error + rot_error + width_error)
                    grasp_error_xyz = np.mean(pos_error)
                    grasp_error_rot = np.mean(rot_error)
                    grasp_error_width = np.mean(width_error)
                else:
                    pos_error = np.mean((x_denorm - grasp_point_denorm) ** 2, axis=1)
                    grasp_error = np.mean(pos_error)
                    grasp_error_xyz = np.mean(pos_error)
                    grasp_error_rot = 0.0
                    grasp_error_width = 0.0

                val_loss += loss.item()
                val_loss_xyz += loss_xyz.item()
                val_loss_rot6d += loss_rot6d.item()
                val_loss_width += loss_width.item()
                val_grasp_error += grasp_error
                val_grasp_error_xyz += grasp_error_xyz
                val_grasp_error_rot += grasp_error_rot
                val_grasp_error_width += grasp_error_width
                val_batch_count += 1

        if train_batch_count > 0:
            avg_train_loss = train_loss / train_batch_count
            avg_train_loss_xyz = train_loss_xyz / train_batch_count
            avg_train_loss_rot6d = train_loss_rot6d / train_batch_count
            avg_train_loss_width = train_loss_width / train_batch_count
        else:
            avg_train_loss = float('inf')
            avg_train_loss_xyz = float('inf')
            avg_train_loss_rot6d = float('inf')
            avg_train_loss_width = float('inf')
            cprint(f"Epoch {epoch + 1}/{num_epochs}, No valid training batches processed", "yellow")

        if val_batch_count > 0:
            avg_val_loss = val_loss / val_batch_count
            avg_val_loss_xyz = val_loss_xyz / val_batch_count
            avg_val_loss_rot6d = val_loss_rot6d / val_batch_count
            avg_val_loss_width = val_loss_width / val_batch_count
            avg_grasp_error = val_grasp_error / val_batch_count
            avg_grasp_error_xyz = val_grasp_error_xyz / val_batch_count
            avg_grasp_error_rot = val_grasp_error_rot / val_batch_count
            avg_grasp_error_width = val_grasp_error_width / val_batch_count
        else:
            avg_val_loss = float('inf')
            avg_val_loss_xyz = float('inf')
            avg_val_loss_rot6d = float('inf')
            avg_val_loss_width = float('inf')
            avg_grasp_error = float('inf')
            avg_grasp_error_xyz = float('inf')
            avg_grasp_error_rot = float('inf')
            avg_grasp_error_width = float('inf')
            cprint(f"Epoch {epoch + 1}/{num_epochs}, No valid validation batches processed", "yellow")

        if train_batch_count > 0 or val_batch_count > 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_message = (
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f} (XYZ: {avg_train_loss_xyz:.4f}, Rot6D: {avg_train_loss_rot6d:.4f}, Width: {avg_train_loss_width:.4f}), "
                f"Val Loss: {avg_val_loss:.4f} (XYZ: {avg_val_loss_xyz:.4f}, Rot6D: {avg_val_loss_rot6d:.4f}, Width: {avg_val_loss_width:.4f}), "
                f"Val Grasp MSE: {avg_grasp_error:.4f} (XYZ: {avg_grasp_error_xyz:.4f}, Rot: {avg_grasp_error_rot:.4f}, Width: {avg_grasp_error_width:.4f}), "
                f"Learning Rate: {current_lr:.6f}"
            )
            cprint(log_message, "cyan")
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "train_loss_xyz": avg_train_loss_xyz,
                    "train_loss_rot6d": avg_train_loss_rot6d,
                    "train_loss_width": avg_train_loss_width,
                    "val_loss": avg_val_loss,
                    "val_loss_xyz": avg_val_loss_xyz,
                    "val_loss_rot6d": avg_val_loss_rot6d,
                    "val_loss_width": avg_val_loss_width,
                    "val_grasp_error": avg_grasp_error,
                    "val_grasp_error_xyz": avg_grasp_error_xyz,
                    "val_grasp_error_rot": avg_grasp_error_rot,
                    "val_grasp_error_width": avg_grasp_error_width,
                    "learning_rate": current_lr
                })

        # Save checkpoint every 200 epochs
        if (epoch + 1) % 200 == 0 and train_batch_count > 0 and val_batch_count > 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch + 1}_trainloss_{avg_train_loss:.4f}_valloss_{avg_val_loss:.4f}_grasperror_{avg_grasp_error:.4f}_{start_time}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pointnet_state_dict': pointnet_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'grasp_error': avg_grasp_error
            }, checkpoint_path)
            cprint(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_filename}", "green")

        # Update best checkpoints for train loss, val loss, and grasp error
        if train_batch_count > 0 and avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            if best_train_loss_filename:
                try:
                    os.remove(os.path.join(checkpoint_dir, best_train_loss_filename))
                except FileNotFoundError:
                    pass
            best_train_loss_filename = f"best_trainloss_{avg_train_loss:.4f}_epoch_{epoch + 1}_{start_time}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pointnet_state_dict': pointnet_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'grasp_error': avg_grasp_error
            }, os.path.join(checkpoint_dir, best_train_loss_filename))
            cprint(f"Saved best train loss model: {best_train_loss:.4f} to {best_train_loss_filename}", "green")

        if val_batch_count > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if best_val_loss_filename:
                try:
                    os.remove(os.path.join(checkpoint_dir, best_val_loss_filename))
                except FileNotFoundError:
                    pass
            best_val_loss_filename = f"best_valloss_{avg_val_loss:.4f}_epoch_{epoch + 1}_{start_time}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pointnet_state_dict': pointnet_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'grasp_error': avg_grasp_error
            }, os.path.join(checkpoint_dir, best_val_loss_filename))
            cprint(f"Saved best val loss model: {best_val_loss:.4f} to {best_val_loss_filename}", "green")

        if val_batch_count > 0 and avg_grasp_error < best_grasp_error:
            best_grasp_error = avg_grasp_error
            if best_grasp_error_filename:
                try:
                    os.remove(os.path.join(checkpoint_dir, best_grasp_error_filename))
                except FileNotFoundError:
                    pass
            best_grasp_error_filename = f"best_grasperror_{avg_grasp_error:.4f}_epoch_{epoch + 1}_{start_time}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pointnet_state_dict': pointnet_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'grasp_error': avg_grasp_error
            }, os.path.join(checkpoint_dir, best_grasp_error_filename))
            cprint(f"Saved best grasp error model: {best_grasp_error:.4f} to {best_grasp_error_filename}", "green")

        scheduler_lr.step()


def get_resized_point_cloud(point_map: np.ndarray, target_size: Tuple[int, int] = (32, 32), order: int = 3) -> Optional[
    np.ndarray]:
    H, W = point_map.shape[1], point_map.shape[2]
    zoom_factor: Tuple[float, float, float] = (1, target_size[0] / H, target_size[1] / W)
    resized_point_map: np.ndarray = zoom(point_map, zoom_factor, order=order)
    if resized_point_map.shape[1:] != target_size:
        return None
    return resized_point_map


def bytes_to_numpy(byte_data: bytes) -> Optional[np.ndarray]:
    try:
        buffer: io.BytesIO = io.BytesIO(byte_data)
        return np.load(buffer)
    except Exception as e:
        cprint(f"Failed to deserialize numpy array: {str(e)}", "red")
        return None


def infer(args: argparse.Namespace) -> Optional[np.ndarray]:
    target_size = tuple(args.target_size)
    grasp_dim = args.grasp_dim
    num_train_timesteps = args.num_train_timesteps
    inference_steps = args.inference_steps
    beta_schedule = args.beta_schedule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load point cloud from Parquet
    data = pd.read_parquet(args.parquet_path, engine="pyarrow")
    required_columns = ['obj_point_map_unfiltered']
    grasp_column = 'top_grasp_r7' if args.use_top_grasp else 'all_grasps_on_top_obj_r7'
    required_columns.append(grasp_column)
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Parquet file must contain columns: {required_columns}")
    point_map = bytes_to_numpy(data.iloc[args.sample_idx]['obj_point_map_unfiltered'])

    # Initialize normalizer and compute per-sample normalization parameters
    normalizer = PointCloudGraspNormalizer(width_min=0.03, width_max=0.07)
    dummy_grasp = np.zeros(3 if args.use_position_only else 10)
    normalized_point_map, _, params_dict = normalizer.normalize(point_map, dummy_grasp, args.use_position_only)
    normalized_point_map = get_resized_point_cloud(normalized_point_map, target_size)
    if normalized_point_map is None:
        cprint(f"Sample {args.sample_idx} point cloud resize failed, size does not match expected {target_size}", "red")
        return None
    normalized_point_map = normalized_point_map.transpose(1, 2, 0)
    valid_points = normalized_point_map.reshape(-1, 3)
    if len(valid_points) == 0:
        cprint(f"Sample {args.sample_idx} point cloud is empty", "red")
        return None

    # Load model and pointnet encoder
    pointnet_encoder = PointNetEncoderXYZ(
        in_channels=in_channels_point_net,
        out_channels=out_channels_point_net,
        use_layernorm=use_layer_norm_point_net,
        final_norm=final_norm_point_net
    ).to(device)
    model = ConditionalUnet(input_dim=grasp_dim, global_cond_dim=out_channels_point_net).to(device)

    # Select scheduler
    if args.inference_scheduler == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
        scheduler.set_timesteps(inference_steps)
        timesteps = torch.linspace(scheduler.config.num_train_timesteps - 1, 0, inference_steps, dtype=torch.long,
                                   device=device)
    elif args.inference_scheduler == "ddpm":
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
        timesteps = reversed(range(scheduler.config.num_train_timesteps))
    else:
        cprint(f"Invalid inference scheduler: {args.inference_scheduler}. Use 'ddim' or 'ddpm'.", "red")
        return None

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        cprint(f"Loaded checkpoint with val_loss={checkpoint['val_loss']:.4f}, epoch={checkpoint['epoch']}", "cyan")
        pointnet_encoder.load_state_dict(checkpoint['pointnet_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
    except (FileNotFoundError, KeyError) as e:
        cprint(f"Error loading checkpoint: {e}", "red")
        return None

    pointnet_encoder.eval()
    model.eval()

    # Inference
    point_cloud = torch.FloatTensor(valid_points).unsqueeze(0).to(device)
    condition = pointnet_encoder(point_cloud)
    x = torch.randn(1, grasp_dim).to(device)
    for t in timesteps:
        with torch.no_grad():
            timesteps_t = torch.tensor([t], device=device).long()
            pred_noise = model(x, timesteps_t, condition)
            x = scheduler.step(pred_noise, timesteps_t, x).prev_sample

    grasp_point = x[0]
    grasp_point_denorm = normalizer.unnormalize(grasp_point, params_dict, args.use_position_only)

    # Convert to 7D format if not position-only
    if not args.use_position_only:
        rot6d = torch.from_numpy(grasp_point_denorm[3:9]).float().to(device)
        axis_angle = rot6d_to_axis_angle(rot6d).cpu().numpy()
        grasp_point_denorm = np.concatenate([grasp_point_denorm[:3], axis_angle, grasp_point_denorm[9:10]])

    cprint("Predicted grasp point: " + str(grasp_point_denorm), "green")

    # Compare with ground truth
    grasp_data = bytes_to_numpy(data.iloc[args.sample_idx][grasp_column])
    if grasp_data is not None and len(grasp_data) > 0:
        if args.use_top_grasp:
            ground_truth = grasp_data[:3] if args.use_position_only else grasp_data
            if not args.use_position_only:
                error = np.mean((grasp_point_denorm - ground_truth) ** 2)
            else:
                error = np.mean((grasp_point_denorm - ground_truth) ** 2)
            cprint(f"Original top grasp point: {ground_truth}", "blue")
            cprint(f"Grasp point MSE: {error:.4f}", "cyan")
        else:
            ground_truth = grasp_data[0][:3] if args.use_position_only else grasp_data[0]
            cprint(f"Original grasp points (first shown): {ground_truth}", "blue")
            errors = []
            for grasp in grasp_data:
                if args.use_position_only:
                    grasp = grasp[:3]
                    error = np.mean((grasp_point_denorm - grasp) ** 2)
                else:
                    error = np.mean((grasp_point_denorm - grasp) ** 2)
                errors.append(error)
            min_error = np.min(errors)
            cprint(f"Best grasp point MSE: {min_error:.4f}", "cyan")
    else:
        cprint("No ground truth grasps available for comparison", "yellow")
        min_error = None
    return grasp_point_denorm

@click.command()
@click.option('--config', type=str, default="config.yaml", help="Path to YAML configuration file")
@click.option('--mode', type=str, default="train", help="Mode: train or infer")
@click.option('--max_samples', type=int, default=None, help="Maximum number of samples to process")
def main(config, mode, max_samples) -> None:
    #parser = argparse.ArgumentParser(description="3D Diffusion Policy for Grasp Point Prediction")
    #parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    #args = parser.parse_args()
    try:
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        cprint(f"Configuration file {config} not found", "red")
        return
    except yaml.YAMLError as e:
        cprint(f"Error parsing YAML configuration: {e}", "red")
        return

    class ConfigNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config_args = ConfigNamespace(
        mode=config.get('mode', 'train'),
        parquet_path=config.get('parquet_path', 'path/to/your/data.parquet'),
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        checkpoint_path=config.get('checkpoint_path', 'checkpoints/best_model.pth'),
        sample_idx=config.get('sample_idx', 0),
        batch_size=config.get('batch_size', 32),
        num_epochs=config.get('num_epochs', 200),
        lr=config.get('lr', 2e-4),
        target_size=config.get('target_size', [32, 32]),
        grasp_dim=config.get('grasp_dim', 3 if config.get('use_position_only', False) else 10),
        num_train_timesteps=config.get('num_train_timesteps', 1000),
        num_workers=config.get('num_workers', 4),
        use_wandb=config.get('use_wandb', False),
        use_top_grasp=config.get('use_top_grasp', False),
        use_position_only=config.get('use_position_only', False),
        inference_steps=config.get('inference_steps', 50),
        beta_schedule=config.get('beta_schedule', 'squaredcos_cap_v2'),
        inference_scheduler=config.get('inference_scheduler', 'ddpm'),
        max_samples=max_samples
    )

    if config_args.mode == "train":
        train(config_args)
    elif config_args.mode == "infer":
        infer(config_args)


if __name__ == "__main__":
    main()