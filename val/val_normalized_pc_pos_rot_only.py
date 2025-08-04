from PIL import Image
import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import click
from viz.grasp_viz import draw_top_grasp_point
from training.utils import load_checkpoint
from training.utils import get_channel_fusion_module, get_resnet, replace_bn_with_gn
from model.conv_unet import ConditionalUnet1D



def predict_normalized_pc_pos_rot_only(nets, noise_scheduler, obj_point_map_normalized, output_dim, device):

    B = 1
    num_diffusion_iters = 100   
    with torch.no_grad():
        # get image features
        #image_features = ema['vision_encoder'](nimages)
        # (2,512)

        # concat with low-dim observations
        #obs_features = torch.cat([image_features, nagent_poses], dim=-1)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = nets['vision_encoder'](obj_point_map_normalized.unsqueeze(0))

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, 1, output_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        naction = naction.detach().to('cpu').numpy()

    return naction


def transform_and_visualize_predicted_grasp(predicted_grasp, label_grasp, obj_max_dist, obj_center, cam_intrinsics, rgb):
    xyz = (predicted_grasp[:3])*obj_max_dist.item()+obj_center.cpu().numpy()
    rot = predicted_grasp[3:9].reshape(2,3)
    rot = gram_schmidt(rot[0], rot[1])
    rot = np.concatenate([np.cross(rot[0], rot[1]), rot], axis=0)
    rot_vec = R.from_matrix(rot).as_rotvec()
    vec_7d = np.append(np.concatenate([xyz, rot_vec], axis=0), [0.07])

    img = draw_top_grasp_point(vec_7d, rgb, cam_intrinsics, color = 'blue')
    img = draw_top_grasp_point(label_grasp, np.array(rgb).transpose(2,0,1)/255.0, cam_intrinsics, color = 'red')
    return img


def gram_schmidt(v1, v2):
    # Normalize first vector
    u1 = v1 / np.linalg.norm(v1)
    
    # Project v2 onto u1 and subtract to get orthogonal component
    proj = np.dot(v2, u1) * u1
    u2_orthogonal = v2 - proj
    
    # Normalize the second vector
    u2 = u2_orthogonal / np.linalg.norm(u2_orthogonal)
    
    return np.stack([u1,u2])

def create_image_grid_pil(images: List[np.ndarray], 
                         images_per_row: int = 4,
                         max_images_per_picture: int = 16,
                         save_dir: str = "output",
                         filename_prefix: str = "image_grid",
                         image_size: Tuple[int, int] = (256, 256),
                         spacing: int = 10,
                         background_color: str = "white") -> List[str]:
    """
    Create and save big pictures using PIL (Pillow) for better memory efficiency.
    
    Args:
        images: List of numpy arrays representing images (H, W, C) or (H, W)
        images_per_row: Number of images per row in each big picture
        max_images_per_picture: Maximum number of images per big picture
        save_dir: Directory to save the output images
        filename_prefix: Prefix for the saved image files
        image_size: Size to resize each image to (width, height)
        spacing: Spacing between images in pixels
        background_color: Background color for the grid
    
    Returns:
        List of saved file paths
    """
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    saved_files = []
    total_images = len(images)
    
    # Calculate number of pictures needed
    num_pictures = (total_images + max_images_per_picture - 1) // max_images_per_picture
    
    for picture_idx in range(num_pictures):
        # Calculate start and end indices for this picture
        start_idx = picture_idx * max_images_per_picture
        end_idx = min(start_idx + max_images_per_picture, total_images)
        
        # Get images for this picture
        picture_images = images[start_idx:end_idx]
        num_images_in_picture = len(picture_images)
        
        # Calculate grid dimensions
        num_rows = (num_images_in_picture + images_per_row - 1) // images_per_row
        
        # Calculate big picture dimensions
        big_width = images_per_row * image_size[0] + (images_per_row - 1) * spacing
        big_height = num_rows * image_size[1] + (num_rows - 1) * spacing
        
        # Create big picture
        big_image = Image.new('RGB', (big_width, big_height), background_color)
        
        # Place images in the grid
        for i, img_array in enumerate(picture_images):
            # Convert numpy array to PIL Image
            if img_array.dtype == np.uint8:
                if len(img_array.shape) == 3:
                    pil_img = Image.fromarray(img_array)
                else:
                    pil_img = Image.fromarray(img_array, mode='L').convert('RGB')
            else:
                # Normalize to 0-255 range
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                
                if len(img_array.shape) == 3:
                    pil_img = Image.fromarray(img_array)
                else:
                    pil_img = Image.fromarray(img_array, mode='L').convert('RGB')
            
            # Resize image
            pil_img = pil_img.resize(image_size, Image.Resampling.LANCZOS)
            
            # Calculate position
            row = i // images_per_row
            col = i % images_per_row
            
            x = col * (image_size[0] + spacing)
            y = row * (image_size[1] + spacing)
            
            # Paste image
            big_image.paste(pil_img, (x, y))
        
        # Save the big picture
        filename = f"{filename_prefix}_{picture_idx:03d}.png"
        filepath = os.path.join(save_dir, filename)
        big_image.save(filepath, quality=95)
        saved_files.append(filepath)
    
    print(f"Created {len(saved_files)} image grid(s) with {total_images} total images")
    print(f"Images saved to: {save_dir}")
    
    return saved_files

@click.command()
@click.option('--checkpoints_dir', type=str, required=True)
@click.option('--data_dir', type=str, required=True)
@click.option('--save_dir', type=str, required=True)
def main(checkpoints_dir, data_dir, save_dir):

    #os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vision_encoder = get_resnet('resnet18').to(device)
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    noise_pred_net = ConditionalUnet1D(
        input_dim=9,
        global_cond_dim=512
    )

    noise_pred_net.to(device)

    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75
    )

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=18*100 #len(dataloader) * num_epochs
    )

    _ = load_checkpoint(nets, ema, optimizer, lr_scheduler, checkpoints_dir, device)
    nets.eval()

    df = pd.read_parquet(data_dir)
    df = df[df['split'] == 'val']

    img_list = []
    for i in range(len(df)):
        datum = df.iloc[i]

        obj_point_map = torch.from_numpy(datum['obj_point_map_unfiltered']).to(device)
        cam_intrinsics = torch.from_numpy(datum['reference_camera_intrinsics']).to(device)
        image = datum['image']
        bbox = datum['obj_bbox']
        top_grasp_r7 = datum['top_grasp_r7']

        obj_point_map_reshaped = obj_point_map.reshape(-1, 3)
        obj_center = obj_point_map_reshaped.mean(dim=0)
        obj_max_dist = torch.norm(obj_point_map_reshaped - obj_center, dim=1).max()
        obj_point_map_normalized = (obj_point_map - obj_center.view(3, 1, 1)) / obj_max_dist

        predicted_grasp = predict_normalized_pc_pos_rot_only(nets, noise_scheduler, obj_point_map_normalized, 9, device)

        img = transform_and_visualize_predicted_grasp(predicted_grasp, top_grasp_r7, obj_max_dist, obj_center, cam_intrinsics, image)

        x_min, y_min, x_max, y_max = bbox
        img = np.array(img)[y_min:y_max, x_min:x_max, :]

        img_list.append(img)

    saved_files = create_image_grid_pil(img_list, save_dir=save_dir)


if __name__ == '__main__':
    main()





    

    
    
    




