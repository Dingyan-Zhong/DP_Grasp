from typing import Optional
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
from scipy.spatial.transform import Rotation as R


def draw_top_grasp_point(grasp_point: torch.Tensor, 
                         rgb: torch.Tensor, 
                         cam_intrinsics: torch.Tensor,
                         save_dir: Optional[str] = None,
                         finger_length: float = 0.07, 
                         tail_length: float = 0.05,
                         color: str = 'red') -> Image.Image:
    """Draw the top grasp point on the image."""
    
    # Convert tensors to numpy arrays
    if isinstance(grasp_point, torch.Tensor):
        grasp_point_npy = grasp_point.detach().cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb_npy = rgb.permute(1, 2, 0).detach().cpu().numpy()
        rgb_npy = (255*rgb_npy).astype(np.uint8)
    if isinstance(cam_intrinsics, torch.Tensor):
        cam_intrinsics_npy = cam_intrinsics.detach().cpu().numpy()

    img = Image.fromarray(rgb_npy)
    draw = ImageDraw.Draw(img)

    if grasp_point_npy.shape[-1] == 7:
        # The Grasp Point is a 7D vector, [x, y, z, axis-angle rotation, width]
        rotation = R.from_rotvec(grasp_point_npy[3:6]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = grasp_point_npy[:3]
        gripper_width = grasp_point_npy[6]

        mid_point_base = np.array([0.0, 0.0, -finger_length, 1.0])
        # tail_point_base is mid_point_base offset by tail_length along Z
        tail_point_base = mid_point_base + np.array([0.0, 0.0, -tail_length, 0.0])

        # Define points as (N, 4) tensors
        left_offset_vector = np.array([0.0, -0.5, 0.0, 0.0])
        left_finger_ends = mid_point_base + left_offset_vector * gripper_width

        left_finger_tips = left_finger_ends + np.array([0.0, 0.0, finger_length, 0.0])

        right_offset_vector = np.array([0.0, 0.5, 0.0, 0.0])
        right_finger_ends = mid_point_base + right_offset_vector * gripper_width
        right_finger_tips = right_finger_ends + np.array([0.0, 0.0, finger_length, 0.0])

        all_points = np.stack([mid_point_base, tail_point_base, left_finger_ends, left_finger_tips, right_finger_ends, right_finger_tips], axis=0)
        all_points = np.matmul(all_points, transform.T)[:, :3]

    elif grasp_point_npy.shape[-1] == 9:
        # The Grasp Point is a 9D vector, [x_positive, y_positive, z_positive, x_negative, y_negative, z_negative, negated z_direction]
        # We don't have "left" or "right" in the grasp point, as the gripper is symmetric. 
        # Positive means the line from the tcp to the finger tip is along the positive direction of tcp frame's y-axis
        # Negative means the line from the tcp to the finger tip is along the negative direction of tcp frame's y-axis

        left_finger_tips = np.array([grasp_point_npy[0], grasp_point_npy[1], grasp_point_npy[2], 1.0])
        right_finger_tips = np.array([grasp_point_npy[3], grasp_point_npy[4], grasp_point_npy[5], 1.0])

        left_finger_ends = left_finger_tips + np.concatenate([finger_length * grasp_point_npy[6:], [0.0]])
        right_finger_ends = right_finger_tips + np.concatenate([finger_length * grasp_point_npy[6:], [0.0]])

        mid_point_base = (left_finger_ends + right_finger_ends) / 2
        tail_point_base = mid_point_base + np.concatenate([tail_length * grasp_point_npy[6:], [0.0]])

        all_points = np.stack([mid_point_base, tail_point_base, left_finger_ends, left_finger_tips, right_finger_ends, right_finger_tips], axis=0)
        
    else:
        raise ValueError(f"Grasp point must be 7D or 9D, but got {grasp_point_npy.shape[-1]}D")

    # Project the points to the image
    fx = cam_intrinsics_npy[0, 0]
    fy = cam_intrinsics_npy[1, 1]
    cx = cam_intrinsics_npy[0, 2]
    cy = cam_intrinsics_npy[1, 2]
    all_points= all_points[:, :2] / all_points[:, 2:3]
    all_points[:, 0] = all_points[:, 0] * fx + cx
    all_points[:, 1] = all_points[:, 1] * fy + cy

    for point in all_points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='white', outline='white')
    
    draw.line([(all_points[0, 0], all_points[0, 1]), (all_points[1, 0], all_points[1, 1])], fill=color, width=5)
    draw.line([(all_points[2, 0], all_points[2, 1]), (all_points[3, 0], all_points[3, 1])], fill=color, width=5)
    draw.line([(all_points[4, 0], all_points[4, 1]), (all_points[5, 0], all_points[5, 1])], fill=color, width=5)
    draw.line([(all_points[2, 0], all_points[2, 1]), (all_points[4, 0], all_points[4, 1])], fill=color, width=5)
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        output_file = save_dir + "/top_grasp.png"
        img.save(output_file)

    return img
