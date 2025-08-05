import torch
from typing import List

def rule_based_grasp_validator(tcp_to_camera_transforms: torch.Tensor, 
                               finger_contact_lengths: torch.Tensor, 
                               gripper_widths: torch.Tensor,  
                               updir_in_cam: torch.Tensor,
                               max_grasp_width: float, 
                               max_grasp_depth: float, 
                               max_approach_deviation_from_updir: float):
    
    length_condition = (finger_contact_lengths > 0.01) & (finger_contact_lengths < max_grasp_depth)
    #logger.info(f"[DEBUG] length_condition shape: {length_condition.shape}")
    #logger.info(f"[DEBUG] length_condition: {length_condition}")
    width_condition = gripper_widths < max_grasp_width + 0.001 # 0.001 is for numerical stability
    #logger.info(f"[DEBUG] width_condition shape: {width_condition.shape}")
    #logger.info(f"[DEBUG] width_condition: {width_condition}")
    approach_dirs = -tcp_to_camera_transforms[:, :3, 2].to(updir_in_cam.dtype)
    cosine_angles = torch.sum(approach_dirs * updir_in_cam, dim=1)
    direction_condition = torch.acos(cosine_angles) < max_approach_deviation_from_updir
    #logger.info(f"[DEBUG] direction_condition shape: {direction_condition.shape}")
    #logger.info(f"[DEBUG] direction_condition: {direction_condition}")
    valid_grasp_indices = torch.where((length_condition & width_condition) & direction_condition)[0]
    #logger.info(f"[DEBUG] valid_grasp_indices shape: {valid_grasp_indices.shape}")

    return valid_grasp_indices

def rule_based_grasp_ranker(tcp_to_camera_transforms: torch.Tensor,
                            finger_contact_lengths: torch.Tensor, 
                            object_id: torch.Tensor,
                            points: List[torch.Tensor],
                            gripper_widths: torch.Tensor,  
                            updir_in_cam: torch.Tensor,
                            max_grasp_width: float, 
                            max_grasp_depth: float):
    
    # prefer dir closer to updir
    approach_dir = -tcp_to_camera_transforms[:, :3, 2]
    dir_deviation = 2*torch.sum(approach_dir.to(updir_in_cam.dtype)*updir_in_cam, dim=1)

    # prefer object center closer to tcp
    obj_centers = []
    obj_max_dist_center_2d = []
    obj_main_axis1 = []
    obj_main_axis2 = []
    obj_min_span = []
    for i in range(len(points)):
        object_points = points[i]
        object_center = object_points.mean(dim=0)
        centered_object_points_2d = object_points[:, :2]-object_center[:2]
        object_max_dist_center_2d = torch.norm(centered_object_points_2d, dim=1).max()
        _, _, singular_vectors = torch.linalg.svd(centered_object_points_2d, full_matrices=False)
        main_axis1 = singular_vectors[0, :]
        main_axis2 = singular_vectors[1, :]
        main_axis1 = torch.cat([main_axis1, torch.tensor([0.0], device=object_points.device, dtype=object_points.dtype)])
        main_axis2 = torch.cat([main_axis2, torch.tensor([0.0], device=object_points.device, dtype=object_points.dtype)])
        projected_2d_points = centered_object_points_2d@singular_vectors.T
        min_coords = torch.min(projected_2d_points, dim=0).values
        max_coords = torch.max(projected_2d_points, dim=0).values
        min_spans = torch.min(max_coords - min_coords)
        obj_centers.append(object_center)
        obj_max_dist_center_2d.append(object_max_dist_center_2d)
        obj_main_axis1.append(main_axis1)
        obj_main_axis2.append(main_axis2)
        obj_min_span.append(min_spans)

    obj_centers = torch.stack(obj_centers)[object_id]
    obj_max_dist_center_2d = torch.stack(obj_max_dist_center_2d)[object_id]
    obj_main_axis1 = torch.stack(obj_main_axis1)[object_id]
    obj_main_axis2 = torch.stack(obj_main_axis2)[object_id]
    obj_min_span = torch.stack(obj_min_span)[object_id]

    tcp_to_object_center_2d = torch.norm(obj_centers[:, :2] - tcp_to_camera_transforms[:, :2, 3], dim=1)
    tcp_to_object_center_ratio = 2*(1-tcp_to_object_center_2d/obj_max_dist_center_2d)

    # prefer y-axis closer to one of the 2d pca components
    y_rot = tcp_to_camera_transforms[:, :3, 1]
    #y_axis_deviation_1 = torch.sum(y_rot*obj_main_axis1, dim=1)
    y_axis_deviation = 1.5*torch.sum(y_rot*obj_main_axis2, dim=1)

    #y_axis_deviation = torch.min(torch.abs(y_axis_deviation_1), torch.abs(y_axis_deviation_2))

    # prefer gripper width closer to min_span
    gripper_width_ratio = torch.clamp(obj_min_span/gripper_widths, 0.0, 1.0)
   
    # prefer finger contact length closer to 0.8*max_grasp_depth
    # Motion planning may be more challenging if the contact length is too close to max_grasp_depth
    # But grasping seems to be more stable if the contact length is longer.

    finger_contact_ratio = 2.5*(1.0-torch.abs(0.95*max_grasp_depth - finger_contact_lengths)/(0.95*max_grasp_depth))
    #logger.info(f"[DEBUG] finger_contact_ratio shape: {finger_contact_ratio.shape}")

    # prefer high object
    object_height_normalized = (obj_centers[:, 2]-obj_centers[:, 2].min())/(obj_centers[:, 2].max()-obj_centers[:, 2].min())
    object_height_score = 5*(1-object_height_normalized)

    score = dir_deviation + tcp_to_object_center_ratio + y_axis_deviation + gripper_width_ratio + finger_contact_ratio + object_height_score

    print(f"[DEBUG] score shape: {score.shape}")
    print(f"dir_deviation shape: {dir_deviation.shape}")
    print(f"tcp_to_object_center_ratio shape: {tcp_to_object_center_ratio.shape}")
    print(f"y_axis_deviation shape: {y_axis_deviation.shape}")
    print(f"gripper_width_ratio shape: {gripper_width_ratio.shape}")
    print(f"finger_contact_ratio shape: {finger_contact_ratio.shape}")


    return (score-score.min())/(score.max()-score.min())