import torch
import numpy as np
import pandas as pd
from io import BytesIO
import os
import boto3
import torchvision.transforms as transforms
import hashlib
from pathlib import Path
from tqdm.auto import tqdm
from pytorch3d.transforms import axis_angle_to_matrix
from DP_Grasp.data.preprocessing_utils import download_and_cache, get_cache_path

class RGBD_R7_Dataset(torch.utils.data.Dataset):
    def __init__(self, s3_path: str, split: str, resize: tuple=(224, 224), cache_dir: str="/home/ubuntu/data_cache"):
        self.s3_path = s3_path
        self.split = split
        self.resize_transform = transforms.Resize(resize)
        self.cache_dir = Path(cache_dir+f"/{s3_path.split('/')[-1]}/{split}")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the parquet table
        table = pd.read_parquet(self.s3_path)
        self.table = table[table["split"] == self.split]
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Pre-download and cache all data
        print(f"Loading {len(self.table)} samples for {split} split...")
        self._cache_all_data()
        print(f"Data caching complete for {split} split!")
    
    def _cache_all_data(self):
        """Download and cache all data files locally."""
        for idx in tqdm(range(len(self.table)), desc=f"Caching {self.split} data"):
            s3_links = self.table.iloc[idx]
            
            # Cache each file type
            for file_type in ["obj_depth_map", "obj_rgb", "top_grasp_r7"]:
                s3_uri = s3_links[file_type]
                cache_path = get_cache_path(self.cache_dir, s3_uri, file_type)
                
                # Only download if not already cached
                if not cache_path.exists():
                    download_and_cache(s3_uri, cache_path, self.s3_client)
    
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        s3_links = self.table.iloc[idx]
        
        # Load data from local cache
        datum = {
            "obj_depth_map": self.resize_transform(
                torch.from_numpy(
                    np.load(get_cache_path(self.cache_dir, s3_links["obj_depth_map"], "obj_depth_map"))
                ).unsqueeze(0)
            ),
            "obj_rgb": self.resize_transform(
                torch.from_numpy(
                    np.load(get_cache_path(self.cache_dir, s3_links["obj_rgb"], "obj_rgb"))
                )
            ),
            "top_grasp_r7": torch.from_numpy(
                np.load(get_cache_path(self.cache_dir, s3_links["top_grasp_r7"], "top_grasp_r7"))
            ).unsqueeze(0),
        }
        
        return datum
    
class PC_R7_Dataset(torch.utils.data.Dataset):
    def __init__(self, s3_path: str, split: str, resize: tuple=(224, 224), cache_dir: str="/home/ubuntu/data_cache"):
        self.s3_path = s3_path
        self.split = split
        self.resize_transform = transforms.Resize(resize)
        self.cache_dir = Path(cache_dir+f"/{s3_path.split('/')[-1]}/{split}")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the parquet table
        table = pd.read_parquet(self.s3_path)
        self.table = table[table["split"] == self.split]

        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Pre-download and cache all data
        print(f"Loading {len(self.table)} samples for {split} split...")
        self._cache_all_data()
        print(f"Data caching complete for {split} split!")

    def _cache_all_data(self):
        """Download and cache all data files locally."""
        for idx in tqdm(range(len(self.table)), desc=f"Caching {self.split} data"):
            s3_links = self.table.iloc[idx]
            
            # Cache each file type
            for file_type in ["obj_point_map_unfiltered", "top_grasp_r7"]:
                s3_uri = s3_links[file_type]
                cache_path = get_cache_path(self.cache_dir, s3_uri, file_type)
                
                # Only download if not already cached
                if not cache_path.exists():
                    download_and_cache(s3_uri, cache_path, self.s3_client)
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        s3_links = self.table.iloc[idx]
        
        # Load data from local cache
        datum = {
            "obj_point_map_unfiltered": self.resize_transform(
                torch.from_numpy(
                    np.load(get_cache_path(self.cache_dir, s3_links["obj_point_map_unfiltered"], "obj_point_map_unfiltered"))
                )
            ),
            "top_grasp_r7": torch.from_numpy(
                np.load(get_cache_path(self.cache_dir, s3_links["top_grasp_r7"], "top_grasp_r7"))
            ).unsqueeze(0),
        }
        
        return datum

class PC_R10_Dataset(torch.utils.data.Dataset):
    def __init__(self, s3_path: str, split: str, resize: tuple=(224, 224), cache_dir: str="/home/ubuntu/data_cache"):
        self.s3_path = s3_path
        self.split = split
        self.resize_transform = transforms.Resize(resize)
        self.cache_dir = Path(cache_dir+f"/{s3_path.split('/')[-1]}/{split}")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the parquet table
        table = pd.read_parquet(self.s3_path)
        self.table = table[table["split"] == self.split]
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Pre-download and cache all data
        print(f"Loading {len(self.table)} samples for {split} split...")
        self._cache_all_data()
        print(f"Data caching complete for {split} split!")
    
    def _cache_all_data(self):
        """Download and cache all data files locally."""
        for idx in tqdm(range(len(self.table)), desc=f"Caching {self.split} data"):
            s3_links = self.table.iloc[idx]
            
            # Cache each file type
            for file_type in ["obj_point_map_unfiltered", "top_grasp_r7"]:
                s3_uri = s3_links[file_type]
                cache_path = get_cache_path(self.cache_dir, s3_uri, file_type)
                
                # Only download if not already cached
                if not cache_path.exists():
                    download_and_cache(s3_uri, cache_path, self.s3_client)

    
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        s3_links = self.table.iloc[idx]
        
        # Load data from local cache
        obj_point_map_unfiltered = self.resize_transform(
                torch.from_numpy(
                    np.load(get_cache_path(self.cache_dir, s3_links["obj_point_map_unfiltered"], "obj_point_map_unfiltered"))
                )
            ),
        
        if isinstance(obj_point_map_unfiltered, tuple):
            obj_point_map_unfiltered = obj_point_map_unfiltered[0]
        
        top_grasp_r7 = torch.from_numpy(
                np.load(get_cache_path(self.cache_dir, s3_links["top_grasp_r7"], "top_grasp_r7"))
            ),
        
        if isinstance(top_grasp_r7, tuple):
            #print("top_grasp_r7 is a tuple")
            #print(top_grasp_r7)
            top_grasp_r7 = top_grasp_r7[0]
        
        obj_point_map_reshaped = obj_point_map_unfiltered.reshape(-1, 3)
        obj_center = obj_point_map_reshaped.mean(dim=0)
        obj_max_dist = torch.norm(obj_point_map_reshaped - obj_center, dim=1).max()
        obj_point_map_normalized = (obj_point_map_unfiltered - obj_center.view(3, 1, 1)) / obj_max_dist

        top_grasp_r10 = torch.zeros(10)
        top_grasp_r10[:3] = (top_grasp_r7[:3]-obj_center) / obj_max_dist
        rot_matrix = axis_angle_to_matrix(top_grasp_r7[3:6])[:3, :3]
        top_grasp_r10[3:6] = rot_matrix[1, :]
        top_grasp_r10[6:9] = rot_matrix[2, :]
        top_grasp_r10[9] = top_grasp_r7[6]
        
        datum = {
            "obj_point_map_normalized": obj_point_map_normalized,
            "top_grasp_r10": top_grasp_r10.unsqueeze(0),
            "obj_center": obj_center,
            "obj_max_dist": obj_max_dist,
        }
        
        return datum
    
class PC_R9_Trans_Only_Dataset(torch.utils.data.Dataset):
    def __init__(self, s3_path: str, split: str, resize: tuple=(224, 224), cache_dir: str="/home/ubuntu/data_cache"):
        self.s3_path = s3_path
        self.split = split
        self.resize_transform = transforms.Resize(resize)
        self.cache_dir = Path(cache_dir+f"/{s3_path.split('/')[-1]}/{split}")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the parquet table
        table = pd.read_parquet(self.s3_path)
        self.table = table[table["split"] == self.split]
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Pre-download and cache all data
        print(f"Loading {len(self.table)} samples for {split} split...")
        self._cache_all_data()
        print(f"Data caching complete for {split} split!")
    
    def _cache_all_data(self):
        """Download and cache all data files locally."""
        for idx in tqdm(range(len(self.table)), desc=f"Caching {self.split} data"):
            s3_links = self.table.iloc[idx]
            
            # Cache each file type
            for file_type in ["obj_point_map_unfiltered", "top_grasp_r7"]:
                s3_uri = s3_links[file_type]
                cache_path = get_cache_path(self.cache_dir, s3_uri, file_type)
                
                # Only download if not already cached
                if not cache_path.exists():
                    download_and_cache(s3_uri, cache_path, self.s3_client)

    
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        s3_links = self.table.iloc[idx]
        
        # Load data from local cache
        obj_point_map_unfiltered = self.resize_transform(
                torch.from_numpy(
                    np.load(get_cache_path(self.cache_dir, s3_links["obj_point_map_unfiltered"], "obj_point_map_unfiltered"))
                )
            ),
        
        if isinstance(obj_point_map_unfiltered, tuple):
            obj_point_map_unfiltered = obj_point_map_unfiltered[0]
        
        top_grasp_r7 = torch.from_numpy(
                np.load(get_cache_path(self.cache_dir, s3_links["top_grasp_r7"], "top_grasp_r7"))
            ),
        
        if isinstance(top_grasp_r7, tuple):
            #print("top_grasp_r7 is a tuple")
            #print(top_grasp_r7)
            top_grasp_r7 = top_grasp_r7[0]
        
        obj_point_map_reshaped = obj_point_map_unfiltered.reshape(-1, 3)
        obj_center = obj_point_map_reshaped.mean(dim=0)
        obj_max_dist = torch.norm(obj_point_map_reshaped - obj_center, dim=1).max()
        obj_point_map_normalized = (obj_point_map_unfiltered - obj_center.view(3, 1, 1)) / obj_max_dist

        top_grasp_r9 = torch.zeros(9)
        top_grasp_r9[:3] = (top_grasp_r7[:3]-obj_center) / obj_max_dist
        rot_matrix = axis_angle_to_matrix(top_grasp_r7[3:6])[:3, :3]
        top_grasp_r9[3:6] = rot_matrix[1, :]
        top_grasp_r9[6:9] = rot_matrix[2, :]
        
        datum = {
            "obj_point_map_normalized": obj_point_map_normalized,
            "top_grasp_r9": top_grasp_r9.unsqueeze(0),
            "obj_center": obj_center,
            "obj_max_dist": obj_max_dist,
        }
        
        return datum
    

def load_np_s3(s3_uri: str, s3_client: boto3.client)->np.ndarray:
    """Load a numpy array from S3."""
    bucket, key = s3_uri[len("s3://") :].split("/", maxsplit=1)
    buffer = BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)  # Rewind the buffer
    
    # Load the NumPy array
    loaded_array = np.load(buffer)
    
    return loaded_array
