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
import torch.nn.functional as F

def enlarge_object_mask(object_mask: torch.Tensor, kernel_size: int = 25):
    """
    Enlarge the object mask by a kernel size.
    """
    mask_for_dilation = object_mask.unsqueeze(0).unsqueeze(0).float()

    # A kxk kernel will expand the mask by (k-1)//2 pixel in each direction.
    padding = (kernel_size - 1) // 2

    # F.max_pool2d with a stride of 1 is equivalent to morphological dilation.
    dilated_mask_float = F.max_pool2d(mask_for_dilation,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding)

    # Convert the dilated mask back to a boolean tensor of shape (H, W)
    dilated_mask = dilated_mask_float.squeeze().to(torch.bool)

    return dilated_mask
    
def get_cache_path(cache_dir: str, s3_uri: str, file_type: str) -> Path:
    """Generate a cache file path based on S3 URI."""
    # Create a hash of the S3 URI to use as filename
    uri_hash = hashlib.md5(s3_uri.encode()).hexdigest()
    return cache_dir / f"{file_type}_{uri_hash}.npy"

def download_and_cache(s3_uri: str, cache_path: Path, s3_client: boto3.client):
    """Download a file from S3 and cache it locally."""
    try:
        bucket, key = s3_uri[len("s3://") :].split("/", maxsplit=1)
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        
        # Load and save the numpy array
        data = np.load(buffer)
        np.save(cache_path, data)
    except Exception as e:
        print(f"Error downloading {s3_uri}: {e}")
        raise

