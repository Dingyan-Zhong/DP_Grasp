import torch
import numpy as np
import pandas as pd
from io import BytesIO
import os
import boto3
import torchvision.transforms as transforms

class RGBD_R7_Dataset(torch.utils.data.Dataset):
    def __init__(self, s3_path: str, split: str, resize: tuple=(224, 224)):
        self.s3_path = s3_path
        table = pd.read_parquet(self.s3_path)
        self.split = split
        self.table = table[table["split"] == self.split]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resize_transform = transforms.Resize(resize)
    
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        s3_links = self.table.iloc[idx]

        # load the data from s3
        s3_client = boto3.client('s3')
        datum = {
            "obj_depth_map": self.resize_transform(torch.from_numpy(load_np_s3(s3_links["obj_depth_map"], s3_client)).unsqueeze(0)),
            "obj_rgb": self.resize_transform(torch.from_numpy(load_np_s3(s3_links["obj_rgb"], s3_client))),
            "top_grasp_r7": torch.from_numpy(load_np_s3(s3_links["top_grasp_r7"], s3_client)),
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
