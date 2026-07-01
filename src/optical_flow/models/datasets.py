from pathlib import Path
import os
import io
import bisect

import boto3
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.io as tv_io


class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, ext):
        # accept either a local path or an s3:// URI
        self.raw_data_dir = data_dir
        self.ext = ext

        self.all_filenames = []
        self.dir_paths = []  # for local: Path; for s3: (bucket, prefix)

        self.pair_offsets = [0]
        self.file_offsets = [0]

        total_pairs = 0
        total_files = 0

        # S3 handling
        self.s3 = False
        if isinstance(data_dir, str) and data_dir.startswith("s3://"):
            self.s3 = True
            # parse s3://bucket/prefix...
            no_scheme = data_dir[len("s3://") :]
            parts = no_scheme.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            prefix = prefix.rstrip('/') + '/'

            self.s3_client = boto3.client('s3')

            # try to list immediate sub-prefixes (directories)
            resp = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            common = resp.get('CommonPrefixes', [])
            if common:
                sub_prefixes = [c['Prefix'] for c in common]
            else:
                # no subfolders; treat the given prefix as a single directory
                sub_prefixes = [prefix]

            for sub in sorted(sub_prefixes):
                # paginate through objects under this sub-prefix and pick images with matching ext
                keys = []
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket, Prefix=sub):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key.lower().endswith(f".{ext.lower()}"):
                            keys.append(key)

                keys = sorted(keys)
                num_images = len(keys)
                if num_images == 0:
                    continue

                total_pairs += num_images - 1
                total_files += num_images

                self.dir_paths.append((bucket, sub))
                self.all_filenames.extend(keys)
                self.pair_offsets.append(total_pairs)
                self.file_offsets.append(total_files)
            return

        # Local filesystem handling
        self.data_dir = Path(data_dir)
        for sub_dir in sorted(self.data_dir.iterdir()):
            images = sorted([f.name for f in sub_dir.glob(f"*.{ext}")])
            num_images = len(images)
            if num_images == 0:
                continue

            total_pairs += num_images - 1
            total_files += num_images

            self.dir_paths.append(sub_dir)
            self.all_filenames.extend(images)
            self.pair_offsets.append(total_pairs)
            self.file_offsets.append(total_files)

    def __len__(self):
        return self.pair_offsets[-1]

    def __getitem__(self, idx):
        if idx < 0:
            raise ValueError("Cannot provide sample for a negative index")
        if idx >= self.pair_offsets[-1]:
            raise ValueError("Cannot provide sample for an index greater than the length")

        dir_idx = bisect.bisect_right(self.pair_offsets, idx) - 1

        local_pair_idx = idx - self.pair_offsets[dir_idx]

        file_idx = self.file_offsets[dir_idx]

        file_a = self.all_filenames[file_idx + local_pair_idx]
        file_b = self.all_filenames[file_idx + local_pair_idx + 1]

        dir_path = self.dir_paths[dir_idx]

        if getattr(self, 's3', False):
            bucket, _ = dir_path
            resp1 = self.s3_client.get_object(Bucket=bucket, Key=file_a)
            data1 = resp1['Body'].read()
            im1_pil = Image.open(io.BytesIO(data1)).convert('RGB')
            im1 = transforms.ToTensor()(im1_pil)

            resp2 = self.s3_client.get_object(Bucket=bucket, Key=file_b)
            data2 = resp2['Body'].read()
            im2_pil = Image.open(io.BytesIO(data2)).convert('RGB')
            im2 = transforms.ToTensor()(im2_pil)

        else:
            raw1 = tv_io.read_file(dir_path / file_a)
            im1 = tv_io.decode_image(raw1).float() / 255.0

            raw2 = tv_io.read_file(dir_path / file_b)
            im2 = tv_io.decode_image(raw2).float() / 255.0

        return torch.cat([im1, im2], dim=0), []

class SintelDataset(OpticalFlowDataset):
    def __init__(self, data_dir=None, ext="png"):
        if data_dir is None:
            if os.environ.get('ENV') == 'prod' or os.environ.get('USE_S3', '').lower() in ('1', 'true'):
                data_dir = 's3://ml-models-515121257519-us-east-1-an/optical-flow/mpi_sintel/training/final'
            else:
                data_dir = 'mpi_sintel/training/final'
        super().__init__(data_dir, ext)


class YoutubeDataset(OpticalFlowDataset):
    def __init__(self, data_dir=None, ext="jpg"):
        if data_dir is None:
            if os.environ.get('ENV') == 'prod' or os.environ.get('USE_S3', '').lower() in ('1', 'true'):
                data_dir = 's3://ml-models-515121257519-us-east-1-an/optical-flow/frames'
            else:
                data_dir = 'frames'
        super().__init__(data_dir, ext)