from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.io as tv_io
import bisect


class SintelDataset(Dataset):
    def __init__(self, data_dir="path_to_sintel"):
        self.data_dir = data_dir
        self.image_pairs = [] # List of paths to consecutive frames

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # 1. Load image_t and image_t+1 using PIL or OpenCV
        # 2. Transform them to Tensors
        # 3. Concatenate them so splitted_im can break them apart
        img_combined = torch.cat([img1, img2], dim=0) 
        dummy_target = torch.tensor(0) 
        
        return img_combined, dummy_target


class YoutubeDataset(Dataset):
    def __init__(self, data_dir="frames"):
        self.data_dir = Path(data_dir)
        
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.all_filenames = []
        self.dir_paths = []

        self.pair_offsets = [0]
        self.file_offsets = [0]

        total_pairs = 0
        total_files = 0

        for sub_dir in sorted(self.data_dir.iterdir()):

            images = sorted([f.name for f in sub_dir.glob("*.jpg")])
            num_images = len(images)
            
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
            return ValueError("Cannot provide sample for and index greater than the length")

        dir_idx = bisect.bisect_right(self.pair_offsets, idx) - 1

        local_pair_idx = idx - self.pair_offsets[dir_idx]

        file_idx = self.file_offsets[dir_idx]

        file_a_name = self.all_filenames[file_idx + local_pair_idx]
        file_b_name = self.all_filenames[file_idx + local_pair_idx + 1]

        dir_path = self.dir_paths[dir_idx]
        
        im1 = tv_io.decode_image(dir_path / file_a_name).float() / 255. 
        im2 = tv_io.decode_image(dir_path / file_b_name).float() / 255.

        return torch.cat([im1,im2], dim=0), []