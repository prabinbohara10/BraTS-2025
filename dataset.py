import os
import torch
from torch.utils.data import Dataset, DataLoader

from utils.utils import load_npy

# Dataset class Definition: takes list of image and label raw numpy filepaths and create a Torch Dataset 
# TODO: parameterize usage of patch vs whole image
class PatchDataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list, patch_size):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        self.patch_size = patch_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        
        image = load_npy(img_path)
        mask = load_npy(mask_path)
        
        if image is None or mask is None:
            raise RuntimeError(f"Failed to load image or mask at index {idx}")
        
        #img_patch, mask_patch = extract_patch(image, mask, self.patch_size)
        # img_patch, mask_patch = augment_image(img_patch, mask_patch)

        # Convert to torch tensor
        img_tensor = torch.from_numpy(image).permute(3, 0, 1, 2).float()  # [D, H, W, C] → [C, D, H, W]
        mask_tensor = torch.from_numpy(mask).permute(3, 0, 1, 2).float()  # [D, H, W]

        return img_tensor, mask_tensor