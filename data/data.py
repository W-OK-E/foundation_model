import os
import json
import numpy as np
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
from .transforms import get_transforms

class SEGDataset(Dataset):
    def __init__(self, root_dir, split="train", 
                train_im_size=(512, 512), dataset_name=None, 
                mean=(0,0,0), std=(1,1,1), cls_weights=None):
        """
        Minimal Segmentation Dataset.
        Reflect pads images to train_im_size (multiples of 128) using Transforms.
        """
        self.root_dir = root_dir
        self.split = split
        self.dataset_name = dataset_name
        
        # Load filenames from split.json
        split_file = os.path.join(root_dir, 'split.json')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"split.json not found in {root_dir}")
            
        with open(split_file, 'r') as f:
            self.images = json.load(f)[split]
            
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # Initialize transforms (PadIfNeeded handles the reflect padding)
        t_train, t_val = get_transforms(img_size=train_im_size, mean=mean, std=std)
        self.transform = t_train if split == "train" else t_val

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        image = iio.imread(os.path.join(self.image_dir, img_name))
        mask = iio.imread(os.path.join(self.mask_dir, img_name))
        
        # Convert RGB mask to grayscale if needed
        if mask.ndim == 3:
            mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            
        return image, mask.long(), self.dataset_name