import os
import json
import torch
import ipdb
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable


def read_split_3d(json_file: str, split: str = "train"):
    """Read split information from JSON file."""
    with open(json_file, "r") as f:
        split_dict = json.load(f)
    image_names = split_dict[split]
    return image_names


class Dataset3D(Dataset):
    """
    Dataset class for 3D volumetric data (images and masks).
    
    Handles loading .pt files with shapes:
    - Images: 3 x D x H x W
    - Masks: 1 x D x H x W
    
    Optionally extracts sub-volumes of specified dimensions and applies normalization.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        target_depth: Optional[int] = None,
        target_height: Optional[int] = 128,
        target_width: Optional[int] = 128,
        extraction_mode: str = "center",
        transform: Optional[Callable] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        dry_run: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing 'images', 'masks', and 'split.json'
            split: Which split to load ("train", "val", "test")
            target_depth: Depth of extracted volume. If None, use full depth
            target_height: Height of extracted volume (default: 128)
            target_width: Width of extracted volume (default: 128)
            extraction_mode: How to extract sub-volumes ("center", "random")
            transform: Optional transform to apply to volumes
            mean: Mean per channel for normalization. Shape: (3,)
            std: Std per channel for normalization. Shape: (3,)
            dry_run: If True, use only first sample for testing
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.target_depth = target_depth
        self.target_height = target_height
        self.target_width = target_width
        self.extraction_mode = extraction_mode
        self.transform = transform
        
        # Read split information from JSON file
        split_json = os.path.join(root_dir, 'split.json')
        self.image_files = read_split_3d(split_json, split=split)
        
        # Apply dry_run if needed
        if dry_run:
            self.image_files = self.image_files[:1]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No files found in split '{split}' from {split_json}")
        
        # Handle normalization
        if mean is None:
            mean = (0.0, 0.0, 0.0)
        if std is None:
            std = (1.0, 1.0, 1.0)
        
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1, 1)

        print("="*70)
        print("3D Dataset initialized")
        print("="*70)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (image, mask) tensors
        """
        # Load image and mask
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)
        
        image = torch.load(image_path)  # 3 x D x H x W
        mask = torch.load(mask_path)   # 1 x D x H x W --> D x H x W
        
        # Convert to numpy for processing
        image = image.numpy() if isinstance(image, torch.Tensor) else image
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        
        # Extract sub-volumes if target dimensions specified
        if (self.target_depth is not None or 
            self.target_height != image.shape[2] or 
            self.target_width != image.shape[3]):
            if mask.ndim == 3:
                mask = np.expand_dims(mask, axis=0)  # D x H x W --> 1 x D x H x W
            image, mask = self._extract_volume(image, mask)
        
        # Apply normalization
        image = (image - self.mean) / (self.std + 1e-7)
        
        # Convert back to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        if(mask.ndim == 4):
            mask = mask.squeeze(0)  # 1 x D x H x W --> D x H x W

        return image, mask 
    
    def _extract_volume(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract sub-volumes from full volume.
        
        Args:
            image: Full image volume (C x D x H x W)
            mask: Full mask volume (1 x D x H x W)
        
        Returns:
            Extracted (image, mask) numpy arrays
        """
        c, d, h, w = image.shape
        
        # Determine target dimensions
        target_d = self.target_depth if self.target_depth is not None else d
        target_h = self.target_height
        target_w = self.target_width
        
        # Validate dimensions
        if target_d > d or target_h > h or target_w > w:
            raise ValueError(
                f"Target volume ({target_d} x {target_h} x {target_w}) "
                f"exceeds source volume ({d} x {h} x {w})"
            )
        
        # Calculate starting positions
        if self.extraction_mode == "center":
            start_d = (d - target_d) // 2
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
        elif self.extraction_mode == "random":
            start_d = np.random.randint(0, d - target_d + 1)
            start_h = np.random.randint(0, h - target_h + 1)
            start_w = np.random.randint(0, w - target_w + 1)
        else:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")
        
        # Extract sub-volumes
        image_extracted = image[
            :,
            start_d:start_d + target_d,
            start_h:start_h + target_h,
            start_w:start_w + target_w
        ]
        
        mask_extracted = mask[
            :,
            start_d:start_d + target_d,
            start_h:start_h + target_h,
            start_w:start_w + target_w
        ]
        
        return image_extracted, mask_extracted
    
    def get_volume_shape(self, idx: int = 0) -> Tuple[int, int, int, int]:
        """Get the shape of a volume without extracting it."""
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = torch.load(image_path)
        return image.shape


# Example usage and testing
if __name__ == "__main__":
    # Create dataset with full volumes from training split
    dataset_train = Dataset3D(
        root_dir='/home/asavari/foundation_model/datasets/cholec_8k_3d',
        split='train',
    )
    
    # Create dataset with extracted sub-volumes (depth=56, height=128, width=128) from validation split
    dataset_val = Dataset3D(
        root_dir='/home/asavari/foundation_model/datasets/cholec_8k_3d',
        split='val',
        target_depth=56,
        target_height=128,
        target_width=128,
        extraction_mode="center",
    )
    
    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Val dataset size: {len(dataset_val)}")
    print(f"Volume shape: {dataset_train.get_volume_shape()}")
    
    # Load a sample
    image, mask = dataset_train[0]
    print(f"Loaded image shape: {image.shape}")
    print(f"Loaded mask shape: {mask.shape}")