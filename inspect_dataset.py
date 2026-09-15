import torch
import os

img_path = "/home/asavari/foundation_model/datasets/cataract_3d/images/000000.pt"
mask_path = "/home/asavari/foundation_model/datasets/cataract_3d/masks/000000.pt"

img = torch.load(img_path)
mask = torch.load(mask_path)

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image min: {img.min()}, max: {img.max()}, mean: {img.mean()}")

print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask unique values: {torch.unique(mask)}")
