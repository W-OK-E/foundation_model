import torch
import os

mask_dir = "/home/asavari/foundation_model/datasets/cataract_3d/masks"
all_unique = set()
for f in sorted(os.listdir(mask_dir))[:10]:
    mask = torch.load(os.path.join(mask_dir, f))
    unique_vals = torch.unique(mask).tolist()
    all_unique.update(unique_vals)

print(f"Unique values in first 10 masks: {sorted(list(all_unique))}")
