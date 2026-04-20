import torch
import os
from tqdm import tqdm

img_dir = "/home/asavari/foundation_model/datasets/cataract_3d/images"
all_means = []
all_stds = []

# Process first 10 images for speed
files = sorted(os.listdir(img_dir))[:10]
for f in tqdm(files):
    img = torch.load(os.path.join(img_dir, f))  # Shape: [128, 360, 640, 3]
    # Normalize to 0-1
    img = img.float() / 255.0
    # Mean per channel (last dim)
    # We want mean over D, H, W
    mean = img.mean(dim=(0, 1, 2))
    std = img.std(dim=(0, 1, 2))
    all_means.append(mean)
    all_stds.append(std)

final_mean = torch.stack(all_means).mean(dim=0)
final_std = torch.stack(all_stds).mean(dim=0)

print(f"Mean per channel (0-1): {final_mean.tolist()}")
print(f"Std per channel (0-1): {final_std.tolist()}")
