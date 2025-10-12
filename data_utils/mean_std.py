import os
import argparse

from pkg_resources import require
import numpy as np
from PIL import Image

MAX_IMAGES = 100

parser = argparse.ArgumentParser()
parser.add_argument("--folder",required = True, help = "Path to the folder containing the images")
args = parser.parse_args()

IMG_DIR = args.folder
# Lists to accumulate pixel values
means = []
stds = []

# Get all image files
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Process first 100 (or fewer) images
for i, filename in enumerate(image_files[:MAX_IMAGES]):
    img_path = os.path.join(IMG_DIR, filename)
    img = Image.open(img_path).convert("RGB")  # convert to RGB to ensure 3 channels
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]

    # Compute per-channel mean and std
    means.append(arr.mean(axis=(0, 1)))
    stds.append(arr.std(axis=(0, 1)))

# Convert lists to arrays
means = np.array(means)
stds = np.array(stds)

# Compute dataset-level mean and std
dataset_mean = means.mean(axis=0)
dataset_std = stds.mean(axis=0)

print(f"✅ Mean per channel: {dataset_mean}")
print(f"✅ Std per channel: {dataset_std}")
