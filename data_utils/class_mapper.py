import os
import cv2
import numpy as np

from tqdm import tqdm
# Paths
mask_dir = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/masks_old"
gt_img_dir = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/images"
output_dir = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/masks_converted"
os.makedirs(output_dir, exist_ok=True)

mask_files = [f for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg"))]

all_unique_vals = set()

mask_single_vals = {}
for mask_file in tqdm(mask_files):
    gt_img = cv2.imread(os.path.join(gt_img_dir,mask_file),cv2.IMREAD_COLOR_RGB)
    gt_bg = cv2.cvtColor((gt_img == 0).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    mask_rgb = cv2.imread(os.path.join(mask_dir, mask_file))
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

    mask_val = ((mask_rgb[:, :, 0].astype(np.uint16) +
                 mask_rgb[:, :, 1].astype(np.uint16) +
                 mask_rgb[:, :, 2].astype(np.uint16)) // 3).astype(np.uint8)
    mask_val[gt_bg] = 0
    mask_single_vals[mask_file] = mask_val
    all_unique_vals.update(np.unique(mask_val))

all_unique_vals = sorted(list(all_unique_vals))
print("All Unique vals:",all_unique_vals)

val_to_class = {val: idx for idx, val in enumerate(all_unique_vals)}

print("Global mapping of pixel values -> class integers:")
for val, cls in val_to_class.items():
    print(f"Pixel value {val} -> Class {cls}")

for mask_file, mask_val in mask_single_vals.items():
    mask_class = np.vectorize(val_to_class.get)(mask_val).astype(np.uint8)
    out_path = os.path.join(output_dir, mask_file)
    cv2.imwrite(out_path, mask_class)

print("\nAll masks converted to single-channel integer masks with consistent mapping.")