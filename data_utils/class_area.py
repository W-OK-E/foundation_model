import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

cataracts_image_dir = '/mnt/data/omkumar/foundation_phase1/datasets/SASVi/semantic_segmentation/cholec80/images'
cataracts_mask_dir = '/mnt/data/omkumar/foundation_phase1/datasets/SASVi/semantic_segmentation/cholec80/masks_mapped'

num_classes = 12
class_names = ["Black Background","Abdominal Wall","Liver",
            "Gastrointestinal Tract","Fat","Grasper","Connective Tissue",
            "Blood","Cystic Duct","L-hook Electrocautery",
            "Gallbladder","Hepatic Vein"]

class_image_areas = defaultdict(list)

print("Processing: \n")

for image_dir,mask_dir in tqdm(zip(os.listdir(cataracts_image_dir),os.listdir(cataracts_mask_dir))):
    image_dir = os.path.join(cataracts_image_dir,image_dir)
    mask_dir = os.path.join(cataracts_mask_dir,mask_dir)

    images = sorted(os.listdir(image_dir))


    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            continue

        H, W = mask.shape
        img_area = H * W

        for cls_id in range(num_classes):
            pixels = np.sum(mask == cls_id)

            if pixels == 0:
                continue

            area = pixels/img_area * 100.0
            class_image_areas[cls_id].append(area)

print("\nPer-image aggregated class area statistics:\n")
print(f"{'Class':30s} | {'Min':>10s} | {'Mean':>10s} | {'Max':>10s} | Images")

for cls_id in range(num_classes):
    areas = class_image_areas.get(cls_id, [])

    if len(areas) == 0:
        print(f"{class_names[cls_id]:30s} | {'-':>10s} | {'-':>10s} | {'-':>10s} | 0")
        continue

    areas = np.array(areas)

    print(
        f"{class_names[cls_id]:30s} | "
        f"{areas.min():10.6f} | "
        f"{areas.mean():10.6f} | "
        f"{areas.max():10.6f} | "
        f"{len(areas)}"
    )