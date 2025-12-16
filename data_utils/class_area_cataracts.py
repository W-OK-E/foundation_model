import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# cataracts_image_dir = '/mnt/data/omkumar/foundation_phase1/datasets/cataract1k/images'
# cataracts_mask_dir = '/mnt/data/omkumar/foundation_phase1/datasets/cataract1k/masks_mapped'

num_classes = 11
class_names = ['bg','pupil', 'cornea', 'skin', 'iris', 'surgical_instrument', 'hand', 'speculum', 'other_instruments', 'lens', 'misc', 'misc']

class_image_areas = defaultdict(list)


image_dir = '/mnt/data/omkumar/foundation_phase1/datasets/cataract1k/images/case_2000'
mask_dir = '/mnt/data/omkumar/foundation_phase1/datasets/cataract1k/masks_mapped/case_2000'

images = sorted(os.listdir(image_dir))
masks = sorted(os.listdir(mask_dir))

for img_name,mask_name in zip(images,masks):
    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)

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