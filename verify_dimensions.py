
import os
import imageio as iio
from PIL import Image
import numpy as np

IMAGE_DIR = "/home/asavari/foundation_model/datasets/INBreast/images"
MASK_DIR = "/home/asavari/foundation_model/datasets/INBreast/masks"

def verify_dimensions():
    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.dcm')])
    masks = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])

    print(f"Found {len(images)} images and {len(masks)} masks.")

    mismatches = []
    missing_masks = []
    
    for img_file in images:
        basename = os.path.splitext(img_file)[0]
        mask_file = f"{basename}.png"
        
        if mask_file not in masks:
            missing_masks.append(basename)
            continue
            
        img_path = os.path.join(IMAGE_DIR, img_file)
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        try:
            ds = iio.imread(img_path)
            # DICOM rows is height, columns is width
            img_shape = (ds.shape[0], ds.shape[1])
            
            with Image.open(mask_path) as m:
                # PIL size is (width, height)
                mask_shape = (m.height, m.width)
                
            if img_shape != mask_shape:
                mismatches.append(f"{basename}: Image {img_shape} != Mask {mask_shape}")
                
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    if mismatches:
        print("\nDimension mismatches found:")
        for m in mismatches:
            print(m)
    else:
        print("\nAll found pairs have matching dimensions.")

    if missing_masks:
        print(f"\nMissing masks for {len(missing_masks)} images.")

if __name__ == "__main__":
    verify_dimensions()
