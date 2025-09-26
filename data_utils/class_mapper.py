import cv2
import numpy as np
import os

# Mapping of pixel values to class IDs
pixel_to_class = {
    0: 0,
    43: 1,
    76: 2,
    126: 3,
    127: 4,
    156: 5,
    160: 6,
    171: 7,
    188: 8,
    200: 9,
    221: 10,
    225: 11
}

def remap_image(img):
    """Remap pixel values in the given image according to pixel_to_class."""
    # Initialize new image with zeros (same shape as grayscale)
    remapped = np.zeros(img.shape, dtype=np.uint8)
    
    for pixel_value, class_id in pixel_to_class.items():
        remapped[img == pixel_value] = class_id
    
    return remapped

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            # Read as grayscale (single channel is enough for masks)
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue
            
            remapped_img = remap_image(img)
            
            # Save with same name in output dir
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, remapped_img)
            print(f"Processed: {filename} -> {out_path}")

# Example usage
if __name__ == "__main__":
    input_dir = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/masks"   # replace with your input folder
    output_dir = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/masks" # replace with your output folder
    process_images(input_dir, output_dir)
