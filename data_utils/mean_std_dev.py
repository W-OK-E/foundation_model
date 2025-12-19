#!/usr/bin/env python3
import os
import cv2
import numpy as np
import sys
import imageio as iio
import tqdm

def main():
   
    dataset_name = sys.argv[1]
    image_dir = f"datasets/{dataset_name}/images"
    output_file = f"datasets/{dataset_name}/images_shape.txt"

    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist.")
        sys.exit(1)

    image_extensions = (".png", ".tif",".dcm")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    sum_channels = np.zeros(3, dtype=np.float64)
    sum_sq_channels = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for img_file in tqdm.tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = iio.imread(img_path) 
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
        
        img = img.astype(np.float32) / 255.0  

        if(len(img.shape) == 2):
            img = img.reshape(*img.shape,1)
        h, w, c = img.shape
        total_pixels += h * w

        sum_channels += img.reshape(-1, c).sum(axis=0)
        sum_sq_channels += (img.reshape(-1, c) ** 2).sum(axis=0)

    mean_per_channel = sum_channels / total_pixels
    std_per_channel = np.sqrt(sum_sq_channels / total_pixels - mean_per_channel**2)

    mean_per_channel = np.round(mean_per_channel, 8)
    std_per_channel = np.round(std_per_channel, 8)

    with open(output_file, "w") as f:
        f.write(f"mean_per_channel: {mean_per_channel.tolist()}\n")
        f.write(f"std_per_channel: {std_per_channel.tolist()}\n")

    print(f"Results saved in {output_file}")
    print(f"mean_per_channel: {mean_per_channel}")
    print(f"std_per_channel: {std_per_channel}")

if __name__ == "__main__":
    main()