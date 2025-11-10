#!/usr/bin/env python3
"""
color_to_index_mapper.py
--------------------------------
Finds all unique RGB colors across a folder of images,
creates a color→index mapping, and converts each image
into an integer-labeled mask.

Usage:
    python color_to_index_mapper.py \
        --input_dir /path/to/color_masks \
        --output_dir /path/to/index_masks \
        --save_mapper /path/to/color_mapper.json
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def find_unique_colors(image_paths):
    """Return a sorted list of all unique RGB colors across given images."""
    unique_colors = set()
    for path in tqdm(image_paths, desc="Collecting unique colors"):
        img = np.array(Image.open(path).convert("RGB"))
        # Reshape to (N, 3) and get unique rows
        colors = np.unique(img.reshape(-1, 3), axis=0)
        for c in colors:
            unique_colors.add(tuple(c))
    return sorted(list(unique_colors))


def map_colors_to_indices(unique_colors):
    """Return dict mapping color tuples to integer indices."""
    return {color: idx for idx, color in enumerate(unique_colors)}


def convert_image_to_indices(img_path, color_to_idx):
    """Convert one color image to an integer-labeled mask."""
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w, _ = img.shape
    indexed_mask = np.zeros((h, w), dtype=np.int32)

    # Build lookup array for speed
    color_to_idx_np = {np.array(k, dtype=np.uint8).tobytes(): v for k, v in color_to_idx.items()}

    # Flatten and map colors
    flat = img.reshape(-1, 3)
    for i, rgb in enumerate(flat):
        idx = color_to_idx_np.get(rgb.tobytes(), -1)  # -1 if unseen color
        indexed_mask.flat[i] = idx

    return Image.fromarray(indexed_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to folder with RGB masks")
    parser.add_argument("--output_dir", required=True, help="Where to save integer masks")
    parser.add_argument("--save_mapper", default="color_mapper.json", help="JSON file to store color-index mapping")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get all image paths
    image_paths = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        print("❌ No images found in", args.input_dir)
        return

    # 1️⃣ Collect unique colors
    unique_colors = find_unique_colors(image_paths)
    print(f"✅ Found {len(unique_colors)} unique colors")

    # 2️⃣ Create mapping
    color_to_idx = map_colors_to_indices(unique_colors)

    # 3️⃣ Save mapper
    with open(args.save_mapper, "w") as f:
        json.dump({"mapping": {str(k): v for k, v in color_to_idx.items()}}, f, indent=2)
    print(f"💾 Saved color mapping to {args.save_mapper}")

    # 4️⃣ Convert all images
    for img_path in tqdm(image_paths, desc="Converting images"):
        indexed_img = convert_image_to_indices(img_path, color_to_idx)
        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        indexed_img.save(out_path)

    print(f"🎉 Done! Integer masks saved in {args.output_dir}")


if __name__ == "__main__":
    main()
