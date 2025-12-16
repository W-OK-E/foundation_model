#!/usr/bin/env python3
"""
Convert RGB images to single-channel images with unique integer mappings.
Each unique RGB color gets a unique integer ID.
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path


def rgb_to_int_direct(rgb_image):
    """
    Convert RGB to single integer using bit packing: R*65536 + G*256 + B
    
    This gives a unique integer for each RGB combination.
    Range: 0 to 16,777,215 (2^24 - 1)
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) with values 0-255
        
    Returns:
        Single channel image of shape (H, W) with unique integers
    """
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    return r.astype(np.uint32) * 65536 + g.astype(np.uint32) * 256 + b.astype(np.uint32)


def int_to_rgb_direct(int_image):
    """
    Convert single integer back to RGB
    
    Args:
        int_image: numpy array of shape (H, W) with uint32 values
        
    Returns:
        RGB image of shape (H, W, 3)
    """
    int_image = int_image.astype(np.uint32)
    r = (int_image // 65536) % 256
    g = (int_image // 256) % 256
    b = int_image % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def rgb_to_int_indexed(rgb_image):
    """
    Convert RGB to single integer using a color palette/index.
    
    This assigns sequential IDs (0, 1, 2, ...) to unique colors.
    More memory efficient but IDs depend on color discovery order.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3)
        
    Returns:
        (indexed_image, color_map) where:
            - indexed_image: shape (H, W) with indices
            - color_map: dict mapping index -> RGB tuple
    """
    h, w = rgb_image.shape[:2]
    
    # Reshape to list of RGB tuples
    pixels = rgb_image.reshape(-1, 3)
    
    # Find unique colors and their indices
    unique_colors, inverse_indices = np.unique(pixels, axis=0, return_inverse=True)
    
    # Create color map
    color_map = {i: tuple(color) for i, color in enumerate(unique_colors)}
    
    # Reshape back to image
    indexed_image = inverse_indices.reshape(h, w)
    
    return indexed_image, color_map


def int_to_rgb_indexed(indexed_image, color_map):
    """
    Convert indexed image back to RGB using color map
    
    Args:
        indexed_image: numpy array of shape (H, W) with indices
        color_map: dict mapping index -> RGB tuple
        
    Returns:
        RGB image of shape (H, W, 3)
    """
    h, w = indexed_image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx, rgb in color_map.items():
        mask = indexed_image == idx
        rgb_image[mask] = rgb
    
    return rgb_image


def main():
    parser = argparse.ArgumentParser(
        description="Convert RGB images to single-channel with unique integer IDs"
    )
    parser.add_argument("input", help="Input RGB image path")
    parser.add_argument("output", help="Output path (will save as .npy)")
    parser.add_argument(
        "--method",
        choices=["direct", "indexed"],
        default="direct",
        help="Conversion method: 'direct' (bit packing) or 'indexed' (palette)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by converting back to RGB"
    )
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading {args.input}...")
    img = Image.open(args.input).convert('RGB')
    rgb_array = np.array(img)
    print(f"Image shape: {rgb_array.shape}")
    
    # Convert based on method
    if args.method == "direct":
        print("\nUsing direct bit-packing method (R*65536 + G*256 + B)...")
        int_image = rgb_to_int_direct(rgb_array)
        print(f"Output shape: {int_image.shape}")
        print(f"Output dtype: {int_image.dtype}")
        print(f"Value range: [{int_image.min()}, {int_image.max()}]")
        print(f"Unique values: {len(np.unique(int_image))}")
        
        # Save
        np.save(args.output, int_image)
        print(f"\nSaved to: {args.output}")
        
        # Verify if requested
        if args.verify:
            print("\nVerifying conversion...")
            rgb_reconstructed = int_to_rgb_direct(int_image)
            if np.array_equal(rgb_array, rgb_reconstructed):
                print("✓ Verification successful! Conversion is lossless.")
            else:
                print("✗ Verification failed! Reconstruction doesn't match original.")
    
    else:  # indexed
        print("\nUsing indexed palette method...")
        indexed_image, color_map = rgb_to_int_indexed(rgb_array)
        print(f"Output shape: {indexed_image.shape}")
        print(f"Output dtype: {indexed_image.dtype}")
        print(f"Number of unique colors: {len(color_map)}")
        print(f"Index range: [{indexed_image.min()}, {indexed_image.max()}]")
        
        # Save both indexed image and color map
        output_path = Path(args.output)
        np.save(output_path.with_suffix('.npy'), indexed_image)
        np.save(output_path.with_stem(output_path.stem + '_colormap').with_suffix('.npy'), 
                color_map)
        print(f"\nSaved indexed image to: {output_path.with_suffix('.npy')}")
        print(f"Saved color map to: {output_path.with_stem(output_path.stem + '_colormap').with_suffix('.npy')}")
        
        # Verify if requested
        if args.verify:
            print("\nVerifying conversion...")
            rgb_reconstructed = int_to_rgb_indexed(indexed_image, color_map)
            if np.array_equal(rgb_array, rgb_reconstructed):
                print("✓ Verification successful! Conversion is lossless.")
            else:
                print("✗ Verification failed! Reconstruction doesn't match original.")


if __name__ == "__main__":
    main()


# Example usage in code:
"""
# Method 1: Direct bit-packing (most straightforward)
rgb_img = np.array(Image.open('image.png').convert('RGB'))
int_img = rgb_to_int_direct(rgb_img)
# Each RGB (r,g,b) -> integer = r*65536 + g*256 + b
# Reversible: rgb_img_restored = int_to_rgb_direct(int_img)

# Method 2: Indexed palette (more compact)
indexed_img, color_map = rgb_to_int_indexed(rgb_img)
# Each unique color gets sequential ID: 0, 1, 2, ...
# Reversible: rgb_img_restored = int_to_rgb_indexed(indexed_img, color_map)
"""