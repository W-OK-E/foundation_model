"""
Split image and mask pairs into train (80%) and validation (20%) folders.

Usage example:
python split_train_val.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --out-train-image-dir /path/to/train/images \
  --out-train-mask-dir /path/to/train/masks \
  --out-val-image-dir /path/to/val/images \
  --out-val-mask-dir /path/to/val/masks \
  --train-split 0.8 --seed 42
"""

import argparse
import shutil
from pathlib import Path
import random


def gather_image_files(img_dir, exts=None):
    """Gather all image files from image directory."""
    exts = exts or {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = []
    for p in Path(img_dir).rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='Split image-mask pairs into train/val directories'
    )
    parser.add_argument('--image-dir', required=True, help='Source image directory')
    parser.add_argument('--mask-dir', required=True, help='Source mask directory')
    parser.add_argument('--out-train-image-dir', required=True, help='Output train image directory')
    parser.add_argument('--out-train-mask-dir', required=True, help='Output train mask directory')
    parser.add_argument('--out-val-image-dir', required=True, help='Output val image directory')
    parser.add_argument('--out-val-mask-dir', required=True, help='Output val mask directory')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data to use for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Gather all image files
    image_files = gather_image_files(args.image_dir)
    print(f'Found {len(image_files)} images in {args.image_dir}')

    if not image_files:
        print('No images found. Exiting.')
        return

    # Shuffle and split
    shuffled = image_files.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * args.train_split)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]

    print(f'Train: {len(train_images)} images ({100*args.train_split:.1f}%)')
    print(f'Val: {len(val_images)} images ({100*(1-args.train_split):.1f}%)')

    # Create output directories
    Path(args.out_train_image_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_train_mask_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_val_image_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_val_mask_dir).mkdir(parents=True, exist_ok=True)

    mask_dir = Path(args.mask_dir)

    # Copy train files
    print('\nCopying train files...')
    for i, img_path in enumerate(train_images, 1):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            shutil.copy2(img_path, Path(args.out_train_image_dir) / img_path.name)
            shutil.copy2(mask_path, Path(args.out_train_mask_dir) / mask_path.name)
            if i % 100 == 0 or i == len(train_images):
                print(f'  Copied {i}/{len(train_images)} train pairs')
        else:
            print(f'  Warning: mask not found for {img_path.name}')

    # Copy val files
    print('\nCopying val files...')
    for i, img_path in enumerate(val_images, 1):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            shutil.copy2(img_path, Path(args.out_val_image_dir) / img_path.name)
            shutil.copy2(mask_path, Path(args.out_val_mask_dir) / mask_path.name)
            if i % 100 == 0 or i == len(val_images):
                print(f'  Copied {i}/{len(val_images)} val pairs')
        else:
            print(f'  Warning: mask not found for {img_path.name}')

    print('\nDone!')
    print(f'Train: {len(list(Path(args.out_train_image_dir).glob("*")))} files copied')
    print(f'Val: {len(list(Path(args.out_val_image_dir).glob("*")))} files copied')


if __name__ == '__main__':
    main()
