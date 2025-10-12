#!/usr/bin/env python3
"""
Stack multi-class masks into per-image .npy files for multi-label training.

Sample Usage:
    python scripts/stack_masks.py \
        --masks-dir "/mnt/data/omkumar/datasets/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/" \
        --out-dir /mnt/data/omkumar/foundation_phase1/datasets/IDRID/masks_npy \
        --resize 512 512

The script expects the directory to contain several subdirectories (one per class)
with mask files in .tif (or .png) format. Files that share the same basename across
subfolders (e.g. IDRiD_01_MA.tif, IDRiD_01_HE.tif, ...) will be stacked in channel
order sorted by subfolder name.

Saves one .npy per image: <out_dir>/<basename>.npy with shape (H, W, C) and dtype uint8.
Each channel is 0/1 (binarized). If masks differ in size, they will be resized to the
first encountered reference size using nearest neighbor.

"""

import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import sys


def find_mask_files_per_class(masks_dir: Path):
    """Return dict[class_name] -> list of file paths"""
    classes = {}
    for entry in sorted(masks_dir.iterdir()):
        if entry.is_dir():
            files = [p for p in sorted(entry.iterdir()) if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')]
            if files:
                classes[entry.name] = files
    return classes


def build_basename_index(classes_dict):
    """Return dict[basename] -> dict[class_name] -> path"""
    index = {}
    for cls, paths in classes_dict.items():
        for p in paths:
            name = p.stem  # basename without extension
            if name not in index:
                index[name] = {}
            index[name][cls] = p
    return index


def load_and_prepare(img_path: Path, size=None):
    im = Image.open(img_path)
    if size is not None and im.size != size:
        im = im.resize(size, resample=Image.NEAREST)
    arr = np.array(im)
    # If grayscale or palette, convert to single channel
    if arr.ndim == 3:
        # take first channel if multiple channels in mask
        arr = arr[..., 0]
    # binarize: anything > 0 -> 1
    arr = (arr > 0).astype(np.uint8)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--resize', nargs=2, type=int, help='optional H W to resize all masks to', default=None)
    parser.add_argument('--index-csv', type=str, default=None, help='optional CSV index file path to write')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not masks_dir.exists():
        print('Masks directory does not exist:', masks_dir, file=sys.stderr)
        sys.exit(2)

    classes = find_mask_files_per_class(masks_dir)
    if not classes:
        print('No class subdirectories with mask files found in', masks_dir, file=sys.stderr)
        sys.exit(2)

    print('Found classes:', list(classes.keys()))

    index = build_basename_index(classes)
    print(f'Found {len(index)} unique basenames across classes')

    csv_rows = []
    ref_size = None
    if args.resize:
        ref_size = (args.resize[1], args.resize[0])  # PIL expects (W,H)

    for basename, cls_map in sorted(index.items()):
        # determine number of channels = number of classes
        channels = []
        # sort classes for deterministic channel order
        for cls in sorted(classes.keys()):
            p = cls_map.get(cls)
            if p is None:
                # missing mask -> use zeros
                channels.append(None)
            else:
                channels.append(p)

        # determine size from the first existing mask if not resizing
        if ref_size is None:
            for c in channels:
                if c is not None:
                    with Image.open(c) as im:
                        ref_size = im.size  # (W,H)
                    break
            if ref_size is None:
                print('No mask files found for', basename, 'skipping', file=sys.stderr)
                continue

        H = ref_size[1]
        W = ref_size[0]
        stacked = np.zeros((H, W, len(channels)), dtype=np.uint8)

        for i, c in enumerate(channels):
            if c is None:
                continue
            arr = load_and_prepare(c, size=ref_size)
            if arr.shape != (H, W):
                print(f'Resized mask shape mismatch for {basename} class {i}', file=sys.stderr)
            stacked[..., i] = arr

        out_path = out_dir / f"{basename}.npy"
        if args.dry_run:
            print('Would save', out_path, 'shape', stacked.shape)
        else:
            np.save(out_path, stacked)
        csv_rows.append((basename, str(out_path), stacked.shape[0], stacked.shape[1], stacked.shape[2]))

    if args.index_csv:
        with open(args.index_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['basename', 'npy_path', 'H', 'W', 'C'])
            writer.writerows(csv_rows)

    print('Done. Saved', len(csv_rows), '.npy files to', out_dir)


if __name__ == '__main__':
    main()
