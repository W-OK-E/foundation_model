"""
Crop masks to their non-zero bounding box, extract patches from both
image and mask, and save only patches where the mask has foreground.

Usage example:
python extract_patches_idrid.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --out-image-dir /path/to/out/images \
  --out-mask-dir /path/to/out/masks \
  --patch-size 1024 --stride 512

The script expects masks as single-channel images where background==0.
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np


def find_bbox(mask_arr, background=0):
    """Return bbox (minr, maxr, minc, maxc) inclusive-exclusive that contains any pixel != background.
    If no foreground found, return None.
    """
    if mask_arr.ndim == 3:
        # take any channel
        m = np.any(mask_arr != background, axis=2)
    else:
        m = mask_arr != background

    rows = np.where(np.any(m, axis=1))[0]
    cols = np.where(np.any(m, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    minr, maxr = rows[0], rows[-1] + 1
    minc, maxc = cols[0], cols[-1] + 1
    return minr, maxr, minc, maxc


def pad_to_size(img_arr, target_h, target_w, pad_value=0):
    h, w = img_arr.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return img_arr
    # pad bottom and right
    if img_arr.ndim == 2:
        padded = np.pad(img_arr, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
    else:
        padded = np.pad(img_arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=pad_value)
    return padded


def sliding_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] + patch_size < length:
        starts.append(length - patch_size)
    return starts


def process_pair(image_path, mask_path, out_image_dir, out_mask_dir, patch_size, stride, bg_value=0):
    img = Image.open(image_path)
    mask = Image.open(mask_path).convert('L')

    img_arr = np.array(img)
    mask_arr = np.array(mask)

    bbox = find_bbox(mask_arr, background=bg_value)
    if bbox is None:
        return 0
    minr, maxr, minc, maxc = bbox

    # crop to bbox
    img_crop = img_arr[minr:maxr, minc:maxc]
    mask_crop = mask_arr[minr:maxr, minc:maxc]

    crop_h, crop_w = mask_crop.shape[:2]

    y_starts = sliding_starts(crop_h, patch_size, stride)
    x_starts = sliding_starts(crop_w, patch_size, stride)

    saved = 0
    base = Path(image_path).stem
    for yi in y_starts:
        for xi in x_starts:
            y0, x0 = yi, xi
            y1, x1 = yi + patch_size, xi + patch_size
            img_patch = img_crop[y0:y1, x0:x1]
            mask_patch = mask_crop[y0:y1, x0:x1]

            # pad if needed
            img_patch = pad_to_size(img_patch, patch_size, patch_size, pad_value=0)
            mask_patch = pad_to_size(mask_patch, patch_size, patch_size, pad_value=bg_value)

            if np.any(mask_patch != bg_value):
                # save
                out_img_name = f"{base}_y{minr + y0:06d}_x{minc + x0:06d}.png"
                out_mask_name = f"{base}_y{minr + y0:06d}_x{minc + x0:06d}.png"
                out_img_path = out_image_dir / out_img_name
                out_mask_path = out_mask_dir / out_mask_name

                # ensure directory exists
                out_image_dir.mkdir(parents=True, exist_ok=True)
                out_mask_dir.mkdir(parents=True, exist_ok=True)

                # convert arrays back to images and save
                if img_patch.ndim == 2:
                    Image.fromarray(img_patch).save(out_img_path)
                else:
                    Image.fromarray(img_patch).save(out_img_path)

                Image.fromarray(mask_patch).save(out_mask_path)
                saved += 1

    return saved


def gather_paths(img_dir, mask_dir, exts=None):
    exts = exts or {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    mask_files = []
    for p in Path(mask_dir).rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            mask_files.append(p)
    pairs = []
    img_dir = Path(img_dir)
    for m in mask_files:
        # try to find corresponding image by stem
        candidate = None
        for ext in exts:
            t = img_dir / (m.stem + ext)
            if t.exists():
                candidate = t
                break
        if candidate is None:
            # try any file with same stem under image dir
            found = list(img_dir.rglob(m.stem + '.*'))
            if found:
                candidate = found[0]
        if candidate is not None:
            pairs.append((candidate, m))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--mask-dir', required=True)
    parser.add_argument('--out-image-dir', required=True)
    parser.add_argument('--out-mask-dir', required=True)
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--bg-value', type=int, default=0)
    parser.add_argument('--max-files', type=int, default=None,
                        help='Optional: limit number of files to process (for testing)')
    args = parser.parse_args()

    patch_size = args.patch_size
    stride = args.stride or max(1, patch_size // 2)

    pairs = gather_paths(args.image_dir, args.mask_dir)
    if args.max_files:
        pairs = pairs[: args.max_files]

    out_image_dir = Path(args.out_image_dir)
    out_mask_dir = Path(args.out_mask_dir)

    total_saved = 0
    total_pairs = len(pairs)
    print(f'Found {total_pairs} image-mask pairs to process. Patch size={patch_size}, stride={stride}')

    for i, (img_p, m_p) in enumerate(pairs, 1):
        saved = process_pair(img_p, m_p, out_image_dir, out_mask_dir, patch_size, stride, bg_value=args.bg_value)
        total_saved += saved
        if i % 50 == 0 or i == total_pairs:
            print(f'Processed {i}/{total_pairs}, saved patches so far: {total_saved}')

    print(f'Done. Total patches saved: {total_saved}')


if __name__ == '__main__':
    main()
