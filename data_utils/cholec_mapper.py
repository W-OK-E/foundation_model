import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- YOUR MAPPING (cleaned to plain ints) ----
RGB_TO_CLASS = {
    (0, 50, 128): 0,
    (111, 74, 0): 1,
    (127, 127, 127): 2,
    (169, 255, 184): 3,
    (170, 255, 0): 4,
    (186, 183, 75): 5,
    (210, 140, 140): 6,
    (231, 70, 156): 7,
    (255, 0, 0): 8,
    (255, 85, 0): 9,
    (255, 114, 114): 10,
    (255, 160, 165): 11,
    (255, 255, 0): 12,
}

# --------------------------------------------

def convert_mask(src_path, dst_path):
    """Convert one RGB mask to single-channel uint8 mask."""
    img = np.array(Image.open(src_path).convert("RGB"))
    h, w, _ = img.shape

    out = np.zeros((h, w), dtype=np.uint8)

    for rgb, cls in RGB_TO_CLASS.items():
        mask = np.all(img == rgb, axis=-1)
        out[mask] = cls

    # Optional: sanity check for unmapped pixels
    if np.any(~np.isin(out, list(RGB_TO_CLASS.values()))):
        pass  # or raise RuntimeError(f"Unmapped colors in {src_path}")

    Image.fromarray(out, mode="L").save(dst_path)


def process_folder(src_dir, dst_dir, num_workers=8):
    os.makedirs(dst_dir, exist_ok=True)

    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for f in files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            futures.append(executor.submit(convert_mask, src, dst))

        for fut in as_completed(futures):
            fut.result()  # propagate exceptions

    print(f"Converted {len(files)} masks → {dst_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert RGB segmentation masks to single-channel class IDs"
    )
    parser.add_argument("src_dir", help="Source directory with RGB masks")
    parser.add_argument("dst_dir", help="Destination directory for converted masks")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    args = parser.parse_args()
    process_folder(args.src_dir, args.dst_dir, num_workers=args.workers)