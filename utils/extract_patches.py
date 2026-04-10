import os
import argparse
from PIL import Image
import random


def extract_random_patches(
    folder1, folder2, output1, output2, h, w, num_patches, seed=None
):
    if seed is not None:
        random.seed(seed)

    os.makedirs(output1, exist_ok=True)
    os.makedirs(output2, exist_ok=True)

    files = sorted(os.listdir(folder1))
    common_files = [f for f in files if os.path.isfile(os.path.join(folder2, f))]
    print(f"Found {len(common_files)} matching image pairs")

    count = 0
    for fname in common_files:
        img1 = Image.open(os.path.join(folder1, fname)).convert("RGB")
        img2 = Image.open(os.path.join(folder2, fname)).convert("RGB")

        img1_w, img1_h = img1.size
        img2_w, img2_h = img2.size

        max_x1 = max(0, img1_w - w)
        max_y1 = max(0, img1_h - h)
        max_x2 = max(0, img2_w - w)
        max_y2 = max(0, img2_h - h)

        for _ in range(max(1, num_patches // len(common_files) + 1)):
            if max_x1 == 0 and max_y1 == 0 and max_x2 == 0 and max_y2 == 0:
                patch1 = img1
                patch2 = img2
                x1, y1, x2, y2 = 0, 0, 0, 0
            else:
                x1 = random.randint(0, max_x1) if max_x1 > 0 else 0
                y1 = random.randint(0, max_y1) if max_y1 > 0 else 0
                x2 = random.randint(0, max_x2) if max_x2 > 0 else 0
                y2 = random.randint(0, max_y2) if max_y2 > 0 else 0

                patch1 = img1.crop((x1, y1, x1 + w, y1 + h))
                patch2 = img2.crop((x2, y2, x2 + w, y2 + h))

            base = os.path.splitext(fname)[0]
            patch1.save(os.path.join(output1, f"{base}_{count:05d}.png"))
            patch2.save(os.path.join(output2, f"{base}_{count:05d}.png"))
            count += 1

            if count >= num_patches:
                print(f"Saved {count} patches")
                return

    print(f"Saved {count} patches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract random patches from image pairs"
    )
    parser.add_argument(
        "--folder1", type=str, required=True, help="Path to first image folder"
    )
    parser.add_argument(
        "--folder2", type=str, required=True, help="Path to second image folder"
    )
    parser.add_argument(
        "--output1", type=str, required=True, help="Path to save patches from folder1"
    )
    parser.add_argument(
        "--output2", type=str, required=True, help="Path to save patches from folder2"
    )
    parser.add_argument("--h", type=int, required=True, help="Patch height")
    parser.add_argument("--w", type=int, required=True, help="Patch width")
    parser.add_argument(
        "--num_patches",
        type=int,
        default=100,
        help="Total number of patches to extract",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    extract_random_patches(
        args.folder1,
        args.folder2,
        args.output1,
        args.output2,
        args.h,
        args.w,
        args.num_patches,
        args.seed,
    )
