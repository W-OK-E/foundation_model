import os
import json
import random
import argparse
import numpy as np
import imageio.v3 as iio


def split_and_create_json(
    directory,
    output_dir=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    viz_ratio=0.05,
    seed=42,
):
    if output_dir is None:
        output_dir = directory

    images_dir = os.path.join(directory, "images")
    mask_dir = os.path.join(directory, "masks")

    img_files = [
        f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))
    ]
    mask_files = [
        f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))
    ]

    assert len(img_files) == len(mask_files), "Mismatch between images and masks"

    if not img_files:
        print("No files found in the directory.")
        return

    random.seed(seed)
    random.shuffle(img_files)

    total = len(img_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = img_files[:train_end]
    val_files = img_files[train_end:val_end]
    test_files = img_files[val_end:]
    viz_files = random.sample(img_files, int(viz_ratio * total))

    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_files))

    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_files))

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_files))

    with open(os.path.join(output_dir, "viz.txt"), "w") as f:
        f.write("\n".join(viz_files))

    first_image_path = os.path.join(images_dir, img_files[0])
    try:
        if first_image_path.lower().endswith((".nii", ".nii.gz")):
            import nibabel as nib

            image = nib.load(first_image_path)
            shape = image.shape
        elif first_image_path.lower().endswith((".pt", ".pth")):
            image = torch.load(first_image_path)
            shape = tuple(image.shape)
        else:
            image = iio.imread(first_image_path)
            shape = image.shape

        with open(os.path.join(output_dir, "image_shape.txt"), "w") as f:
            f.write(f"{img_files[0]}: {shape}\n")
        print(f"Wrote image shape to image_shape.txt: {shape}")
    except Exception as e:
        print(f"Failed to read {first_image_path}: {e}")

    splits = {}
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".txt"):
            split_name = os.path.splitext(file_name)[0]
            if split_name in ("viz", "image_shape.txt"):
                continue
            with open(os.path.join(output_dir, file_name), "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if split_name == "test":
                splits["viz"] = lines[:8]
            splits[split_name] = lines

    output_path = os.path.join(output_dir, "split.json")
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Saved split.json at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset and create split.json")
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing images/ and masks/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to input dir)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--viz_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_create_json(
        args.dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        viz_ratio=args.viz_ratio,
        seed=args.seed,
    )
