import os
import random
import argparse
import json
import numpy as np
import imageio.v3 as iio


def split_dataset(
    directory,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    output_dir=".",
    viz_ratio=0.05
):
    images_dir = os.path.join(directory, "images")
    masks_dir = os.path.join(directory, "masks")

    img_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
    ])

    mask_files = sorted([
        f for f in os.listdir(masks_dir)
        if os.path.isfile(os.path.join(masks_dir, f))
    ])

    assert len(img_files) == len(mask_files), "Images and masks count mismatch!"

    if not img_files:
        raise RuntimeError("No images found.")

    random.seed(42)
    random.shuffle(img_files)

    total = len(img_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        "train": img_files[:train_end],
        "val": img_files[train_end:val_end],
        "test": img_files[val_end:]
    }

    os.makedirs(output_dir, exist_ok=True)

    for split, files in splits.items():
        with open(os.path.join(output_dir, f"{split}.txt"), "w") as f:
            f.write("\n".join(files))

    print(f"✔ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    viz_size = max(1, int(viz_ratio * total))
    viz_files = random.sample(img_files, viz_size)

    with open(os.path.join(output_dir, "viz.txt"), "w") as f:
        f.write("\n".join(viz_files))

    print(f"✔ Viz samples: {len(viz_files)}")

    first_image_path = os.path.join(images_dir, img_files[0])
    image = iio.imread(first_image_path)

    with open(os.path.join(output_dir, "image_shape.txt"), "w") as f:
        f.write(f"{img_files[0]}: {image.shape}\n")

    print(f"✔ Image shape saved: {image.shape}")

    return splits


def create_split_json(folder_path, output_file="split.json"):
    splits = {}

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".txt"):
            continue

        split_name = os.path.splitext(file_name)[0]
        if split_name in ["viz", "image_shape"]:
            continue

        with open(os.path.join(folder_path, file_name), "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        splits[split_name] = lines

        if split_name == "test":
            splits["viz"] = lines[:8]

    output_path = os.path.join(folder_path, output_file)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"split.json created at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset and generate split.json")
    parser.add_argument("--dir", type=str, required=True, help="Dataset directory containing images/ and masks/")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")

    args = parser.parse_args()

    split_dataset(args.dir, output_dir=args.output_dir)
    create_split_json(args.output_dir)