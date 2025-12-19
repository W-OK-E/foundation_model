import os
import random
import argparse
import numpy as np
import imageio.v3 as iio

def check_num_classes(mask_files):
    num_classes = 0
    for file in mask_files:
        im = iio.imread(file)
        classes = len(np.unique(im))
        if(classes > num_classes):
            num_classes = classes
    return num_classes
    
def split_dataset(
    directory, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15,
    output_dir=".",
    viz_ratio=0.05
):
    images_dir = os.path.join(directory,'images')
    mask_dir = os.path.join(directory,'masks')


    # List all files (excluding directories)
    img_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

    assert(len(img_files) == len(mask_files))

    if not img_files:
        print("No files found in the directory.")
        return

    # Shuffle the files
    random.seed(42)
    random.shuffle(img_files)

    # Compute split indices
    total = len(img_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = img_files[:train_end]
    val_files = img_files[train_end:val_end]
    test_files = img_files[val_end:]

    # Write to respective files
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_files))

    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_files))

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_files))
    
    print(f"Done! Saved {len(train_files)} train, {len(val_files)} val, and {len(test_files)} test files.")

    viz_size = int(viz_ratio * total)

    viz_files = random.sample(img_files, viz_size)

    with open(os.path.join(output_dir, "viz.txt"), "w") as f:
        f.write("\n".join(viz_files))
    
    print(f"Saved {len(viz_files)} files to viz.txt.")

    # Inspect the first image and save its shape
    first_image_path = os.path.join(images_dir, img_files[0])
    try:
        image = iio.imread(first_image_path)
        shape = image.shape
        

        shape_str = f"{img_files[0]}: {image.shape}"
        with open(os.path.join(output_dir, "image_shape.txt"), "w") as f:
            f.write(shape_str + "\n")
        print(f"Wrote shape of {img_files[0]} to image_shape.txt: {image.shape}")
    except Exception as e:
        print(f"Failed to read {first_image_path}: {e}")

# Argument parsing
parser = argparse.ArgumentParser(description="Randomly split dataset into train, val, and test sets.")
parser.add_argument("--dir", type=str, required=True, help="Directory containing the dataset files.")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the split files.")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dir:
        split_dataset(args.dir, output_dir=args.output_dir)
    else:
        print("Please provide a directory using --dir argument.")
