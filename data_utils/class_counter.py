import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import imageio.v3 as iio

def count_segmentation_classes(mask_dir, exts=(".png", ".jpg", ".bmp",".tif")):
    """
    Reads segmentation mask images in a directory and determines the number of unique classes.
    
    Args:
        mask_dir (str): Path to the directory containing segmentation masks.
        exts (tuple): Allowed file extensions.
        
    Returns:
        unique_classes (set): Set of unique class IDs across all masks.
    """
    unique_classes = set()

    for fname in tqdm(os.listdir(mask_dir)):
        if fname.lower().endswith(exts):
            path = os.path.join(mask_dir, fname)
            # mask = np.array(Image.open(path))
            mask = iio.imread(path)
            unique_classes.update(np.unique(mask))

    return unique_classes

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required = True,help = "The folder containing the image files")

if __name__ == "__main__":
    args = parser.parse_args()

    mask_folder = args.dir   # <= change this to your folder
    classes = count_segmentation_classes(mask_folder)

    print(f"Unique class IDs found: {sorted(classes)}")
    print(f"Number of classes: {len(classes)}")
