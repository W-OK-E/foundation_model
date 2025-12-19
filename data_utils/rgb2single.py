import cv2
import numpy as np
import os
import tqdm

def rgb_mask_to_binary(mask_path, out_path, threshold=128):
    """
    Convert a black/white RGB mask to single-channel 0/1 mask.

    Args:
        mask_path (str): path to RGB mask
        out_path (str): output path (.png recommended)
        threshold (int): grayscale threshold
    """
    # Read image (BGR)
    img = cv2.imread(mask_path)

    if img is None:
        raise FileNotFoundError(mask_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold → {0, 1}
    binary = (gray >= threshold).astype(np.uint8)

    # Save (0/1 values)
    cv2.imwrite(out_path, binary)

# Example
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=".")
args = parser.parse_args()

dir_path  = args.dir
for im in tqdm.tqdm(os.listdir(dir_path)):
    if im.endswith(".png"):
        mask_path = os.path.join(dir_path, im)
        out_path = os.path.join(dir_path, im)
        rgb_mask_to_binary(mask_path, out_path)

