import numpy as np
import os
import imageio.v3 as iio
import argparse
from tqdm import tqdm
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--dir",help = "Path to the directory containing Masks")
parser.add_argument("--nclass",type = int,help = "Number of classes")
args = parser.parse_args()

mask_dir = args.dir
n_classes = args.nclass
counts = np.zeros(n_classes)

total_pixels = 0
for fname in tqdm(os.listdir(mask_dir)):
    mask = iio.imread(os.path.join(mask_dir, fname))
    total_pixels += mask.size
    for c in range(n_classes):
        counts[c] += np.sum(mask == c)

freq = counts / total_pixels
median_freq = np.median(freq)
cls_weights = median_freq/freq
print("Frequencies:", freq)
print("Class weights:",cls_weights)
