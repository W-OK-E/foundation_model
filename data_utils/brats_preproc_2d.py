import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm


data = "/home/asavari/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output = "/home/asavari/BraTS2020_TrainingData/BraTS_Processed"

img_size = 128
modalities = ["flair", "t1", "t1ce", "t2"]

os.makedirs(output, exist_ok = True)
os.makedirs(os.path.join(output, "images"), exist_ok = True)
os.makedirs(os.path.join(output, "masks"), exist_ok = True)

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize(volume):
    volume = volume.astype(np.float32)
    mean = volume.mean()
    std = volume.std()
    if std == 0:
        return volume
    return (volume - mean) / std

def resize_slice(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def resize_mask(mask, size):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

def remap_labels(mask):
    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3
    return mask

img_count = 0
patients = sorted(os.listdir(data))

for patient in tqdm(patients, desc="Processing patients..."):
    patient_dir = os.path.join(data, patient)

    if not os.path.isdir(patient_dir):
        continue

    volumes = []
    for mod in modalities:
        mod_path = os.path.join(patient_dir, f"{patient}_{mod}.nii")
        vol = load_nifti(mod_path)
        vol = normalize(vol)
        volumes.append(vol)
    
    image = np.stack(volumes, axis=-1)

    mask_path = os.path.join(patient_dir, f"{patient}_seg.nii")
    mask = load_nifti(mask_path).astype(np.uint8)
    mask = remap_labels(mask)

    H, W, D, C = image.shape # transpose in dataset

    for slice_idx in range(D):
        img_slice = image[:, :, slice_idx, :]
        mask_slice = mask[:, :, slice_idx]

        if np.max(mask_slice) == 0:
            continue

        img_resized = np.zeros((img_size, img_size, C), dtype=np.float32)
        for c in range(C):
            img_resized[:, :, c] = resize_slice(img_slice[:, :, c], img_size)

        mask_resized = resize_mask(mask_slice, img_size)

        np.save(
            os.path.join(output, "images", f"img_{img_count}.npy"),
            img_resized
        )

        np.save(
            os.path.join(output, "masks", f"mask_{img_count}.npy"),
            mask_resized
        )

        img_count += 1

print(f"Preprocessing complete. Total slices saved: {img_count}")