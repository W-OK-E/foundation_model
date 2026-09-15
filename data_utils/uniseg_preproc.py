import pathlib
import torch
import numpy as np
import nibabel as nib
import json
import pickle
from tqdm import tqdm
from collections import OrderedDict
import argparse


def make_labels_continuous(mask):
    unique_vals = np.unique(mask)
    mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
    new_mask = np.zeros_like(mask)

    for val, new_val in mapping.items():
        new_mask[mask == val] = new_val

    return new_mask, mapping


def save_nifti(array, path):
    nii = nib.Nifti1Image(array, np.eye(4))
    nib.save(nii, path)


def main(dataset_name):

    input_root = pathlib.Path("datasets") / dataset_name
    output_root = pathlib.Path("UniSeg/datasets") / dataset_name

    images_dir = input_root / "images"
    masks_dir = input_root / "masks"

    imagesTr = output_root / "imagesTr"
    labelsTr = output_root / "labelsTr"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.pt"))
    case_ids = []

    print(f"\nProcessing dataset: {dataset_name}")

    modality_count = None
    label_values = set()

    for img_path in tqdm(image_files):

        case_id = img_path.stem
        mask_path = masks_dir / f"{case_id}.pt"

        if not mask_path.exists():
            continue

        img_tensor = torch.load(img_path, map_location="cpu")
        mask_tensor = torch.load(mask_path, map_location="cpu")

        img = img_tensor.numpy()
        mask = mask_tensor.numpy()

        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]

        if mask.ndim != 3:
            raise ValueError(f"Mask must be (D,H,W), got {mask.shape}")

        if img.ndim != 4:
            raise ValueError(f"Image must be (C,D,H,W), got {img.shape}")

        C = img.shape[0]

        if modality_count is None:
            modality_count = C

        mask, mapping = make_labels_continuous(mask)
        label_values.update(np.unique(mask))

        for c in range(C):
            save_nifti(
                img[c].astype(np.float32),
                imagesTr / f"{case_id}_{c:04d}.nii.gz"
            )

        save_nifti(
            mask.astype(np.int16),
            labelsTr / f"{case_id}.nii.gz"
        )

        case_ids.append(case_id)

    print("Creating dataset.json...")

    dataset_json = OrderedDict()
    dataset_json["name"] = dataset_name
    dataset_json["description"] = dataset_name
    dataset_json["tensorImageSize"] = "4D"
    dataset_json["reference"] = ""
    dataset_json["licence"] = ""
    dataset_json["release"] = "0.0"

    dataset_json["modality"] = {
        str(i): f"Modality_{i}" for i in range(modality_count)
    }

    dataset_json["labels"] = {
        str(int(i)): f"Class_{i}" for i in sorted(label_values)
    }

    dataset_json["numTraining"] = len(case_ids)
    dataset_json["numTest"] = 0

    dataset_json["training"] = [
        {
            "image": f"./imagesTr/{cid}.nii.gz",
            "label": f"./labelsTr/{cid}.nii.gz"
        }
        for cid in case_ids
    ]

    dataset_json["test"] = []

    with open(output_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("Creating splits_final.pkl...")

    rng = np.random.default_rng(42)
    perm = rng.permutation(case_ids)
    split_idx = int(0.8 * len(case_ids))

    split = OrderedDict()
    split["train"] = perm[:split_idx].tolist()
    split["val"] = perm[split_idx:].tolist()

    with open(output_root / "splits_final.pkl", "wb") as f:
        pickle.dump([split], f)

    print(f"\nFinished preprocessing {dataset_name}")
    print(f"Saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()

    main(args.dataset_name)