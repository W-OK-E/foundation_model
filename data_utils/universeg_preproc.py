import pathlib
import torch
import numpy as np
import PIL.Image
from tqdm import tqdm


def process_image(path, size=(128,128)):
    img = PIL.Image.open(path).convert("L")
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    return img


def process_mask(path, size=(128,128), num_classes=2):
    mask = PIL.Image.open(path)
    mask = mask.resize(size, resample=PIL.Image.NEAREST)
    mask = np.array(mask)

    one_hot = []
    for c in range(num_classes):
        one_hot.append((mask == c).astype(np.float32))
    mask = np.stack(one_hot)

    return mask


def preprocess_and_save(input_root, output_root, dataset_name, num_classes, size=(128,128), support_frac=0.7):
    
    input_root = pathlib.Path(input_root)
    img_dir = input_root / "images"
    mask_dir = input_root / "masks"

    samples = []

    for img_path in tqdm(sorted(img_dir.iterdir())):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue

        img = process_image(img_path, size)
        mask = process_mask(mask_path, size, num_classes)

        img = img[None]

        samples.append(
            (torch.from_numpy(img),
             torch.from_numpy(mask))
        )

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(samples))
    split_idx = int(support_frac * len(samples))

    support = [samples[i] for i in perm[:split_idx]]
    test = [samples[i] for i in perm[split_idx:]]

    out_dir = pathlib.Path(output_root) / "datasets" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(support, out_dir / "support.pt")
    torch.save(test, out_dir / "test.pt")

    print(f"Saved dataset to {out_dir}")

def main():
    preprocess_and_save(
    input_root="datasets/INBreast",
    output_root="UniverSeg",
    dataset_name="INBreast",
    num_classes=3
)

if __name__ == '__main__':
    main()