#!/usr/bin/env python3
import os
import random
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
import hydra

def detect_num_classes(mask_dir):
    mask_files = glob(os.path.join(mask_dir, "*.png"))
    num_classes = 0
    for mfile in mask_files:
        mask = cv2.imread(mfile, cv2.IMREAD_UNCHANGED)
        num_classes = max(num_classes, mask.max())
    return num_classes + 1 

def random_colors(num_classes):
    colors = []
    for _ in range(num_classes):
        colors.append([random.randint(0, 255) for _ in range(3)])
    return colors

def visualize(model, dataset_name, out_dir, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    viz_file = f"datasets/{dataset_name}/viz.txt"
    img_dir = f"datasets/{dataset_name}/images"
    mask_dir = f"datasets/{dataset_name}/masks"

    num_classes = detect_num_classes(mask_dir)
    colors = random_colors(num_classes)

    with open(viz_file, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    for fname in tqdm(filenames):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        overlay = np.zeros_like(img)
        for c in range(num_classes):
            overlay[mask == c] = colors[c]

        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        save_path = os.path.join(out_dir, fname)
        cv2.imwrite(save_path, blended)

def main(dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_dir = f"datasets/{dataset_name}/masks"
    num_classes = int(detect_num_classes(mask_dir))

    cfg_dict = {
        "instance": {
            "_target_": "models.network.ElitNet.ElitNet",
            "in_channels": 3,
            "num_classes": num_classes,
            "layers": [4, 8, 16],
            "kernel_sz": 3,
            "up_mode": "up_conv",
            "conv_bridge": True,
            "shortcut": True,
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    model = instantiate(cfg.instance)
    model.to(device)
    model.eval()

    ckpt_path = f"checkpoints/{dataset_name}_ELiTNet/best_dice_ckpt.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if "final" not in k}
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)

    out_dir = f"visuals/{dataset_name}"
    visualize(model, dataset_name, out_dir, device)
    print(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="Dataset name (IDRiD, etc.)")
    args = parser.parse_args()
    main(args.dataset_name)
