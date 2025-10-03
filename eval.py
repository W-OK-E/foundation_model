#!/usr/bin/env python3
import os
import torch
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.network.ElitNet import ElitNet
from metrics.seg_metrics import SegmentationMetrics

def detect_num_classes(mask_dir):
    mask_files = glob(os.path.join(mask_dir, "*.png"))
    num_classes = 0
    for mfile in mask_files:
        mask = cv2.imread(mfile, cv2.IMREAD_UNCHANGED)
        num_classes = max(num_classes, mask.max())
    return num_classes + 1

def evaluate(dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_dir = f"datasets/{dataset_name}/images"
    mask_dir = f"datasets/{dataset_name}/masks"
    ckpt_path = f"checkpoints/{dataset_name}_ELiTNet/best_dice_ckpt.ckpt"
    out_dir = f"reports/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    filenames = sorted(os.listdir(img_dir))

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

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint
    model_state_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and "final" not in k}
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)

    metrics = SegmentationMetrics(num_classes=num_classes,
                                  class_names=[str(i) for i in range(num_classes)],
                                  ignore_index=255)

    with torch.no_grad():
        for fname in tqdm(filenames):
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            img_tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float().to(device) / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

            logits = model(img_tensor)
            metrics.update(logits, mask_tensor)

    results = metrics.compute()

    out_path = os.path.join(out_dir, f"{dataset_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics report saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="Dataset name folder (e.g., IDRiD)")
    args = parser.parse_args()
    evaluate(args.dataset_name)