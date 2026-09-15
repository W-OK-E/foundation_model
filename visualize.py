#!/usr/bin/env python3
import os
import random
import cv2
import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate


def get_transforms(img_size: tuple = (512, 512)):
    """
    Returns the train and validation transforms for image segmentation tasks.
    
    Args:
        img_size (tuple): The size(hxw) to which images will be padded, must be a multiple of 128.
    """
    orig_h,orig_w = img_size
    pad_h = (orig_h / 128)
    pad_w = (orig_w / 128)
    if pad_h != int(pad_h):
        pad_h = (int(pad_h) + 1) * 128
    else:
        pad_h = orig_h
    if pad_w != int(pad_w):
        pad_w = (int(pad_w) + 1) * 128
    else:
        pad_w = orig_w
    
    
    tf = A.Compose([
            A.PadIfNeeded(min_height=pad_h, min_width=pad_w, border_mode=cv2.BORDER_REFLECT, p=1),
            A.Normalize(
            mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
            ),
            ToTensorV2()
        ], is_check_shapes=False)
    
    return tf

def detect_num_classes(mask_dir):
    mask_files = glob(os.path.join(mask_dir, "*.png"))
    num_classes = 0
    for mfile in mask_files:
        mask = cv2.imread(mfile, cv2.IMREAD_UNCHANGED)
        num_classes = max(num_classes, mask.max())
    return num_classes + 1 

def visualize()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_name = args.dataset_name
    img_path =  Path(args.img_path)

    mask_dir = f"datasets/{dataset_name}/masks"
    mask_file = os.path.join(mask_dir,img_path.stem)

    im = cv2.imread(img_path)
    mask = cv2.imread(mask_file)

    transforms = get_transforms(im.shape[:2])
    im,mask = transforms(image = im,mask = mask)

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
    parser.add_argument("--img_path",required=True,help="Path to the input Image")
    args = parser.parse_args()
    main(args)
