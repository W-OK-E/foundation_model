"""
Run a trained model on a list of images using a saved run folder (that contains
`config.yaml` and checkpoints). Applies the same transforms used in training
and saves two outputs per input image: `pred` and `gt` (if GT provided).

Usage:
python visualize.py \
  --run-dir /path/to/run_folder \
  --images-file /path/to/images.txt \
  --out-dir /path/to/out_dir \
  [--gt-dir /path/to/gts] \
  [--device cuda]

`images-file` should be a text file with one image path per line (absolute or relative).
If `--gt-dir` is provided, the script will look for a mask with the same base name
inside that directory.

The script expects the saved `config.yaml` inside `--run-dir` (this is the file
that `train.py` copies into the run directory).
"""

import argparse
import os
from pathlib import Path
import sys
sys.path.append('/mnt/data/omkumar/foundation_phase1')
import json
import tqdm

from omegaconf import OmegaConf
import torch
import numpy as np
from PIL import Image
import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Import project internals
sys.path.insert(0, os.getcwd())
from models.module import ElitLightModel
from data.transforms import get_transforms


def find_checkpoint(run_dir: Path):
    # Prefer last.ckpt then best_dice_ckpt.ckpt then any .ckpt
    candidates = [run_dir / 'last.ckpt', run_dir / 'best_dice_ckpt.ckpt']
    for c in candidates:
        if c.exists():
            return str(c)
    # any ckpt
    for c in run_dir.glob('*.ckpt'):
        return str(c)
    return None


def load_model_from_run(cfg, run_dir: Path, device='cpu'):
    model = ElitLightModel(cfg.model)
    ckpt_path = find_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f'No checkpoint found in {run_dir}')
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        # assume checkpoint is a raw state dict
        state = ckpt
    # adapt keys if they are prefixed (common when using Lightning)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try stripping 'model.' prefix
        new_state = {}
        for k, v in state.items():
            nk = k
            if k.startswith('model.'):
                nk = k[len('model.'):]
            new_state[nk] = v
        model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model


def apply_transforms_and_predict(model, img_path: Path, gt_path: Path, cfg, device='cpu'):
    # Load images (unnormalized original)
    img = iio.imread(str(img_path))
    if gt_path is not None and gt_path.exists():
        gt = iio.imread(str(gt_path))
    else:
        # create an empty mask with same spatial dims if GT is not present
        gt = np.zeros(img.shape[:2], dtype=np.uint8)

    # Obtain transforms like in train._viz_from_split
    tf, _ = get_transforms(img.shape[:2])
    transformer = tf(image=img, mask=gt)
    img_t = transformer['image']
    gt_t = transformer['mask']

    # Add batch and send to device
    with torch.no_grad():
        inp = img_t.unsqueeze(0).float().to(device)
        out = model.model(inp)
        # out -> logits
        if out.dim() == 4 and out.size(1) > 1:
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            # binary case
            out_sig = torch.sigmoid(out)
            pred = (out_sig.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)

    # gt_t may be torch tensor
    if hasattr(gt_t, 'cpu'):
        gt_np = gt_t.cpu().numpy().astype(np.uint8)
    else:
        gt_np = np.asarray(gt_t).astype(np.uint8)

    return img, pred, gt_np


def main():
    parser = argparse.ArgumentParser(description='Run model on a list of images from a run folder')
    parser.add_argument('--run-dir', required=True, help='Run folder that contains config.yaml and checkpoints')
    parser.add_argument('--data-dir', required=True, help="The folder containing the dataset")
    parser.add_argument('--images-file', required=True, help='Text file with one image path per line')
    parser.add_argument('--out-dir', required=True, help='Directory to save outputs (pred and gt)')
    parser.add_argument('--gt-dir', default=None, help='Optional GT masks directory (same basenames expected)')
    parser.add_argument('--device', default='cuda', help='Device to run model on (cpu or cuda)')
    parser.add_argument('--save-original', action='store_true', help='Also save original image for reference')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    cfg_path = run_dir / 'config.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Config not found at {cfg_path}')

    cfg = OmegaConf.load(str(cfg_path))

    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    print(f'Loading model from run {run_dir} onto device {device}')
    model = load_model_from_run(cfg, run_dir, device=device)

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / 'pred'
    gt_out_dir = out_dir / 'gt'
    orig_dir = out_dir / 'orig'
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_original:
        orig_dir.mkdir(parents=True, exist_ok=True)

    # Read image list
    img_list = []
    with open(args.images_file, 'r') as f:
        img_count = 0
        for line in f:
            img_count += 1  
            p = line.strip()
            if not p:
                continue
            img_list.append(Path(os.path.join(data_dir,"images",p)))
            if(img_count == 10):
                break

    print(f'Found {len(img_list)} images to process')

    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    for i, img_path in enumerate(img_list, 1):
        if not img_path.exists():
            print(f'Warning: image not found {img_path}, skipping')
            continue
        gt_path = None
        if gt_dir is not None:
            candidate = gt_dir / img_path.name
            if candidate.exists():
                gt_path = candidate
        img_unnorm, pred, gt_np = apply_transforms_and_predict(model, img_path, gt_path, cfg, device=device)

        stem = img_path.stem
        pred_path = pred_dir / f"{stem}_pred.png"
        gt_path_out = gt_out_dir / f"{stem}_gt.png"
        orig_out_path = orig_dir / f"{stem}_orig.png"

        # Save unnormalized original image
        orig_dir.mkdir(parents=True, exist_ok=True)
        try:
            # ensure uint8
            img_to_save = img_unnorm
            if img_to_save.dtype != np.uint8:
                img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
            iio.imwrite(str(orig_out_path), img_to_save)
        except Exception:
            # fallback to PIL
            Image.fromarray(img_unnorm).save(str(orig_out_path))

        # Build colormap / norm from config
        try:
            class_names = list(cfg.dataset.class_names)
        except Exception:
            class_names = None
        if class_names is not None:
            n_classes = len(class_names)
        else:
            try:
                n_classes = int(cfg.dataset.num_classes)
            except Exception:
                n_classes = int(model.cfg.model.network.instance.num_classes)

        # Colors fallback (same as train.py COLORS)
        COLORS = [
            "black", "red", "green", "blue", "yellow", "magenta", "cyan",
            "orange", "purple", "brown", "pink", "lime", "teal", "navy",
            "maroon", "olive", "coral", "gold", "turquoise", "violet"
        ]
        cmap = ListedColormap(COLORS[:max(1, n_classes)])
        norm = BoundaryNorm(np.arange(max(1, n_classes) + 1) - 0.5, max(1, n_classes))

        # Save prediction using matplotlib to preserve colormap
        fig = plt.figure(frameon=False)
        plt.axis('off')
        plt.imshow(pred, cmap=cmap, norm=norm)
        plt.savefig(str(pred_path), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save GT using the same cmap/norm
        fig = plt.figure(frameon=False)
        plt.axis('off')
        plt.imshow(gt_np, cmap=cmap, norm=norm)
        plt.savefig(str(gt_path_out), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        if i % 50 == 0 or i == len(img_list):
            print(f'Processed {i}/{len(img_list)}')

    print('Done. Outputs saved under', out_dir)


if __name__ == '__main__':
    main()
