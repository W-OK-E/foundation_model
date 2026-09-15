"""
Compare predictions from multiple run folders side-by-side.

Given multiple folders (each produced by `run_on_list.py`), each containing
subfolders `orig/`, `gt/`, and `pred/`, this script builds a grid of images
with columns: Input | GT | pred_from_folder1 | pred_from_folder2 | ...
and rows corresponding to different sample stems.

Usage:
python compare_preds_grid.py \
  --folders /path/runA /path/runB /path/runC \
  --samples-file samples.txt \
  --out /path/to/compare.png --max-samples 10

If `--samples-file` is omitted, the script will take the intersection of
stems present in the first folder's `orig/` (or `pred/`) directory and
use up to `--max-samples` of them (sorted).

"""

import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def gather_stems_from_folder(folder: Path, orig_sub='orig', pred_sub='pred'):
    orig_dir = folder / orig_sub
    pred_dir = folder / pred_sub
    stems = set()
    if orig_dir.exists():
        for p in orig_dir.iterdir():
            if p.is_file():
                stems.add(p.stem.rsplit('_orig', 1)[0] if p.stem.endswith('_orig') else p.stem)
    elif pred_dir.exists():
        for p in pred_dir.iterdir():
            if p.is_file():
                stems.add(p.stem.rsplit('_pred', 1)[0] if p.stem.endswith('_pred') else p.stem)
    return stems


def read_image_for_display(path: Path):
    try:
        arr = iio.imread(str(path))
        # convert grayscale to 2D
        if arr.ndim == 3 and arr.shape[2] == 4:
            # drop alpha
            arr = arr[..., :3]
        return arr
    except Exception:
        return None


def build_colormap(n_classes):
    COLORS = [
        "black", "red", "green", "blue", "yellow", "magenta", "cyan",
        "orange", "purple", "brown", "pink", "lime", "teal", "navy",
        "maroon", "olive", "coral", "gold", "turquoise", "violet"
    ]
    cmap = ListedColormap(COLORS[:max(1, n_classes)])
    norm = BoundaryNorm(np.arange(max(1, n_classes) + 1) - 0.5, max(1, n_classes))
    return cmap, norm


def main():
    parser = argparse.ArgumentParser(description='Compare prediction folders in a grid')
    parser.add_argument('--folders', nargs='+', required=True,
                        help='Paths to run output folders (each should contain orig/, gt/, pred/)')
    parser.add_argument('--samples-file', default=None,
                        help='Optional: text file with one stem per line to plot')
    parser.add_argument('--out', required=True, help='Output image path (PNG)')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--orig-sub', default='orig', help='Subfolder name for originals')
    parser.add_argument('--gt-sub', default='gt', help='Subfolder name for GT masks')
    parser.add_argument('--pred-sub', default='pred', help='Subfolder name for predictions')
    parser.add_argument('--ncols', type=int, default=None, help='Optional: force number of columns')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    folders = [Path(f) for f in args.folders]
    for f in folders:
        if not f.exists():
            raise FileNotFoundError(f'Folder not found: {f}')

    # Get sample stems
    if args.samples_file:
        with open(args.samples_file, 'r') as fh:
            stems = [ln.strip() for ln in fh if ln.strip()]
    else:
        # take intersection of stems across all folders (using first folder as base)
        base_stems = gather_stems_from_folder(folders[0], orig_sub=args.orig_sub, pred_sub=args.pred_sub)
        common = base_stems
        for f in folders[1:]:
            stems_f = gather_stems_from_folder(f, orig_sub=args.orig_sub, pred_sub=args.pred_sub)
            common = common.intersection(stems_f)
        stems = sorted(common)

    if not stems:
        raise RuntimeError('No sample stems found')

    stems = stems[: args.max_samples]

    n_rows = len(stems)
    n_pred_folders = len(folders)
    n_cols = 2 + n_pred_folders  # Input, GT, preds...
    if args.ncols:
        n_cols = args.ncols

    # try to infer number of classes from first folder's gt if available
    # default to 3 classes to create a cmap
    n_classes = 3
    first_gt = (folders[0] / args.gt_sub)
    if first_gt.exists():
        for p in first_gt.iterdir():
            if p.is_file():
                # try to read unique values from mask
                try:
                    m = iio.imread(str(p))
                    vals = np.unique(m)
                    if vals.size > 1:
                        n_classes = int(vals.max()) + 1
                        break
                except Exception:
                    continue

    cmap, norm = build_colormap(n_classes)

    # Create figure
    fig_w = n_cols * 4
    fig_h = n_rows * 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for r, stem in enumerate(stems):
        # Input
        ax = axes[r, 0]
        # try orig/<stem>_orig.* else orig/<stem>.* else images/...
        orig_candidates = list((folders[0] / args.orig_sub).glob(f'{stem}*'))
        orig_img = None
        for c in orig_candidates:
            if c.exists():
                orig_img = read_image_for_display(c)
                if orig_img is not None:
                    break
        if orig_img is None:
            # fallback: try pred or gt from first folder
            cand = (folders[0] / args.pred_sub / f'{stem}_pred.png')
            if cand.exists():
                orig_img = read_image_for_display(cand)
        if orig_img is None:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
            ax.axis('off')
        else:
            ax.imshow(orig_img)
            ax.axis('off')
        if r == 0:
            ax.set_title('Input')

        # GT
        ax = axes[r, 1]
        gt_img = None
        gt_candidate = (folders[0] / args.gt_sub / f'{stem}_gt.png')
        if gt_candidate.exists():
            gt_img = read_image_for_display(gt_candidate)
        else:
            # try any file with stem in gt dirs
            for f in folders:
                cand = (f / args.gt_sub).glob(f'{stem}*')
                for c in cand:
                    if c.exists():
                        gt_img = read_image_for_display(c)
                        break
                if gt_img is not None:
                    break
        if gt_img is None:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
            ax.axis('off')
        else:
            ax.imshow(gt_img, cmap=cmap, norm=norm)
            ax.axis('off')
        if r == 0:
            ax.set_title('GT')

        # Predictions from each folder
        for ci, folder in enumerate(folders, start=2):
            ax = axes[r, ci]
            pred_img = None
            pred_candidate = (folder / args.pred_sub / f'{stem}_pred.png')
            if pred_candidate.exists():
                pred_img = read_image_for_display(pred_candidate)
            else:
                # try any file matching stem
                for p in (folder / args.pred_sub).glob(f'{stem}*'):
                    if p.exists():
                        pred_img = read_image_for_display(p)
                        break
            if pred_img is None:
                ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
                ax.axis('off')
            else:
                ax.imshow(pred_img, cmap=cmap, norm=norm)
                ax.axis('off')
            if r == 0:
                ax.set_title(f'Pred: {folder.name}')

    plt.tight_layout()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outp), dpi=args.dpi)
    plt.close(fig)
    print(f'Saved comparison grid to {outp}')


if __name__ == '__main__':
    main()
