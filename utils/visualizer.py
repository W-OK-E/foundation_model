import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm

COLORS = [
    "black",        # 0
    "red",          # 1
    "green",        # 2
    "blue",         # 3
    "yellow",       # 4
    "magenta",      # 5
    "cyan",         # 6
    "orange",       # 7
    "purple",       # 8
    "brown",        # 9
    "pink",         # 10
    "lime",         # 11
    "teal",         # 12
    "navy",         # 13
    "maroon",       # 14
    "olive",        # 15
    "coral",        # 16
    "gold",         # 17
    "turquoise",    # 18
    "violet"        # 19
]

def visualize(cfg, orig:torch.Tensor,mask:torch.Tensor,pred_arr:np.ndarray,idx:int,val_step:int = -1):    
    # Create figure
    print(cfg.ckpt_dir_path)
    import ipdb
    ipdb.set_trace()
    if(val_step == -1):
        out_dir = os.path.join(cfg.ckpt_dir_path,"viz", f'val_step_{val_step}')
    else:
        out_dir = os.path.join(cfg.ckpt_dir_path,"viz", "post_training")

    os.makedirs(out_dir,exist_ok=True)
    # Example class names and colors
    class_names = cfg.dataset.class_names
    colors = COLORS[:len(class_names)]

    # Create a discrete colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, len(class_names))

    # Use GridSpec to allocate space: 3 images + 1 for colorbar
    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1,1,1,0.1], wspace=0.3)

    # --- Original ---
    ax0 = fig.add_subplot(gs[0, 0])

    #Obrain Arrays
    orig = orig.permute(1,2,0).cpu().numpy() #Assuming that the orig tensor is a batch of original tensors.
    mask = mask.cpu().numpy()
    
    if orig is not None:
        ax0.imshow(np.array(orig))
    else:
        ax0.text(0.5, 0.5, "Original not found", ha="center")
    ax0.set_title("Original")
    ax0.axis("off")

    # --- Mask ---
    ax1 = fig.add_subplot(gs[0, 1])
    if mask is not None:
        im_mask = ax1.imshow(np.array(mask), cmap=cmap, norm=norm)
    else:
        ax1.text(0.5, 0.5, "Mask not found", ha="center")
    ax1.set_title("Ground Truth Mask")
    ax1.axis("off")

    # --- Prediction ---
    ax2 = fig.add_subplot(gs[0, 2])
    # print("Unique Values in pred_arr",np.unique(pred_arr))
    # print("Cmap looks like:",cmap)
    # import ipdb
    # ipdb.set_trace()
    if pred_arr is not None:
        im_pred = ax2.imshow(pred_arr, cmap=cmap, norm=norm)
    else:
        ax2.text(0.5, 0.5, "Prediction not available", ha="center")
    ax2.set_title("Prediction")
    ax2.axis("off")

    # --- Colorbar in separate axis ---
    ax_cbar = fig.add_subplot(gs[0, 3])
    cb = plt.colorbar(im_mask, cax=ax_cbar, ticks=range(len(class_names)))
    cb.ax.set_yticklabels(class_names)
    cb.set_label("Classes")

    out_path = os.path.join(out_dir, f"viz_{idx:03d}.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)