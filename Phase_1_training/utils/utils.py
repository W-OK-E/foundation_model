import os
import cv2
import yaml
import torch
from torchvision import transforms
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def predictions_to_classes(predictions):
    """
    Convert model output to predicted class indices
    predictions: (B, num_classes, H, W)
    returns: (B, H, W) with predicted class indices
    """
    return torch.argmax(predictions, dim=1)

def get_reflect_pad_transform(base_dir, config_path):
    """
    Reads a sample image and returns a torchvision.transforms.Pad object
    with padding_mode='reflect' to resize image to target shape from config.
    """

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg["dataset"]["image_size"]  # e.g., (2944, 4352)

    # Pick a sample image
    img_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))]
    if not img_files:
        raise FileNotFoundError(f"No image files found in {base_dir}")
    sample_path = os.path.join(base_dir, img_files[0])

    # Read image
    img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read {sample_path}")
    
    orig_h, orig_w = img.shape[:2]

    # Calculate padding amounts
    pad_h_total = target_h - orig_h
    pad_w_total = target_w - orig_w
    if pad_h_total < 0 or pad_w_total < 0:
        raise ValueError(f"Target shape {target_h}x{target_w} is smaller than image shape {orig_h}x{orig_w}")

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    # Return transform
    return transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), padding_mode='reflect')



def count_classes_in_segmentation_masks(folder_path, image_extensions=(".png", ".jpg", ".jpeg", ".tif")):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    unique_values = set()

    image_files = [f for f in folder.glob("*") if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} image files in {folder_path}")

    for image_path in tqdm(image_files, desc="Processing masks"):
        # Load image as grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"⚠️ Warning: Could not read image {image_path}")
            continue

        # Flatten and get unique pixel values
        unique = np.unique(image)
        unique_values.update(unique.tolist())

    sorted_classes = sorted(unique_values)
    print(f"\n✅ Unique pixel values found across all masks: {sorted_classes}")
    print(f"🧪 Number of classes: {len(sorted_classes)}")

    return sorted_classes


def show_image_pairs(pairs, titles=('Left', 'Right'), cmap=None):
    """
    Display a list of image pairs side by side.
    
    Parameters:
    - pairs: List of tuples [(img1, img2), ...]
    - titles: Titles for left and right images
    - cmap: Matplotlib colormap or None
    """
    n = len(pairs)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(8, 4 * n))

    if n == 1:
        axs = np.array([axs])  # Ensure axs is 2D

    for i, (left_img, right_img) in enumerate(pairs):
        axs[i, 0].imshow(left_img, cmap=cmap)
        axs[i, 0].set_title(titles[0])
        axs[i, 0].axis('off')

        axs[i, 1].imshow(right_img, cmap=cmap)
        axs[i, 1].set_title(titles[1])
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_dice_on_each_class(gt,pred,num_classes = 6):
    dice_score = DiceScore(num_classes = 2,average = "micro")
    jack_index = JaccardIndex(task = 'multiclass',num_classes=6,average = "micro")

    #Since this function is to calculate the metrics for each class, we have to make 
    #sure that the predcition mask does indeed have multiple classes.
    #So the prediction we are recieving is gonna be B x C x H x W
    assert(len(pred.shape) == 4)
    assert(pred.shape[1] == num_classes,"The number of classes specified donot match with data.")

    pred_processed = predictions_to_classes(pred)
    # if(len(gt.shape) > 3):
    #     print(gt.shape)
    #     gt.squeeze(1)
    
    print(gt.shape,pred_processed.shape)
    dice_scores = []
    jack_scores = []
    for i in range(num_classes):
        class_mask = np.zeros_like(gt.cpu().numpy())
        class_mask[gt.cpu().numpy() == i] = 1
        pred_mask = np.zeros_like(pred_processed.cpu().numpy())
        pred_mask[pred_processed.cpu().numpy() == i] =1
        # print(class_mask.shape,pred_mask.shape)
        cls_dice_score = dice_score(torch.tensor(class_mask),torch.tensor(pred_mask))
        cls_jac_score = jack_index(torch.tensor(class_mask),torch.tensor(pred_mask))

        # print("Appending one class index")
        jack_scores.append(cls_jac_score)
        dice_scores.append(cls_dice_score)

    return dice_scores,jack_scores
# Example usage:
# pad_transform = get_reflect_pad_transform("/path/to/train/images", "/path/to/config.yaml")
# padded_img = pad_transform(torchvision_image_tensor)
