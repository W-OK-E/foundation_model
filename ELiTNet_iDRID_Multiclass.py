import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import random
import numpy as np
import matplotlib.pyplot as plt

import json
import cv2
from tqdm import tqdm
from PIL import Image


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF 
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

# Custom transform for green channel extraction
class GreenChannelExtraction(ImageOnlyTransform):
    """Extract green channel from RGB image and replicate to 3 channels"""
    def __init__(self, always_apply=False, p=1.0):
        super(GreenChannelExtraction, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        # Extract green channel (index 1)
        green_channel = img[:, :, 1]
        # Replicate to 3 channels
        return np.stack([green_channel, green_channel, green_channel], axis=2)
    
    def get_transform_init_args_names(self):
        return ()

# ============================================================================
# DATASET AND CLASS CONFIGURATION - MODIFY THESE VARIABLES AS NEEDED
# ============================================================================

# Dataset configuration - defined externally
DATASET_PATH = "/home/dipayan/Anupam/dataset/6.IDRiD"
TRAIN_IMAGE_DIR = os.path.join(DATASET_PATH, "training/images")
VAL_IMAGE_DIR = os.path.join(DATASET_PATH, "validation/images")
TRAIN_GT_BASE_DIR = os.path.join(DATASET_PATH, "training/ground_truth")
VAL_GT_BASE_DIR = os.path.join(DATASET_PATH, "validation/ground_truth")

# File format configuration
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Class configuration - defined externally
# Add or remove classes as needed for your specific dataset
# For MULTICLASS segmentation, Background is automatically added as class 0
ALL_CLASS_NAMES = [
    "Background",        # Background is class 0 (automatically generated)
    "Microaneurysms",    # Class 1
    "Haemorrhages",      # Class 2
    "Hard Exudates",     # Class 3
    "Soft Exudates",     # Class 4
    "Optic Disc"         # Class 5 - Keep or remove based on dataset availability
]

# For metrics computation, specify which classes to exclude (typically background)
EXCLUDE_FROM_METRICS = ["Background"]  # Classes to exclude from IoU/F1 computation

NUM_OUTPUT_CHANNELS = len(ALL_CLASS_NAMES)  # Total number of classes including background

# File naming patterns for each class
# Update these patterns based on your dataset's naming convention
FILENAME_PATTERNS = {
    # "Background": "_gt.png",         # Add pattern for Background masks
    "Microaneurysms": "_MA.tif",
    "Haemorrhages": "_HE.tif",
    "Hard Exudates": "_EX.tif",
    "Soft Exudates": "_SE.tif",
    "Optic Disc": "_OD.tif"
}

# ============================================================================


def _list_image_files(directory, extensions):
    """Return sorted list of image filenames filtered by extension."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Image directory not found: {directory}")

    return sorted(
        [
            entry
            for entry in os.listdir(directory)
            if entry.lower().endswith(tuple(ext.lower() for ext in extensions))
            and os.path.isfile(os.path.join(directory, entry))
        ]
    )


class MultiClassSegDataset(Dataset):
    """
    Dataset for multiclass segmentation where each pixel belongs to exactly one class.
    The output mask is a single-channel integer tensor with values [0, num_classes-1].
    Background (class 0) is assigned to pixels that don't belong to any lesion class.
    """
    def __init__(
        self,
        image_dir,
        gt_base_dir,
        image_names_list,
        class_names,
        filename_patterns,
        transform=None,
    ):
        super(MultiClassSegDataset, self).__init__()
        self.image_dir = image_dir
        self.gt_base_dir = gt_base_dir
        self.images = image_names_list
        self.class_names = class_names
        self.filename_patterns = filename_patterns
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Load the fundus image
        image_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image shape to initialize mask
        h, w = image.shape[:2]
        
        # Create a single-channel mask initialized with 0 (background)
        final_mask = np.zeros((h, w), dtype=np.int64)
        
        # Get base filename without extension
        base_name = os.path.splitext(self.images[index])[0]
        
        # Load masks for each class (skip background at index 0)
        # Priority order: later classes override earlier ones if there's overlap
        for class_idx, class_name in enumerate(self.class_names):
            if class_name == "Background":
                continue  # Skip background, it's already 0
                
            class_dir = os.path.join(self.gt_base_dir, class_name)
            
            # Check if the directory exists
            if os.path.exists(class_dir):
                patterns = self.filename_patterns.get(class_name, [])
                if isinstance(patterns, str):
                    patterns = [patterns]

                mask_path = None
                for pattern in patterns:
                    pattern_has_ext_placeholder = "{ext" in pattern
                    if pattern_has_ext_placeholder:
                        for ext in SUPPORTED_MASK_EXTENSIONS:
                            try:
                                candidate = pattern.format(
                                    base=base_name,
                                    image=base_name,
                                    ext=ext.lstrip("."),
                                )
                            except KeyError:
                                continue
                            candidate_path = os.path.join(class_dir, candidate)
                            if os.path.exists(candidate_path):
                                mask_path = candidate_path
                                break
                        if mask_path is not None:
                            break
                    else:
                        formatted = pattern
                        try:
                            formatted = pattern.format(base=base_name, image=base_name)
                        except KeyError:
                            formatted = pattern

                        if formatted == pattern:
                            candidate = f"{base_name}{pattern}"
                        else:
                            candidate = formatted

                        candidate_path = os.path.join(class_dir, candidate)
                        if os.path.exists(candidate_path):
                            mask_path = candidate_path
                            break

                if mask_path and os.path.exists(mask_path):
                    # Load and process mask
                    loaded_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if loaded_mask is not None:
                        # Convert to binary mask
                        if len(loaded_mask.shape) == 2:
                            binary_mask = loaded_mask.astype(np.float32)
                        elif len(loaded_mask.shape) == 3:
                            if loaded_mask.shape[2] == 1:
                                binary_mask = loaded_mask[:, :, 0].astype(np.float32)
                            else:
                                binary_mask = cv2.cvtColor(loaded_mask, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        
                        # Assign class index to pixels where mask is positive
                        binary_mask = (binary_mask > 0).astype(bool)
                        final_mask[binary_mask] = class_idx

        # Apply transformations if specified
        if self.transform:
            # For multiclass, mask is a single-channel image
            transformed = self.transform(image=image, mask=final_mask)
            image = transformed["image"]
            final_mask = transformed["mask"]

        # Convert to tensors
        if not isinstance(final_mask, torch.Tensor):
            final_mask = torch.from_numpy(final_mask).long()
        else:
            final_mask = final_mask.long()

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        return image, final_mask, self.images[index]

def get_dataloaders(
    train_image_dir,
    val_image_dir,
    train_gt_base_dir,
    val_gt_base_dir,
    class_names,
    filename_patterns,
    num_output_channels,
    img_size=2048,
    batch_size=1,
):
    """
    Creates dataloaders for multiclass segmentation dataset.
    
    Args:
        train_image_dir: Path to training images
        val_image_dir: Path to validation images
        train_gt_base_dir: Base path to training ground truth masks
        val_gt_base_dir: Base path to validation ground truth masks
        class_names: List of ALL class names (including Background)
        filename_patterns: Dictionary of filename patterns for each class
        num_output_channels: Number of classes (should equal len(class_names))
        img_size: Image size for resizing
        batch_size: Batch size for dataloaders
    """
    # Get image filenames filtered by supported formats
    train_image_names = _list_image_files(train_image_dir, SUPPORTED_IMAGE_EXTENSIONS)
    val_image_names = _list_image_files(val_image_dir, SUPPORTED_IMAGE_EXTENSIONS)
    
    # For multiclass segmentation, we only need one mask per image
    # No additional_targets needed since mask is single-channel
    
    # Define transforms for multiclass segmentation
    train_transforms = A.Compose([
        # GreenChannelExtraction(p=1.0),  # Extract green channel first
        # A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=128, pad_width_divisor=128, 
        #               border_mode=cv2.BORDER_REFLECT, p=1),
        # A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.CenterCrop(height=3840, width=3840, pad_if_needed=True, border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.Resize(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        # A.PlasmaBrightnessContrast(brightness_range=(-0.5, 0.5),
        #                            contrast_range=(-0.3, 0.3),
        #                            plasma_size=512,    # More detailed pattern
        #                            roughness=0.7,      # Smoother transitions
        #                            p=1.0),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
        A.Normalize(
            mean=(0, 0, 0), 
            std=(1.0, 1.0, 1.0), 
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        # GreenChannelExtraction(p=1.0),  # Extract green channel first
        A.CenterCrop(height=3840, width=3840, pad_if_needed=True, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Resize(height=img_size, width=img_size, p=1.0),
        # A.PlasmaBrightnessContrast(brightness_range=(-0.5, 0.5),
        #                            contrast_range=(-0.3, 0.3),
        #                            plasma_size=512,    # More detailed pattern
        #                            roughness=0.7,      # Smoother transitions
        #                            p=1.0),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
        A.Normalize(
            mean=(0, 0, 0), 
            std=(1.0, 1.0, 1.0), 
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = MultiClassSegDataset(
        train_image_dir, 
        train_gt_base_dir, 
        train_image_names,
        class_names,
        filename_patterns,
        train_transforms,
    )
    
    val_dataset = MultiClassSegDataset(
        val_image_dir, 
        val_gt_base_dir, 
        val_image_names,
        class_names,
        filename_patterns,
        val_transforms,
    )
    
    # Create dataloaders
    g_seed = torch.Generator()
    g_seed.manual_seed(0)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=8
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=8
    )
    
    return train_loader, val_loader
    
#Splitting into train and val sets
random.seed(53)
np.random.seed(53)
torch.manual_seed(53)
'''np.random.shuffle(image_names)
N = len(image_names)
train_len = int(0.9 * N)
val_len = N - train_len'''
#train_image_names = image_names[:train_len]
#val_image_names = image_names[train_len:]
#print(f"No. of training images = {train_len}")
#print(f"No. of validation images = {val_len}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms,datasets
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from focal_loss.focal_loss import FocalLoss
import torchvision
import sys
#from torchsummary import summary
from ptflops import get_model_complexity_info
from torchinfo import summary
from torchstat import stat


import matplotlib.pyplot as plt
import time
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth
from ptflops import get_model_complexity_info
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from ELiTNetV2 import EliTNet2D



# encoder = ConvNextEncoder(in_channels=1, stem_features=12, depths=[3, 3, 9, 3], widths=[12, 24, 48, 96])
# # encoder = ConvNexStage(16, 62, depth=1)
# # encoder = ConvNextStem(16, 96)
# image = torch.rand(256, 1, 2048)
# print(encoder(image).shape)
#decoder = ConvNextDecoder(in_channels=1, stem_features=12, depths=[3, 3, 9, 3], widths=[12, 24, 48, 96])
#latent = encoder(image)
#print(decoder(latent).shape)
# autoencoder = ConvNextForTSPrediction(in_channels=1, stem_features=20, depths=[3, 3, 9, 3], widths=[20, 40, 80, 160])
# print(autoencoder(image).shape)
def train(model, optimizer, loader, epoch, device='cuda'):
    model.train()
    loss_ep = 0
    jaccard_indx = 0
    f1_ep = 0
    
    for idx, (images, masks, _) in enumerate(tqdm(loader, desc=f"EPOCH {epoch}")):
        images = images.to(device)
        masks = masks.to(device).long()  # Shape: [batch_size, H, W] with values [0, num_classes-1]
        
        outputs = model(images)  # Shape: [batch_size, num_classes, H, W]
        
        # Use CrossEntropy loss for multiclass segmentation
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_ep += loss.item()
        
        # Get predicted class for each pixel
        probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        preds = torch.argmax(probs, dim=1)  # Get predicted class indices [batch_size, H, W]
        
        # Convert to one-hot for metrics calculation
        # Exclude specified classes from metrics (e.g., background)
        if len(EXCLUDE_FROM_METRICS) > 0:
            # Find indices of classes to include in metrics
            include_indices = [i for i, name in enumerate(ALL_CLASS_NAMES) if name not in EXCLUDE_FROM_METRICS]
            if len(include_indices) > 0:
                # Create binary masks for each included class
                probs_for_metrics = []
                masks_for_metrics = []
                for class_idx in include_indices:
                    probs_for_metrics.append((preds == class_idx).unsqueeze(1).float())
                    masks_for_metrics.append((masks == class_idx).unsqueeze(1).float())
                
                probs_for_metrics = torch.cat(probs_for_metrics, dim=1)  # [B, num_included_classes, H, W]
                masks_for_metrics = torch.cat(masks_for_metrics, dim=1)
            else:
                # If all classes excluded, use all classes
                probs_for_metrics = F.one_hot(preds, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()
                masks_for_metrics = F.one_hot(masks, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()
        else:
            # Use all classes
            probs_for_metrics = F.one_hot(preds, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()
            masks_for_metrics = F.one_hot(masks, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()

        targets_for_metrics = masks_for_metrics.int()

        tp, fp, fn, tn = smp.metrics.get_stats(
            probs_for_metrics,
            targets_for_metrics,
            mode='multilabel',  # Use multilabel mode for per-class metrics
            threshold=0.5,
        )

        # then compute metrics with required reduction (see metric docs)
        jaccard_indx += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_ep += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
    # Calculate epoch averages
    train_loss = loss_ep / len(loader)
    # train_dice_coeff = dice_coeff_ep / len(loader)
    train_jac_indx = jaccard_indx / len(loader)
    f1 = f1_ep / len(loader)
    
    return train_loss, train_jac_indx, f1

def validate(model, loader, epoch, device='cuda'):
    model.eval()
    loss_ep = 0
    jaccard_indx = 0
    f1_ep = 0
    
    with torch.no_grad():
        for idx, (images, masks, _) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device).long()  # Shape: [batch_size, H, W] with values [0, num_classes-1]
            
            outputs = model(images)  # Shape: [batch_size, num_classes, H, W]
            
            # Use CrossEntropy loss for multiclass segmentation
            loss = criterion(outputs, masks)
            
            loss_ep += loss.item()
            
            # Get predicted class for each pixel
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            preds = torch.argmax(probs, dim=1)  # Get predicted class indices [batch_size, H, W]
            
            # Convert to one-hot for metrics calculation
            # Exclude specified classes from metrics (e.g., background)
            if len(EXCLUDE_FROM_METRICS) > 0:
                # Find indices of classes to include in metrics
                include_indices = [i for i, name in enumerate(ALL_CLASS_NAMES) if name not in EXCLUDE_FROM_METRICS]
                if len(include_indices) > 0:
                    # Create binary masks for each included class
                    probs_for_metrics = []
                    masks_for_metrics = []
                    for class_idx in include_indices:
                        probs_for_metrics.append((preds == class_idx).unsqueeze(1).float())
                        masks_for_metrics.append((masks == class_idx).unsqueeze(1).float())
                    
                    probs_for_metrics = torch.cat(probs_for_metrics, dim=1)  # [B, num_included_classes, H, W]
                    masks_for_metrics = torch.cat(masks_for_metrics, dim=1)
                else:
                    # If all classes excluded, use all classes
                    probs_for_metrics = F.one_hot(preds, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()
                    masks_for_metrics = F.one_hot(masks, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()
            else:
                # Use all classes
                probs_for_metrics = F.one_hot(preds, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()
                masks_for_metrics = F.one_hot(masks, num_classes=len(ALL_CLASS_NAMES)).permute(0, 3, 1, 2).float()

            targets_for_metrics = masks_for_metrics.int()

            tp, fp, fn, tn = smp.metrics.get_stats(
                probs_for_metrics,
                targets_for_metrics,
                mode='multilabel',  # Use multilabel mode for per-class metrics
                threshold=0.5,
            )

            # then compute metrics with required reduction (see metric docs)
            jaccard_indx += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_ep += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
    # Calculate epoch averages
    val_loss = loss_ep / len(loader)
    # val_dice_coeff = dice_coeff_ep / len(loader)
    val_jac_indx = jaccard_indx / len(loader)
    val_f1 = f1_ep / len(loader)
    
    return val_loss, val_jac_indx, val_f1


# a=sum((original_img*generated_img).flatten())
# b=(sum(generated_img.flatten()))+(sum(original_img.flatten()))


# from torchmetrics.functional import jaccard_index as ji
# from torchmetrics.functional import dice
# from torchmetrics.functional import f1_score

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
img_size = 1024
batch_size = 4

# Use the external configuration
train_loader, val_loader = get_dataloaders(
    TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TRAIN_GT_BASE_DIR, VAL_GT_BASE_DIR,
    ALL_CLASS_NAMES, FILENAME_PATTERNS, NUM_OUTPUT_CHANNELS, img_size, batch_size,
)

#model = MeDiAUNET(in_channels = 3, out_channels = 2 ,features = [64, 128, 256, 512])
#model = MeDiAUNET(in_channels = 3, out_channels = 2)
#model = SUMNet_all_bn(in_ch=3,out_ch=2)
model = EliTNet2D(
    in_c=3,
    n_classes=NUM_OUTPUT_CHANNELS,
    layers=[4, 8, 16, 24],
    k_sz=3,
    up_mode='pixelshuffle',
    pool='conv',
    conv_bridge=True,
    shortcut=True,
    skip_conn=True,
    residual=True,
    causal=False,
)
#model = UNet(in_c=3, n_classes=2, layers=[4,8,16,32], conv_bridge=True, shortcut=True)
model.to(device)

# Modified training and validation functions for multiclass segmentation
class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal loss for multiclass segmentation"""
    def __init__(self, weight=None):
        super(DiceFocalLoss, self).__init__()
        # Use 'multiclass' mode instead of 'multilabel'
        self.focal = FocalLoss("multiclass")
        self.dice = DiceLoss("multiclass")

    def forward(self, inputs, targets, smooth=1):
        # Focal and Dice loss for multiclass segmentation
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)        
        # Combine Focal and Dice loss
        return focal_loss + 0.3*dice_loss


class WeightedCrossEntropyDiceLoss(nn.Module):
    """Weighted CrossEntropy + Dice Loss for multiclass segmentation"""
    def __init__(self, class_weights=None, dice_weight=0.5):
        super(WeightedCrossEntropyDiceLoss, self).__init__()
        self.class_weights = class_weights
        self.dice_weight = dice_weight
        
    def cross_entropy_loss(self, inputs, targets):
        """CrossEntropy loss with class weights"""
        # inputs: [B, C, H, W], targets: [B, H, W]
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')
        return ce_loss
    
    def dice_loss(self, inputs, targets, smooth=1e-5):
        """Dice loss with class weights for multiclass"""
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Calculate dice for each class
        dice_scores = []
        for c in range(num_classes):
            input_c = probs[:, c].flatten()
            target_c = targets_one_hot[:, c].flatten()
            
            intersection = (input_c * target_c).sum()
            dice_score = (2. * intersection + smooth) / (input_c.sum() + target_c.sum() + smooth)
            dice_scores.append(dice_score)
        
        # Convert to tensor and apply weights
        dice_scores = torch.stack(dice_scores)
        
        if self.class_weights is not None:
            # Weight the dice scores
            weighted_dice = dice_scores * self.class_weights
            dice_loss = 1 - weighted_dice.mean()
        else:
            dice_loss = 1 - dice_scores.mean()
            
        return dice_loss
    
    def forward(self, inputs, targets):
        ce = self.cross_entropy_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return (1-self.dice_weight) * ce + self.dice_weight * dice
    
from monai.losses import GeneralizedDiceFocalLoss
    
def calculate_class_weights(dataloader, num_classes=None, device='cuda', method='inverse_sqrt'):
    """
    Calculate class weights based on class frequency in the dataset for multiclass segmentation.
    
    Args:
        dataloader: Training data loader
        num_classes: Number of classes (auto-detected if None)
        device: Device for computation
        method: Weighting method
            - 'inverse': 1 / frequency (gives very high weights to rare classes)
            - 'inverse_sqrt': 1 / sqrt(frequency) (more balanced, recommended)
            - 'effective_samples': Effective number of samples method
            - 'none': Equal weights for all classes
    
    Returns:
        class_weights: Tensor of shape [num_classes]
    """
    print(f"Calculating class weights using method: {method}...")
    class_counts = None
    total_pixels = 0
    
    with torch.no_grad():
        for idx, (images, masks, _) in enumerate(tqdm(dataloader, desc="Computing class weights")):
            masks = masks.to(device).long()  # [B, H, W] with values [0, num_classes-1]
            batch_size, h, w = masks.shape

            if class_counts is None:
                if num_classes is None:
                    num_classes = masks.max().item() + 1
                class_counts = torch.zeros(num_classes, device=device)
            
            # Count pixels for each class
            for class_idx in range(num_classes):
                class_counts[class_idx] += (masks == class_idx).sum()
            
            total_pixels += batch_size * h * w
    
    if class_counts is None or num_classes is None:
        raise ValueError("Unable to compute class weights: dataloader returned no samples.")

    # Calculate class frequencies
    eps = 1e-7
    class_frequencies = class_counts / total_pixels
    
    # Calculate weights based on method
    if method == 'none':
        class_weights = torch.ones(num_classes, device=device)
    elif method == 'inverse':
        # Standard inverse frequency weighting
        class_weights = 1.0 / (class_frequencies + eps)
    elif method == 'inverse_sqrt':
        # Sqrt of inverse frequency (less aggressive, more stable)
        class_weights = 1.0 / torch.sqrt(class_frequencies + eps)
    elif method == 'effective_samples':
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + eps)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights so they sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\n" + "="*60)
    print("CLASS WEIGHT ANALYSIS")
    print("="*60)
    print(f"Total pixels analyzed: {total_pixels:,}")
    print(f"\nPer-class statistics:")
    print(f"{'Class':<20} {'Count':<15} {'Frequency':<12} {'Weight':<10}")
    print("-"*60)
    
    class_names = ALL_CLASS_NAMES
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        count = int(class_counts[i].item())
        freq = class_frequencies[i].item()
        weight = class_weights[i].item()
        print(f"{class_name:<20} {count:<15,} {freq:<12.6f} {weight:<10.4f}")
    
    print("="*60)
    print(f"Weight sum: {class_weights.sum().item():.4f} (should be {num_classes})")
    print(f"Weight range: [{class_weights.min().item():.4f}, {class_weights.max().item():.4f}]")
    print("="*60 + "\n")
    
    return class_weights

# Calculate class weights from training data
print("Computing class weights from training data...")
# Options: 'inverse_sqrt' (recommended), 'inverse', 'effective_samples', 'none'
class_weights = calculate_class_weights(
    train_loader, 
    num_classes=NUM_OUTPUT_CHANNELS, 
    device=device,
    method='effective_samples'  # More balanced than 'inverse'
)

# criterion = nn.CrossEntropyLoss()
# Use weighted CrossEntropy + Dice loss for multiclass segmentation
# Move class_weights to CPU for MONAI compatibility if needed
# criterion = GeneralizedDiceFocalLoss(
#     sigmoid=False,  # Use softmax for multiclass
#     to_onehot_y=True,  # Convert targets to one-hot
#     focal_weight=class_weights.cpu() if class_weights is not None else None
# )
# Alternative: Use custom weighted loss
criterion = WeightedCrossEntropyDiceLoss(class_weights=class_weights, dice_weight=0.5)
#criterion = DiceFocalLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.005)  # Reduced from 0.002 for stability
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'max',factor=0.5,patience=20,verbose = True, min_lr = 1e-6)

num_epochs = 400

save_checkpoint = True
checkpoint_freq = 5
load_from_checkpoint = False
load_pretrained = False
checkpoint_dir = "/home/dipayan/Anupam/IDRiD_MC/ELiTNetV2/weights"
save_dir = "/home/dipayan/Anupam/IDRiD_MC/ELiTNetV2/weights"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plot_history = True

model_path = "/home/dipayan/Anupam/IDRiD_MC/ELiTNetV2/weights"

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)

# model = getattr(model, model)
# model = WrappedModel(model)

if load_from_checkpoint:
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_weight.tar"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print('Model Loaded')
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if load_pretrained:
    checkpoint = torch.load(os.path.join(model_path, "best_weight.tar"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print('Model Loaded')
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    

def visualize_predictions(model, dataloader, device, save_path, epoch, num_samples=3):
    """
    Visualize model predictions compared to ground truth for multiclass segmentation
    """
    model.eval()
    # Get classes to visualize (exclude those in EXCLUDE_FROM_METRICS)
    vis_class_names = [name for name in ALL_CLASS_NAMES if name not in EXCLUDE_FROM_METRICS]
    vis_class_indices = [i for i, name in enumerate(ALL_CLASS_NAMES) if name not in EXCLUDE_FROM_METRICS]
    num_vis_classes = len(vis_class_names)
    
    with torch.no_grad():
        for idx, (images, masks, filenames) in enumerate(dataloader):
            if idx >= num_samples:  # Only visualize first few samples
                break
                
            images = images.to(device)
            masks = masks.to(device).long()  # [B, H, W]
            
            # Get predictions
            outputs = model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # [B, H, W]
            
            # Convert to numpy for visualization
            image = images[0].cpu().permute(1, 2, 0).numpy()
            # Denormalize image (assuming normalization with mean=0, std=1)
            image = np.clip(image, 0, 1)
            
            gt_mask = masks[0].cpu().numpy()  # [H, W]
            pred_mask = predictions[0].cpu().numpy()  # [H, W]

            # Create visualization
            n_cols = num_vis_classes + 1  # Original image + visualization classes
            fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))

            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title(f'Original Image\n{filenames[0]}', fontsize=10)
            axes[0, 0].axis('off')

            # Hide unused plots in first row beyond the first cell
            for col in range(1, n_cols):
                axes[0, col].axis('off')

            # Ground truth masks (per class)
            for c, (class_idx, class_name) in enumerate(zip(vis_class_indices, vis_class_names)):
                gt_class_mask = (gt_mask == class_idx).astype(np.float32)
                axes[1, c + 1].imshow(gt_class_mask, cmap='gray')
                axes[1, c + 1].set_title(f'GT: {class_name}', fontsize=10)
                axes[1, c + 1].axis('off')

            # Predicted masks (per class)
            for c, (class_idx, class_name) in enumerate(zip(vis_class_indices, vis_class_names)):
                pred_class_mask = (pred_mask == class_idx).astype(np.float32)
                axes[2, c + 1].imshow(pred_class_mask, cmap='gray')
                axes[2, c + 1].set_title(f'Pred: {class_name}', fontsize=10)
                axes[2, c + 1].axis('off')
            
            # Hide the first column in GT/Pred rows
            axes[1, 0].axis('off')
            axes[2, 0].axis('off')
            
            # Add overall title
            fig.suptitle(f'Epoch {epoch} - Sample {idx+1}', fontsize=16, y=0.98)
            
            plt.tight_layout()
            
            # Save the visualization
            sample_save_path = os.path.join(save_path, f'epoch_{epoch}_sample_{idx+1}.png')
            plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'Saved visualization: {sample_save_path}')

def create_overlay_visualization(model, dataloader, device, save_path, epoch, num_samples=2):
    """
    Create overlay visualization showing predictions on top of original images for multiclass segmentation
    """
    model.eval()
    # Get classes to visualize (exclude those in EXCLUDE_FROM_METRICS)
    vis_class_names = [name for name in ALL_CLASS_NAMES if name not in EXCLUDE_FROM_METRICS]
    vis_class_indices = [i for i, name in enumerate(ALL_CLASS_NAMES) if name not in EXCLUDE_FROM_METRICS]
    num_vis_classes = len(vis_class_names)
    cmap = plt.cm.get_cmap('tab10', num_vis_classes)
    colors = [cmap(i)[:3] for i in range(num_vis_classes)]
    
    with torch.no_grad():
        for idx, (images, masks, filenames) in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device).long()  # [B, H, W]
            
            # Get predictions
            outputs = model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # [B, H, W]
            
            # Convert to numpy
            image = images[0].cpu().permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)
            
            gt_mask = masks[0].cpu().numpy()  # [H, W]
            pred_mask = predictions[0].cpu().numpy()  # [H, W]
            
            # Create overlay images
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=12)
            axes[0, 0].axis('off')
            
            # Ground truth overlay
            gt_overlay = image.copy()
            for c, class_idx in enumerate(vis_class_indices):
                mask = (gt_mask == class_idx)
                for channel in range(3):
                    gt_overlay[:, :, channel][mask] = gt_overlay[:, :, channel][mask] * 0.6 + colors[c][channel] * 0.4
            
            axes[0, 1].imshow(gt_overlay)
            axes[0, 1].set_title('Ground Truth Overlay', fontsize=12)
            axes[0, 1].axis('off')
            
            # Prediction overlay
            pred_overlay = image.copy()
            for c, class_idx in enumerate(vis_class_indices):
                mask = (pred_mask == class_idx)
                for channel in range(3):
                    pred_overlay[:, :, channel][mask] = pred_overlay[:, :, channel][mask] * 0.6 + colors[c][channel] * 0.4
            
            axes[0, 2].imshow(pred_overlay)
            axes[0, 2].set_title('Prediction Overlay', fontsize=12)
            axes[0, 2].axis('off')
            
            # Individual class comparisons - create combined masks
            gt_combined = np.zeros_like(gt_mask, dtype=np.float32)
            pred_combined = np.zeros_like(pred_mask, dtype=np.float32)
            
            for c, class_idx in enumerate(vis_class_indices):
                gt_combined[gt_mask == class_idx] = c + 1
                pred_combined[pred_mask == class_idx] = c + 1
            
            axes[1, 0].imshow(gt_combined, cmap='hot')
            axes[1, 0].set_title('GT: All Classes Combined', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(pred_combined, cmap='hot')
            axes[1, 1].set_title('Pred: All Classes Combined', fontsize=12)
            axes[1, 1].axis('off')
            
            # Difference map
            diff_map = (gt_mask != pred_mask).astype(np.float32)
            axes[1, 2].imshow(diff_map, cmap='Reds')
            axes[1, 2].set_title('Difference Map (Misclassified)', fontsize=12)
            axes[1, 2].axis('off')
            
            # Add legend for colors
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=vis_class_names[i]) for i in range(num_vis_classes)]
            fig.legend(
                handles=legend_elements,
                loc='center',
                bbox_to_anchor=(0.5, 0.02),
                ncol=max(1, min(num_vis_classes, 5)),
            )
            
            plt.suptitle(f'Epoch {epoch} - Overlay Visualization - {filenames[0]}', fontsize=16)
            plt.tight_layout()
            
            # Save the visualization
            overlay_save_path = os.path.join(save_path, f'epoch_{epoch}_overlay_{idx+1}.png')
            plt.savefig(overlay_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'Saved overlay visualization: {overlay_save_path}')

# Add this to your imports at the top
import matplotlib.patches as patches

train_loss_history, val_loss_history, train_dice_coeff_history, dice_coeff_history, train_jcrd_indx_history, jacrd_indx_history, train_f1_history, f1_history = [], [], [], [], [], [], [], []
best_dice_coeff = 0
train_loader_len  = len(train_loader)
val_loader_len = len(val_loader)


visualization_dir = "/home/dipayan/Anupam/IDRiD_MC/ELiTNetV2/visualizations"
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)
    

loop = tqdm(range(1, num_epochs + 1), leave = True)
for epoch in loop:
    train_loss, train_jac_indx, train_f1 = train(model, optimizer, train_loader, epoch, device)
    val_loss, jac_indx, f1 = validate(model, val_loader, epoch, device)
    #if(epoch > 20):
    scheduler.step(jac_indx)
    #scheduler.step()
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    # train_dice_coeff_history.append(train_dice_coeff.cpu())
    # dice_coeff_history.append(dice_coeff.cpu())
    train_jcrd_indx_history.append(train_jac_indx.cpu())
    jacrd_indx_history.append(jac_indx.cpu())
    train_f1_history.append(train_f1.cpu())
    f1_history.append(f1.cpu())
    
    print(f"Train loss = {train_loss} ::train_jac_index = {train_jac_indx} :: train_f1_score = {train_f1} :: Val Loss = {val_loss} :: Jaccard Index = {jac_indx} :: F1 Score = {f1}")
    
    if jac_indx > best_dice_coeff:
        torch.save({
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
        },os.path.join(checkpoint_dir,"best_weight.tar"))
        print('model saved')
        best_dice_coeff = jac_indx
        
        # Generate visualizations when model is saved
        print("Generating visualizations...")
        try:
            # Create visualizations from validation set
            visualize_predictions(model, val_loader, device, visualization_dir, epoch, num_samples=3)
            create_overlay_visualization(model, val_loader, device, visualization_dir, epoch, num_samples=2)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    if save_checkpoint and epoch % checkpoint_freq == 0:
        torch.save({
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "epoch":epoch
        },os.path.join(save_dir,"checkpoint.tar"))
    
    
fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (30, 10))
ax[0].plot(range(1, num_epochs + 1), train_loss_history, label = "Train loss")
ax[0].plot(range(1, num_epochs + 1), val_loss_history, label = "Val loss")
ax[1].plot(range(1, num_epochs + 1), train_f1_history, label = "Train Dice Coefficient" )
ax[1].plot(range(1, num_epochs + 1), f1_history, label = "Dice Coefficient" )
ax[2].plot(range(1, num_epochs + 1), train_jcrd_indx_history, label = "Train Jaccard Index")
ax[2].plot(range(1, num_epochs + 1), jacrd_indx_history, label = "Jaccard Index")
ax[0].legend(fontsize = 20)
ax[1].legend(fontsize = 20)
ax[2].legend(fontsize = 20)
plt.savefig('/home/dipayan/Anupam/IDRiD_MC/ELiTNetV2.png')
plt.show()

# ============================================================================
# USAGE EXAMPLE FOR CUSTOM DATASETS
# ============================================================================
"""
To use this code with a different dataset or different classes, modify the 
configuration variables at the top of the file:

IMPORTANT: This is a MULTICLASS segmentation setup where each pixel belongs 
to exactly ONE class (mutually exclusive). Background is automatically class 0.

1. Update DATASET_PATH to point to your dataset location
2. Modify ALL_CLASS_NAMES list to include "Background" and your specific classes
   - Background should be the first element (class 0)
   - List other classes in order (they will be assigned indices 1, 2, 3, ...)
3. Update FILENAME_PATTERNS dictionary to match your file naming convention
   - No need to specify pattern for Background (automatically assigned)
4. NUM_OUTPUT_CHANNELS will automatically update based on ALL_CLASS_NAMES length

Example for a custom dataset:

DATASET_PATH = "/path/to/your/dataset"
ALL_CLASS_NAMES = [
    "Background",    # Class 0 - automatically assigned to non-lesion pixels
    "Class1",        # Class 1
    "Class2",        # Class 2
    "Class3"         # Class 3
]
FILENAME_PATTERNS = {
    "Class1": "_C1.png",
    "Class2": "_C2.png",
    "Class3": "_C3.png"
}

The rest of the code will automatically adapt to your configuration.

Key differences from multilabel segmentation:
- Each pixel can belong to only ONE class (not multiple)
- Output is a single-channel mask with integer class indices [0, num_classes-1]
- Loss function is CrossEntropy (not BCE)
- Activation is softmax (not sigmoid)
- Background pixels are assigned class 0
"""