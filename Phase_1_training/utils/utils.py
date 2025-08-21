import os
import cv2
import yaml
from torchvision import transforms

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

# Example usage:
# pad_transform = get_reflect_pad_transform("/path/to/train/images", "/path/to/config.yaml")
# padded_img = pad_transform(torchvision_image_tensor)
