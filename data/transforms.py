import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int = 512):
    """
    Returns the train and validation transforms for image segmentation tasks.
    
    Args:
        img_size (int): The size to which images will be resized. Default is 512.
    """
    train_transforms = A.Compose([
            A.PadIfNeeded(min_height=768, min_width=768, p=1),
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.Rotate(limit = 90, p = 0.5),
            A.Normalize(
            mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
            ),
            ToTensorV2()
        ])
    val_transforms = A.Compose([
            A.PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.Resize(height = img_size, width = img_size),
            A.Normalize(
            mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
            ),
            ToTensorV2()
        ])
    
    return train_transforms, val_transforms