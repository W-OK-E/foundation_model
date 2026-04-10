import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: tuple = (512, 512),  mean = (0, 0, 0), std = (1.0, 1.0, 1.0), get_size = False):
    """
    Returns the train and validation transforms for image segmentation tasks.
    
    If the image is smaller than img_size, it will be reflect padded to 
    the nearest multiple of 128 >= img_size.
    If it is larger, it will be randomly cropped (during training) to that size.
    """
    target_h, target_w = img_size
    
    # Pad size must be a multiple of 128
    pad_h = int((target_h + 127) // 128 * 128)
    pad_w = int((target_w + 127) // 128 * 128)
    
    if get_size:
        return pad_h, pad_w

    train_transforms = A.Compose([
            # Pad if image is smaller than target
            A.PadIfNeeded(min_height=pad_h, min_width=pad_w, border_mode=cv2.BORDER_REFLECT, p=1),
            # Random crop if image is larger than target
            A.RandomCrop(height=pad_h, width=pad_w, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ], is_check_shapes=False)
    
    val_transforms = A.Compose([
            # Validation usually uses padding to mult of 128 without cropping to retain full resolution
            A.PadIfNeeded(min_height=pad_h, min_width=pad_w, border_mode=cv2.BORDER_REFLECT, p=1),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ], is_check_shapes=False)
    
    return train_transforms, val_transforms