import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    
    #Alright so it must pad it to be of size that is a multiple of 128, and then other optional transformations
    #can be applied.
    train_transforms = A.Compose([
            A.PadIfNeeded(min_height=pad_h, min_width=pad_w, border_mode = cv2.BORDER_REFLECT, p=1),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.Normalize(
            mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
            ),
            ToTensorV2()
        ], is_check_shapes=False)
    
    val_transforms = A.Compose([
            A.PadIfNeeded(min_height=pad_h, min_width=pad_w, border_mode=cv2.BORDER_REFLECT, p=1),
            A.Normalize(
            mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
            ),
            ToTensorV2()
        ], is_check_shapes=False)
    
    return train_transforms, val_transforms