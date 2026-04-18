import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 3D transforms — requires MONAI
try:
    from monai.transforms import (
        Compose,
        EnsureTyped,
        Orientationd,
        SpatialCropd,
        RandFlipd,
        RandAxisFlipd,
        RandAffined,
        NormalizeIntensityd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        ResizeWithPadOrCropd,
    )
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


def get_transforms(img_size: tuple = (512, 512), mean=(0, 0, 0), std=(1.0, 1.0, 1.0), get_size=False):
    """
    Returns the train and validation transforms for image segmentation tasks.
    
    Args:
        img_size (tuple): The minimum size(hxw) to which images will be padded.
    """
    orig_h, orig_w = img_size
    
    # Calculate nearest multiple of 128 >= img_size
    pad_h = ((orig_h + 127) // 128) * 128
    pad_w = ((orig_w + 127) // 128) * 128
    
    if get_size:
        return pad_h, pad_w
        
    # Standard padding that ensures multiples of 128 even if image exceeds target size
    padding = A.PadIfNeeded(
        min_height=pad_h,
        min_width=pad_w,
        border_mode=cv2.BORDER_REFLECT,
        p=1
    )
    
    train_transforms = A.Compose([
        padding,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ], is_check_shapes=False)
    
    val_transforms = A.Compose([
        padding,
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ], is_check_shapes=False)
    
    return train_transforms, val_transforms


def get_transforms_3d(phase):
    """
    Returns MONAI transforms for 3D volumetric medical image segmentation.

    Args:
        phase (str): One of 'train', 'val', or 'test'/'infer'.

    Returns:
        monai.transforms.Compose: The transform pipeline for that phase.
    """
    if not _MONAI_AVAILABLE:
        raise ImportError(
            "MONAI is required for 3D transforms. Install it with: pip install monai"
        )

    if phase == 'train':
        transform = Compose([
            EnsureTyped(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            SpatialCropd(keys=["image", "mask"], roi_center=(70, 120, 120), roi_size=[140, 180, 180]),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            RandAxisFlipd(keys=["image", "mask"], prob=0.5),
            RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(2, 1, 2)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])
    elif phase == 'val':
        transform = Compose([
            EnsureTyped(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            SpatialCropd(keys=["image", "mask"], roi_center=(70, 120, 120), roi_size=[140, 180, 180]),
            ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=[192, 192, 192]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
    else:  # test / inference — mask key not available
        transform = Compose([
            EnsureTyped(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            SpatialCropd(keys="image", roi_center=(70, 120, 120), roi_size=[140, 180, 180]),
            ResizeWithPadOrCropd(keys="image", spatial_size=[192, 192, 192]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

    return transform