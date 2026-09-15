# Errors

- The tensor is 5D (B, C, D, H, W) because 3D data was being used but the code was trying to permute it like a 4D tensor. Basically, loss function is written for 2D but it is being fed 3D data.
    - Fix: (focal_weighted_dice_multi_class.py) Made the loss dimension-aware by handling both 4D and 5D tensors and adjusted *permute* accordingly.
- Feature maps mismatch (48 vs 49) i.e. encoder-decoder spatial sizes are not perfectly aligned due to pooling + upsampling. Seen in operations like add, multiply, concat.
    - Fix: (blocksv2_3d.py) Explicitly aligned shapes using interpolation before combining tensors
- Model is changing its spatial resolution due to pooling, upsampling and padding inconsistencies. Even after fixing internal alignment, output size is not equal to input label size.
    - Fix: (models/module.py) Resized prediction to match target before computing loss.