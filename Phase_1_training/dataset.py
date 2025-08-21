import os
import torch
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, name: str, data_dir: str, img_sz: tuple, is_train: bool = True,transform = None,cache_dir="preprocessed",conversion = False):
        self.name = name
        self.data_dir = data_dir
        self.img_sz = img_sz
        self.is_train = is_train

        split = "train" if is_train else "test"
        self.raw_image_dir = os.path.join(data_dir, split, "images")
        self.raw_mask_dir = os.path.join(data_dir, split, "labels")

        self.cache_dir = os.path.join(data_dir, cache_dir, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.im_paths = sorted([
            f for f in os.listdir(self.raw_image_dir) if f.endswith(('.png', '.jpg', '.jpeg','.tif'))
        ])

        self.image_transform = transform
        self.conversion = conversion

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        fname = self.im_paths[index]
        pt_path = os.path.join(self.cache_dir, fname.replace('.jpg', '.pt').replace('.png', '.pt').replace('.tif', '.pt'))

        if os.path.exists(pt_path):
            return torch.load(pt_path)

        img_path = os.path.join(self.raw_image_dir, fname)
        mask_path = os.path.join(self.raw_mask_dir, fname)

        # Load image
        if self.conversion:
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.open(img_path)

        # FIXED: Handle mask properly
        mask_pil = Image.open(mask_path)
        mask_array = self.image_transform(mask_pil).cpu().numpy()
        if self.conversion:
            # If it's RGB but all channels have same values, extract from one channel
            if len(mask_array.shape) == 3:
                # Check if all channels are identical (R=G=B for each pixel)
                if np.allclose(mask_array[:,:,0], mask_array[:,:,1]) and np.allclose(mask_array[:,:,1], mask_array[:,:,2]):
                    mask_array = mask_array[:,:,0]  # Take red channel
                else:
                    # If channels are different, you might need custom logic here
                    # For now, assume red channel contains class info
                    mask_array = mask_array[:,:,0]
            
            mask = torch.from_numpy(mask_array).long()
        else:
            # If not converting, assume it's already in correct format
            mask = torch.from_numpy(mask_array).long()

        # Apply transform to image
        if self.image_transform is not None:
            image = self.image_transform(image)
            image = torch.tensor(image,dtype=torch.float32)
        
        # Remove any extra dimensions from mask
        if mask.dim() > 2:
            mask = mask.squeeze()
        
        # print(f"Saving image and mask: {image.shape}, {mask.shape}")
        # print(f"Mask unique values: {torch.unique(mask)}")  # Debug: check class values
        
        torch.save((image, mask), pt_path)
        return image, mask
    

if __name__ == "__main__":

    with open("/mnt/data/omkumar/foundation_phase1/Phase_1_training/configs/us_nerve.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    img_sz=cfg["dataset"]["image_size"]
    transforms = []
    if(img_sz is not None):
        transforms.append(T.Resize(tuple(img_sz)))
    transforms.append(T.ToTensor())

    train_dataset = CustomDataset(
                    name=cfg["dataset"]["name"],
                    data_dir=cfg["dataset"]["path"],
                    img_sz=img_sz,
                    is_train=True,
                    transform = T.Compose(transforms),
                    cache_dir=cfg["dataset"]["cache_dir"],
                    conversion = cfg['dataset']['conversion']
                    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    for sample in train_loader:
        print(sample[1].shape)

