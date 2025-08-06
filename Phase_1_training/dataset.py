import os
import torch
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, name: str, data_dir: str, img_sz: tuple, is_train: bool = True,transform = None,cache_dir="preprocessed"):
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
            f for f in os.listdir(self.raw_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.image_transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        fname = self.im_paths[index]
        pt_path = os.path.join(self.cache_dir, fname.replace('.jpg', '.pt').replace('.png', '.pt'))

        # Load from .pt if it exists
        if os.path.exists(pt_path):
            return torch.load(pt_path)

        # Otherwise, preprocess and save
        img_path = os.path.join(self.raw_image_dir, fname)
        mask_path = os.path.join(self.raw_mask_dir, fname)

        image = Image.open(img_path).convert("RGB").resize(self.img_sz)
        mask = Image.open(mask_path).convert("L").resize(self.img_sz, Image.NEAREST)

        if(self.image_transform is not None):
            image = self.image_transform(image)  # Tensor [3, H, W]
        
        mask = torch.from_numpy(np.array(mask)).long()  # Tensor [H, W]

        torch.save((image, mask), pt_path)

        return image, mask


if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    train_dataset = CustomDataset(
                    name=cfg["dataset"]["name"],
                    data_dir=cfg["dataset"]["path"],
                    img_sz=tuple(cfg["dataset"]["image_size"]),
                    is_train=True,
                    transform = T.Compose([
                                            T.Resize(cfg["dataset"]["image_size"]),
                                            T.ToTensor()
                                        ]),
                    cache_dir=cfg["dataset"]["cache_dir"]
                    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)


    for img,labels in train_loader:
        print(img.shape,labels.shape)
        break