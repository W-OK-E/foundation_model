import os
import numpy as np
import imageio.v3 as iio
from torch.utils.data import Dataset
from .transforms import get_transforms

def read_split(text_file: str):
    with open(text_file, "r") as f:
        lines = f.readlines()
    image_names = []
    ann_names = []
    for line in lines:
        line = line.strip().split(",")
        img_name = line[0]
        ann_name = img_name #Here both image and annotation have same name, change this if otherwise
        image_names.append(img_name)
        ann_names.append(ann_name)
    return image_names, ann_names

class SEGDataset(Dataset):
    def __init__(self, root_dir, split = 'train', img_size = (512,512)):
        super(SEGDataset, self).__init__()
        self.image_dir = os.path.join(root_dir,'images')
        self.ann_dir = os.path.join(root_dir,'masks')
        self.images, self.anns = read_split(os.path.join(root_dir,f'{split}.txt'))
        train_transforms, val_transforms = get_transforms(img_size = img_size)
        if(split == "train"):
            self.transform = train_transforms
        elif(split == 'val'):
            self.transform = val_transforms
        else:
            self.transform = None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        ann_path = os.path.join(self.ann_dir, self.anns[index])
        
        image = iio.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        mask = iio.imread(ann_path)
        
        if(mask is None):
            raise ValueError(f"Mask not found at {ann_path}")
        
        if mask.ndim == 3:
            mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        print("Checking the shape of the images:",mask.shape,image.shape)
        if self.transform is not None:
            transformer = self.transform(image = image, mask = mask)
            image, mask = transformer["image"], transformer["mask"]
            
        return image, mask
    

