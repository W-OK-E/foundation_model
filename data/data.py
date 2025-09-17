import os
import cv2
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
    def __init__(self, image_dir, ann_dir, image_names_list, ann_names_list, transform = None):
        super(SEGDataset, self).__init__()
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.images = image_names_list
        self.anns = ann_names_list
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        ann_path = os.path.join(self.ann_dir, self.anns[index])
        
        image = cv2.imread(image_path)
        if(image is not None):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Image not found at {image_path}")
        
        mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if(mask is not None):
            mask[mask > 1] = 1
            mask[mask < 1] = 0
        else:
            raise ValueError(f"Mask not found at {ann_path}")
        
        if self.transform is not None:
            transformer = self.transform(image = image, mask = mask)
            image, mask = transformer["image"], transformer["mask"]
            
        return image, mask
    

class IDRiD(SEGDataset):
    def __init__(self, root_dir, split = "train",img_size = 512, transform = None):
        """
        The root dir should contain two folders named "images" and "masks" 
        and a text file named "train.txt" or "val.txt" or "test.txt"
        depending on the split, which contains the train-val-test split information.
        """
        image_dir = os.path.join(root_dir, "images")
        ann_dir = os.path.join(root_dir, "masks")
        image_names, ann_names = read_split(os.path.join(root_dir, f"{split}.txt"))

        train_transforms, val_transforms = get_transforms(img_size=512)

        if split == "train":
            super(IDRiD, self).__init__(image_dir, ann_dir, image_names, ann_names, train_transforms)
        elif split == "val":
            super(IDRiD, self).__init__(image_dir, ann_dir, image_names, ann_names, val_transforms)
        elif split == "test":    
            super(IDRiD, self).__init__(image_dir, ann_dir, image_names, ann_names, None)
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")  

class AMD(SEGDataset):
    def __init__(self, root_dir, split = "train", transform = None):
        """
        The root dir should contain two folders named "images" and "masks" 
        and a text file named "train.txt" or "val.txt" or "test.txt"
        depending on the split, which contains the train-val-test split information.
        """
        image_dir = os.path.join(root_dir, "images")
        ann_dir = os.path.join(root_dir, "masks")
        image_names, ann_names = read_split(os.path.join(root_dir, f"{split}.txt"))

        train_transforms, val_transforms = get_transforms(img_size=512)

        if split == "train":
            super(AMD, self).__init__(image_dir, ann_dir, image_names, ann_names, train_transforms)
        elif split == "val":
            super(AMD, self).__init__(image_dir, ann_dir, image_names, ann_names, val_transforms)
        elif split == "test":    
            super(AMD, self).__init__(image_dir, ann_dir, image_names, ann_names, transform)
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        
class Refuge(SEGDataset):
    def __init__(self, root_dir, split = "train", transform = None):
        """
        The root dir should contain two folders named "images" and "masks" 
        and a text file named "train.txt" or "val.txt" or "test.txt"
        depending on the split, which contains the train-val-test split information.
        """
        image_dir = os.path.join(root_dir, "images")
        ann_dir = os.path.join(root_dir, "masks")
        image_names, ann_names = read_split(os.path.join(root_dir, f"{split}.txt"))

        train_transforms, val_transforms = get_transforms(img_size=512)

        if split == "train":
            super(Refuge, self).__init__(image_dir, ann_dir, image_names, ann_names, train_transforms)
        elif split == "val":
            super(Refuge, self).__init__(image_dir, ann_dir, image_names, ann_names, val_transforms)
        elif split == "test":    
            super(Refuge, self).__init__(image_dir, ann_dir, image_names, ann_names, transform)
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        
class ChaseDB(SEGDataset):
    def __init__(self, root_dir, split = "train", transform = None):
        """
        The root dir should contain two folders named "images" and "masks" 
        and a text file named "train.txt" or "val.txt" or "test.txt"
        depending on the split, which contains the train-val-test split information.
        """
        image_dir = os.path.join(root_dir, "images")
        ann_dir = os.path.join(root_dir, "masks")
        image_names, ann_names = read_split(os.path.join(root_dir, f"{split}.txt"))

        train_transforms, val_transforms = get_transforms(img_size=512)
        
        if split == "train":
            super(ChaseDB, self).__init__(image_dir, ann_dir, image_names, ann_names, train_transforms)
        elif split == "val":
            super(ChaseDB, self).__init__(image_dir, ann_dir, image_names, ann_names, val_transforms)
        elif split == "test":    
            super(ChaseDB, self).__init__(image_dir, ann_dir, image_names, ann_names, transform)
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")

class HRF(SEGDataset):
    def __init__(self, root_dir, split = "train", transform = None):
        """
        The root dir should contain two folders named "images" and "masks" 
        and a text file named "train.txt" or "val.txt" or "test.txt"
        depending on the split, which contains the train-val-test split information.
        """
        image_dir = os.path.join(root_dir, "images")
        ann_dir = os.path.join(root_dir, "masks")
        image_names, ann_names = read_split(os.path.join(root_dir, f"{split}.txt"))

        train_transforms, val_transforms = get_transforms(img_size=512)

        if split == "train":
            super(HRF, self).__init__(image_dir, ann_dir, image_names, ann_names, train_transforms)
        elif split == "val":
            super(HRF, self).__init__(image_dir, ann_dir, image_names, ann_names, val_transforms)
        elif split == "test":    
            super(HRF, self).__init__(image_dir, ann_dir, image_names, ann_names, transform)
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        

class DRIVE(SEGDataset):
    def __init__(self, root_dir, split = "train", transform = None):
        """
        The root dir should contain two folders named "images" and "masks" 
        and a text file named "train.txt" or "val.txt" or "test.txt"
        depending on the split, which contains the train-val-test split information.
        """
        image_dir = os.path.join(root_dir, "images")
        ann_dir = os.path.join(root_dir, "masks")
        image_names, ann_names = read_split(os.path.join(root_dir, f"{split}.txt"))

        train_transforms, val_transforms = get_transforms(img_size=512)

        if split == "train":
            super(DRIVE, self).__init__(image_dir, ann_dir, image_names, ann_names, train_transforms)
        elif split == "val":
            super(DRIVE, self).__init__(image_dir, ann_dir, image_names, ann_names, val_transforms)
        elif split == "test":    
            super(DRIVE, self).__init__(image_dir, ann_dir, image_names, ann_names, transform)
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")