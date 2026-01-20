import os
import json
import random
import numpy as np
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
from .transforms import get_transforms
from loss.boundary_loss.dataloader import dist_map_transform

def read_split(json_file: str,split = "train"):
    with open(json_file, "r") as f:
        split_dict = json.load(f)
    image_names = ann_names = split_dict[split]
    return image_names, ann_names

class SEGDataset(Dataset):
    def __init__(
            self, root_dir, split = 'train', img_size = (512,512), 
            multi_label = False, dry_run = False, mean = None,
            std = None, pt_files = False,dist_map = False,num_classes = 0,
            extract_patches = False, patch_size = (256,256)):

        super(SEGDataset, self).__init__()
        self.image_dir = os.path.join(root_dir,'images')
        self.multi_label = multi_label
        self.pt_files = pt_files
        self.dist_map = dist_map
        self.img_size = img_size
        self.extract_patches = extract_patches
        self.patch_size = patch_size

        if(self.extract_patches):
            print("="*15)
            print("Patch-Wise Training On")
            print("="*15)
            self.img_size = patch_size
            
        if(multi_label):
            print("="*15)
            print("Multi-Label Training")
            print("="*15)
            self.ann_dir = os.path.join(root_dir,'masks')
        elif(self.pt_files):
            self.ann_dir = os.path.join(root_dir,'masks_pt')
        else:
            self.ann_dir = os.path.join(root_dir,"masks")

        self.images, self.anns = read_split(os.path.join(root_dir,'split.json'),split = split)
        
        sample_ann_ext = os.listdir(self.ann_dir)[0].split('.')[1]
        curr_ext = self.anns[0].split('.')[1]

        if(sample_ann_ext != curr_ext):
            self.anns = [x.replace(curr_ext,sample_ann_ext) for x in self.anns]
        if(dry_run):
            self.images = self.images[:10]
            self.anns = self.anns[:10]
        
        # #NOTE: Only in place for Cholec, reducing the total dataset size:
        # self.images = self.images[:40]
        # self.images = self.images[:40]
        if(multi_label):
            ext = self.anns[0].split('.')[1]
            self.anns = [x.replace(ext,'pt') for x in self.anns]

        if(pt_files):
            ext = self.anns[0].split('.')[1]
            self.anns = [x.replace(ext,'pt') for x in self.anns]
            self.images = [x.replace(ext,'pt') for x in self.images]
            
        if(mean is None):
            mean = (0.0,0.0,0.0)
            std = (1.0,1.0,1.0)
            
        train_transforms, val_transforms = get_transforms(img_size = self.img_size,mean=mean,std=std)
        
        print("Transforms:",train_transforms,"Validation_transforms:",val_transforms)
        if(split == "train"):
            self.transform = train_transforms
        elif(split == 'val' or split == "test"):
            self.transform = val_transforms
        else:
            self.transform = None
        if(dist_map):
            ann_path = os.path.join(self.ann_dir, self.anns[0])
            if(self.multi_label):
                mask = torch.load(ann_path).numpy() #Empirically torch.load was faster than np.load
            elif(self.pt_files):#FIXME: Change the pt_files from H X W X 1 to H X W
                mask = torch.load(ann_path).numpy()[:,:,0] #NOTE: Monkey Patched to handle masks of shape h x w x 1
            else:
                mask = iio.imread(ann_path)
            resolution = [1]*len(mask.shape) 

            #NOTE: The resolution is basically how the pixel values change along each dimension, dx, dy and dz
            #We can't calculate it, it depends on how the image was acquired, for brats we checked it was [1,1,1]
            #For example, if the mask is of shape H X W X C, then the resolution is [1,1,1]
            #If the mask is of shape H X W X C X D, then the resolution is [1,1,1,1] and so on
            self.label_transform = dist_map_transform(resolution,num_classes)
        # print("")

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = {}
        
        image_path = os.path.join(self.image_dir, self.images[index])
        ann_path = os.path.join(self.ann_dir, self.anns[index])
        
        image = None
        if(self.pt_files):
            image = torch.load(image_path).numpy()
        else:
            image = iio.imread(image_path)

        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        
        if(self.multi_label):
            mask = torch.load(ann_path).numpy() #Empirically torch.load was faster than np.load
        elif(self.pt_files):#FIXME: Change the pt_files from H X W X 1 to H X W
            mask = torch.load(ann_path).numpy()[:,:,0] #NOTE: Monkey Patched to handle masks of shape h x w x 1
        else:
            mask = iio.imread(ann_path)
            
            if(mask is None):
                raise ValueError(f"Mask not found at {ann_path}")
            
            if mask.ndim == 3:
                mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
        # Patch Extraction Logic
        if self.extract_patches:
            h, w = image.shape[:2]
            ph, pw = self.patch_size
            
            if h <= ph or w <= pw:
                # If image is smaller than patch, we can't extract a random patch easily
                pass
            else:
                max_attempts = 50
                found_foreground = False
                
                for _ in range(max_attempts):
                    y = random.randint(0, h - ph)
                    x = random.randint(0, w - pw)
                    
                    candidate_mask = mask[y:y+ph, x:x+pw]
                    # Check if there is any foreground pixel (value > 0)
                    if np.any(candidate_mask > 0):
                        image = image[y:y+ph, x:x+pw]
                        mask = candidate_mask
                        found_foreground = True
                        break
                
                # If no foreground found after max_attempts, take a random one
                if not found_foreground:
                    y = random.randint(0, h - ph)
                    x = random.randint(0, w - pw)
                    image = image[y:y+ph, x:x+pw]
                    mask = mask[y:y+ph, x:x+pw]

        if self.transform is not None:
            transformer = self.transform(image = image, mask = mask)
            image, mask = transformer["image"], transformer["mask"].long()
        
        if(self.dist_map):
            dst_map_transformer = self.label_transform(mask)
            data["dist_map"] = dst_map_transformer
        data["image"] = image
        data["mask"] = mask
        
        return data
    

#This is the sample MRI Dataset Class
"""
class MRI_SEGDataset(keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()    

            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()
        
            for j in range(VOLUME_SLICES):
                 X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                 X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                 y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
                    
        # Generate masks
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y
        
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)
"""