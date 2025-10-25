import os
import json
import numpy as np
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
from .transforms import get_transforms

def read_split(json_file: str,split = "train"):
    with open(json_file, "r") as f:
        split_dict = json.load(f)
    image_names = ann_names = split_dict[split]
    return image_names, ann_names

class SEGDataset(Dataset):
    def __init__(
            self, root_dir, split = 'train', img_size = (512,512), 
            multi_label = False, dry_run = False, mean = None,std = None):
        super(SEGDataset, self).__init__()
        self.image_dir = os.path.join(root_dir,'images')
        self.multi_label = multi_label

        if(multi_label):
            print("="*15)
            print("Multi-Label Training")
            print("="*15)
            self.ann_dir = os.path.join(root_dir,'masks_pt')
        else:
            self.ann_dir = os.path.join(root_dir,"masks")

        self.images, self.anns = read_split(os.path.join(root_dir,'split.json'),split = split)
        if(dry_run):
            self.images = self.images[:1]
            self.anns = self.anns[:1]
        
        # #NOTE: Only in place for Cholec, reducing the total dataset size:
        # self.images = self.images[:40]
        # self.images = self.images[:40]
        if(multi_label):
            ext = self.anns[0].split('.')[1]
            self.anns = [x.replace(ext,'pt') for x in self.anns]

        if(mean is None):
            mean = (0.0,0.0,0.0)
            std = (1.0,1.0,1.0)
            
        train_transforms, val_transforms = get_transforms(img_size = img_size,mean=mean,std=std)
        if(split == "train"):
            self.transform = train_transforms
        elif(split == 'val' or split == "test"):
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
        
        if(self.multi_label):
            mask = torch.load(ann_path).numpy() #Empirically torch.load was faster than np.load
        else:
            mask = iio.imread(ann_path)
            
            if(mask is None):
                raise ValueError(f"Mask not found at {ann_path}")
            
            if mask.ndim == 3:
                mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        if self.transform is not None:
            transformer = self.transform(image = image, mask = mask)
            image, mask = transformer["image"], transformer["mask"]
        return image, mask
    

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