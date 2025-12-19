import os
import pydicom
import numpy as np
import cv2

from tqdm import tqdm
from skimage import exposure
# from tensorflow.keras.preprocessing.image import img_to_array
dataset_path = '/mnt/data/omkumar/foundation_phase1/datasets/INBreast/INBreast/AllDICOMs'


import pandas as pd

metadata_path = '/mnt/data/omkumar/foundation_phase1/datasets/INBreast/INBreast/INbreast.csv'
metadata = pd.read_csv(metadata_path,delimiter=';')


# Final Shape - 4096, 3328

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    
    image = exposure.equalize_hist(image)
    
    image = image / np.max(image)
    
    image = np.stack((image,)*3, axis=-1)
    
    return image

def load_and_preprocess_dicom(metadata, dataset_path):
    X = []
    y = []
    
    # for index, row in metadata.iterrows():
    #     print(row['File Name'],end=' ')
    #     file_path = os.path.join(dataset_path, str(row['File Name']))
    #     print(file_path)
    #     import ipdb
    #     ipdb.set_trace()
    #     for j in tqdm(os.listdir(img)):
    #         if str(row['File Name'])==j.split('_')[0]:
    im_path = '/mnt/data/omkumar/foundation_phase1/datasets/INBreast/INBreast/AllDICOMs'
    im_shape = None
    for img in os.listdir(im_path):
        img_path = os.path.join(im_path,img)
        dicom = pydicom.dcmread(img_path)
        image = dicom.pixel_array
    
    # processed_image = preprocess_image(image)
        if im_shape is None:
            im_shape = image.shape
        else:
            if()
        print("Shape of the processed image:", image.shape)
        X.append(np.array(image))
        # if row['Less annotation status']=='NO ANNOTATION (NORMAL)':
        #     y.append(0)
        # else:
        #     y.append(1)  
    X = np.array(X)
    # y = np.array(y)
    return X  #y
    

X, y = load_and_preprocess_dicom(metadata, dataset_path)
