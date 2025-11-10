import os
import cv2
import numpy as np
from tqdm import tqdm

global_class_map =  {
        0: 0,
        6247: 1,
        1938167: 2,
        3486383: 3,
        5610240: 4,
        6760942: 5,
        7392408: 6,
        7661279: 7,
        10043233: 8,
        11563695: 9,
        11830002: 10,
        12349696: 11,
        14533958: 12
    }

img_dir = '/mnt/data/omkumar/foundation_phase1/datasets/cataract1k/masks'

for case_dir in os.listdir(img_dir):
    case_dir_path = os.path.join(img_dir,case_dir)
    for im in tqdm(os.listdir(case_dir_path)):
        print("Image path:",os.path.join(case_dir_path,im))
        import sys
        sys.exit(0)
        img = cv2.imread(os.path.join(case_dir_path,im))
        img = img.reshape(-1,3)
        print(np.unique(img))
        img_colors_int = img[:,0] << 16 | img[:,1] << 8 | img[:,2] 
        print("Flattened Shape:",img_colors_int.shape)
        for color,idx in global_class_map.items():
            img_colors_int[img_colors_int == color] = idx
        print(np.unique(img_colors_int))
            