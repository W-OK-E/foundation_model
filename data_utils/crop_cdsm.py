import os
from tqdm import tqdm
import imageio.v2 as iio
import numpy as np
import argparse

def read_and_crop_images(image_dir,mask_dir,perc_crop = 0.2):
    img_dest = image_dir.replace("images","cropped_images")
    mask_dest = mask_dir.replace("masks","cropped_masks")

    os.makedirs(img_dest,exist_ok = True)
    os.makedirs(mask_dest,exist_ok = True)
    images = os.listdir(image_dir)
    for im in tqdm(images):
        src_im_path = os.path.join(image_dir,im)
        src_mask_path = os.path.join(mask_dir,im)

        np_img = iio.imread(src_im_path)
        np_mask = iio.imread(src_mask_path)

        h,w = np_img.shape

        np_img = np_img[:,:-(int(perc_crop*w))]
        np_mask = np_mask[:,:-(int(perc_crop*w))]

        dest_im_path = os.path.join(img_dest,im)
        dest_mask_path = os.path.join(mask_dest,im)    

        iio.imwrite(dest_im_path,np_img)
        iio.imwrite(dest_mask_path,np_mask)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str)
parser.add_argument("--perc_crop", type = float)
args = parser.parse_args()


img_dir = args.data_dir + "/images"
mask_dir = args.data_dir + "/masks"
read_and_crop_images(img_dir,mask_dir,args.perc_crop)
