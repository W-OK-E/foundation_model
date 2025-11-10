import os
import random
import shutil
from tqdm import tqdm
# Set random seed for reproducibility
random.seed(42)

# Paths
src_root = "/mnt/data/omkumar/foundation_phase1/datasets/SASVi/semantic_segmentation/cholec80/images"
mask_src_root = '/mnt/data/omkumar/foundation_phase1/datasets/SASVi/semantic_segmentation/cholec80/segm_ann_updated'
dst_root = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/images"
mask_dest_root = "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/masks"
# Create destination folders
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(dst_root, split), exist_ok=True)

# Get all video folders
video_folders = sorted([f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))])

# Split into train, val, test videos
train_videos = video_folders[:15]
val_videos   = video_folders[15:23]
test_videos  = video_folders[23:25]

print("Train videos:", train_videos)
print("Val videos:", val_videos)
print("Test videos:", test_videos)

def copy_subset(video_list, split, n_total):
    """Copy images from selected videos to destination folder."""
    n_per_video = n_total // len(video_list)
    im_dst_dir = os.path.join(dst_root, split)
    mask_dst_dir = os.path.join(mask_dest_root,split)
    copied = 0
    for video in tqdm(video_list):
        src_dir = os.path.join(src_root, video)
        mask_src_dir = os.path.join(mask_src_root,video)
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        selected = images[:n_per_video]

        for img in selected:
            src_path = os.path.join(src_dir, img)
            src_m_path = os.path.join(mask_src_dir,img)
            # Prefix filename to avoid duplicates
            dst_filename = f"{copied:04d}.png"
            im_dst_path = os.path.join(im_dst_dir, dst_filename)
            mask_dst_path = os.path.join(mask_dst_dir,dst_filename)
            shutil.copy2(src_path, im_dst_path)
            shutil.copy2(src_m_path,mask_dst_path)
            copied += 1

    print(f"Copied {copied} images to {split}/")

# Copy subsets
copy_subset(train_videos, "train", 20000)
copy_subset(val_videos, "val", 1000)
copy_subset(test_videos, "test", 200)

print("✅ Dataset split complete.")
