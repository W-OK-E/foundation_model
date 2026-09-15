import os
import cv2
import numpy as np

paths = {
    "UNeXt":"/data/asavari/UNeXt-pytorch/outputs/SOTA_test_US_Nerve",
    "US_Nerve_ELiTNet_CE":"/home/asavari/foundation_model/checkpoints/US_Nerve_ELiTNet_CE/viz",
    "US_Nerve_ELiTNet_US_Nerve_set2":"/home/asavari/foundation_model/checkpoints/US_Nerve_ELiTNet_US_Nerve_set2/viz"
}

output_path = "/home/asavari/foundation_model/US_Nerve_Visuals"
os.makedirs(output_path, exist_ok=True)

def get_all_images(root_dir):
    files = {}
    for root, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith((".png")):
                files[f] = os.path.join(root, f)
    
    return files

def padding(imgs):
    max_w = max(img.shape[1] for img in imgs)
    padded_imgs = []
    for img in imgs:
        h, w = img.shape[:2]
        if w < max_w:
            pad = max_w - w
            left = pad // 2
            right = pad - left
            img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value = (255,255, 255))
        padded_imgs.append(img)
    
    return padded_imgs

image_dict = {name: get_all_images(path) for name, path in paths.items()}

filelist = set.intersection(*(set(d.keys()) for d in image_dict.values()))


for filename in filelist:
    imgs = []
    labels = []

    for label, d in image_dict.items():
        img = cv2.imread(d[filename])
        if img is None:
            print(f"Could not read {d[filename]}")
            break
        imgs.append(img)
        labels.append(label)
    
    if len(imgs) != 3:
        continue

    imgs = padding(imgs)

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        cv2.putText(img, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

    combined = np.concatenate(imgs, axis = 0)

    cv2.imwrite(os.path.join(output_path, filename), combined)

print(f"Combined visualisation saved at {output_path}")