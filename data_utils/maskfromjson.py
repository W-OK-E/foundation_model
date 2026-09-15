import os
import json
import numpy as np
import cv2

# Path to your directory containing .json files
input_dir = "/path/to/json/files"
output_dir = os.path.join(input_dir, "masks")
os.makedirs(output_dir, exist_ok=True)

# Assign colors or integer labels per class
class_colors = {
    "Pupil": 1,
    "Cornea": 2
}

for file in os.listdir(input_dir):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(input_dir, file)
    with open(json_path, 'r') as f:
        data = json.load(f)

    h, w = data["size"]["height"], data["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)

    num_labels_file = 0
    for obj in data["objects"]:
        class_name = obj["classTitle"]
        num_labels += 1
        color = class_colors.get(class_name, 255)  # unknown → white
        pts = np.array(obj["points"]["exterior"], np.int32)
        cv2.fillPoly(mask, [pts], color)
    
    out_name = os.path.splitext(file)[0] + "_mask.png"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, mask)
    
    print(f"Saved mask: {out_path}")
