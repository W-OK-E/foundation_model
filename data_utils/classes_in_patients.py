import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

mask_root = "../datasets/cataract1k/masks"
output_csv = "classes_in_patients.csv"

records = []

for case in tqdm(sorted(os.listdir(mask_root)), desc="Scanning cases.."):
    case_path = os.path.join(mask_root, case)
    if not os.path.isdir(case_path):
        continue

    case_classes = set()

    for fname in os.listdir(case_path):
        if not fname.endswith(".png"):
            continue

        mask_path = os.path.join(case_path, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            continue

        unique_vals = np.unique(mask)
        case_classes.update(unique_vals.tolist())

    if 0 in case_classes:
        case_classes.remove(0)

    records.append({
        "case_id": case,
        "num_classes": len(case_classes),
        "classes": sorted(list(case_classes))
    })

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"Saved class information to {output_csv}")