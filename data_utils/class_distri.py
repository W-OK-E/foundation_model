#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import json
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from typing import Tuple, List, Dict, Set

# -------------------------
# Utility functions
# -------------------------
def rgb_to_int(rgb: np.ndarray) -> int:
    """Convert RGB triplet (array-like or tuple) to 24-bit integer."""
    # ensure ints
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (r << 16) | (g << 8) | b

def int_to_rgb(i: int) -> Tuple[int, int, int]:
    """Convert 24-bit integer to RGB tuple."""
    return ((i >> 16) & 255, (i >> 8) & 255, i & 255)

# -------------------------
# First-pass worker:
# reads all images in a patient folder and returns set of unique color ints
# -------------------------
def collect_unique_colors_for_patient(patient_folder: str) -> Tuple[str, Set[int]]:
    """
    Process a single patient folder (path string).
    Return (patient_name, set_of_color_ints).
    This function is CPU+I/O bound: it reads images and finds the unique colors per image.
    """
    patient_path = Path(patient_folder)
    patient_name = patient_path.name
    unique_colors: Set[int] = set()

    # List files; skip non-files
    for fname in os.listdir(patient_path):
        fpath = patient_path / fname
        if not fpath.is_file():
            continue
        img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        if img is None:
            # skip invalid image
            continue

        # Compute integer representation for all pixels fast:
        arr = img.reshape(-1, 3)
        # Use numpy to compute ints
        ints = (arr[:,0].astype(np.uint32) << 16) | (arr[:,1].astype(np.uint32) << 8) | arr[:,2].astype(np.uint32)
        print("Unique Ints spotted:",ints)
        uniq_ints = np.unique(ints)
        for i in np.nditer(uniq_ints):
            unique_colors.add(int(i))
    return (patient_name, unique_colors)

# -------------------------
# Second-pass worker:
# maps images for one patient given the global mapping and returns pixel counts
# -------------------------
def map_patient_images(patient_folder: str, save_root: str, global_map: Dict[int, int]) -> Tuple[str, Dict[int, int]]:
    """
    Map every mask image in patient_folder to single-channel class indices (based on global_map).
    Save results under save_root/<patient_name>/ and return per-class pixel counts for that patient.
    """
    patient_path = Path(patient_folder)
    patient_name = patient_path.name
    save_folder = Path(save_root) / patient_name
    save_folder.mkdir(parents=True, exist_ok=True)

    class_pixel_counter: Dict[int, int] = {}

    for fname in os.listdir(patient_path):
        fpath = patient_path / fname
        if not fpath.is_file():
            continue
        img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        h, w = img.shape[:2]
        arr = img.reshape(-1, 3)
        ints = (arr[:,0].astype(int) << 16) | (arr[:,1].astype(int) << 8) | arr[:,2].astype(int)

        # Prepare output array (use uint16 to support many classes)
        mapped_flat = np.zeros_like(ints, dtype=np.uint16)

        # We iterate over the unique ints present in this image to avoid per-pixel python loops
        uniq_ints, uniq_counts = np.unique(ints, return_counts=True)
        for ui, cnt in zip(uniq_ints, uniq_counts):
            ui_int = int(ui)
            # Lookup mapping (assume mapping exists for all colors found)
            mapped_idx = global_map.get(ui_int, 0)  # fallback to 0 (background) if missing
            if(mapped_idx == 0 and ui_int !=0):
                print("Index Missing",ui_int)
            # Set mapped pixels where ints == ui
            mapped_flat[ints == ui] = mapped_idx
            class_pixel_counter[mapped_idx] = class_pixel_counter.get(mapped_idx, 0) + int(cnt)

        mapped_img = mapped_flat.reshape(h, w).astype(np.uint16)  # keep 16-bit to be safe
        # Save mapped image (OpenCV will save 16-bit png if dtype is uint16)
        save_path = save_folder / fname
        # If indices are small, saving as uint8 is OK; but to be safe preserve uint16
        cv2.imwrite(str(save_path), mapped_img)

    return (patient_name, class_pixel_counter)

# -------------------------
# Main: orchestrates two passes and aggregation
# -------------------------
def map_masks_parallel(mask_root: str, dest_root: str = None, workers: int = None, json_name: str = "mask_statistics.json"):
    mask_root = Path(mask_root)
    if dest_root is None:
        dest_root = str(mask_root).replace('segm_ann_updated', 'masks_mapped')
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Collect patient folders
    patient_folders: List[str] = []
    for entry in os.listdir(mask_root):
        p = mask_root / entry
        if p.is_dir():
            patient_folders.append(str(p))
    global_map =  {
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
    # -------------------------
    # 1) First pass: collect unique colors (parallel)
    # -------------------------
    print("PASS 1/2 — collecting unique colors per patient (parallel)...")
    patient_to_colors: Dict[str, Set[int]] = {}
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(collect_unique_colors_for_patient, pf): pf for pf in patient_folders}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            patient_name, colors = fut.result()
            patient_to_colors[patient_name] = colors

    # Merge to global set and create mapping (single-threaded)
    global_colors = set()
    for colors in patient_to_colors.values():
        global_colors.update(colors)

    # Guarantee background black maps to 0
    bg_int = rgb_to_int(np.array([0,0,0], dtype=np.uint8))
    if bg_int not in global_colors:
        global_colors.add(bg_int)

    # Create deterministic mapping: sort ints and assign indices
    sorted_colors = sorted(global_colors)
    global_map: Dict[int, int] = {}
    next_idx = 0
    # Ensure background 0 unless some other order required
    if bg_int in sorted_colors:
        global_map[bg_int] = 0
        next_idx = 1

    for ci in sorted_colors:
        if ci == bg_int:
            continue
        global_map[ci] = next_idx
        next_idx += 1

    print(f"Global classes discovered: {len(global_map)}")

    # -------------------------
    # 2) Second pass: map images using global_map (parallel)
    # -------------------------
    # print("PASS 2/2 — mapping images to class indices (parallel) ...")
    # patient_pixel_counts: Dict[str, Dict[int, int]] = {}
    # # Worker partial that includes the mapping and save root
    # worker_func = partial(map_patient_images, save_root=str(dest_root), global_map=global_map)

    # with ProcessPoolExecutor(max_workers=workers) as exe:
    #     futures = {exe.submit(worker_func, pf): pf for pf in patient_folders}
    #     for fut in tqdm(as_completed(futures), total=len(futures)):
    #         patient_name, counts = fut.result()
    #         patient_pixel_counts[patient_name] = counts

    # # -------------------------
    # # 3) Compute percentages per patient
    # # -------------------------
    # patient_stats: Dict[str, Dict[str, float]] = {}
    # for patient, counts in patient_pixel_counts.items():
    #     total = sum(counts.values()) if counts else 0
    #     if total == 0:
    #         patient_stats[patient] = {}
    #         continue
    #     pct = {str(k): round(v/total*100.0, 6) for k, v in counts.items()}
    #     patient_stats[patient] = pct

    # # Save JSON with mapping and patient stats
    # out = {
    #     "global_class_map": {str(k): int(v) for k, v in global_map.items()},
    #     "patient_statistics": patient_stats
    # }
    # stats_path = dest_root / json_name
    # with open(stats_path, "w") as f:
    #     json.dump(out, f, indent=4)

    print(f"Done. Mapped images saved under: {dest_root}")
    print(f"Statistics saved to: {stats_path}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel mask RGB -> single-channel mapping with per-patient stats")
    parser.add_argument("--dir", required=True, help="Root folder containing patient subfolders of masks")
    parser.add_argument("--dest", default=None, help="Destination root folder for mapped masks (defaults to replace 'masks' with 'masks_mapped')")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--json", default="mask_statistics.json", help="Filename for statistics JSON")
    args = parser.parse_args()
    map_masks_parallel(args.dir, args.dest, args.workers, args.json)
