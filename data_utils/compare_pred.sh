#!/usr/bin/env bash

# -----------------------------
# Global output base directory
# -----------------------------
OUT_DIR_BASE="/mnt/data/omkumar/foundation_phase1/vis_results"

# -----------------------------
# Dataset Directorry
# -----------------------------
DATA_DIR="/mnt/data/omkumar/foundation_phase1/datasets/IDRiD"

# -----------------------------
# Inputs as lists (aligned by index)
# -----------------------------
RUN_DIRS=(
  "/mnt/data/omkumar/foundation_phase1/checkpoints/IDRiD_Focal" #bg was ignored
  "/mnt/data/omkumar/foundation_phase1/checkpoints/IDRiD_Conv2d_focal_dice_weighted_set2" #bg was not ignored
  "/mnt/data/omkumar/foundation_phase1/checkpoints/IDRiD_IDRID",
  "/mnt/data/omkumar/foundation_phase1/checkpoints/IDRiD_IDRID_128",
  "/mnt/data/omkumar/foundation_phase1/checkpoints/IDRiD_mkconv_focal_dice_weighted"
)

IMAGES_FILES=(
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/viz.txt"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/viz.txt"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/viz.txt"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/viz.txt"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/viz.txt"
)


GT_DIRS=(
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/masks"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/masks"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/masks"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/masks"
  "/mnt/data/omkumar/foundation_phase1/datasets/IDRiD/masks"
)

DEVICE="cuda"   # or cpu

# -----------------------------
# List to store per-run output dirs
# -----------------------------
OUT_DIRS=()

# -----------------------------
# Run visualization per folder
# -----------------------------
for i in "${!RUN_DIRS[@]}"; do
  RUN_DIR="${RUN_DIRS[$i]}"
  IMAGES_FILE="${IMAGES_FILES[$i]}"
  GT_DIR="${GT_DIRS[$i]}"

  # Extract stem of RUN_DIR
  STEM="$(basename "$RUN_DIR")"

  # Construct per-run output directory
  OUT_DIR="${OUT_DIR_BASE}/${STEM}"

  # Append to list
  OUT_DIRS+=("$OUT_DIR")

  echo "Running visualization for: $RUN_DIR"
  echo "Output dir: $OUT_DIR"

  uv run /mnt/data/omkumar/foundation_phase1/utils/visualize.py \
    --data-dir "$DATA_DIR" \
    --run-dir "$RUN_DIR" \
    --images-file "$IMAGES_FILE" \
    --out-dir "$OUT_DIR" \
    --gt-dir "$GT_DIR" \
    --device "$DEVICE"

done

# -----------------------------
# Run comparison on outputs
# -----------------------------
# for OUT_DIR in "${OUT_DIRS[@]}"; do
#   echo "Comparing predictions in: $OUT_DIR"

#   uv run /mnt/data/omkumar/foundation_phase1/utils/compare_preds_grid.py \
#     --folder_path "$OUT_DIR"

# done
