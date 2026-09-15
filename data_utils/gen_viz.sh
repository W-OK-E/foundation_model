#!/usr/bin/env bash

# -----------------------------
# Global output base directory
# -----------------------------
OUT_DIR_BASE="/mnt/data/omkumar/foundation_phase1/vis_results"

# -----------------------------
# Dataset Directorry
# -----------------------------
DATA_DIR="/mnt/data/omkumar/foundation_phase1/datasets/Cholec"

# -----------------------------
# Inputs as lists (aligned by index)
# -----------------------------
RUN_DIRS=(
  "/mnt/data/omkumar/foundation_phase1/checkpoints/Cholec_Cholec_set3" 
)

IMAGES_FILES=(
  "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/viz.txt"
)


GT_DIRS=(
  "/mnt/data/omkumar/foundation_phase1/datasets/Cholec/masks"
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
