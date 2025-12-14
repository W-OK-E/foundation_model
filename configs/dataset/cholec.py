"""
Configuration file for Cholec surgical scene segmentation dataset
"""
import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Dataset identification
DATASET_NAME = "Cholec"
GLOBAL_BATCH_SIZE = 8

# Dataset paths - adjust DATA_DIR as needed
DATA_DIR = "/path/to/data"  # Set this to your data directory
DATASET_PATH = os.path.join(DATA_DIR, "Cholec")
TRAIN_IMAGE_DIR = os.path.join(DATASET_PATH, "train")
VAL_IMAGE_DIR = os.path.join(DATASET_PATH, "val")
TEST_IMAGE_DIR = os.path.join(DATASET_PATH, "test")

# File format configuration
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

IMG_SIZE = [480, 854]
TRAIN_IM_SIZE = [512, 896]
NUM_CHANNELS = 3

# Normalization statistics
MEAN_PER_CHANNEL = [0.3782772, 0.23324355, 0.22724209]
STD_PER_CHANNEL = [0.27367861, 0.23258302, 0.21778946]

# ============================================================================
# CLASS CONFIGURATION
# ============================================================================

# List of class names - surgical instrument and anatomy segmentation
ALL_CLASS_NAMES = [
    "Black Background",
    "Abdominal Wall",
    "Liver",
    "Gastrointestinal Tract",
    "Fat",
    "Grasper",
    "Connective Tissue",
    "Blood",
    "Cystic Duct",
    "L-hook Electrocautery",
    "Gallbladder",
    "Hepatic Vein"
]

NUM_CLASSES = len(ALL_CLASS_NAMES)

# Index to ignore in loss calculation
IGNORE_INDEX = None

# Multi-label classification flag
MULTI_LABEL = False

# Class weights for handling class imbalance
CLASS_WEIGHTS = [
    3.07611788e-01, 2.79015445e+03, 3.25865581e+00, 5.79923131e-01,
    8.38769630e-01, 1.92239589e-01, 1.15761036e+00, 7.04655605e-01,
    8.80164325e-01, 3.56306789e+00, 1.83491041e+00, 1.67977953e+01
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Device configuration
DEVICE = 'cuda'

# Training parameters
BATCH_SIZE = GLOBAL_BATCH_SIZE
NUM_EPOCHS = 100

# Dry run mode for debugging
DRY_RUN = False

# Optimizer parameters
LEARNING_RATE = 0.001
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-5

# Scheduler parameters
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4

# Loss function parameters
DICE_WEIGHT = 0.5
CLASS_WEIGHT_METHOD = 'effective_samples'

# Mixed Precision Training
USE_FP16 = False

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

# ELiTNet2D Architecture parameters
MODEL_LAYERS = [4, 8, 16, 24]
MODEL_KERNEL_SIZE = 3
MODEL_UP_MODE = 'pixelshuffle'
MODEL_POOL = 'conv'
MODEL_CONV_BRIDGE = True
MODEL_SHORTCUT = True
MODEL_SKIP_CONN = True
MODEL_RESIDUAL = True
MODEL_CAUSAL = False
MODEL_CONV_MODE = 'Conv2d'

# ============================================================================
# CHECKPOINT AND VISUALIZATION CONFIGURATION
# ============================================================================

# Checkpoint settings
SAVE_CHECKPOINT = True
CHECKPOINT_FREQ = 5
LOAD_FROM_CHECKPOINT = False
LOAD_PRETRAINED = False

# Directory paths
CHECKPOINT_DIR = "./checkpoints/cholec"
SAVE_DIR = "./weights/cholec"
VISUALIZATION_DIR = "./visualizations/cholec"
PLOT_SAVE_PATH = f'./plots/cholec_{IMG_SIZE}_{LEARNING_RATE}_FP16_{USE_FP16}.png'

# Visualization parameters
NUM_VIS_SAMPLES = 3
NUM_OVERLAY_SAMPLES = 2
