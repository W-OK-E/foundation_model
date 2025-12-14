"""
Configuration file for AMD segmentation dataset
"""
import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Dataset identification
DATASET_NAME = "AMD"
GLOBAL_BATCH_SIZE = 2

# Dataset paths - adjust DATA_DIR as needed
DATA_DIR = "/path/to/data"  # Set this to your data directory
DATASET_PATH = os.path.join(DATA_DIR, "AMD")
TRAIN_IMAGE_DIR = os.path.join(DATASET_PATH, "train")
VAL_IMAGE_DIR = os.path.join(DATASET_PATH, "val")
TEST_IMAGE_DIR = os.path.join(DATASET_PATH, "test")

# File format configuration
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

IMG_SIZE = 512
NUM_CHANNELS = 1

# ============================================================================
# CLASS CONFIGURATION
# ============================================================================

# List of class names
ALL_CLASS_NAMES = [
    "IRF",
    "SRF",
    "PED",
    "SHRM",
    "ISOS"
]

NUM_CLASSES = len(ALL_CLASS_NAMES)

# Index to ignore in loss calculation (usually background)
IGNORE_INDEX = 0

# Classes to exclude from metrics
EXCLUDE_FROM_METRICS = []

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Device configuration
DEVICE = 'cuda'

# Training parameters
BATCH_SIZE = GLOBAL_BATCH_SIZE
NUM_EPOCHS = 100

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
CLASS_WEIGHT_METHOD = 'effective_samples'  # Options: 'inverse_sqrt', 'inverse', 'effective_samples', 'none'

# Mixed Precision Training
USE_FP16 = False

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

# ELiTNet2D Architecture parameters
MODEL_LAYERS = [4, 8, 16, 24]
MODEL_KERNEL_SIZE = 3
MODEL_UP_MODE = 'pixelshuffle'  # Options: 'pixelshuffle', 'up_conv', 'transp_conv'
MODEL_POOL = 'conv'  # Options: 'pool', 'conv', False
MODEL_CONV_BRIDGE = True
MODEL_SHORTCUT = True
MODEL_SKIP_CONN = True
MODEL_RESIDUAL = True
MODEL_CAUSAL = False
MODEL_CONV_MODE = 'Conv2d'  # Options: 'Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d'

# ============================================================================
# CHECKPOINT AND VISUALIZATION CONFIGURATION
# ============================================================================

# Checkpoint settings
SAVE_CHECKPOINT = True
CHECKPOINT_FREQ = 5
LOAD_FROM_CHECKPOINT = False
LOAD_PRETRAINED = False

# Directory paths
CHECKPOINT_DIR = "./checkpoints/amd"
SAVE_DIR = "./weights/amd"
VISUALIZATION_DIR = "./visualizations/amd"
PLOT_SAVE_PATH = f'./plots/amd_{IMG_SIZE}_{LEARNING_RATE}_FP16_{USE_FP16}.png'

# Visualization parameters
NUM_VIS_SAMPLES = 3
NUM_OVERLAY_SAMPLES = 2
