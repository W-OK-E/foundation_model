"""
Configuration file for Cataracts segmentation dataset
"""
import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Dataset identification
DATASET_NAME = "Cataracts"
GLOBAL_BATCH_SIZE = 8

# Dataset paths - adjust DATA_DIR as needed
DATA_DIR = "/path/to/data"  # Set this to your data directory
DATASET_PATH = os.path.join(DATA_DIR, "Cataracts")
TRAIN_IMAGE_DIR = os.path.join(DATASET_PATH, "train")
VAL_IMAGE_DIR = os.path.join(DATASET_PATH, "val")
TEST_IMAGE_DIR = os.path.join(DATASET_PATH, "test")

# File format configuration
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

IMG_SIZE = [360, 640]
TRAIN_IM_SIZE = [384, 640]
NUM_CHANNELS = 3

# Normalization statistics
MEAN_PER_CHANNEL = [0.30843202, 0.26594361, 0.21238147]
STD_PER_CHANNEL = [0.30274876, 0.23929678, 0.20821963]

# ============================================================================
# CLASS CONFIGURATION
# ============================================================================

# List of class names
# Note: Total dataset has 36 classes, but subset is used
ALL_CLASS_NAMES = [
    'bg',
    'pupil',
    'cornea',
    'skin',
    'iris',
    'surgical_instrument',
    'hand',
    'speculum',
    'other_instruments',
    'lens',
    'background',
    'misc'
]

NUM_CLASSES = len(ALL_CLASS_NAMES)

# Index to ignore in loss calculation
IGNORE_INDEX = None

# Multi-label classification flag
MULTI_LABEL = False

# Class weights for handling class imbalance
CLASS_WEIGHTS = [
    1.44005782e-02, 6.65201948e+00, 1.00000000e+00, 4.35970199e+01,
    1.36072862e+01, 2.61770459e+00, 2.00894792e-01, 9.98245199e-01,
    2.15347491e-01, 6.16941872e+00, 6.80057322e-01
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
CHECKPOINT_DIR = "./checkpoints/cataract"
SAVE_DIR = "./weights/cataract"
VISUALIZATION_DIR = "./visualizations/cataract"
PLOT_SAVE_PATH = f'./plots/cataract_{IMG_SIZE}_{LEARNING_RATE}_FP16_{USE_FP16}.png'

# Visualization parameters
NUM_VIS_SAMPLES = 3
NUM_OVERLAY_SAMPLES = 2
