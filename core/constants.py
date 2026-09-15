"""
====================================================
RetinaSense
Core Constants
====================================================

Description:
This file stores all constant values used
throughout the RetinaSense project.

====================================================
"""

# ====================================================
# Project Information
# ====================================================

PROJECT_NAME = "RetinaSense"

PROJECT_VERSION = "1.0.0"

AUTHOR = "Sinta Baby"

# ====================================================
# Dataset Names
# ====================================================

DATASET_NAMES = [
    "ODIR",
    "APTOS",
    "GLAUCOMA",
    "ARMD"
]

# ====================================================
# Disease Labels
# ====================================================

DISEASE_LABELS = [
    "Healthy",
    "DR",
    "Glaucoma",
    "AMD"
]

NUM_CLASSES = len(DISEASE_LABELS)

# ====================================================
# Disease Mapping
# ====================================================

DISEASE_TO_INDEX = {
    "Healthy": 0,
    "DR": 1,
    "Glaucoma": 2,
    "AMD": 3
}

INDEX_TO_DISEASE = {
    0: "Healthy",
    1: "DR",
    2: "Glaucoma",
    3: "AMD"
}

# ====================================================
# DR Severity Labels
# ====================================================

DR_SEVERITY = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# ====================================================
# Image Settings
# ====================================================

IMAGE_SIZE = (300, 300)

IMAGE_CHANNELS = 3

# ====================================================
# ImageNet Normalization
# ====================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)

# ====================================================
# Training Settings
# ====================================================

BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

RANDOM_SEED = 42

NUM_WORKERS = 2

# ====================================================
# Model Settings
# ====================================================

MODEL_NAME = "efficientnet_b3"

BEST_MODEL_NAME = "best_model.pth"

# ====================================================
# Supported File Extensions
# ====================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
]

CSV_EXTENSION = ".csv"

EXCEL_EXTENSIONS = [
    ".xlsx",
    ".xls"
]

# ====================================================
# Prediction Settings
# ====================================================

CONFIDENCE_THRESHOLD = 0.50