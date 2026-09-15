"""
====================================================
RetinaSense Configuration
====================================================

Developer : Sinta Baby
"""

from core.constants import *
from core.paths import *

class Config:

    PROJECT_NAME = PROJECT_NAME
    VERSION = PROJECT_VERSION

    IMAGE_SIZE = IMAGE_SIZE
    BATCH_SIZE = BATCH_SIZE

    DATASET_NAMES = DATASET_NAMES

    DISEASE_LABELS = DISEASE_LABELS

    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD

    RAW_DATASET = RAW_DATASET
    EXTRACTED_DATASET = EXTRACTED_DATASET
    PROCESSED_DATASET = PROCESSED_DATASET

config = Config()