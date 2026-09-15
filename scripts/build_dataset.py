"""
==========================================================
RetinaSense

Dataset Builder


Description:
Builds the unified RetinaSense dataset.

==========================================================
"""

import pandas as pd


import sys
sys.path.insert(0, ".")

from core.paths import *
from core.logger import logger

from utils.file_utils import create_directory

from processors.odir_processor import process_odir
from processors.aptos_processor import process_aptos
from processors.glaucoma_processor import process_glaucoma
from processors.armd_processor import process_armd

# =====================================================
# Disease Classes
# =====================================================

CLASSES = [
    "Healthy",
    "DR",
    "Glaucoma",
    "AMD"
]

# =====================================================
# Global Metadata
# =====================================================

master_metadata = []

# =====================================================
# Create Dataset Structure
# =====================================================

def create_dataset_structure():

    print("=" * 70)
    print("Creating RetinaSense Dataset Structure")
    print("=" * 70)

    create_directory(RETINASENSE_DATASET)
    create_directory(TRAIN_DATASET)
    create_directory(VALIDATION_DATASET)
    create_directory(TEST_DATASET)
    create_directory(METADATA)

    for cls in CLASSES:

        create_directory(TRAIN_DATASET / cls)
        create_directory(VALIDATION_DATASET / cls)
        create_directory(TEST_DATASET / cls)

        print(f"✅ {cls}")

    logger.info("RetinaSense dataset folders created.")
    
    


# =====================================================
# Save Metadata
# =====================================================

def save_metadata():

    print("\n" + "=" * 70)
    print("Saving Master Metadata")
    print("=" * 70)

    if len(master_metadata) == 0:

        print("No metadata found.")

        return

    metadata_df = pd.DataFrame(master_metadata)

    metadata_df.to_csv(
        MASTER_METADATA,
        index=False
    )

    print(f"Metadata File : {MASTER_METADATA}")
    print(f"Total Records : {len(metadata_df)}")

    print("\nPreview")

    print(metadata_df.head())
    
    
    logger.info(f"Total Metadata Records : {len(metadata_df)}")
    logger.info("Master metadata saved successfully.")


# =====================================================
# Main Builder
# =====================================================

def build_dataset():

    create_dataset_structure()

    # -----------------------------
    # ODIR
    # -----------------------------
    odir_metadata = process_odir()
    master_metadata.extend(odir_metadata)

    # -----------------------------
    # APTOS
    # -----------------------------
    aptos_metadata = process_aptos()

    if aptos_metadata:
        master_metadata.extend(aptos_metadata)

    # -----------------------------
    # Glaucoma
    # -----------------------------
    glaucoma_metadata = process_glaucoma()

    if glaucoma_metadata:
        master_metadata.extend(glaucoma_metadata)

    # -----------------------------
    # ARMD
    # -----------------------------
    armd_metadata = process_armd()

    if armd_metadata:
        master_metadata.extend(armd_metadata)

    save_metadata()

    print("\n" + "=" * 70)
    print("RetinaSense Dataset Builder Completed")
    print("=" * 70)

# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    build_dataset()