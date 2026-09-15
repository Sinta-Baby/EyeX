"""
==========================================================
RetinaSense

Image Copier

Description:
Copies images into the RetinaSense dataset structure
using the split metadata files.

==========================================================
"""

import shutil
import pandas as pd

from pathlib import Path

from core.paths import *
from core.logger import logger


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
# Create Dataset Folder Structure
# =====================================================

def create_dataset_structure():

    print("=" * 70)
    print("Creating Dataset Folder Structure")
    print("=" * 70)

    datasets = [
        TRAIN_DATASET,
        VALIDATION_DATASET,
        TEST_DATASET
    ]

    # Remove only train/validation/test folders
    # Metadata folder remains untouched
    for dataset in datasets:

        if dataset.exists():
            shutil.rmtree(dataset)

        dataset.mkdir(parents=True, exist_ok=True)

        for disease in CLASSES:

            (dataset / disease).mkdir(
                parents=True,
                exist_ok=True
            )

    logger.info("Dataset folder structure created successfully.")


# =====================================================
# Load Metadata
# =====================================================

def load_metadata(metadata_path):

    print("\n" + "=" * 70)
    print(f"Loading {metadata_path.name}")
    print("=" * 70)

    metadata = pd.read_csv(metadata_path)

    print(f"Total Records : {len(metadata)}")

    logger.info(f"{metadata_path.name} loaded successfully.")

    return metadata


# =====================================================
# Copy Images
# =====================================================

def copy_images(
    metadata,
    destination_folder,
    metadata_path
):

    print("\n" + "=" * 70)
    print(f"Copying Images to {destination_folder.name}")
    print("=" * 70)

    copied = 0
    skipped = 0

    for row_index, row in metadata.iterrows():

        source = Path(row["filepath"])

        disease = row["disease"]

        dataset = row["dataset"]

        if source.exists():

            # Create unique filename
            new_filename = (
                f"{dataset}_{disease}_{row_index}_{source.name}"
            )

            # Update metadata
            metadata.at[row_index, "filename"] = new_filename

            destination = (
                destination_folder
                / disease
                / new_filename
            )

            shutil.copy2(
                source,
                destination
            )

            copied += 1

            if copied % 1000 == 0:
                print(f"Copied {copied} images...")

        else:

            print(f"Missing : {source}")

            skipped += 1

    # Save updated metadata
    metadata.to_csv(
        metadata_path,
        index=False
    )

    print("\nSummary")
    print("-" * 30)
    print(f"Copied Images : {copied}")
    print(f"Skipped Images: {skipped}")

    logger.info(
        f"{destination_folder.name}: "
        f"{copied} images copied, "
        f"{skipped} skipped."
    )


# =====================================================
# Main
# =====================================================

def main():

    create_dataset_structure()

    # -----------------------------
    # Train
    # -----------------------------
    train_metadata = load_metadata(
        TRAIN_METADATA
    )

    copy_images(
        train_metadata,
        TRAIN_DATASET,
        TRAIN_METADATA
    )

    # -----------------------------
    # Validation
    # -----------------------------
    validation_metadata = load_metadata(
        VALIDATION_METADATA
    )

    copy_images(
        validation_metadata,
        VALIDATION_DATASET,
        VALIDATION_METADATA
    )

    # -----------------------------
    # Test
    # -----------------------------
    test_metadata = load_metadata(
        TEST_METADATA
    )

    copy_images(
        test_metadata,
        TEST_DATASET,
        TEST_METADATA
    )

    print("\n" + "=" * 70)
    print("Copy Summary")
    print("=" * 70)

    print(f"Train Images      : {len(train_metadata)}")
    print(f"Validation Images : {len(validation_metadata)}")
    print(f"Test Images       : {len(test_metadata)}")
    print(
        f"Total Images      : "
        f"{len(train_metadata) + len(validation_metadata) + len(test_metadata)}"
    )

    print("\n" + "=" * 70)
    print("Image Copy Completed")
    print("=" * 70)

    logger.info("All images copied successfully.")


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    main()