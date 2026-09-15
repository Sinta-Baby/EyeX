"""
==========================================================
RetinaSense

Dataset Statistics

Description:
Verifies the final RetinaSense dataset after images
have been copied.

Checks:

1. Train images
2. Validation images
3. Test images

Displays:

• Images per disease
• Total images
• Overall summary

==========================================================
"""

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
# Count Images
# =====================================================


def count_images(folder):

    total = 0
    statistics = {}

    for disease in CLASSES:

        disease_folder = folder / disease

        count = len([
            file
            for file in disease_folder.iterdir()
            if file.is_file()
        ])

        statistics[disease] = count
        total += count

    return statistics, total


# =====================================================
# Display Statistics
# =====================================================

def display_statistics(name, folder):

    print("\n" + "=" * 70)
    print(f"{name} Dataset")
    print("=" * 70)

    statistics, total = count_images(folder)

    for disease, count in statistics.items():

        print(f"{disease:<12}: {count}")

    print("-" * 30)
    print(f"Total Images : {total}")

    logger.info(f"{name}: {total} images verified.")

    return total


# =====================================================
# Main
# =====================================================

def main():

    train_total = display_statistics(
        "Training",
        TRAIN_DATASET
    )

    validation_total = display_statistics(
        "Validation",
        VALIDATION_DATASET
    )

    test_total = display_statistics(
        "Test",
        TEST_DATASET
    )

    grand_total = (
        train_total
        + validation_total
        + test_total
    )

    print("\n" + "=" * 70)
    print("RetinaSense Dataset Summary")
    print("=" * 70)

    print(f"Training Images   : {train_total}")
    print(f"Validation Images : {validation_total}")
    print(f"Test Images       : {test_total}")

    print("-" * 30)

    print(f"Grand Total Images : {grand_total}")

    logger.info(f"Grand Total Images : {grand_total}")

    print("\nDataset verification completed successfully.")


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    main()