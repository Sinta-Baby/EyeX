"""
==========================================================
RetinaSense

ODIR Processor

==========================================================
"""

import pandas as pd

from core.paths import *
from core.logger import logger


def get_odir_label(row):
    """
    Returns RetinaSense disease label.
    """

    if row["N"] == 1:
        return "Healthy"

    if row["D"] == 1:
        return "DR"

    if row["G"] == 1:
        return "Glaucoma"

    if row["A"] == 1:
        return "AMD"

    return None

# =====================================================
# Image Exists
# =====================================================

def image_exists(filename):
    """
    Checks whether an image exists.
    """

    image_path = ODIR_IMAGES / filename

    return image_path.exists()


# =====================================================
# Create Metadata Entry
# =====================================================

def create_metadata_entry(filename, disease):
    """
    Creates one metadata record.
    """

    return {

        "filename": filename,

        "filepath": str(ODIR_IMAGES / filename),

        "dataset": "ODIR",

        "disease": disease

    }
    
# =====================================================
# Process ODIR
# =====================================================

def process_odir():

    print("\n" + "=" * 70)
    print("Processing ODIR Dataset")
    print("=" * 70)

    df = pd.read_csv(ODIR_METADATA)

    print(f"Total Records : {len(df)}")

    metadata = []

    skipped = 0

    for _, row in df.iterrows():

        filename = row["filename"]

        disease = get_odir_label(row)

        if disease is None:
            skipped += 1
            continue

        if not image_exists(filename):
            skipped += 1
            continue

        metadata.append(
            create_metadata_entry(
                filename,
                disease
            )
        )

    print("\nSummary")
    print("-" * 30)
    print(f"Valid Images : {len(metadata)}")
    print(f"Skipped      : {skipped}")

    print("\nSample Metadata")

    for item in metadata[:5]:
        print(item)

    print("\nDisease Distribution")
    print("-" * 30)

    distribution = {}

    for item in metadata:
        disease = item["disease"]
        distribution[disease] = distribution.get(disease, 0) + 1

    for disease, count in distribution.items():
        print(f"{disease:<12}: {count}")

    return metadata
    

