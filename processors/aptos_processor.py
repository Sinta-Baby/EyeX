"""
==========================================================
RetinaSense

APTOS Processor

==========================================================
"""




import pandas as pd

from core.paths import *
from core.logger import logger

# =====================================================
# Get APTOS Label
# =====================================================

def get_aptos_label(label):
    """
    Converts APTOS diagnosis to RetinaSense label.
    """

    if label == 0:
        return "Healthy"

    return "DR"


# =====================================================
# Create Metadata Entry
# =====================================================

def create_metadata_entry(filename, filepath, disease):
    """
    Creates one metadata record.
    """

    return {

        "filename": filename,

        "filepath": str(filepath),

        "dataset": "APTOS",

        "disease": disease

    }
    
# =====================================================
# Process Train Split
# =====================================================

def process_train():

    metadata = []

    df = pd.read_csv(APTOS_TRAIN_METADATA)

    print(f"APTOS Train Records : {len(df)}")

    skipped = 0

    for _, row in df.iterrows():

        filename = str(row["id_code"]).strip() + ".png"

        image_path = APTOS_TRAIN_IMAGES / filename

        if not image_path.exists():
            skipped += 1
            continue

        disease = get_aptos_label(row["diagnosis"])

        metadata.append(
            create_metadata_entry(
                filename,
                image_path,
                disease
            )
        )

    print(f"Valid Train Images : {len(metadata)}")
    print(f"Skipped            : {skipped}")

    return metadata



# =====================================================
# Process Validation Split
# =====================================================

def process_validation():

    metadata = []

    df = pd.read_csv(APTOS_VALID_METADATA)

    print(f"APTOS Validation Records : {len(df)}")

    skipped = 0

    for _, row in df.iterrows():

        filename = str(row["id_code"]).strip() + ".png"

        image_path = APTOS_VALID_IMAGES / filename

        if not image_path.exists():
            skipped += 1
            continue

        disease = get_aptos_label(row["diagnosis"])

        metadata.append(
            create_metadata_entry(
                filename,
                image_path,
                disease
            )
        )

    print(f"Valid Validation Images : {len(metadata)}")
    print(f"Skipped                 : {skipped}")

    return metadata

# =====================================================
# Process Test Split
# =====================================================

def process_test():

    metadata = []

    df = pd.read_csv(APTOS_TEST_METADATA)

    print(f"APTOS Test Records : {len(df)}")

    skipped = 0

    for _, row in df.iterrows():

        filename = str(row["id_code"]).strip() + ".png"

        image_path = APTOS_TEST_IMAGES / filename

        if not image_path.exists():
            skipped += 1
            continue

        disease = get_aptos_label(row["diagnosis"])

        metadata.append(
            create_metadata_entry(
                filename,
                image_path,
                disease
            )
        )

    print(f"Valid Test Images : {len(metadata)}")
    print(f"Skipped           : {skipped}")

    return metadata


# =====================================================
# Process APTOS
# =====================================================

def process_aptos():

    print("\n" + "=" * 70)
    print("Processing APTOS Dataset")
    print("=" * 70)

    train_metadata = process_train()

    validation_metadata = process_validation()

    test_metadata = process_test()

    metadata = []

    metadata.extend(train_metadata)
    metadata.extend(validation_metadata)
    metadata.extend(test_metadata)

    print("\nAPTOS Summary")
    print("-" * 30)
    print(f"Total Images : {len(metadata)}")
    
    print("\nDisease Distribution")
    print("-" * 30)

    distribution = {}
    for item in metadata:
        disease = item["disease"]
        distribution[disease] = distribution.get(disease, 0) + 1

    for disease, count in distribution.items():
        print(f"{disease:<12}: {count}")

    logger.info(f"APTOS Total Images : {len(metadata)}")

    return metadata