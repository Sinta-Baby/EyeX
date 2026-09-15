"""
==========================================================
RetinaSense

Glaucoma Processor

==========================================================
"""

from pathlib import Path

from core.paths import *
from core.logger import logger

# =====================================================
# Create Metadata Entry
# =====================================================

def create_metadata_entry(filename, filepath, disease):

    return {

        "filename": filename,

        "filepath": str(filepath),

        "dataset": "GLAUCOMA",

        "disease": disease

    }
    
    
# =====================================================
# Process Folder
# =====================================================

def process_folder(folder_path, disease):

    metadata = []

    image_count = 0

    for image_path in sorted(folder_path.glob("*.jpg")):

        metadata.append(

            create_metadata_entry(

                image_path.name,

                image_path,

                disease

            )

        )

        image_count += 1

    print(f"{folder_path.name:<22}: {image_count}")

    return metadata

# =====================================================
# Process Train
# =====================================================

def process_train():

    print("\nTraining Images")
    print("-" * 30)

    metadata = []

    metadata.extend(

        process_folder(

            GLAUCOMA_TRAIN / "Glaucoma_Negative",

            "Healthy"

        )

    )

    metadata.extend(

        process_folder(

            GLAUCOMA_TRAIN / "Glaucoma_Positive",

            "Glaucoma"

        )

    )

    print(f"\nTotal Train Images : {len(metadata)}")

    return metadata

# =====================================================
# Process Validation
# =====================================================

def process_validation():

    print("\nValidation Images")
    print("-" * 30)

    metadata = []

    metadata.extend(

        process_folder(

            GLAUCOMA_VALIDATION / "Glaucoma_Negative",

            "Healthy"

        )

    )

    metadata.extend(

        process_folder(

            GLAUCOMA_VALIDATION / "Glaucoma_Positive",

            "Glaucoma"

        )

    )

    print(f"\nTotal Validation Images : {len(metadata)}")

    return metadata

# =====================================================
# Process Glaucoma
# =====================================================

def process_glaucoma():

    print("\n" + "=" * 70)
    print("Processing Glaucoma Dataset")
    print("=" * 70)

    train_metadata = process_train()

    validation_metadata = process_validation()

    metadata = []

    metadata.extend(train_metadata)

    metadata.extend(validation_metadata)

    print("\nSummary")
    print("-" * 30)

    print(f"Total Images : {len(metadata)}")

    distribution = {}

    for item in metadata:

        disease = item["disease"]

        distribution[disease] = distribution.get(disease, 0) + 1

    print("\nDisease Distribution")
    print("-" * 30)

    for disease, count in distribution.items():

        print(f"{disease:<12}: {count}")

    logger.info(f"Glaucoma Images : {len(metadata)}")

    return metadata

