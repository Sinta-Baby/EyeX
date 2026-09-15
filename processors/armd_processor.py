"""
==========================================================
RetinaSense

ARMD Processor

==========================================================
"""

from core.paths import *
from core.logger import logger

def create_metadata_entry(filename, filepath, disease):

    return {

        "filename": filename,

        "filepath": str(filepath),

        "dataset": "ARMD",

        "disease": disease

    }
    

def process_folder(folder_path, disease):

    metadata = []

    image_count = 0

    for extension in ("*.jpg", "*.jpeg", "*.png"):

        for image_path in sorted(folder_path.glob(extension)):

            metadata.append(

                create_metadata_entry(

                    image_path.name,

                    image_path,

                    disease

                )

            )

            image_count += 1

    print(f"{folder_path.name:<12}: {image_count}")

    return metadata

def process_train():

    print("\nTraining Images")
    print("-" * 30)

    metadata = []

    metadata.extend(

        process_folder(

            ARMD_TRAIN / "normal",

            "Healthy"

        )

    )

    metadata.extend(

        process_folder(

            ARMD_TRAIN / "amd",

            "AMD"

        )

    )

    print(f"\nTotal Train Images : {len(metadata)}")

    return metadata

def process_validation():

    print("\nValidation Images")
    print("-" * 30)

    metadata = []

    metadata.extend(

        process_folder(

            ARMD_VALIDATION / "normal",

            "Healthy"

        )

    )

    metadata.extend(

        process_folder(

            ARMD_VALIDATION / "amd",

            "AMD"

        )

    )

    print(f"\nTotal Validation Images : {len(metadata)}")

    return metadata


def process_armd():

    print("\n" + "=" * 70)
    print("Processing ARMD Dataset")
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

    logger.info(f"ARMD Images : {len(metadata)}")

    return metadata
