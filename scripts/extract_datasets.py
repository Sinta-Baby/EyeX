"""
=========================================================
RetinaSense

Dataset Extraction Script

Developer : Sinta Baby

Description:
Automatically extracts all dataset ZIP files
into the extracted dataset folder.

=========================================================
"""

import zipfile

from core.paths import RAW_DATASET
from core.paths import EXTRACTED_DATASET
from utils.file_utils import create_directory
from core.logger import logger

print("=" * 70)
print("RetinaSense Dataset Extraction")
print("=" * 70)

# =====================================================
# Dataset Dictionary
# =====================================================

DATASETS = {

    "ODIR": "ODIR.zip",

    "APTOS": "APTOS.zip",

    "GLAUCOMA": "GLAUCOMA.zip",

    "ARMD": "ARMD.zip"

}

# =====================================================
# Create Extracted Folder
# =====================================================

create_directory(EXTRACTED_DATASET)

# =====================================================
# Extract Each Dataset
# =====================================================

for dataset_name, zip_name in DATASETS.items():

    zip_path = RAW_DATASET / zip_name

    output_folder = EXTRACTED_DATASET / dataset_name

    create_directory(output_folder)

    print(f"\nExtracting {dataset_name}...")

    if not zip_path.exists():

        print(f"❌ {zip_name} not found.")

        continue

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        zip_ref.extractall(output_folder)

    print(f"✅ {dataset_name} extracted successfully.")

    logger.info(f"{dataset_name} extracted.")
    

print("\n" + "=" * 70)
print("All datasets processed.")
print("=" * 70)