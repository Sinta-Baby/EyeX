"""
====================================================
RetinaSense

Test CSV Utilities

Developer : Sinta Baby

====================================================
"""

from pathlib import Path

from utils.csv_utils import *
from core.paths import EXTRACTED_DATASET


print("=" * 60)
print("Testing CSV Utilities")
print("=" * 60)

# ====================================================
# Locate ODIR Metadata Automatically
# ====================================================

odir_folder = EXTRACTED_DATASET / "ODIR"

csv_file = None

for item in odir_folder.iterdir():

    if item.is_dir():

        possible_csv = item / "full_df.csv"

        if possible_csv.exists():

            csv_file = possible_csv

            break


# ====================================================
# Check CSV Exists
# ====================================================

if csv_file is None:

    raise FileNotFoundError(
        "Could not locate full_df.csv inside the extracted ODIR dataset."
    )

print(f"\nCSV Found : {csv_file}")

# ====================================================
# Read CSV
# ====================================================

df = read_csv(csv_file)

# ====================================================
# Display Information
# ====================================================

dataset_info(df)

# ====================================================
# Validate Required Columns
# ====================================================

required_columns = [

    "Left-Fundus",
    "Right-Fundus",
    "N",
    "D",
    "G",
    "A"

]

print("\nColumn Validation")
print("-" * 60)

status = validate_columns(df, required_columns)

print(f"\nValidation Status : {status}")