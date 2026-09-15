"""
====================================================
RetinaSense

CSV Utility Functions


Description:
Common CSV operations used throughout
the RetinaSense project.

====================================================
"""

from pathlib import Path
import pandas as pd

from core.logger import logger


# ====================================================
# Read CSV
# ====================================================

def read_csv(file_path: Path):

    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    logger.info(f"Loaded CSV : {file_path.name}")

    return df


# ====================================================
# Save CSV
# ====================================================

def save_csv(df, file_path: Path):

    df.to_csv(file_path, index=False)

    logger.info(f"Saved CSV : {file_path.name}")


# ====================================================
# Display Dataset Information
# ====================================================

def dataset_info(df):

    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names")

    for col in df.columns:
        print(f"• {col}")


# ====================================================
# Validate Required Columns
# ====================================================

def validate_columns(df, required_columns):

    missing = []

    for col in required_columns:

        if col not in df.columns:

            missing.append(col)

    if len(missing) == 0:

        logger.info("Required columns verified.")

        return True

    logger.error(f"Missing Columns : {missing}")

    return False


# ====================================================
# Merge DataFrames
# ====================================================

def merge_dataframes(dataframes):

    merged = pd.concat(dataframes, ignore_index=True)

    logger.info("DataFrames merged successfully.")

    return merged