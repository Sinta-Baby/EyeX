"""
==========================================================
RetinaSense

Dataset Splitter

Description:
Splits the master metadata into:

1. Train
2. Validation
3. Test

using stratified sampling.

==========================================================
"""
import pandas as pd

from sklearn.model_selection import train_test_split

from core.paths import *

from core.logger import logger


# =====================================================
# Load Master Metadata
# =====================================================

def load_metadata():

    print("=" * 70)
    print("Loading Master Metadata")
    print("=" * 70)

    metadata = pd.read_csv(MASTER_METADATA)

    print(f"Total Records : {len(metadata)}")

    print("\nPreview")

    print(metadata.head())

    logger.info("Master metadata loaded successfully.")

    return metadata

# =====================================================
# Split Dataset
# =====================================================

def split_dataset(metadata):

    print("\n" + "=" * 70)
    print("Splitting Dataset")
    print("=" * 70)

    # -----------------------------
    # Train (80%) and Temp (20%)
    # -----------------------------
    train_df, temp_df = train_test_split(

        metadata,

        test_size=0.20,

        stratify=metadata["disease"],

        random_state=42

    )

    # -----------------------------
    # Validation (10%) and Test (10%)
    # -----------------------------
    validation_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        stratify=temp_df["disease"],

        random_state=42

    )

    print(f"Train Records      : {len(train_df)}")
    print(f"Validation Records : {len(validation_df)}")
    print(f"Test Records       : {len(test_df)}")

    total = len(metadata)

    print("\nSplit Ratio")
    print("-" * 30)
    print(f"Train      : {len(train_df)/total:.1%}")
    print(f"Validation : {len(validation_df)/total:.1%}")
    print(f"Test       : {len(test_df)/total:.1%}")
    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)
    
    logger.info("Dataset split completed successfully.")

    return train_df, validation_df, test_df


# =====================================================
# Display Dataset Distribution
# =====================================================

def display_distribution(train_df, validation_df, test_df):

    print("\n" + "=" * 70)
    print("Dataset Distribution")
    print("=" * 70)

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df
    }

    for name, df in datasets.items():

        print(f"\n{name}")

        print("-" * 30)

        distribution = df["disease"].value_counts().sort_index()

        for disease, count in distribution.items():

            print(f"{disease:<12}: {count}")

        print(f"\nTotal : {len(df)}")


# =====================================================
# Save Split Metadata
# =====================================================

def save_split_metadata(train_df, validation_df, test_df):

    print("\n" + "=" * 70)
    print("Saving Split Metadata")
    print("=" * 70)

    train_df.to_csv(
        TRAIN_METADATA,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_METADATA,
        index=False
    )

    test_df.to_csv(
        TEST_METADATA,
        index=False
    )

    print(f"Train Metadata      : {TRAIN_METADATA}")
    print(f"Validation Metadata : {VALIDATION_METADATA}")
    print(f"Test Metadata       : {TEST_METADATA}")

    logger.info("Split metadata saved successfully.")
    

# =====================================================
# Main
# =====================================================

def main():

    metadata = load_metadata()

    train_df, validation_df, test_df = split_dataset(metadata)

    display_distribution(
    train_df,
    validation_df,
    test_df
    )
    save_split_metadata(
    train_df,
    validation_df,
    test_df
    )

    print("\n" + "=" * 70)
    print("Dataset Split Completed")
    print("=" * 70)
    
# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    main()