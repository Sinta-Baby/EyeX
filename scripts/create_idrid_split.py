import json
import random
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.idrid_patch_dataset import IDRiDPatchDataset


# --------------------------------------------------
# Settings
# --------------------------------------------------

DATASET_ROOT = Path(
    "datasets/IDRiD/IDRiD_Segmentation"
)

SEED = 42
VALIDATION_RATIO = 0.20


# --------------------------------------------------
# Load IDRiD training images
# --------------------------------------------------

dataset = IDRiDPatchDataset(
    dataset_root=DATASET_ROOT,
    split="a. Training Set",
    patch_size=512,
    stride=384
)


image_ids = [
    image_path.stem
    for image_path in dataset.images
]


# --------------------------------------------------
# Shuffle image IDs
# --------------------------------------------------

random.seed(SEED)

random.shuffle(image_ids)


# --------------------------------------------------
# Create train/validation split
# --------------------------------------------------

validation_count = round(
    len(image_ids) * VALIDATION_RATIO
)

validation_ids = sorted(
    image_ids[:validation_count]
)

train_ids = sorted(
    image_ids[validation_count:]
)


# --------------------------------------------------
# Save split
# --------------------------------------------------

split_data = {
    "seed": SEED,
    "validation_ratio": VALIDATION_RATIO,
    "train": train_ids,
    "validation": validation_ids
}


output_dir = Path("results/idrid")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    output_dir /
    "train_val_split.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        split_data,
        file,
        indent=4
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nIDRiD TRAIN / VALIDATION SPLIT")
print("==============================")

print("Total training images:", len(image_ids))
print("Training images:", len(train_ids))
print("Validation images:", len(validation_ids))

print("\nTraining IDs")
print("------------")

print(
    ", ".join(train_ids)
)

print("\nValidation IDs")
print("--------------")

print(
    ", ".join(validation_ids)
)

print("\nSplit saved to:")
print(output_file)

print("\nSplit completed successfully.")