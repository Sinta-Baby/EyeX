import json
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.idrid_patch_dataset import IDRiDPatchDataset


DATASET_ROOT = Path(
    "datasets/IDRiD/IDRiD_Segmentation"
)

SPLIT_FILE = Path(
    "results/idrid/train_val_split.json"
)


# --------------------------------------------------
# Load split
# --------------------------------------------------

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as file:

    split_data = json.load(file)


train_ids = split_data["train"]
validation_ids = split_data["validation"]


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = IDRiDPatchDataset(
    dataset_root=DATASET_ROOT,
    split="a. Training Set",
    patch_size=512,
    stride=384,
    image_ids=train_ids
)


validation_dataset = IDRiDPatchDataset(
    dataset_root=DATASET_ROOT,
    split="a. Training Set",
    patch_size=512,
    stride=384,
    image_ids=validation_ids
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nIDRiD TRAIN / VALIDATION DATASET TEST")
print("=====================================")

print(
    "Training images:",
    len(train_dataset.images)
)

print(
    "Training patches:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset.images)
)

print(
    "Validation patches:",
    len(validation_dataset)
)


# --------------------------------------------------
# Check image IDs
# --------------------------------------------------

train_set = {
    image.stem
    for image in train_dataset.images
}

validation_set = {
    image.stem
    for image in validation_dataset.images
}


overlap = train_set.intersection(
    validation_set
)


print("\nData leakage check")
print("------------------")

print(
    "Training/Validation overlap:",
    len(overlap)
)

if len(overlap) == 0:

    print(
        "✓ No image-level data leakage"
    )

else:

    print(
        "✗ DATA LEAKAGE DETECTED"
    )

    print(
        "Overlapping IDs:",
        sorted(overlap)
    )


# --------------------------------------------------
# Sample check
# --------------------------------------------------

train_sample = train_dataset[0]
val_sample = validation_dataset[0]


print("\nSample check")
print("------------")

print(
    "Training sample:",
    train_sample["image_id"],
    train_sample["image"].shape,
    train_sample["mask"].shape
)

print(
    "Validation sample:",
    val_sample["image_id"],
    val_sample["image"].shape,
    val_sample["mask"].shape
)


print("\nSplit dataset test completed.")