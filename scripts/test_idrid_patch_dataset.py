import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.idrid_patch_dataset import (
    IDRiDPatchDataset
)


DATASET_ROOT = Path(
    "datasets/IDRiD/IDRiD_Segmentation"
)


dataset = IDRiDPatchDataset(
    dataset_root=DATASET_ROOT,
    split="a. Training Set",
    patch_size=512,
    stride=384
)


print("\nIDRiD PATCH DATASET TEST")
print("========================")

print(
    "Original images:",
    len(dataset.images)
)

print(
    "Total patches:",
    len(dataset)
)


sample = dataset[0]


print("\nFirst patch")
print("-----------")

print(
    "Image ID:",
    sample["image_id"]
)

print(
    "Position:",
    sample["x"],
    sample["y"]
)

print(
    "Image shape:",
    sample["image"].shape
)

print(
    "Mask shape:",
    sample["mask"].shape
)

print(
    "Valid masks:",
    sample["valid"]
)

print(
    "Image min:",
    sample["image"].min().item()
)

print(
    "Image max:",
    sample["image"].max().item()
)

print(
    "Mask values:",
    sample["mask"].unique()
)

print("\nPatch dataset test completed.")