import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.idrid_dataset import IDRiDDataset


DATASET_ROOT = Path(
    "datasets/IDRiD/IDRiD_Segmentation"
)


dataset = IDRiDDataset(
    dataset_root=DATASET_ROOT,
    split="a. Training Set",
    image_size=(512, 512)
)


print("\nIDRiD DATASET LOADER TEST")
print("=========================")

print("Number of images:", len(dataset))


sample = dataset[0]

print("\nSample information")
print("------------------")

print(
    "Image ID:",
    sample["image_id"]
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

print("\nLoader test completed.")