from pathlib import Path
import cv2
import numpy as np


DATASET_ROOT = Path("datasets/IDRiD/IDRiD_Segmentation")

MASK_ROOT = (
    DATASET_ROOT
    / "2. All Segmentation Groundtruths"
    / "a. Training Set"
)


MASK_FOLDERS = {
    "Microaneurysms": "1. Microaneurysms",
    "Haemorrhages": "2. Haemorrhages",
    "Hard Exudates": "3. Hard Exudates",
    "Soft Exudates": "4. Soft Exudates",
}


print("\nIDRiD MASK INSPECTION")
print("====================")


for lesion, folder_name in MASK_FOLDERS.items():

    folder = MASK_ROOT / folder_name

    mask_files = sorted(folder.glob("*.tif"))

    if not mask_files:
        print(f"\n{lesion}: No masks found")
        continue

    mask_path = mask_files[0]

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    print(f"\n{lesion}")
    print("File   :", mask_path.name)
    print("Shape  :", mask.shape)
    print("Dtype  :", mask.dtype)

    unique_values = np.unique(mask)

    print("Unique pixel values:")

    if len(unique_values) <= 20:
        print(unique_values)
    else:
        print(
            f"{len(unique_values)} unique values"
        )