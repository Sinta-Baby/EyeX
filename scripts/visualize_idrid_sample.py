import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Allow Python to find the EyeX project folders
# --------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

DATASET_ROOT = Path(
    "datasets/IDRiD/IDRiD_Segmentation"
)

IMAGE_PATH = (
    DATASET_ROOT
    / "1. Original Images"
    / "a. Training Set"
    / "IDRiD_01.jpg"
)

MASK_ROOT = (
    DATASET_ROOT
    / "2. All Segmentation Groundtruths"
    / "a. Training Set"
)


# --------------------------------------------------
# Lesion folders
# --------------------------------------------------

LESIONS = {
    "Microaneurysms": (
        "1. Microaneurysms",
        "MA"
    ),
    "Haemorrhages": (
        "2. Haemorrhages",
        "HE"
    ),
    "Hard Exudates": (
        "3. Hard Exudates",
        "EX"
    ),
    "Soft Exudates": (
        "4. Soft Exudates",
        "SE"
    )
}


# --------------------------------------------------
# Load original image
# --------------------------------------------------

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise ValueError(
        f"Could not read image: {IMAGE_PATH}"
    )

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# Create visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Fundus")
axes[0, 0].axis("off")


# --------------------------------------------------
# Display lesion masks
# --------------------------------------------------

positions = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1)
]


for (lesion_name, (folder_name, suffix)), position in zip(
    LESIONS.items(),
    positions
):

    mask_path = (
        MASK_ROOT
        / folder_name
        / f"IDRiD_01_{suffix}.tif"
    )

    ax = axes[position]

    if not mask_path.exists():

        ax.text(
            0.5,
            0.5,
            "Not Annotated",
            ha="center",
            va="center",
            fontsize=16
        )

        ax.set_title(
            lesion_name
        )

        ax.axis("off")

        continue


    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    # Convert 0/76 → 0/1
    mask = mask > 0

    ax.imshow(
        mask,
        cmap="gray"
    )

    ax.set_title(
        lesion_name
    )

    ax.axis("off")


# --------------------------------------------------
# Empty sixth panel
# --------------------------------------------------

axes[1, 2].axis("off")


plt.tight_layout()


# --------------------------------------------------
# Save result
# --------------------------------------------------

output_dir = Path("results/idrid")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir
    / "idrid_sample_masks.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nVisualization saved to: {output_path}"
)