import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.idrid_patch_dataset import IDRiDPatchDataset


DATASET_ROOT = Path(
    "datasets/IDRiD/IDRiD_Segmentation"
)


dataset = IDRiDPatchDataset(
    dataset_root=DATASET_ROOT,
    split="a. Training Set",
    patch_size=512,
    stride=384
)


print("\nIDRiD PATCH DISTRIBUTION ANALYSIS")
print("=================================")

print("Original images:", len(dataset.images))
print("Total patches:", len(dataset))


background_patches = 0
retina_patches = 0
lesion_patches = 0

lesion_counts = {
    "MA": 0,
    "HE": 0,
    "EX": 0,
    "SE": 0
}


# --------------------------------------------------
# Cache images and masks
# --------------------------------------------------

image_cache = {}
mask_cache = {}


# --------------------------------------------------
# Analyze the EXACT patches used by the dataset
# --------------------------------------------------

for image_path, x, y in dataset.patch_index:

    image_id = image_path.stem

    # ----------------------------------------------
    # Load image only once
    # ----------------------------------------------

    if image_id not in image_cache:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            continue

        image_cache[image_id] = image

    image = image_cache[image_id]

    # ----------------------------------------------
    # Crop patch
    # ----------------------------------------------

    patch = image[
        y:y + dataset.patch_size,
        x:x + dataset.patch_size
    ]

    if patch.shape[:2] != (
        dataset.patch_size,
        dataset.patch_size
    ):
        continue

    # ----------------------------------------------
    # Background / retina check
    # ----------------------------------------------

    patch_float = (
        patch.astype(np.float32) / 255.0
    )

    mean_pixel = patch_float.mean()

    if mean_pixel < 0.03:

        background_patches += 1

    else:

        retina_patches += 1

    # ----------------------------------------------
    # Load masks only once per image
    # ----------------------------------------------

    if image_id not in mask_cache:

        mask_cache[image_id] = {}

        for lesion in [
            "MA",
            "HE",
            "EX",
            "SE"
        ]:

            mask = dataset._load_mask(
                image_id,
                lesion
            )

            mask_cache[
                image_id
            ][lesion] = mask

    masks = mask_cache[image_id]

    # ----------------------------------------------
    # Check lesions
    # ----------------------------------------------

    patch_has_lesion = False

    for lesion in [
        "MA",
        "HE",
        "EX",
        "SE"
    ]:

        mask = masks[lesion]

        # Missing annotation
        if mask is None:
            continue

        mask_patch = mask[
            y:y + dataset.patch_size,
            x:x + dataset.patch_size
        ]

        if np.any(mask_patch > 0):

            lesion_counts[
                lesion
            ] += 1

            patch_has_lesion = True

    if patch_has_lesion:

        lesion_patches += 1


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nPatch distribution")
print("------------------")

print(
    "Mostly background patches:",
    background_patches
)

print(
    "Useful retina patches:",
    retina_patches
)

print(
    "Total:",
    background_patches + retina_patches
)

print(
    "Patches containing at least one lesion:",
    lesion_patches
)


print("\nLesion-containing patches")
print("-------------------------")

for lesion, count in lesion_counts.items():

    print(
        f"{lesion}: {count}"
    )


print("\nAnalysis completed.")