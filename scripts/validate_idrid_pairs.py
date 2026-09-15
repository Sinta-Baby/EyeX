from pathlib import Path
import cv2
import numpy as np


DATASET_ROOT = Path("datasets/IDRiD/IDRiD_Segmentation")

IMAGE_ROOT = (
    DATASET_ROOT
    / "1. Original Images"
    / "a. Training Set"
)

MASK_ROOT = (
    DATASET_ROOT
    / "2. All Segmentation Groundtruths"
    / "a. Training Set"
)


LESIONS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}


print("\nIDRiD IMAGE-MASK VALIDATION")
print("===========================")


# Check first 5 images
image_files = sorted(IMAGE_ROOT.glob("*.jpg"))[:5]


for image_path in image_files:

    image_id = image_path.stem

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    print("\n" + "-" * 50)
    print("Image:", image_path.name)

    if image is None:
        print("ERROR: Image could not be read")
        continue

    image_height, image_width = image.shape[:2]

    print(
        "Image size:",
        image_width,
        "x",
        image_height
    )

    for lesion, folder_name in LESIONS.items():

        mask_path = (
            MASK_ROOT
            / folder_name
            / f"{image_id}_{lesion}.tif"
        )

        if not mask_path.exists():

            print(
                f"{lesion}: mask not available"
            )

            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        if mask is None:

            print(
                f"{lesion}: ERROR reading mask"
            )

            continue

        mask_height, mask_width = mask.shape[:2]

        same_size = (
            image_width == mask_width
            and image_height == mask_height
        )

        lesion_pixels = np.sum(mask > 0)

        print(
            f"{lesion}: "
            f"mask={mask_width}x{mask_height} | "
            f"same size={same_size} | "
            f"lesion pixels={lesion_pixels}"
        )


print("\nValidation completed.")