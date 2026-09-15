from pathlib import Path


# --------------------------------------------------
# IDRiD dataset location
# --------------------------------------------------

DATASET_ROOT = Path("datasets/IDRiD/IDRiD_Segmentation")

IMAGE_ROOT = DATASET_ROOT / "1. Original Images"
MASK_ROOT = DATASET_ROOT / "2. All Segmentation Groundtruths"


LESIONS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc"
}


def get_image_ids(folder):
    """Get image IDs from JPG files."""
    
    image_ids = set()

    for file in folder.glob("*.jpg"):
        image_ids.add(file.stem)

    return image_ids


def get_mask_ids(folder, suffix):
    """Get image IDs from mask files."""
    
    mask_ids = set()

    for file in folder.glob("*.tif"):
        name = file.stem

        if name.endswith(suffix):
            image_id = name.replace(suffix, "")
            mask_ids.add(image_id)

    return mask_ids


def check_split(split_name):
    
    print("\n" + "=" * 50)
    print(split_name.upper())
    print("=" * 50)

    image_folder = IMAGE_ROOT / split_name

    # Get original images
    image_ids = get_image_ids(image_folder)

    print("Original images :", len(image_ids))

    for lesion, folder_name in LESIONS.items():

        mask_folder = MASK_ROOT / split_name / folder_name

        mask_ids = get_mask_ids(
            mask_folder,
            "_" + lesion
        )

        matched = image_ids.intersection(mask_ids)
        missing = image_ids - mask_ids

        print(
            f"{lesion:3} masks        : {len(mask_ids):3} "
            f"| matched: {len(matched):3}"
        )

        if missing:
            print(
                f"     Missing masks : {sorted(missing)}"
            )


# --------------------------------------------------
# Run checks
# --------------------------------------------------

print("\nIDRiD DATASET VERIFICATION")
print("===========================")

check_split("a. Training Set")
check_split("b. Testing Set")

print("\nVerification completed.")