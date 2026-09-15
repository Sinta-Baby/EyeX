from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class IDRiDDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        split="a. Training Set",
        image_size=(512, 512)
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size

        # Original fundus images
        self.image_dir = (
            self.dataset_root
            / "1. Original Images"
            / split
        )

        # Ground-truth masks
        self.mask_dir = (
            self.dataset_root
            / "2. All Segmentation Groundtruths"
            / split
        )

        self.lesion_folders = {
            "MA": "1. Microaneurysms",
            "HE": "2. Haemorrhages",
            "EX": "3. Hard Exudates",
            "SE": "4. Soft Exudates",
        }

        # Get all original images
        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

    def __len__(self):
        return len(self.images)

    def _load_mask(self, image_id, lesion):

        folder_name = self.lesion_folders[lesion]

        mask_path = (
            self.mask_dir
            / folder_name
            / f"{image_id}_{lesion}.tif"
        )

        # Some lesions don't have annotations
        if not mask_path.exists():
            return None

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        if mask is None:
            return None

        # IDRiD uses 0 for background
        # and 76 for lesion pixels.
        mask = (mask > 0).astype(np.float32)

        return mask

    def __getitem__(self, index):

        image_path = self.images[index]

        image_id = image_path.stem

        # -------------------------
        # Load fundus image
        # -------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # OpenCV BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------
        # Resize image
        # -------------------------

        width, height = self.image_size

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        # Convert 0-255 → 0-1
        image = image.astype(
            np.float32
        ) / 255.0

        # HWC → CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # -------------------------
        # Load lesion masks
        # -------------------------

        masks = []
        valid_masks = []

        for lesion in ["MA", "HE", "EX", "SE"]:

            mask = self._load_mask(
                image_id,
                lesion
            )

            if mask is None:

                # No annotation available
                mask = np.zeros(
                    (height, width),
                    dtype=np.float32
                )

                valid = 0.0

            else:

                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

                valid = 1.0

            masks.append(mask)
            valid_masks.append(valid)

        masks = np.stack(
            masks,
            axis=0
        )

        masks = torch.tensor(
            masks,
            dtype=torch.float32
        )

        valid_masks = torch.tensor(
            valid_masks,
            dtype=torch.float32
        )

        return {
            "image": image,
            "mask": masks,
            "valid": valid_masks,
            "image_id": image_id
        }