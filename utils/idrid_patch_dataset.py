from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class IDRiDPatchDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        split="a. Training Set",
        patch_size=512,
        stride=384,
        image_ids=None
    ):

        self.dataset_root = Path(dataset_root)
        self.split = split

        self.patch_size = patch_size
        self.stride = stride

        # -------------------------
        # Original images
        # -------------------------

        self.image_dir = (
            self.dataset_root
            / "1. Original Images"
            / split
        )

        # -------------------------
        # Ground-truth masks
        # -------------------------

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

        # -------------------------
        # Get all original images
        # -------------------------

        all_images = sorted(
            self.image_dir.glob("*.jpg")
        )

        # -------------------------
        # Optional image filtering
        # -------------------------

        if image_ids is not None:

            image_ids = set(image_ids)

            self.images = [
                image_path
                for image_path in all_images
                if image_path.stem in image_ids
            ]

        else:

            self.images = all_images

        # -------------------------
        # Create patch index
        # -------------------------

        self.patch_index = []

        for image_path in self.images:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR
            )

            if image is None:
                continue

            height, width = image.shape[:2]

            for y in range(
                0,
                height - patch_size + 1,
                stride
            ):

                for x in range(
                    0,
                    width - patch_size + 1,
                    stride
                ):

                    self.patch_index.append(
                        (
                            image_path,
                            x,
                            y
                        )
                    )

    def __len__(self):

        return len(self.patch_index)

    def _load_mask(
        self,
        image_id,
        lesion
    ):

        folder_name = self.lesion_folders[
            lesion
        ]

        mask_path = (
            self.mask_dir
            / folder_name
            / f"{image_id}_{lesion}.tif"
        )

        # Annotation doesn't exist
        if not mask_path.exists():

            return None

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        if mask is None:

            return None

        # IDRiD:
        # 0 = background
        # 76 = lesion
        #
        # Convert to:
        # 0 = background
        # 1 = lesion

        mask = (
            mask > 0
        ).astype(np.float32)

        return mask

    def __getitem__(self, index):

        image_path, x, y = (
            self.patch_index[index]
        )

        image_id = image_path.stem

        # -------------------------
        # Load original image
        # -------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------
        # Crop image patch
        # -------------------------

        image = image[
            y:y + self.patch_size,
            x:x + self.patch_size
        ]

        # Convert to 0-1
        image = (
            image.astype(np.float32)
            / 255.0
        )

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

        for lesion in [
            "MA",
            "HE",
            "EX",
            "SE"
        ]:

            mask = self._load_mask(
                image_id,
                lesion
            )

            if mask is None:

                # Missing annotation
                mask = np.zeros(
                    (
                        self.patch_size,
                        self.patch_size
                    ),
                    dtype=np.float32
                )

                valid = 0.0

            else:

                # Crop same location
                mask = mask[
                    y:y + self.patch_size,
                    x:x + self.patch_size
                ]

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
            "image_id": image_id,
            "x": x,
            "y": y
        }