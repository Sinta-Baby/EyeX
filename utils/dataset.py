"""
==========================================================
RetinaSense

Custom Dataset

Description:
PyTorch Dataset class for loading retinal fundus images
using metadata CSV files.

==========================================================
"""

from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.constants import DISEASE_TO_INDEX


class RetinaDataset(Dataset):
    """
    Custom Dataset for RetinaSense.
    """

    def __init__(
        self,
        metadata_file: str,
        dataset_root: str,
        split: str,
        transforms: Optional[object] = None
    ):

        self.metadata = pd.read_csv(metadata_file)

        self.dataset_root = Path(dataset_root)

        self.split = split

        self.transforms = transforms

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        filename = row["filename"]

        disease = row["disease"]

        image_path = (
            self.dataset_root /
            self.split /
            disease /
            filename
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transforms:

            image = self.transforms(
                image=image
            )["image"]

        label = DISEASE_TO_INDEX[disease]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label