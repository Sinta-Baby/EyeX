"""
==========================================================
RetinaSense

DataLoader

Description:
Creates PyTorch DataLoaders for
training, validation and testing.

==========================================================
"""

import torch

from torch.utils.data import DataLoader

from core.constants import (
    BATCH_SIZE,
    NUM_WORKERS
)

from utils.dataset import RetinaDataset

from utils.transforms import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms
)


# =====================================================
# DataLoader Builder
# =====================================================

def get_dataloaders(dataset_root):
    """
    Returns train, validation and test dataloaders.
    """

    train_dataset = RetinaDataset(
        metadata_file=f"{dataset_root}/metadata/train_metadata.csv",
        dataset_root=dataset_root,
        split="train",
        transforms=get_train_transforms()
    )

    validation_dataset = RetinaDataset(
        metadata_file=f"{dataset_root}/metadata/validation_metadata.csv",
        dataset_root=dataset_root,
        split="validation",
        transforms=get_validation_transforms()
    )

    test_dataset = RetinaDataset(
        metadata_file=f"{dataset_root}/metadata/test_metadata.csv",
        dataset_root=dataset_root,
        split="test",
        transforms=get_test_transforms()
    )

    use_cuda = torch.cuda.is_available()

    common_loader_args = {

        "batch_size": BATCH_SIZE,

        "num_workers": NUM_WORKERS,

        "pin_memory": use_cuda,

        "persistent_workers":
            NUM_WORKERS > 0

    }

    train_loader = DataLoader(

        train_dataset,

        shuffle=True,

        drop_last=True,

        **common_loader_args

    )

    validation_loader = DataLoader(

        validation_dataset,

        shuffle=False,

        drop_last=False,

        **common_loader_args

    )

    test_loader = DataLoader(

        test_dataset,

        shuffle=False,

        drop_last=False,

        **common_loader_args

    )

    print("\n" + "=" * 70)

    print("DataLoader Summary")

    print("=" * 70)

    print(f"Train Images      : {len(train_dataset)}")

    print(f"Validation Images : {len(validation_dataset)}")

    print(f"Test Images       : {len(test_dataset)}")

    print(f"Batch Size        : {BATCH_SIZE}")

    print(f"Workers           : {NUM_WORKERS}")

    print(f"CUDA              : {use_cuda}")

    print("=" * 70)

    return (

        train_loader,

        validation_loader,

        test_loader

    )