"""
==========================================================
RetinaSense

Image Transformations

Description:
Defines image preprocessing and augmentation
pipelines for training, validation, and testing.

Framework:
Albumentations

==========================================================
"""

import albumentations as A

from albumentations.pytorch import ToTensorV2

from core.constants import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD
)


# =====================================================
# Training Transform
# =====================================================

def get_train_transforms():
    """
    Returns image augmentation pipeline
    used during training.
    """

    return A.Compose([

        A.Resize(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1]
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.Rotate(
            limit=10,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2()

    ])


# =====================================================
# Validation Transform
# =====================================================

def get_validation_transforms():
    """
    Returns preprocessing pipeline
    used during validation.
    """

    return A.Compose([

        A.Resize(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1]
        ),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2()

    ])


# =====================================================
# Test Transform
# =====================================================

def get_test_transforms():
    """
    Returns preprocessing pipeline
    used during testing.
    """

    return A.Compose([

        A.Resize(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1]
        ),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2()

    ])