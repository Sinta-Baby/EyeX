"""
==========================================================
RetinaSense

EfficientNet-B3 Model

Description:
Creates the EfficientNet-B3 model for RetinaSense.

==========================================================
"""

import torch.nn as nn

from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights
)

from core.constants import NUM_CLASSES


def build_model(pretrained=True):
    """
    Builds EfficientNet-B3.
    """

    if pretrained:

        weights = EfficientNet_B3_Weights.DEFAULT

    else:

        weights = None

    model = efficientnet_b3(
        weights=weights
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model