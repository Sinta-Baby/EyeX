import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)

from core.constants import NUM_CLASSES


def build_model(pretrained=True):

    if pretrained:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
    else:
        weights = None

    model = convnext_tiny(
        weights=weights
    )

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model