import torch.nn as nn
from torchvision.models import (
    swin_t,
    Swin_T_Weights
)

from core.constants import NUM_CLASSES


def build_model(pretrained=True):

    if pretrained:
        weights = Swin_T_Weights.DEFAULT
    else:
        weights = None

    model = swin_t(
        weights=weights
    )

    in_features = model.head.in_features

    model.head = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model