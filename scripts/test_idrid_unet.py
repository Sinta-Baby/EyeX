import sys
from pathlib import Path

import torch


# --------------------------------------------------
# Allow Python to find EyeX folders
# --------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from models.idrid_unet import IDRiDUNet


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = IDRiDUNet(
    num_classes=4
)


# --------------------------------------------------
# Create a fake batch
# --------------------------------------------------

input_image = torch.randn(
    2,
    3,
    512,
    512
)


# --------------------------------------------------
# Run model
# --------------------------------------------------

output = model(
    input_image
)


# --------------------------------------------------
# Display shapes
# --------------------------------------------------

print("\nIDRiD U-NET TEST")
print("================")

print(
    "Input shape :",
    input_image.shape
)

print(
    "Output shape:",
    output.shape
)

print("\nU-Net test completed.")