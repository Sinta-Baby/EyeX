"""
==========================================================
RetinaSense

Single Image Prediction

Description:
Predicts the disease class for one fundus image
using the trained EfficientNet-B3 model and
generates a Grad-CAM visualization.

==========================================================
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.efficientnet_b3 import build_model

from core.constants import (
    INDEX_TO_DISEASE,
    IMAGE_SIZE
)

from utils.transforms import get_test_transforms

from inference.gradcam import GradCAM


# ======================================================
# Predict Image
# ======================================================

def predict_image(image_path):

    # ------------------------------------------
    # Device
    # ------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------
    # Check Image
    # ------------------------------------------

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # ------------------------------------------
    # Read Image
    # ------------------------------------------

    original_image = cv2.imread(
        str(image_path)
    )

    if original_image is None:

        raise ValueError(
            "Unable to read image."
        )

    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------
    # Preprocessing
    # ------------------------------------------

    transforms = get_test_transforms()

    transformed = transforms(
        image=original_image
    )

    image_tensor = transformed[
        "image"
    ]

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)

    # ------------------------------------------
    # Build Model
    # ------------------------------------------

    model = build_model(
        pretrained=False
    )

    # ------------------------------------------
    # Load Trained Model
    # ------------------------------------------

    model_path = (
        "saved_models/best_model.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)

    model.eval()

    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    disease = INDEX_TO_DISEASE[
        predicted_class
    ]

    # ------------------------------------------
    # Grad-CAM
    # ------------------------------------------

    target_layer = model.features[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    cam = gradcam.generate(
        image_tensor,
        predicted_class
    )

    gradcam.close()

    # ------------------------------------------
    # Create Heatmap
    # ------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------
    # Resize Original Image
    # ------------------------------------------

    display_image = cv2.resize(
        original_image,
        (
            IMAGE_SIZE[1],
            IMAGE_SIZE[0]
        )
    )

    # ------------------------------------------
    # Create Overlay
    # ------------------------------------------

    overlay = cv2.addWeighted(
        display_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    # ------------------------------------------
    # Return Results
    # ------------------------------------------

    return {
        "disease": disease,
        "confidence": confidence,
        "original_image": display_image,
        "heatmap": heatmap,
        "overlay": overlay
    }


# ======================================================
# Test Prediction
# ======================================================

if __name__ == "__main__":

    image_path = (
        "datasets/RetinaSense_Dataset/test/AMD/"
        "ARMD_AMD_1016_img_1255.jpg"
    )

    result = predict_image(
        image_path
    )

    print("=" * 70)
    print("RetinaSense Prediction")
    print("=" * 70)

    print(
        f"Prediction : "
        f"{result['disease']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print("=" * 70)